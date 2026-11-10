#!/usr/bin/env python3
"""
app.py - Gradio command-centre dashboard for the Vital wristband.

It polls the webhook receiver (server.py) every couple of seconds and shows live vitals,
trends and a flashing alert log. Modes:

    python dashboard/app.py                  UI on :7860, polls http://127.0.0.1:8000
                                             (starts server.py in-process if nothing is listening)
    python dashboard/app.py --simulate       ...and run a virtual wristband (no hardware needed)
    python dashboard/app.py --share          ...and create a public *.gradio.live link
    python dashboard/app.py --combined       ONE port for UI + webhook (/api/telemetry) - used
                                             for Hugging Face Spaces / Render / Docker deploys

Environment: WRISTBAND_SERVER_URL (default http://127.0.0.1:8000), WRISTBAND_API_KEY.
"""
from __future__ import annotations

import argparse
import html
import inspect
import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

SERVER_URL = os.environ.get("WRISTBAND_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
POLL_S = 2.0
CRITICAL = {"FALL", "SOS", "MED_MISSED", "VITALS_ABNORMAL"}
EVENT_META = {  # icon, label, css class
    "FALL": ("🚨", "Fall detected", "crit"),
    "SOS": ("🆘", "SOS button", "crit"),
    "MED_MISSED": ("💊", "Medication missed", "crit"),
    "VITALS_ABNORMAL": ("❤️‍🩹", "Abnormal vitals", "crit"),
    "LOW_BATTERY": ("🪫", "Low battery", "warn"),
    "MED_REMINDER": ("⏰", "Medication reminder", "info"),
    "MED_TAKEN": ("✅", "Medication taken", "ok"),
    "FALL_CANCELLED": ("👍", "Fall cancelled by user", "ok"),
    "BOOT": ("🔌", "Device boot", "info"),
}

CSS = """
.vw-banner{padding:14px 18px;border-radius:12px;font-weight:700;font-size:1.05rem;display:flex;gap:10px;align-items:center}
.vw-banner.ok{background:#e7f6ec;color:#11633a;border:1px solid #a8dcb9}
.vw-banner.crit{background:#c62828;color:#fff;animation:vwflash 1s steps(2,start) infinite}
.vw-banner.off{background:#eceff1;color:#455a64;border:1px solid #cfd8dc}
@keyframes vwflash{0%{background:#c62828}50%{background:#ff5252}100%{background:#c62828}}
.vw-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.vw-card{border-radius:14px;padding:14px 16px;border:1px solid var(--border-color-primary);background:var(--block-background-fill)}
.vw-card .lbl{font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;opacity:.7}
.vw-card .val{font-size:2.1rem;font-weight:800;line-height:1.15;font-variant-numeric:tabular-nums}
.vw-card .unit{font-size:.95rem;font-weight:600;opacity:.65;margin-left:4px}
.vw-card .sub{font-size:.8rem;opacity:.7}
.vw-card.normal{border-left:6px solid #2e7d32}.vw-card.warn{border-left:6px solid #f9a825}
.vw-card.crit{border-left:6px solid #c62828;background:rgba(198,40,40,.08)}.vw-card.na{border-left:6px solid #90a4ae}
.vw-log{width:100%;border-collapse:collapse;font-size:.92rem}
.vw-log th{text-align:left;padding:6px 8px;border-bottom:2px solid var(--border-color-primary);opacity:.75}
.vw-log td{padding:7px 8px;border-bottom:1px solid var(--border-color-primary);vertical-align:top}
.vw-log tr.crit.open td{background:rgba(198,40,40,.14);animation:vwrow 1.2s ease-in-out infinite}
.vw-log tr.crit.acked td{opacity:.6}
.vw-log tr.warn td{background:rgba(249,168,37,.12)}
.vw-log tr.ok td{background:rgba(46,125,50,.07)}
@keyframes vwrow{0%,100%{background:rgba(198,40,40,.10)}50%{background:rgba(198,40,40,.32)}}
.vw-pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.75rem;font-weight:700}
.vw-pill.open{background:#c62828;color:#fff}.vw-pill.acked{background:#90a4ae;color:#fff}
"""


# Gradio 6 moved css/theme from gr.Blocks() to launch()/mount_gradio_app(); support both.
_STYLE_IN_BLOCKS = "css" in inspect.signature(gr.Blocks.__init__).parameters


def _style() -> dict:
    return {"css": CSS, "theme": gr.themes.Soft(primary_hue="red", neutral_hue="slate")}


# ----------------------------------------------------------------------------- server plumbing
def _get(path: str, **params):
    try:
        r = requests.get(SERVER_URL + path, params=params, timeout=2)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def _post(path: str, **params):
    try:
        r = requests.post(SERVER_URL + path, params=params, timeout=3)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_server(simulate: bool) -> None:
    """Start server.py in a background thread if nothing is listening on SERVER_URL."""
    if _get("/health"):
        print(f"[app] using existing receiver at {SERVER_URL}")
        return
    import uvicorn
    import server
    port = int(SERVER_URL.rsplit(":", 1)[-1])
    cfg = uvicorn.Config(server.create_app(simulate, SERVER_URL), host="0.0.0.0", port=port, log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    for _ in range(50):
        if _port_open(port):
            break
        time.sleep(0.1)
    print(f"[app] started embedded receiver on :{port} (simulate={simulate})")


# ----------------------------------------------------------------------------- rendering
def _cls(v, lo_warn, hi_warn, lo_crit, hi_crit):
    if v is None:
        return "na"
    if v < lo_crit or v > hi_crit:
        return "crit"
    if v < lo_warn or v > hi_warn:
        return "warn"
    return "normal"


def _card(label, value, unit, cls, sub=""):
    v = "--" if value is None else value
    return (f'<div class="vw-card {cls}"><div class="lbl">{label}</div>'
            f'<div class="val">{v}<span class="unit">{unit}</span></div><div class="sub">{sub}</div></div>')


def render_cards(s: dict) -> str:
    if not s:
        return '<div class="vw-cards">' + "".join(
            _card(n, None, u, "na", "waiting for device") for n, u in
            (("Heart rate", "bpm"), ("SpO₂", "%"), ("Resp. rate", "br/min"), ("Battery", "V"))) + "</div>"
    hr, spo2, rr, bv, bp = (s.get(k) for k in ("hr", "spo2", "rr", "battery_v", "battery_pct"))
    gate = "motion gate active" if s.get("motion") else "at rest"
    return '<div class="vw-cards">' + "".join([
        _card("Heart rate", None if hr is None else f"{hr:.0f}", "bpm", _cls(hr, 50, 110, 40, 130), gate),
        _card("SpO₂", None if spo2 is None else f"{spo2:.1f}", "%", _cls(spo2, 94, 100.1, 90, 101), "pulse oximetry"),
        _card("Resp. rate", None if rr is None else f"{rr:.0f}", "br/min", _cls(rr, 10, 22, 8, 28),
              "PPG 0.1–0.5 Hz band"),
        _card("Battery", None if bv is None else f"{bv:.2f}", "V",
              _cls(bp if bp is not None else 50, 20, 101, 10, 101), "" if bp is None else f"{bp} %"),
    ]) + "</div>"


def render_banner(s: dict, alerts: list) -> str:
    if s is None:
        return f'<div class="vw-banner off">⚠️ Receiver offline ({html.escape(SERVER_URL)})</div>'
    if not s:
        return '<div class="vw-banner off">⌛ No data yet from this device</div>'
    age = time.time() - s["last_seen"]
    open_crit = [a for a in alerts if a["event"] in CRITICAL and not a["acked"]]
    if open_crit:
        a = open_crit[0]
        icon, label, _ = EVENT_META.get(a["event"], ("⚠️", a["event"], "crit"))
        extra = f" (+{len(open_crit) - 1} more)" if len(open_crit) > 1 else ""
        return (f'<div class="vw-banner crit">{icon} {label.upper()} — {html.escape(a["device_id"])} at '
                f'{datetime.fromtimestamp(a["received"]).strftime("%H:%M:%S")}{extra}. Acknowledge below.</div>')
    if age > 120:
        return f'<div class="vw-banner off">📡 Device silent for {age / 60:.0f} min (WiFi is duty-cycled)</div>'
    sim = " · simulated" if s.get("sim") else ""
    return (f'<div class="vw-banner ok">✅ {html.escape(s["device_id"])} OK — last update {age:.0f}s ago'
            f'{sim}</div>')


def render_log(alerts: list) -> str:
    if not alerts:
        return "<p style='opacity:.7'>No events yet.</p>"
    rows = []
    for a in alerts:
        icon, label, cls = EVENT_META.get(a["event"], ("•", a["event"], "info"))
        state = ""
        if a["event"] in CRITICAL:
            state = '<span class="vw-pill acked">ACK</span>' if a["acked"] else '<span class="vw-pill open">OPEN</span>'
        vit = " · ".join(x for x in (
            f"HR {a['hr']:.0f}" if a["hr"] is not None else "",
            f"SpO₂ {a['spo2']:.1f}" if a["spo2"] is not None else "",
            f"RR {a['rr']:.0f}" if a["rr"] is not None else "",
            f"p(fall) {a['fall_prob']:.2f}" if a["fall_prob"] is not None else "",
            f"peak {a['peak_g']:.1f} g" if a["peak_g"] is not None else "") if x)
        rows.append(
            f'<tr class="{cls} {"acked" if a["acked"] else "open"}"><td>#{a["id"]}</td>'
            f'<td>{datetime.fromtimestamp(a["received"]).strftime("%H:%M:%S")}</td>'
            f'<td>{icon} <b>{label}</b> {state}</td><td>{html.escape(a["device_id"])}</td>'
            f'<td>{vit}<br><small>{html.escape(a["detail"] or "")}</small></td></tr>')
    return ('<table class="vw-log"><thead><tr><th>ID</th><th>Time</th><th>Event</th><th>Device</th>'
            '<th>Details</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def history_frames(dev: str, minutes: float):
    rows = _get("/api/history", device_id=dev, minutes=minutes) or []
    df = pd.DataFrame(rows, columns=["received", "hr", "spo2", "rr", "battery_v", "battery_pct", "event"])
    df["time"] = pd.to_datetime(df["received"], unit="s", utc=True).dt.tz_convert(None)
    def one(col):
        d = df[["time", col]].dropna()
        return d.rename(columns={col: "value"})
    return one("hr"), one("spo2"), one("rr")


# ----------------------------------------------------------------------------- UI
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Vital Wristband · Command Center", **(_style() if _STYLE_IN_BLOCKS else {})) as demo:
        gr.Markdown("# 🩺 Vital Wristband — Command Center\n"
                    "Live fall detection, vitals and medication adherence from ESP32 wristbands.")
        with gr.Row():
            device = gr.Dropdown(label="Device", choices=[], value=None, allow_custom_value=True, scale=2)
            window = gr.Radio(["5", "15", "60"], value="15", label="Trend window (min)", scale=2)
            status = gr.Markdown("", elem_id="vw-status")
        banner = gr.HTML()
        cards = gr.HTML(render_cards({}))
        with gr.Row():
            hr_plot = gr.LinePlot(x="time", y="value", title="Heart rate (bpm)", height=220, y_lim=[40, 160])
            spo2_plot = gr.LinePlot(x="time", y="value", title="SpO₂ (%)", height=220, y_lim=[84, 100])
            rr_plot = gr.LinePlot(x="time", y="value", title="Respiratory rate (br/min)", height=220, y_lim=[5, 35])
        gr.Markdown("## 🔔 Alert log")
        with gr.Row():
            ack_id = gr.Number(label="Alert ID", precision=0, scale=1)
            ack_btn = gr.Button("Acknowledge", variant="primary", scale=1)
            ack_all = gr.Button("Acknowledge all open", scale=1)
        log = gr.HTML()
        with gr.Accordion("🧪 Demo / test controls (inject events as the selected device)", open=False):
            with gr.Row():
                sim_buttons = {ev: gr.Button(lbl) for ev, lbl in (
                    ("FALL", "Simulate fall"), ("SOS", "SOS press"), ("MED_MISSED", "Missed medication"),
                    ("VITALS_ABNORMAL", "Abnormal vitals"), ("MED_TAKEN", "Medication taken"))}
            gr.Markdown(f"Webhook endpoint for the ESP32: `POST {SERVER_URL}/api/telemetry` "
                        "(set `WEBHOOK_URL` in `include/config.h` to your PC's LAN IP).")

        def refresh(dev, win):
            devs = _get("/api/devices")
            if devs is None:
                return (gr.update(), render_banner(None, []), render_cards({}), None, None, None,
                        render_log([]), "🔴 receiver offline")
            names = [d["device_id"] for d in devs]
            if not dev and names:
                dev = names[0]
            snap = _get("/api/latest", device_id=dev) if dev else {}
            alerts = _get("/api/alerts", limit=40) or []
            hr, sp, rr = history_frames(dev, float(win)) if dev else (None, None, None)
            return (gr.update(choices=names, value=dev), render_banner(snap or {}, alerts), render_cards(snap or {}),
                    hr, sp, rr, render_log(alerts),
                    f"🟢 receiver online · {len(names)} device(s) · {datetime.now().strftime('%H:%M:%S')}")

        outs = [device, banner, cards, hr_plot, spo2_plot, rr_plot, log, status]
        timer = gr.Timer(POLL_S)
        timer.tick(refresh, [device, window], outs, show_progress="hidden")
        demo.load(refresh, [device, window], outs)
        device.change(refresh, [device, window], outs, show_progress="hidden")
        window.change(refresh, [device, window], outs, show_progress="hidden")

        def do_ack(i, dev, win):
            if i:
                _post(f"/api/alerts/{int(i)}/ack")
            return refresh(dev, win)

        def do_ack_all(dev, win):
            for a in _get("/api/alerts", limit=200) or []:
                if a["event"] in CRITICAL and not a["acked"]:
                    _post(f"/api/alerts/{a['id']}/ack")
            return refresh(dev, win)

        ack_btn.click(do_ack, [ack_id, device, window], outs)
        ack_all.click(do_ack_all, [device, window], outs)
        for ev, btn in sim_buttons.items():
            btn.click(lambda dev, win, ev=ev: (_post(f"/api/simulate/{ev}", device_id=dev or "WB-SIM"),
                                               refresh(dev or "WB-SIM", win))[1],
                      [device, window], outs)
    return demo


def main() -> None:
    global SERVER_URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--simulate", action="store_true", help="run a virtual wristband in the embedded receiver")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--combined", action="store_true", help="serve UI + webhook API on one port")
    a = ap.parse_args()
    on_space = bool(os.environ.get("SPACE_ID"))
    if on_space:                                   # Hugging Face Spaces: one port, always simulate
        a.combined, a.simulate = True, True

    if a.combined:
        import uvicorn
        import server
        SERVER_URL = f"http://127.0.0.1:{a.port}"
        api = server.create_app(a.simulate, SERVER_URL)
        app = gr.mount_gradio_app(api, build_ui(), path="/", **({} if _STYLE_IN_BLOCKS else _style()))
        print(f"[app] combined mode: UI + webhook on http://0.0.0.0:{a.port}  (POST /api/telemetry)")
        uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")
    else:
        ensure_server(a.simulate)
        build_ui().queue().launch(server_name="0.0.0.0", server_port=a.port, share=a.share,
                                  **({} if _STYLE_IN_BLOCKS else _style()))


if __name__ == "__main__":
    main()
