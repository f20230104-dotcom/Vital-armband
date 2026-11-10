#!/usr/bin/env python3
"""
device_simulator.py - A virtual wristband that speaks the same HTTP/JSON protocol as the
ESP32 firmware (src/tasks/alert_task.cpp). Used for the hosted demo and for testing the
dashboard without hardware. It reproduces the firmware's behaviour at a high level:

  * TELEMETRY heartbeat every `period` seconds (HR / SpO2 / RR / battery)
  * scripted scenarios: fall (with tachycardic response), SOS button, medication
    reminder -> taken or missed, sustained abnormal vitals, low battery

Run standalone against a server:
    python dashboard/device_simulator.py --url http://127.0.0.1:8000 --device WB-01
"""
from __future__ import annotations

import argparse
import math
import random
import threading
import time
from datetime import datetime, timezone

import requests


class VirtualWristband:
    def __init__(self, device_id: str = "WB-SIM", seed: int | None = None):
        self.device_id = device_id
        self.rng = random.Random(seed)
        self.t0 = time.time()
        self.seq = 0
        self.hr, self.spo2, self.rr = 72.0, 97.5, 15.0
        self.battery_v = 4.10
        self.stress = 0.0                     # >0 after a fall / during abnormal episode
        self.hypoxia = 0.0

    # ---- physiology-ish random walk -------------------------------------------------
    def step(self, dt: float) -> None:
        r = self.rng
        circadian = 3 * math.sin((time.time() - self.t0) / 600)
        hr_target = 70 + circadian + 45 * self.stress
        self.hr += (hr_target - self.hr) * 0.15 + r.gauss(0, 1.2)
        spo2_target = 97.5 - 9 * self.hypoxia
        self.spo2 += (spo2_target - self.spo2) * 0.2 + r.gauss(0, 0.25)
        self.spo2 = min(100.0, self.spo2)
        rr_target = 15 + 9 * self.stress + 6 * self.hypoxia
        self.rr += (rr_target - self.rr) * 0.15 + r.gauss(0, 0.3)
        self.stress = max(0.0, self.stress - dt / 120)
        self.hypoxia = max(0.0, self.hypoxia - dt / 90)
        self.battery_v = max(3.30, self.battery_v - dt * 2.5e-5 + r.gauss(0, 0.002))

    def battery_pct(self) -> int:
        # LiPo 1S approximate curve 3.3 V -> 0 %, 4.2 V -> 100 %
        return int(max(0, min(100, (self.battery_v - 3.3) / 0.9 * 100)))

    def base(self, event: str, priority: int = 0) -> dict:
        self.seq += 1
        return {
            "device_id": self.device_id, "event": event, "priority": priority,
            "hr": round(self.hr), "spo2": round(self.spo2, 1), "rr": round(self.rr, 1),
            "battery_v": round(self.battery_v, 2), "battery_pct": self.battery_pct(),
            "seq": self.seq, "uptime_s": int(time.time() - self.t0),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sim": True,
        }

    def event_payload(self, event: str) -> dict:
        r = self.rng
        if event == "FALL":
            self.stress = 1.0
            p = self.base("FALL", 3)
            p.update(fall_prob=round(r.uniform(0.86, 0.99), 3), peak_g=round(r.uniform(2.8, 6.5), 2),
                     hr=round(self.hr + r.uniform(15, 35)),
                     detail="Free-fall %.0f ms -> impact, no response to countdown" % r.uniform(180, 450))
        elif event == "SOS":
            self.stress = max(self.stress, 0.6)
            p = self.base("SOS", 3)
            p["detail"] = "SOS button held 2 s"
        elif event == "MED_MISSED":
            p = self.base("MED_MISSED", 2)
            p["detail"] = r.choice(["08:00 Metformin 500 mg", "13:00 Amlodipine 5 mg", "21:00 Atorvastatin 20 mg"]) \
                + " not confirmed within 5 min"
        elif event == "MED_TAKEN":
            p = self.base("MED_TAKEN", 1)
            p["detail"] = "Medication confirmed by button press"
        elif event == "VITALS_ABNORMAL":
            self.hypoxia = 1.0
            p = self.base("VITALS_ABNORMAL", 2)
            p.update(spo2=round(r.uniform(86, 89.5), 1), hr=round(r.uniform(118, 135)),
                     detail="SpO2 < 90 % for > 30 s")
        elif event == "LOW_BATTERY":
            p = self.base("LOW_BATTERY", 1)
            p.update(battery_v=3.42, battery_pct=13, detail="Battery below 15 %")
        else:
            p = self.base(event)
        return p


def run(url: str, device_id: str = "WB-SIM", period: float = 3.0, scenario_every: tuple = (40, 90),
        api_key: str = "", stop: threading.Event | None = None) -> None:
    dev = VirtualWristband(device_id)
    headers = {"X-API-Key": api_key} if api_key else {}
    endpoint = url.rstrip("/") + "/api/telemetry"
    next_event = time.time() + 15
    first = True
    while not (stop and stop.is_set()):
        dev.step(period)
        msgs = [dev.base("BOOT", 1) | {"detail": "simulated firmware v1.0"}] if first else []
        first = False
        msgs.append(dev.base("TELEMETRY"))
        if time.time() >= next_event:
            ev = dev.rng.choices(["FALL", "SOS", "MED_MISSED", "VITALS_ABNORMAL", "MED_TAKEN", "LOW_BATTERY"],
                                 weights=[4, 2, 3, 2, 3, 1])[0]
            msgs.append(dev.event_payload(ev))
            next_event = time.time() + dev.rng.uniform(*scenario_every)
        for m in msgs:
            try:
                requests.post(endpoint, json=m, headers=headers, timeout=3)
            except requests.RequestException:
                pass                      # server not up yet - keep going, like the firmware's retry queue
        time.sleep(period)


def start_background(url: str, device_id: str = "WB-SIM", api_key: str = "") -> threading.Thread:
    t = threading.Thread(target=run, kwargs=dict(url=url, device_id=device_id, api_key=api_key), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--device", default="WB-SIM")
    ap.add_argument("--period", type=float, default=3.0)
    a = ap.parse_args()
    print(f"[sim] {a.device} -> {a.url}  (Ctrl+C to stop)")
    run(a.url, a.device, a.period)
