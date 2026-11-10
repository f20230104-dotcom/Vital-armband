#!/usr/bin/env python3
"""
server.py - Webhook receiver for the Vital wristband.

The ESP32 alert_task POSTs JSON to /api/telemetry, e.g.

    {"device_id": "WB-01", "event": "FALL", "priority": 3, "hr": 112, "spo2": 95,
     "rr": 18, "battery_v": 3.82, "fall_prob": 0.97, "peak_g": 4.3, "seq": 41}

Messages are stored in SQLite and exposed to the Gradio dashboard (app.py):

    POST /api/telemetry            ingest one message (alias: POST /webhook)
    GET  /api/devices              known devices + last-seen
    GET  /api/latest?device_id=    most recent vitals snapshot for a device
    GET  /api/history?device_id=&minutes=30
    GET  /api/alerts?limit=50      non-telemetry events, newest first
    POST /api/alerts/{id}/ack      acknowledge an alert (caregiver action)
    POST /api/simulate/{event}     inject a simulated event (demo / testing)
    GET  /health

Run:  python dashboard/server.py --port 8000 [--simulate]
Optional shared secret: set WRISTBAND_API_KEY and send it as header "X-API-Key".
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

EventType = Literal["TELEMETRY", "FALL", "FALL_CANCELLED", "SOS", "MED_REMINDER", "MED_TAKEN",
                    "MED_MISSED", "VITALS_ABNORMAL", "LOW_BATTERY", "BOOT"]

CRITICAL_EVENTS = {"FALL", "SOS", "MED_MISSED", "VITALS_ABNORMAL"}
# Mirrors the firmware's priorities (include/types.h) for payloads that omit "priority".
DEFAULT_PRIORITY = {"FALL": 3, "SOS": 3, "MED_MISSED": 2, "VITALS_ABNORMAL": 2, "LOW_BATTERY": 1, "BOOT": 1}
API_KEY = os.environ.get("WRISTBAND_API_KEY", "")
DB_PATH = Path(os.environ.get("WRISTBAND_DB", Path(__file__).resolve().parent / "wristband.db"))


class Telemetry(BaseModel):
    device_id: str = Field(..., max_length=32)
    event: EventType = "TELEMETRY"
    priority: Optional[int] = Field(None, ge=0, le=3)
    hr: Optional[float] = Field(None, ge=0, le=300)
    spo2: Optional[float] = Field(None, ge=0, le=100)
    rr: Optional[float] = Field(None, ge=0, le=80)
    battery_v: Optional[float] = Field(None, ge=0, le=5)
    battery_pct: Optional[int] = Field(None, ge=0, le=100)
    fall_prob: Optional[float] = Field(None, ge=0, le=1)
    peak_g: Optional[float] = None
    motion: Optional[bool] = None
    detail: Optional[str] = Field(None, max_length=200)
    seq: Optional[int] = None
    uptime_s: Optional[int] = None
    ts: Optional[str] = None          # device RTC time (ISO-8601), informational
    sim: Optional[bool] = None


# ----------------------------------------------------------------------------- storage
class Store:
    def __init__(self, path: Path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, received REAL NOT NULL, device_id TEXT NOT NULL,
            event TEXT NOT NULL, priority INTEGER, hr REAL, spo2 REAL, rr REAL, battery_v REAL,
            battery_pct INTEGER, fall_prob REAL, peak_g REAL, motion INTEGER, detail TEXT,
            seq INTEGER, uptime_s INTEGER, ts TEXT, sim INTEGER, acked INTEGER DEFAULT 0)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_dev_time ON messages(device_id, received)")
        self.db.commit()

    def insert(self, t: Telemetry) -> int:
        d = t.model_dump()
        if d["priority"] is None:
            d["priority"] = DEFAULT_PRIORITY.get(d["event"], 0)
        with self.lock:
            cur = self.db.execute(
                """INSERT INTO messages (received, device_id, event, priority, hr, spo2, rr, battery_v,
                   battery_pct, fall_prob, peak_g, motion, detail, seq, uptime_s, ts, sim)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (time.time(), d["device_id"], d["event"], d["priority"], d["hr"], d["spo2"], d["rr"],
                 d["battery_v"], d["battery_pct"], d["fall_prob"], d["peak_g"], d["motion"], d["detail"],
                 d["seq"], d["uptime_s"], d["ts"], d["sim"]))
            self.db.commit()
            # keep the DB bounded (~1 week of 10 s telemetry for a few devices)
            if cur.lastrowid % 1000 == 0:
                self.db.execute("DELETE FROM messages WHERE event='TELEMETRY' AND received < ?",
                                (time.time() - 7 * 86400,))
                self.db.commit()
            return int(cur.lastrowid)

    def q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def ack(self, msg_id: int) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE messages SET acked=1 WHERE id=?", (msg_id,))
            self.db.commit()
            return cur.rowcount > 0


store = Store(DB_PATH)
_sim_threads: list = []


def _check_key(key: Optional[str]) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing X-API-Key")


def latest_snapshot(device_id: str) -> dict:
    """Most recent non-null value of each vital, plus device status."""
    rows = store.q("SELECT * FROM messages WHERE device_id=? ORDER BY id DESC LIMIT 200", (device_id,))
    if not rows:
        return {}
    snap = {"device_id": device_id, "last_seen": rows[0]["received"], "last_event": rows[0]["event"],
            "sim": bool(rows[0]["sim"])}
    for k in ("hr", "spo2", "rr", "battery_v", "battery_pct", "motion", "uptime_s"):
        snap[k] = next((r[k] for r in rows if r[k] is not None), None)
    snap["open_alerts"] = store.q(
        "SELECT COUNT(*) AS n FROM messages WHERE device_id=? AND acked=0 AND event IN (%s)"
        % ",".join("?" * len(CRITICAL_EVENTS)), (device_id, *CRITICAL_EVENTS))[0]["n"]
    return snap


# ----------------------------------------------------------------------------- API
def create_app(simulate: bool = False, sim_target: Optional[str] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        if simulate:
            from device_simulator import start_background
            _sim_threads.append(start_background(sim_target or "http://127.0.0.1:8000", api_key=API_KEY))
        yield

    app = FastAPI(title="Vital Wristband Webhook Receiver", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ok": True, "time": time.time(), "messages": store.q("SELECT COUNT(*) n FROM messages")[0]["n"]}

    @app.post("/api/telemetry")
    @app.post("/webhook")
    def ingest(t: Telemetry, x_api_key: Optional[str] = Header(None)):
        _check_key(x_api_key)
        msg_id = store.insert(t)
        if t.event != "TELEMETRY":
            print(f"[server] {t.device_id} {t.event} prio={t.priority if t.priority is not None else DEFAULT_PRIORITY.get(t.event, 0)} hr={t.hr} spo2={t.spo2} "
                  f"rr={t.rr} detail={t.detail!r}", flush=True)
        return {"ok": True, "id": msg_id}

    @app.get("/api/devices")
    def devices():
        return store.q("""SELECT device_id, MAX(received) AS last_seen, COUNT(*) AS messages
                          FROM messages GROUP BY device_id ORDER BY last_seen DESC""")

    @app.get("/api/latest")
    def latest(device_id: str):
        snap = latest_snapshot(device_id)
        if not snap:
            raise HTTPException(404, "unknown device")
        return snap

    @app.get("/api/history")
    def history(device_id: str, minutes: float = Query(30, gt=0, le=7 * 24 * 60)):
        return store.q("""SELECT received, hr, spo2, rr, battery_v, battery_pct, event FROM messages
                          WHERE device_id=? AND received>=? ORDER BY received""",
                       (device_id, time.time() - minutes * 60))

    @app.get("/api/alerts")
    def alerts(limit: int = Query(50, ge=1, le=500), device_id: Optional[str] = None):
        sql = "SELECT * FROM messages WHERE event!='TELEMETRY'"
        args: tuple = ()
        if device_id:
            sql += " AND device_id=?"
            args = (device_id,)
        return store.q(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))

    @app.post("/api/alerts/{msg_id}/ack")
    def ack(msg_id: int):
        if not store.ack(msg_id):
            raise HTTPException(404, "no such alert")
        return {"ok": True}

    @app.post("/api/simulate/{event}")
    def simulate_event(event: str, device_id: str = "WB-SIM"):
        from device_simulator import VirtualWristband
        ev = event.upper()
        if ev not in ("FALL", "SOS", "MED_MISSED", "VITALS_ABNORMAL", "LOW_BATTERY", "MED_TAKEN"):
            raise HTTPException(400, "unsupported event")
        payload = VirtualWristband(device_id).event_payload(ev)
        msg_id = store.insert(Telemetry(**payload))
        return {"ok": True, "id": msg_id, "payload": payload}

    return app


app = create_app(simulate=os.environ.get("WRISTBAND_SIMULATE") == "1")


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so the ESP32 on your LAN can reach it")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--simulate", action="store_true", help="run a virtual wristband that posts here")
    a = ap.parse_args()
    uvicorn.run(create_app(a.simulate, f"http://127.0.0.1:{a.port}"), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
