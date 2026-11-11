#!/usr/bin/env python3
"""
record_wrist_data.py - Record labelled WRIST IMU trials from the wristband (hardware phase).

Flash the logger firmware first:   pio run -e esp32dev-logger -t upload
Then record one trial per run, e.g.:

    python ml_training/record_wrist_data.py --port /dev/cu.usbserial-0001 \
        --subject W01 --activity F01 --seconds 15          # forward fall onto a mat
    python ml_training/record_wrist_data.py --port ... --subject W01 --activity D01 --seconds 60

Activity codes follow SisFall (D01-D19 daily activities, F01-F15 falls; see
ml_training/fetch_dataset.py SISFALL_ACTIVITIES). Each trial is saved as
ml_training/data/wrist/<subject>/<activity>_<subject>_R<nn>.csv with columns
t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,mark at 50 Hz, and is picked up automatically by
`python ml_training/fetch_dataset.py --wrist-dir ml_training/data/wrist`.

Safety: falls must be performed by healthy volunteers onto crash mats, never by elderly users.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "wrist"
HEADER = "t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,mark"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial port of the wristband (e.g. /dev/cu.usbserial-0001, COM5)")
    ap.add_argument("--subject", required=True, help="volunteer id, e.g. W01")
    ap.add_argument("--activity", required=True, help="SisFall-style code: D01..D19 or F01..F15")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--baud", type=int, default=115200)
    a = ap.parse_args()
    if not re.fullmatch(r"[DF]\d{2}", a.activity):
        print("activity must look like D05 or F12")
        return 1
    try:
        import serial  # pyserial
    except ImportError:
        print("pip install pyserial")
        return 1

    folder = OUT / a.subject
    folder.mkdir(parents=True, exist_ok=True)
    rep = 1 + len(list(folder.glob(f"{a.activity}_{a.subject}_R*.csv")))
    path = folder / f"{a.activity}_{a.subject}_R{rep:02d}.csv"

    rows, bad = [], 0
    with serial.Serial(a.port, a.baud, timeout=1) as port:
        port.reset_input_buffer()
        print(f"recording {a.activity} for {a.seconds:.0f} s ... (perform the activity now)")
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            line = port.readline().decode("ascii", errors="ignore").strip()
            if not line or line.startswith(("#", "[")):
                continue                                   # header / firmware log lines
            parts = line.split(",")
            if len(parts) != 8:
                bad += 1
                continue
            rows.append(line)
    if len(rows) < 50:
        print(f"only {len(rows)} samples received - is the logger firmware running on {a.port}?")
        return 1
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    print(f"saved {len(rows)} samples ({len(rows) / 50:.1f} s @50 Hz, {bad} malformed lines) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
