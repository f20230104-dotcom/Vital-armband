#!/usr/bin/env python3
"""
trigger_analysis.py - How often does the firmware's threshold trigger fire for real falls?

In the firmware the CNN only runs after the state machine triggers, so the system's recall is
(trigger recall) x (CNN recall on triggered windows). This script measures both on the held-out
test subjects (SisFall + real-wrist UCI) for several trigger rules, using the deployed int8
model's predictions saved by train_model.py (models/test_predictions.npy).

    python ml_training/trigger_analysis.py   ->  ml_training/results/trigger.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_model as tm  # noqa: E402

IMP = 64
RULES = {
    "free fall <0.5 g (60 ms) then impact >2.7 g (spec)": dict(ff_g=0.5, ff_n=3, imp_g=2.7, need_ff=True),
    "free fall <0.6 g then impact >2.0 g": dict(ff_g=0.6, ff_n=2, imp_g=2.0, need_ff=True),
    "impact >2.7 g only": dict(ff_g=0, ff_n=0, imp_g=2.7, need_ff=False),
    "spec path OR impact >2.0 g (deployed)": dict(ff_g=0.5, ff_n=3, imp_g=2.0, need_ff=False),
}


def longest_run(b: np.ndarray) -> int:
    best = run = 0
    for v in b:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def fires(m: np.ndarray, ff_g, ff_n, imp_g, need_ff) -> bool:
    hits = np.where(m[IMP - 10:IMP + 11] > imp_g)[0]
    if not len(hits):
        return False
    if not need_ff:
        return True
    t = IMP - 10 + hits[0]
    return longest_run(m[max(0, t - 50):t] < ff_g) >= ff_n


def main() -> int:
    X, y, groups, source, activity, split, meta = tm.load_data()
    te_idx, p = np.load(tm.OUT / "test_predictions.npy")
    te_idx = te_idx.astype(int)
    mag = np.linalg.norm(X[:, :, :3], axis=2)
    out = {}
    for name, rule in RULES.items():
        out[name] = {}
        for src_name, src in (("SisFall test", "sisf"), ("UCI real-wrist test", "uci")):
            sel = source[te_idx] == src
            idx, pp, yy = te_idx[sel], p[sel], y[te_idx][sel]
            trig = np.array([fires(mag[i], **rule) for i in idx])
            alert = trig & (pp >= 0.5)
            out[name][src_name] = {
                "trigger_recall_falls": float(trig[yy == 1].mean()),
                "trigger_rate_adl_peaks": float(trig[yy == 0].mean()),
                "system_recall": float(alert[yy == 1].mean()),
                "system_false_alarm_rate_adl_peaks": float(alert[yy == 0].mean()),
            }
            r = out[name][src_name]
            print(f"{name:<50} {src_name:<20} trigger recall {r['trigger_recall_falls']:.3f}  "
                  f"system recall {r['system_recall']:.3f}  false alarms on ADL peaks "
                  f"{r['system_false_alarm_rate_adl_peaks']:.3f}")
    (Path(__file__).resolve().parent / "results" / "trigger.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
