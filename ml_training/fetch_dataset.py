#!/usr/bin/env python3
"""
fetch_dataset.py - Download, parse and window a public fall-detection dataset.

Primary source : SisFall (Sucerquia, Lopez & Vargas-Bonilla, Sensors 2017)
                 38 subjects (23 young SA01-SA23, 15 elderly SE01-SE15),
                 19 ADL types (D01-D19) and 15 fall types (F01-F15),
                 ADXL345 accel + ITG3200 gyro + MMA8451Q accel @ 200 Hz.
Fallback       : physics-inspired synthetic WRIST data (free-fall valley < 0.5 g,
                 impact spike 2.5-7 g, post-impact stillness, orientation change).

Output         : ml_training/data/dataset.npz
                 X      float32 (N, 128, 6)  ax,ay,az [g], gx,gy,gz [dps / GYRO_SCALE]
                 y      int8    (N,)         0 = ADL, 1 = FALL
                 groups <U8     (N,)         subject id (for subject-wise splits)
                 source <U4     (N,)         "sisf" | "syn"

The window convention mirrors the firmware (src/tasks/motion_task.cpp):
50 Hz, 128 samples (2.56 s), with the impact peak located near index IMPACT_INDEX.

Usage:
    python ml_training/fetch_dataset.py                 # SisFall + synthetic wrist augmentation
    python ml_training/fetch_dataset.py --source synthetic
    python ml_training/fetch_dataset.py --no-augment    # SisFall only
"""
from __future__ import annotations

import argparse


import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- constants
FS_HZ = 50                 # target sample rate (must match config.h MOTION_SAMPLE_HZ)
WIN = 128                  # samples per window (must match config.h FALL_WINDOW_SAMPLES)
IMPACT_INDEX = 64          # where the impact peak sits in a window (config.h FALL_IMPACT_INDEX)
GYRO_SCALE = 500.0         # dps -> model units (config.h FALL_GYRO_SCALE_DPS)
SISFALL_FS = 200

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_FILE = DATA_DIR / "dataset.npz"

# The official UdeA link has been offline for a while; the GitHub mirror holds the
# untouched raw SisFall text files (SisFall/SAxx/Dyy_SAxx_Rzz.txt).
SISFALL_URLS = [
    "http://sistemic.udea.edu.co/wp-content/uploads/2016/10/SisFall_dataset.zip",
    "https://codeload.github.com/Sibonile7/Elderly-Fall-Detection-using-AI-SisFall-Dataset-/zip/refs/heads/main",
]
SISFALL_FILE_RE = re.compile(r"(?:^|/)(?P<act>[DF]\d{2})_(?P<subj>S[AE]\d{2})_R(?P<rep>\d{2})\.txt$")

# ADXL345: +-16 g, 13-bit ; ITG3200: +-2000 dps, 16-bit (SisFall Readme.txt)
ADXL_TO_G = (2 * 16) / (2 ** 13)
ITG_TO_DPS = (2 * 2000) / (2 ** 16)


# =========================================================================== download
def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"[fetch] trying {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vital-wristband-ml/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done, t0 = 0, time.time()
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - t0 > 2:
                    t0 = time.time()
                    pct = f"{100 * done / total:5.1f}%" if total else ""
                    print(f"[fetch]   {done / 1e6:7.1f} MB {pct}", flush=True)
        if not zipfile.is_zipfile(tmp):
            print("[fetch]   not a zip archive, skipping")
            tmp.unlink(missing_ok=True)
            return False
        tmp.rename(dest)
        print(f"[fetch]   saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as exc:  # network errors, HTTP 4xx/5xx, auth walls ...
        print(f"[fetch]   failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def fetch_sisfall() -> Path | None:
    """Return a path to a SisFall zip, downloading it if needed."""
    local = RAW_DIR / "sisfall.zip"
    if local.exists() and zipfile.is_zipfile(local):
        print(f"[fetch] using cached {local}")
        return local
    for url in SISFALL_URLS:
        if _download(url, local):
            return local
    return None


# =========================================================================== SisFall parsing
def _parse_sisfall_txt(text: str) -> np.ndarray:
    """Parse one SisFall trial -> (T, 6) float array at 200 Hz [g, dps]."""
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(";")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append([float(p) for p in parts[:6]])
        except ValueError:
            continue
    raw = np.asarray(rows, dtype=np.float32)
    if raw.size == 0:
        return raw.reshape(0, 6)
    out = np.empty_like(raw)
    out[:, :3] = raw[:, :3] * ADXL_TO_G
    out[:, 3:] = raw[:, 3:] * ITG_TO_DPS
    return out


def _decimate(sig: np.ndarray, factor: int) -> np.ndarray:
    """Block-average decimation (simple anti-alias low-pass + downsample)."""
    n = (len(sig) // factor) * factor
    return sig[:n].reshape(-1, factor, sig.shape[1]).mean(axis=1)


def _to_model_units(sig: np.ndarray) -> np.ndarray:
    out = sig.astype(np.float32).copy()
    out[:, 3:] /= GYRO_SCALE
    return out


def _peak_window(sig: np.ndarray, peak: int, offset: int) -> np.ndarray | None:
    start = peak - IMPACT_INDEX - offset
    if start < 0 or start + WIN > len(sig):
        return None
    return sig[start:start + WIN]


def load_sisfall(zip_path: Path, rng: np.random.Generator, max_subjects: int | None = None):
    X, y, groups = [], [], []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if SISFALL_FILE_RE.search(n)]
        if not names:
            raise RuntimeError("zip does not contain SisFall trial files")
        subjects = sorted({SISFALL_FILE_RE.search(n)["subj"] for n in names})
        if max_subjects:
            subjects = subjects[:max_subjects]
        print(f"[sisfall] {len(names)} trial files, using {len(subjects)} subjects")
        for i, name in enumerate(names):
            m = SISFALL_FILE_RE.search(name)
            if m["subj"] not in subjects:
                continue
            sig = _parse_sisfall_txt(zf.read(name).decode("utf-8", errors="ignore"))
            if len(sig) < SISFALL_FS * 3:
                continue
            sig = _to_model_units(_decimate(sig, SISFALL_FS // FS_HZ))
            mag = np.linalg.norm(sig[:, :3], axis=1)
            is_fall = m["act"].startswith("F")
            if is_fall:
                peak = int(np.argmax(mag))
                for off in (-10, -5, 0, 5, 10):           # temporal jitter augmentation
                    w = _peak_window(sig, peak, off)
                    if w is not None:
                        X.append(w); y.append(1); groups.append(m["subj"])
            else:
                # Hard negatives: windows centred on the strongest ADL peaks (these are the
                # windows the firmware's threshold trigger would actually hand to the model).
                order = np.argsort(mag)[::-1]
                used: list[int] = []
                for p in order:
                    if len(used) >= 3:
                        break
                    if all(abs(p - u) > WIN for u in used):
                        used.append(int(p))
                        w = _peak_window(sig, int(p), int(rng.integers(-10, 11)))
                        if w is not None:
                            X.append(w); y.append(0); groups.append(m["subj"])
                # plus one random window for general coverage
                if len(sig) > WIN:
                    s = int(rng.integers(0, len(sig) - WIN))
                    X.append(sig[s:s + WIN]); y.append(0); groups.append(m["subj"])
            if (i + 1) % 500 == 0:
                print(f"[sisfall]   parsed {i + 1}/{len(names)} files", flush=True)
    return np.stack(X).astype(np.float32), np.asarray(y, np.int8), np.asarray(groups)


# =========================================================================== synthetic wrist data
def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rand_unit(rng) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _gravity_track(rng, n, g0, wobble_deg=5.0, freq=0.0):
    """Gravity vector in sensor frame, gently rotating (arm movement)."""
    axis = _rand_unit(rng)
    t = np.arange(n) / FS_HZ
    ang = np.deg2rad(wobble_deg) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28)) if freq else \
        np.deg2rad(wobble_deg) * np.cumsum(rng.normal(0, 0.05, n)) / max(1, np.sqrt(n))
    G = np.stack([_rot(axis, a) @ g0 for a in ang])
    w = np.gradient(ang) * FS_HZ                               # rad/s around axis
    gyro = np.rad2deg(np.outer(w, axis))
    return G, gyro


def _activity(rng, kind: str, n: int, g0: np.ndarray):
    """Return (acc[g], gyro[dps]) for an ADL segment of length n."""
    t = np.arange(n) / FS_HZ
    noise_a, noise_g = 0.015, 1.5
    if kind == "rest":
        G, W = _gravity_track(rng, n, g0, 2.0)
        lin = np.zeros((n, 3))
    elif kind in ("walk", "stairs", "jog"):
        f = {"walk": rng.uniform(0.8, 1.1), "stairs": rng.uniform(0.7, 0.9), "jog": rng.uniform(1.3, 1.6)}[kind]
        amp = {"walk": rng.uniform(0.15, 0.45), "stairs": rng.uniform(0.2, 0.5), "jog": rng.uniform(0.6, 1.4)}[kind]
        swing = {"walk": 25, "stairs": 20, "jog": 45}[kind]
        G, W = _gravity_track(rng, n, g0, swing, f)             # arm swing at stride freq
        d = _rand_unit(rng)
        step = np.sin(2 * np.pi * 2 * f * t + rng.uniform(0, 6.28))
        step = np.sign(step) * np.abs(step) ** 1.5             # peaky heel strikes
        lin = np.outer(amp * step, d) + rng.normal(0, amp * 0.1, (n, 3))
    elif kind == "gesture":                                     # waving / eating / typing
        G, W = _gravity_track(rng, n, g0, rng.uniform(20, 70), rng.uniform(0.5, 2.5))
        lin = rng.normal(0, rng.uniform(0.05, 0.3), (n, 3))
        lin = np.apply_along_axis(lambda c: np.convolve(c, np.ones(5) / 5, "same"), 0, lin)
    else:
        raise ValueError(kind)
    acc = G + lin + rng.normal(0, noise_a, (n, 3))
    gyro = W + rng.normal(0, noise_g, (n, 3))
    return acc, gyro


def _impact(rng, n_pre, n_imp, peak_g, direction, ring=True):
    """Damped impact spike along `direction`, length n_imp."""
    k = np.arange(n_imp)
    env = np.exp(-k / max(1.0, n_imp / 4))
    osc = np.cos(2 * np.pi * k / rng.uniform(3, 6)) if ring else np.ones(n_imp)
    spike = peak_g * env * osc
    return np.outer(spike, direction) + rng.normal(0, 0.15 * peak_g, (n_imp, 3)) * env[:, None]


def synth_window(rng: np.random.Generator, label: int) -> np.ndarray:
    g0 = _rand_unit(rng)
    if label == 0:
        kind = rng.choice(["rest", "walk", "stairs", "jog", "gesture",
                           "sit_hard", "jump", "slap", "lie_down", "trip_recover"],
                          p=[0.08, 0.14, 0.08, 0.12, 0.12, 0.12, 0.1, 0.1, 0.07, 0.07])
        if kind in ("rest", "walk", "stairs", "jog", "gesture"):
            acc, gyr = _activity(rng, kind, WIN, g0)
        else:
            acc, gyr = _activity(rng, rng.choice(["walk", "rest", "gesture"]), WIN, g0)
            c = IMPACT_INDEX + int(rng.integers(-10, 11))
            if kind == "sit_hard":                      # small dip then 1.6-2.6 g bump, then still
                acc[c - 8:c] *= rng.uniform(0.6, 0.85)
                acc[c:c + 6] += _impact(rng, 0, 6, rng.uniform(0.7, 1.6), _rand_unit(rng))
                post, pg = _activity(rng, "rest", WIN - c - 6, g0)
                acc[c + 6:], gyr[c + 6:] = post, pg
            elif kind == "jump":                        # real free-fall + big landing, keeps moving
                ff = int(rng.integers(8, 18))
                acc[c - ff:c] = rng.normal(0, 0.08, (ff, 3)) + g0 * rng.uniform(0.05, 0.35)
                acc[c:c + 5] += _impact(rng, 0, 5, rng.uniform(1.5, 4.0), g0)
                post, pg = _activity(rng, rng.choice(["walk", "jog", "gesture"]), WIN - c - 5, g0)
                acc[c + 5:], gyr[c + 5:] = post, pg
            elif kind == "slap":                        # wrist hits table / clap: spike, no free-fall
                acc[c:c + 3] += _impact(rng, 0, 3, rng.uniform(2.0, 6.0), _rand_unit(rng))
                gyr[c - 5:c] += rng.normal(0, 150, (5, 3))
            elif kind == "lie_down":                    # slow orientation change, no impact
                g1 = _rand_unit(rng)
                m = int(rng.integers(30, 60))
                for i in range(m):
                    a = (i + 1) / m
                    v = (1 - a) * g0 + a * g1
                    acc[c - m // 2 + i] = v / np.linalg.norm(v) + rng.normal(0, 0.05, 3)
                tail, tg = _activity(rng, "rest", WIN - (c + m - m // 2), g1)
                acc[c + m - m // 2:], gyr[c + m - m // 2:] = tail, tg
                gyr[c - m // 2:c + m - m // 2] += rng.normal(0, 40, (m, 3))
            elif kind == "trip_recover":                # brief partial free-fall, stumble, keeps walking
                ff = int(rng.integers(4, 9))
                acc[c - ff:c] *= rng.uniform(0.35, 0.6)
                acc[c:c + 4] += _impact(rng, 0, 4, rng.uniform(1.0, 2.2), _rand_unit(rng))
                gyr[c - ff:c + 4] += rng.normal(0, 120, (ff + 4, 3))
        x = np.hstack([acc, gyr / GYRO_SCALE])
        return x.astype(np.float32)

    # ---------------- FALL
    style = rng.choice(["forward", "backward", "lateral", "syncope", "from_chair"], p=[0.3, 0.2, 0.2, 0.15, 0.15])
    c = IMPACT_INDEX + int(rng.integers(-10, 11))
    pre_kind = "rest" if style in ("syncope", "from_chair") else rng.choice(["walk", "rest", "stairs", "gesture"])
    acc, gyr = _activity(rng, pre_kind, WIN, g0)
    ff = int(rng.integers(8, 25)) if style != "syncope" else int(rng.integers(15, 35))   # 160-500 ms (syncope: slower)
    g1 = _rand_unit(rng)                                                                  # final (lying) orientation
    ff_level = rng.uniform(0.05, 0.45) if style != "syncope" else rng.uniform(0.3, 0.55)
    axis = np.cross(g0, g1); axis = axis / (np.linalg.norm(axis) + 1e-9)
    tot_ang = np.arccos(np.clip(np.dot(g0, g1), -1, 1))
    for i in range(ff):                                 # free-fall valley + body rotation
        a = (i + 1) / ff
        acc[c - ff + i] = ff_level * (_rot(axis, a * tot_ang) @ g0) + rng.normal(0, 0.06, 3)
        gyr[c - ff + i] = np.rad2deg(axis * tot_ang * FS_HZ / ff) + rng.normal(0, 30, 3)
    n_imp = int(rng.integers(4, 10))
    peak = rng.uniform(2.6, 7.0) if style != "syncope" else rng.uniform(2.3, 3.8)
    acc[c:c + n_imp] = g1 + _impact(rng, 0, n_imp, peak, _rand_unit(rng) * 0.5 + g1 * 0.5)
    gyr[c:c + n_imp] = rng.normal(0, 200, (n_imp, 3))
    rest = WIN - (c + n_imp)
    post_acc, post_gyr = _activity(rng, "rest", rest, g1)          # lying still at new orientation
    if rng.random() < 0.25:                                        # some struggle / attempts to move
        post_acc += rng.normal(0, 0.08, post_acc.shape)
        post_gyr += rng.normal(0, 15, post_gyr.shape)
    acc[c + n_imp:], gyr[c + n_imp:] = post_acc, post_gyr
    x = np.hstack([acc, gyr / GYRO_SCALE])
    return x.astype(np.float32)


def make_synthetic(n_per_class: int, rng: np.random.Generator, n_subjects: int = 20):
    X, y, groups = [], [], []
    for label in (0, 1):
        for i in range(n_per_class):
            X.append(synth_window(rng, label)); y.append(label); groups.append(f"SYN{i % n_subjects:02d}")
    return np.stack(X), np.asarray(y, np.int8), np.asarray(groups)


# =========================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "sisfall", "synthetic"], default="auto")
    ap.add_argument("--no-augment", action="store_true", help="do not add synthetic wrist windows to SisFall")
    ap.add_argument("--synthetic-per-class", type=int, default=2500)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    if args.source in ("auto", "sisfall"):
        zp = fetch_sisfall()
        if zp is not None:
            try:
                Xs, ys, gs = load_sisfall(zp, rng, args.max_subjects)
                parts.append((Xs, ys, gs, "sisf"))
                print(f"[sisfall] {len(ys)} windows  (falls={int(ys.sum())}, adl={int((ys == 0).sum())})")
            except Exception as exc:
                print(f"[sisfall] parse failed: {exc}")
        if not parts and args.source == "sisfall":
            print("[error] SisFall unavailable and --source sisfall was requested")
            return 1
        if not parts:
            print("[fallback] SisFall unavailable -> generating synthetic wrist dataset only")

    if args.source == "synthetic" or not parts or not args.no_augment:
        Xn, yn, gn = make_synthetic(args.synthetic_per_class, rng)
        parts.append((Xn, yn, gn, "syn"))
        print(f"[synthetic] {len(yn)} windows (falls={int(yn.sum())})")

    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts]).astype("<U8")
    source = np.concatenate([np.full(len(p[1]), p[3]) for p in parts]).astype("<U4")
    np.savez_compressed(OUT_FILE, X=X, y=y, groups=groups, source=source,
                        fs=FS_HZ, win=WIN, impact_index=IMPACT_INDEX, gyro_scale=GYRO_SCALE)
    mag = np.linalg.norm(X[:, :, :3], axis=2)
    print(f"[done] {OUT_FILE}  X={X.shape}  falls={int(y.sum())}  adl={int((y == 0).sum())}")
    print(f"[check] fall windows: min|a| median={np.median(mag[y == 1].min(1)):.2f} g, "
          f"max|a| median={np.median(mag[y == 1].max(1)):.2f} g")
    return 0


if __name__ == "__main__":
    sys.exit(main())
