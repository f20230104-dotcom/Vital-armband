#!/usr/bin/env python3
"""
train_model.py - Train the compact 1D-CNN fall classifier and quantize it to int8 TFLite.

Inputs : ml_training/data/dataset.npz     SisFall (waist, 38 subjects) + synthetic wrist windows
         ml_training/data/uci_wrist.npz   UCI falls dataset, RIGHT-WRIST sensor (17 subjects), optional
Outputs: ml_training/models/fall_cnn.keras        float model
         ml_training/models/fall_cnn_int8.tflite  full-integer model for TFLite Micro
         ml_training/models/metrics.json          evaluation report (consumed by export_to_c.py)
         ml_training/models/history.json          training curves
         ml_training/models/selftest.npz          two test windows for the on-device boot self-test

Splits are SUBJECT-WISE everywhere (no person appears in both train and test):
  * SisFall + synthetic: 20 % of subjects held out for test, 15 % of the rest for validation
  * UCI wrist:           5 of 17 volunteers held out for test (always used for evaluation;
                         its other 12 volunteers are added to training with --with-uci)

Architecture (TFLite-Micro friendly ops only: RESHAPE, CONV_2D, MAX_POOL_2D, MEAN,
FULLY_CONNECTED, LOGISTIC). Conv1D is written as Conv2D with (k,1) kernels so the
converter emits no EXPAND_DIMS/SQUEEZE, which older TFLM builds lack.

The deployed recipe (defaults) was chosen by ml_training/experiments.py (ablation study):
SisFall + synthetic + UCI wrist training subjects, with random 3-D rotation augmentation so the
network does not depend on how the IMU is mounted in the strap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dataset.npz"
UCI = HERE / "data" / "uci_wrist.npz"
OUT = HERE / "models"
SEED = 7


# ----------------------------------------------------------------------------- model
def build_model(win: int = 128, ch: int = 6, batch_size: int | None = None) -> tf.keras.Model:
    L = tf.keras.layers
    inp = L.Input(shape=(win, ch), batch_size=batch_size, name="imu")
    x = L.Reshape((win, 1, ch))(inp)
    x = L.Conv2D(8, (7, 1), padding="same", activation="relu")(x)
    x = L.MaxPooling2D((2, 1))(x)                       # 64
    x = L.Conv2D(16, (5, 1), padding="same", activation="relu")(x)
    x = L.MaxPooling2D((2, 1))(x)                       # 32
    x = L.Conv2D(24, (5, 1), padding="same", activation="relu")(x)
    x = L.MaxPooling2D((2, 1))(x)                       # 16
    x = L.Conv2D(32, (3, 1), padding="same", activation="relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(16, activation="relu")(x)
    out = L.Dense(1, activation="sigmoid", name="p_fall")(x)
    return tf.keras.Model(inp, out, name="fall_cnn")


def macs(win: int = 128) -> int:
    """Multiply-accumulates of one inference (conv + dense layers)."""
    t, total, cin = win, 0, 6
    for cout, k in ((8, 7), (16, 5), (24, 5), (32, 3)):
        total += t * k * cin * cout
        cin = cout
        t = t // 2 if cout != 32 else t
    return total + 32 * 16 + 16


# ----------------------------------------------------------------------------- data
def load_data():
    """All windows plus subject-wise train / val / test index arrays."""
    d = np.load(DATA)
    X, y, groups, source = d["X"], d["y"].astype(np.int32), d["groups"].astype("<U12"), d["source"]
    activity = d["activity"] if "activity" in d.files else np.full(len(y), "?")
    meta = {"fs": int(d["fs"]), "samples": int(d["win"]), "impact_index": int(d["impact_index"]),
            "gyro_scale": float(d["gyro_scale"])}
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED + 1)
    tr2, va2 = next(gss2.split(X[tr], y[tr], groups[tr]))
    va, tr = tr[va2], tr[tr2]
    split = {"train": tr, "val": va, "test": te, "uci_train": np.array([], int), "uci_val": np.array([], int)}

    if UCI.exists():
        u = np.load(UCI)
        n0 = len(y)
        X = np.concatenate([X, u["X"]])
        y = np.concatenate([y, u["y"].astype(np.int32)])
        groups = np.concatenate([groups, u["groups"].astype("<U12")])
        source = np.concatenate([source, np.full(len(u["y"]), "uci")])
        activity = np.concatenate([activity, u["activity"]])
        ui = np.arange(n0, len(y))
        gu = GroupShuffleSplit(n_splits=1, test_size=5 / 17, random_state=SEED)
        utr, ute = next(gu.split(ui, y[ui], groups[ui]))
        gu2 = GroupShuffleSplit(n_splits=1, test_size=2 / 12, random_state=SEED + 1)
        utr2, uva2 = next(gu2.split(utr, y[ui][utr], groups[ui][utr]))
        split["uci_train"], split["uci_val"] = ui[utr[utr2]], ui[utr[uva2]]
        split["test"] = np.concatenate([te, ui[ute]])
    return X, y, groups, source, activity, split, meta


def train_val_indices(split: dict, sources: np.ndarray, use: set[str], with_uci: bool):
    tr = split["train"][np.isin(sources[split["train"]], list(use))]
    va = split["val"][np.isin(sources[split["val"]], list(use))]
    if with_uci:
        tr = np.concatenate([tr, split["uci_train"]])
        va = np.concatenate([va, split["uci_val"]])
    return tr, va


# ----------------------------------------------------------------------------- training
def _random_rotation(n: int, max_deg: float) -> tf.Tensor:
    """n random rotation matrices: uniform random axis, angle uniform in [-max_deg, max_deg]."""
    axis = tf.math.l2_normalize(tf.random.normal((n, 3)), axis=1)
    ang = tf.random.uniform((n, 1, 1), -max_deg, max_deg) * (np.pi / 180.0)
    x, y_, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = tf.zeros_like(x)
    K = tf.stack([tf.stack([zero, -z, y_], 1), tf.stack([z, zero, -x], 1), tf.stack([-y_, x, zero], 1)], 1)
    eye = tf.eye(3, batch_shape=[n])
    return eye + tf.sin(ang) * K + (1 - tf.cos(ang)) * tf.matmul(K, K)


def make_augment(rot_deg: float):
    def aug(xb, yb):
        n = tf.shape(xb)[0]
        if rot_deg > 0:                      # same rotation for accel and gyro (rigid sensor)
            R = _random_rotation(n, rot_deg)
            acc = tf.einsum("btc,bdc->btd", xb[:, :, :3], R)
            gyr = tf.einsum("btc,bdc->btd", xb[:, :, 3:], R)
            xb = tf.concat([acc, gyr], axis=2)
        g = tf.random.uniform((n, 1, 1), 0.9, 1.1)            # sensor gain tolerance
        return xb * g + tf.random.normal(tf.shape(xb), stddev=0.02), yb
    return aug


def train_cnn(X, y, tr, va, *, epochs=40, batch=64, seed=SEED, rot_deg=0.0, verbose=0):
    tf.keras.utils.set_random_seed(seed)
    model = build_model(X.shape[1], X.shape[2])
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.Precision(name="precision")])
    pos = y[tr].mean()
    cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}                    # balance classes
    ds = (tf.data.Dataset.from_tensor_slices((X[tr], y[tr])).shuffle(len(tr), seed=seed)
          .batch(batch).map(make_augment(rot_deg)).prefetch(2))
    hist = model.fit(ds, validation_data=(X[va], y[va]), epochs=epochs, class_weight=cw, verbose=verbose,
                     callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True),
                                tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=4)])
    return model, {k: [float(v) for v in vals] for k, vals in hist.history.items()}


# ----------------------------------------------------------------------------- evaluation
def metrics(y_true, p, thr=0.5) -> dict:
    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, p))
    return out


def print_metrics(title: str, m: dict) -> None:
    c = m["confusion"]
    print(f"  {title:<24} n={m['n']:<6} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
          f"rec={m['recall']:.4f} f1={m['f1']:.4f} spec={m['specificity']:.4f} "
          f"auc={m.get('auc', float('nan')):.4f} [tn={c['tn']} fp={c['fp']} fn={c['fn']} tp={c['tp']}]")


def by_source(y, p, source, thr=0.5) -> dict:
    rep = {"all": metrics(y, p, thr)}
    for s in np.unique(source):
        m = source == s
        rep[str(s)] = metrics(y[m], p[m], thr)
    return rep


def to_int8_tflite(model, X_rep: np.ndarray) -> bytes:
    def rep():
        for i in range(0, min(len(X_rep), 1000)):
            yield [X_rep[i:i + 1]]

    # Fixed batch of 1 -> fully static shapes, so the graph has no SHAPE / STRIDED_SLICE / PACK ops.
    fixed = build_model(model.input_shape[1], model.input_shape[2], batch_size=1)
    fixed.set_weights(model.get_weights())
    conv = tf.lite.TFLiteConverter.from_keras_model(fixed)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep                          # calibration for activation ranges
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def to_float_tflite(model) -> bytes:
    fixed = build_model(model.input_shape[1], model.input_shape[2], batch_size=1)
    fixed.set_weights(model.get_weights())
    return tf.lite.TFLiteConverter.from_keras_model(fixed).convert()


def tflite_predict(tfl: bytes, X: np.ndarray):
    it = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=
                             tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = ind["quantization"]
    s_out, z_out = outd["quantization"]
    probs = np.empty(len(X), np.float32)
    for i, x in enumerate(X):
        if ind["dtype"] == np.int8:
            it.set_tensor(ind["index"], np.clip(np.round(x / s_in + z_in), -128, 127).astype(np.int8)[None])
        else:
            it.set_tensor(ind["index"], x[None].astype(np.float32))
        it.invoke()
        o = it.get_tensor(outd["index"]).astype(np.float32)[0, 0]
        probs[i] = (o - z_out) * s_out if outd["dtype"] == np.int8 else o
    ops = sorted({d["op_name"] for d in it._get_ops_details()}) if hasattr(it, "_get_ops_details") else []
    return probs, (float(s_in), int(z_in), float(s_out), int(z_out)), ops


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--rot-aug", type=float, default=180.0,
                    help="max random rotation (deg) applied to training windows; 0 = off")
    ap.add_argument("--no-uci", action="store_true", help="do not train on the UCI wrist training volunteers")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--convert-only", action="store_true", help="skip training, reload models/fall_cnn.keras")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"[error] {DATA} not found - run fetch_dataset.py first")
        return 1
    X, y, groups, source, activity, split, meta = load_data()
    with_uci = UCI.exists() and not args.no_uci
    use = {"sisf"} if args.no_synthetic else {"sisf", "syn", "wrst"}
    tr, va = train_val_indices(split, source, use, with_uci)
    te = split["test"]
    print(f"[data] {len(y)} windows, {len(set(groups))} subjects, sources "
          f"{dict(zip(*np.unique(source, return_counts=True)))}")
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)} | uci in training: {with_uci} | "
          f"rotation aug: {args.rot_aug} deg")

    OUT.mkdir(parents=True, exist_ok=True)
    if args.convert_only:
        model = tf.keras.models.load_model(OUT / "fall_cnn.keras")
    else:
        model, hist = train_cnn(X, y, tr, va, epochs=args.epochs, batch=args.batch, rot_deg=args.rot_aug, verbose=2)
        model.save(OUT / "fall_cnn.keras")
        (OUT / "history.json").write_text(json.dumps(hist))

    print("\n[eval] float32 model, held-out subjects")
    p_float = model.predict(X[te], verbose=0).ravel()
    report = {"float": by_source(y[te], p_float, source[te], args.threshold)}
    for k, m in report["float"].items():
        print_metrics(f"float32 / {k}", m)

    print("\n[quant] full-integer int8 TFLite (calibrated on 1000 training windows)")
    tfl = to_int8_tflite(model, X[tr][np.random.default_rng(0).permutation(len(tr))])
    (OUT / "fall_cnn_int8.tflite").write_bytes(tfl)
    p_q, qparams, ops = tflite_predict(tfl, X[te])
    report["int8"] = by_source(y[te], p_q, source[te], args.threshold)
    for k, m in report["int8"].items():
        print_metrics(f"int8 / {k}", m)
    agree = float(((p_q >= args.threshold) == (p_float >= args.threshold)).mean())
    print(f"  float/int8 decision agreement: {agree:.4f}")

    report.update({
        "threshold": args.threshold,
        "recipe": {"rot_aug_deg": args.rot_aug, "with_uci": with_uci, "sources": sorted(use),
                   "epochs_max": args.epochs, "batch": args.batch, "optimizer": "Adam 2e-3",
                   "early_stopping": "val_loss patience 8", "lr_schedule": "x0.5 on plateau (4)"},
        "tflite_bytes": len(tfl),
        "params": int(model.count_params()),
        "macs": macs(meta["samples"]),
        "quant": {"in_scale": qparams[0], "in_zero": qparams[1], "out_scale": qparams[2], "out_zero": qparams[3]},
        "ops": ops,
        "float_int8_agreement": agree,
        "train_windows": int(len(tr)),
        "test_subjects": sorted(map(str, set(groups[te]))),
        "dataset_sources": {str(k): int(v) for k, v in zip(*np.unique(source, return_counts=True))},
        "window": meta,
    })
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    np.save(OUT / "test_predictions.npy", np.stack([te, p_q]))

    # one confident fall + one confident ADL from the test set for the device boot self-test
    te_y = y[te]
    fi = te[np.where(te_y == 1)[0][np.argmax(p_q[te_y == 1])]]
    ai = te[np.where(te_y == 0)[0][np.argmin(p_q[te_y == 0])]]
    np.savez(OUT / "selftest.npz", fall=X[fi], adl=X[ai])
    print(f"\n[done] tflite={len(tfl)} bytes, params={model.count_params()}, MACs={macs()}, ops={ops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
