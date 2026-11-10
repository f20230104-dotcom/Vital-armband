#!/usr/bin/env python3
"""
train_model.py - Train a compact 1D-CNN fall classifier and quantize it to int8 TFLite.

Input  : ml_training/data/dataset.npz           (from fetch_dataset.py)
Output : ml_training/models/fall_cnn.keras        float model
         ml_training/models/fall_cnn_int8.tflite  full-integer model for TFLite Micro
         ml_training/models/metrics.json          evaluation report (consumed by export_to_c.py)
         ml_training/models/selftest.npz          a few test windows for the on-device boot self-test

Architecture (TFLite-Micro friendly ops only: RESHAPE, CONV_2D, MAX_POOL_2D, MEAN,
FULLY_CONNECTED, LOGISTIC). Conv1D is expressed as Conv2D with (k,1) kernels so the
converter does not emit EXPAND_DIMS/SQUEEZE, which older TFLM builds lack.

Splits are SUBJECT-WISE (no subject appears in both train and test) to avoid the
optimistic leakage that window-level random splits produce on SisFall.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dataset.npz"
OUT = HERE / "models"


def build_model(win: int, ch: int, batch_size: int | None = None) -> tf.keras.Model:
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


def metrics(y_true, p, thr=0.5) -> dict:
    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_metrics(title: str, m: dict) -> None:
    c = m["confusion"]
    print(f"  {title:<26} n={m['n']:<6} acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
          f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  spec={m['specificity']:.4f}  "
          f"[tn={c['tn']} fp={c['fp']} fn={c['fn']} tp={c['tp']}]")


def to_int8_tflite(model, X_rep: np.ndarray) -> bytes:
    def rep():
        for i in range(0, min(len(X_rep), 500)):
            yield [X_rep[i:i + 1]]

    # Fixed batch of 1 -> fully static shapes, so the graph contains no SHAPE / STRIDED_SLICE /
    # PACK ops (dynamic-reshape plumbing that TFLite Micro does not need).
    fixed = build_model(model.input_shape[1], model.input_shape[2], batch_size=1)
    fixed.set_weights(model.get_weights())
    conv = tf.lite.TFLiteConverter.from_keras_model(fixed)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_predict(tfl: bytes, X: np.ndarray):
    it = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=
                             tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = ind["quantization"]
    s_out, z_out = outd["quantization"]
    probs = np.empty(len(X), np.float32)
    for i, x in enumerate(X):
        q = np.clip(np.round(x / s_in + z_in), -128, 127).astype(np.int8)[None]
        it.set_tensor(ind["index"], q)
        it.invoke()
        probs[i] = (it.get_tensor(outd["index"]).astype(np.float32)[0, 0] - z_out) * s_out
    ops = sorted({d["op_name"] for d in it._get_ops_details()}) if hasattr(it, "_get_ops_details") else []
    return probs, (float(s_in), int(z_in), float(s_out), int(z_out)), ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--convert-only", action="store_true", help="skip training, reload models/fall_cnn.keras")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"[error] {DATA} not found - run fetch_dataset.py first")
        return 1
    tf.keras.utils.set_random_seed(args.seed)
    d = np.load(DATA)
    X, y, groups, source = d["X"], d["y"].astype(np.int32), d["groups"], d["source"]
    print(f"[data] X={X.shape}  falls={y.sum()}  adl={(y == 0).sum()}  subjects={len(set(groups))}  "
          f"sources={dict(zip(*np.unique(source, return_counts=True)))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed + 1)
    tr2, va2 = next(gss2.split(X[tr_idx], y[tr_idx], groups[tr_idx]))
    va_idx, tr_idx = tr_idx[va2], tr_idx[tr2]
    print(f"[split] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}  "
          f"test subjects={sorted(set(groups[te_idx]))}")

    if args.convert_only:
        model = tf.keras.models.load_model(OUT / "fall_cnn.keras")
    else:
        model = train(args, X, y, tr_idx, va_idx)
    report = evaluate_and_export(args, d, model, X, y, groups, source, tr_idx, te_idx)
    return 0 if report else 1


def train(args, X, y, tr_idx, va_idx):
    model = build_model(X.shape[1], X.shape[2])
    model.summary(print_fn=lambda s: print("   " + s))
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.Precision(name="precision")])
    pos = y[tr_idx].mean()
    cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}

    # light augmentation: random gain + noise (sensor tolerance, strap tightness)
    def aug(xb, yb):
        g = tf.random.uniform((tf.shape(xb)[0], 1, 1), 0.9, 1.1)
        return xb * g + tf.random.normal(tf.shape(xb), stddev=0.02), yb

    ds = (tf.data.Dataset.from_tensor_slices((X[tr_idx], y[tr_idx])).shuffle(len(tr_idx), seed=args.seed)
          .batch(args.batch).map(aug).prefetch(2))
    model.fit(ds, validation_data=(X[va_idx], y[va_idx]), epochs=args.epochs, class_weight=cw, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=4)])

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / "fall_cnn.keras")
    return model


def evaluate_and_export(args, d, model, X, y, groups, source, tr_idx, te_idx):
    print("\n[eval] float32 model (held-out subjects)")
    p_float = model.predict(X[te_idx], verbose=0).ravel()
    report = {"float": {"all": metrics(y[te_idx], p_float, args.threshold)}}
    print_metrics("float32 / all", report["float"]["all"])
    for src in np.unique(source[te_idx]):
        m = source[te_idx] == src
        report["float"][str(src)] = metrics(y[te_idx][m], p_float[m], args.threshold)
        print_metrics(f"float32 / {src}", report["float"][str(src)])

    print("\n[quant] converting to full-integer int8 TFLite ...")
    tfl = to_int8_tflite(model, X[tr_idx][np.random.default_rng(0).permutation(len(tr_idx))])
    (OUT / "fall_cnn_int8.tflite").write_bytes(tfl)
    p_q, qparams, ops = tflite_predict(tfl, X[te_idx])
    report["int8"] = {"all": metrics(y[te_idx], p_q, args.threshold)}
    print_metrics("int8 / all", report["int8"]["all"])
    for src in np.unique(source[te_idx]):
        m = source[te_idx] == src
        report["int8"][str(src)] = metrics(y[te_idx][m], p_q[m], args.threshold)
        print_metrics(f"int8 / {src}", report["int8"][str(src)])
    agree = float(((p_q >= args.threshold) == (p_float >= args.threshold)).mean())
    print(f"  float/int8 decision agreement: {agree:.4f}")

    report.update({
        "threshold": args.threshold,
        "tflite_bytes": len(tfl),
        "params": int(model.count_params()),
        "quant": {"in_scale": qparams[0], "in_zero": qparams[1], "out_scale": qparams[2], "out_zero": qparams[3]},
        "ops": ops,
        "float_int8_agreement": agree,
        "test_subjects": sorted(map(str, set(groups[te_idx]))),
        "dataset_sources": {str(k): int(v) for k, v in zip(*np.unique(source, return_counts=True))},
        "window": {"fs": int(d["fs"]), "samples": int(d["win"]), "impact_index": int(d["impact_index"]),
                   "gyro_scale": float(d["gyro_scale"])},
    })
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))

    # pick one confident fall + one confident ADL from the test set for the device boot self-test
    te_y, te_p = y[te_idx], p_q
    fi = te_idx[np.where(te_y == 1)[0][np.argmax(te_p[te_y == 1])]]
    ai = te_idx[np.where(te_y == 0)[0][np.argmin(te_p[te_y == 0])]]
    np.savez(OUT / "selftest.npz", fall=X[fi], adl=X[ai])
    print(f"\n[done] tflite={len(tfl)} bytes, params={model.count_params()}, ops={ops}")
    print(f"       wrote {OUT / 'fall_cnn_int8.tflite'} and {OUT / 'metrics.json'}")
    return report


if __name__ == "__main__":
    sys.exit(main())
