#!/usr/bin/env python3
"""
experiments.py - The evaluation study behind the deployed fall model.

    python ml_training/experiments.py            (~15-25 min on a laptop CPU)

Every split is subject-wise. Test sets: 12 held-out SisFall/synthetic subjects (from
train_model.load_data) + 5 held-out UCI wrist volunteers that no model ever trains on.

  E1  Baselines vs CNN: threshold rules (free fall + impact + stillness), the firmware's
      rule-based score, a Random Forest on 42 hand-crafted features, the CNN (float, int8)
  E2  Ablation on training data and augmentation (5 recipes, same test subjects)
  E3  5-fold subject-wise cross-validation of the deployed recipe
  E4  Decision-threshold sweep (precision / recall / F1) for the int8 model
  E5  Per-activity analysis (which falls are missed, which daily activities false-alarm)
  E6  Model cost: parameters, MACs, flash size (float vs int8), host latency

Outputs: ml_training/results/experiments.json and ml_training/results/fig_*.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_model as tm  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
IMP = 64
NAVY, ORANGE, SKY, RED, GREY = "#101141", "#FCB017", "#76C2E5", "#E31C24", "#8A8FA3"
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#444", "figure.dpi": 150})

SISFALL_NAMES = {
    "D01": "Walk slowly", "D02": "Walk quickly", "D03": "Jog slowly", "D04": "Jog quickly",
    "D05": "Stairs slowly", "D06": "Stairs quickly", "D07": "Sit half-chair slowly", "D08": "Sit half-chair quickly",
    "D09": "Sit low chair slowly", "D10": "Sit low chair quickly", "D11": "Collapse into chair",
    "D12": "Lie down slowly", "D13": "Lie down quickly", "D14": "Roll in bed", "D15": "Bend knees",
    "D16": "Bend at waist", "D17": "Get in/out of car", "D18": "Stumble", "D19": "Jump to reach",
    "F01": "Fwd slip", "F02": "Back slip", "F03": "Lateral slip", "F04": "Fwd trip", "F05": "Fwd trip (jog)",
    "F06": "Vertical faint", "F07": "Faint w/ hands", "F08": "Fwd getting up", "F09": "Lateral getting up",
    "F10": "Fwd sitting down", "F11": "Back sitting down", "F12": "Lateral sitting down",
    "F13": "Fwd faint seated", "F14": "Back faint seated", "F15": "Lateral faint seated"}


# ----------------------------------------------------------------------------- baselines
def threshold_rules(X):
    """Firmware state machine applied to a window: free fall (>=3 samples < 0.5 g) within 1 s
    before an impact > 2.7 g, then <10 % 'moving' samples in the post-impact part."""
    mag = np.linalg.norm(X[:, :, :3], axis=2)
    gyr = np.linalg.norm(X[:, :, 3:], axis=2) * 500.0
    out = np.zeros(len(X), bool)
    for i in range(len(X)):
        m = mag[i]
        imp = np.where(m[IMP - 10:IMP + 11] > 2.7)[0]
        if not len(imp):
            continue
        t_imp = IMP - 10 + imp[0]
        pre = m[max(0, t_imp - 50):t_imp] < 0.5
        run = best = 0
        for v in pre:
            run = run + 1 if v else 0
            best = max(best, run)
        if best < 3:
            continue
        post = slice(t_imp + 25, 128)
        moving = (np.abs(m[post] - 1) > 0.15) | (gyr[i][post] > 40)
        out[i] = moving.mean() <= 0.10
    return out.astype(float)


def heuristic_score(X):
    """Python port of FallDetector::heuristicScore (src/drivers/fall_detector.cpp)."""
    mag = np.linalg.norm(X[:, :, :3], axis=2)
    min_pre = mag[:, IMP - 30:IMP].min(1)
    peak = mag[:, IMP - 5:IMP + 10].max(1)
    pre = X[:, :IMP - 30, :3].sum(1)
    post = X[:, IMP + 25:, :3].sum(1)
    cos = (pre * post).sum(1) / (np.linalg.norm(pre, axis=1) * np.linalg.norm(post, axis=1) + 1e-6)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    sd = mag[:, IMP + 25:].std(1)
    c = lambda v: np.clip(v, 0, 1)  # noqa: E731
    return 0.25 * c((0.7 - min_pre) / 0.4) + 0.25 * c((peak - 2) / 2) + 0.25 * c((0.15 - sd) / 0.1) + 0.25 * c(ang / 45)


def features(X):
    """42 hand-crafted features: per channel (6 + |a| magnitude) mean, std, min, max, range, energy."""
    mag = np.linalg.norm(X[:, :, :3], axis=2)[:, :, None]
    S = np.concatenate([X, mag], axis=2)
    return np.concatenate([S.mean(1), S.std(1), S.min(1), S.max(1), S.max(1) - S.min(1), (S ** 2).mean(1)], axis=1)


# ----------------------------------------------------------------------------- figures
def bar_groups(ax, labels, series: dict, colors, ylim=(0.5, 1.0), fmt="{:.2f}"):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(labels))
    for k, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + (k - (n - 1) / 2) * w, vals, w, label=name, color=colors[k])
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, min(v, ylim[1]) + 0.004, fmt.format(v), ha="center",
                    va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=11, ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.0))


def main() -> int:
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    X, y, groups, source, activity, split, meta = tm.load_data()
    te = split["test"]
    te_main = te[source[te] != "uci"]
    te_uci = te[source[te] == "uci"]
    sets = {"SisFall + synthetic test (12 subj.)": te_main, "UCI real-wrist test (5 subj.)": te_uci}
    R: dict = {"test_sizes": {k: int(len(v)) for k, v in sets.items()}}
    print(f"[data] train/val/test windows: {len(split['train'])}/{len(split['val'])}/{len(te)}; "
          f"uci train {len(split['uci_train'])}")

    # ---- E2 ablation (also yields the models used by E1/E4/E5) ---------------------------
    recipes = {
        "A  SisFall only": dict(use={"sisf"}, uci=False, rot=0),
        "B  + synthetic wrist": dict(use={"sisf", "syn"}, uci=False, rot=0),
        "C  + rotation aug.": dict(use={"sisf", "syn"}, uci=False, rot=180),
        "D  + UCI wrist train": dict(use={"sisf", "syn"}, uci=True, rot=0),
        "E  + UCI + rotation (deployed)": dict(use={"sisf", "syn"}, uci=True, rot=180),
    }
    R["ablation"] = {}
    models = {}
    for name, r in recipes.items():
        tr, va = tm.train_val_indices(split, source, r["use"], r["uci"])
        t1 = time.time()
        model, hist = tm.train_cnn(X, y, tr, va, rot_deg=r["rot"])
        p = model.predict(X[te], verbose=0).ravel()
        res = {k: tm.metrics(y[v], p[np.isin(te, v)]) for k, v in sets.items()}
        res["train_windows"] = int(len(tr))
        res["epochs"] = len(hist["loss"])
        R["ablation"][name] = res
        models[name] = (model, hist, tr)
        print(f"[E2] {name:<32} " + "  ".join(f"{k[:7]} F1={m['f1']:.3f} rec={m['recall']:.3f}"
                                               for k, m in res.items() if isinstance(m, dict))
              + f"  ({time.time() - t1:.0f}s)")

    final_name = "E  + UCI + rotation (deployed)"
    model, hist, tr_final = models[final_name]
    R["final_history"] = hist

    # ---- E1 baselines on both test sets --------------------------------------------------
    tr_b, _ = tm.train_val_indices(split, source, {"sisf", "syn"}, True)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0)
    rf.fit(features(X[tr_b]), y[tr_b])
    tfl8 = tm.to_int8_tflite(model, X[tr_final][np.random.default_rng(0).permutation(len(tr_final))])
    tflf = tm.to_float_tflite(model)
    p_cnn = model.predict(X[te], verbose=0).ravel()
    t1 = time.time()
    p_int8, _, _ = tm.tflite_predict(tfl8, X[te])
    lat_us = (time.time() - t1) / len(te) * 1e6
    p_rf = rf.predict_proba(features(X[te]))[:, 1]
    rf_nodes = sum(e.tree_.node_count for e in rf.estimators_)
    base = {
        "Threshold rules": threshold_rules(X[te]),
        "Rule-based score": heuristic_score(X[te]),
        "Random Forest (42 feat.)": p_rf,
        "1D-CNN float32": p_cnn,
        "1D-CNN int8 (deployed)": p_int8,
    }
    R["baselines"] = {n: {k: tm.metrics(y[v], p[np.isin(te, v)]) for k, v in sets.items()} for n, p in base.items()}
    R["baselines_cost"] = {
        "Threshold rules": "~10 comparisons", "Rule-based score": "~1k FLOPs",
        "Random Forest (42 feat.)": f"{rf_nodes:,} tree nodes (~{rf_nodes * 12 / 1e6:.1f} MB)",
        "1D-CNN float32": f"{len(tflf) / 1024:.1f} KB", "1D-CNN int8 (deployed)": f"{len(tfl8) / 1024:.1f} KB"}
    for n, r in R["baselines"].items():
        print(f"[E1] {n:<26} " + "  ".join(f"{k[:7]} acc={m['accuracy']:.3f} F1={m['f1']:.3f}" for k, m in r.items()))

    # ---- E3 5-fold subject-wise CV of the deployed recipe --------------------------------
    pool = np.concatenate([split["train"], split["val"], split["uci_train"], split["uci_val"], te])
    pool = pool[np.isin(source[pool], ["sisf", "syn", "uci"])]
    gkf = GroupKFold(n_splits=5)
    folds = []
    for f, (a, b) in enumerate(gkf.split(pool, y[pool], groups[pool])):
        tr_f, te_f = pool[a], pool[b]
        rng = np.random.default_rng(f)
        subj = np.unique(groups[tr_f])
        val_subj = rng.choice(subj, size=max(1, len(subj) // 8), replace=False)
        va_f, tr_f = tr_f[np.isin(groups[tr_f], val_subj)], tr_f[~np.isin(groups[tr_f], val_subj)]
        m_f, _ = tm.train_cnn(X, y, tr_f, va_f, rot_deg=180, seed=100 + f)
        p_f = m_f.predict(X[te_f], verbose=0).ravel()
        folds.append(tm.metrics(y[te_f], p_f) | {"test_subjects": int(len(np.unique(groups[te_f])))})
        print(f"[E3] fold {f + 1}: acc={folds[-1]['accuracy']:.4f} f1={folds[-1]['f1']:.4f} "
              f"rec={folds[-1]['recall']:.4f} prec={folds[-1]['precision']:.4f}")
    R["cv"] = {"folds": folds, **{f"{k}_mean": float(np.mean([f[k] for f in folds])) for k in
                                   ("accuracy", "precision", "recall", "f1", "specificity", "auc")},
               **{f"{k}_std": float(np.std([f[k] for f in folds])) for k in
                  ("accuracy", "precision", "recall", "f1", "specificity", "auc")}}

    # ---- E4 threshold sweep (int8, all test windows) -----------------------------------
    ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
    R["threshold_sweep"] = [{"t": float(t), **{k: tm.metrics(y[te], p_int8, t)[k]
                                                for k in ("precision", "recall", "f1", "specificity")}} for t in ths]

    # ---- E5 per-activity (int8) ---------------------------------------------------------
    act_te = activity[te]
    per = {}
    for a in np.unique(act_te):
        m = act_te == a
        yy = y[te][m]
        hit = (p_int8[m] >= 0.5).astype(int)
        per[str(a)] = {"n": int(m.sum()), "label": int(yy[0]), "rate": float((hit == yy).mean())}
    R["per_activity"] = per

    # ---- E6 cost ------------------------------------------------------------------------
    R["cost"] = {"params": int(model.count_params()), "macs": tm.macs(), "int8_bytes": len(tfl8),
                 "float_bytes": len(tflf), "host_latency_us_per_window": lat_us,
                 "agreement_float_int8": float(((p_int8 >= .5) == (p_cnn >= .5)).mean())}
    R["runtime_min"] = (time.time() - t0) / 60
    (RES / "experiments.json").write_text(json.dumps(R, indent=2))
    figures(R, X, y, activity, source)
    print(f"[done] {RES / 'experiments.json'} in {R['runtime_min']:.1f} min")
    return 0


def figures(R, X, y, activity, source):
    k_main, k_uci = list(R["test_sizes"])

    # baselines
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(R["baselines"])
    short = ["Threshold\nrules", "Rule-based\nscore", "Random\nForest", "CNN\nfloat32", "CNN int8\n(deployed)"]
    for ax, key, title in ((axs[0], k_main, "SisFall + synthetic test"), (axs[1], k_uci, "UCI real-wrist test")):
        bar_groups(ax, short, {m: [R["baselines"][n][key][mk] for n in names] for m, mk in
                               (("Accuracy", "accuracy"), ("Recall", "recall"), ("F1", "f1"))},
                   [NAVY, ORANGE, SKY], ylim=(0.0, 1.05))
        ax.set_title(title, pad=34, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RES / "fig_baselines.png", bbox_inches="tight")
    plt.close(fig)

    # ablation
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ab = R["ablation"]
    labels = [n.split("  ")[0] + "\n" + n.split("  ")[1].replace(" (deployed)", "\n(deployed)") for n in ab]
    bar_groups(ax, labels, {"F1 SisFall+synthetic test": [ab[n][k_main]["f1"] for n in ab],
                            "F1 UCI real-wrist test": [ab[n][k_uci]["f1"] for n in ab],
                            "Recall UCI real-wrist": [ab[n][k_uci]["recall"] for n in ab]},
               [NAVY, ORANGE, RED], ylim=(0.6, 1.02), fmt="{:.3f}")
    fig.tight_layout()
    fig.savefig(RES / "fig_ablation.png", bbox_inches="tight")
    plt.close(fig)

    # CV
    fig, ax = plt.subplots(figsize=(10, 4.2))
    folds = R["cv"]["folds"]
    bar_groups(ax, [f"Fold {i + 1}" for i in range(len(folds))],
               {"Accuracy": [f["accuracy"] for f in folds], "Precision": [f["precision"] for f in folds],
                "Recall": [f["recall"] for f in folds]}, [NAVY, SKY, ORANGE], ylim=(0.85, 1.01), fmt="{:.3f}")
    fig.tight_layout()
    fig.savefig(RES / "fig_cv.png", bbox_inches="tight")
    plt.close(fig)

    # threshold sweep
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ts = R["threshold_sweep"]
    t = [r["t"] for r in ts]
    for k, c in (("precision", NAVY), ("recall", ORANGE), ("f1", RED)):
        ax.plot(t, [r[k] for r in ts], marker="o", ms=4, lw=2.4, color=c, label=k.capitalize() if k != "f1" else "F1")
    ax.axvline(0.5, color=GREY, ls="--", lw=1.4)
    ax.text(0.51, ax.get_ylim()[0] + 0.01, "deployed\nthreshold 0.5", color=GREY, fontsize=11)
    ax.set_xlabel("Decision threshold on p(fall)")
    ax.legend(frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(RES / "fig_threshold.png", bbox_inches="tight")
    plt.close(fig)

    # per-activity SisFall
    per = R["per_activity"]
    falls = sorted(a for a in per if a.startswith("F") and per[a]["label"] == 1)
    adls = sorted(a for a in per if a.startswith("D"))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    axs[0].barh([SISFALL_NAMES[a] for a in falls], [per[a]["rate"] for a in falls], color=ORANGE)
    axs[0].set_xlim(0.5, 1.0)
    axs[0].set_title("Falls detected (recall) by fall type", fontweight="bold")
    axs[0].invert_yaxis()
    axs[1].barh([SISFALL_NAMES[a] for a in adls], [1 - per[a]["rate"] for a in adls], color=NAVY)
    axs[1].set_title("False-alarm rate by daily activity", fontweight="bold")
    axs[1].invert_yaxis()
    for ax in axs:
        ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(RES / "fig_per_activity.png", bbox_inches="tight")
    plt.close(fig)

    # training curves
    h = R["final_history"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ep = np.arange(1, len(h["loss"]) + 1)
    axs[0].plot(ep, h["loss"], color=NAVY, lw=2.2, label="train")
    axs[0].plot(ep, h["val_loss"], color=ORANGE, lw=2.2, label="validation")
    axs[0].set_title("Binary cross-entropy loss", fontweight="bold")
    axs[1].plot(ep, h["accuracy"], color=NAVY, lw=2.2, label="train")
    axs[1].plot(ep, h["val_accuracy"], color=ORANGE, lw=2.2, label="validation")
    axs[1].set_title("Accuracy", fontweight="bold")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig_training.png", bbox_inches="tight")
    plt.close(fig)

    # dataset: example windows
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    tt = (np.arange(128) - IMP) / 50
    picks = [("SisFall F01 forward slip (waist)", (source == "sisf") & (activity == "F01")),
             ("UCI 901 forward fall (wrist)", (source == "uci") & (activity == "UCI-901")),
             ("SisFall D19 jump (daily activity)", (source == "sisf") & (activity == "D19"))]
    for ax, (title, m) in zip(axs, picks):
        i = np.where(m)[0][len(np.where(m)[0]) // 2]
        ax.plot(tt, np.linalg.norm(X[i, :, :3], axis=1), color=NAVY, lw=2)
        ax.axhline(0.5, color=ORANGE, ls="--", lw=1.4)
        ax.axhline(2.7, color=RED, ls="--", lw=1.4)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("time from window centre (s)")
    axs[0].set_ylabel("|a| (g)")
    fig.tight_layout()
    fig.savefig(RES / "fig_windows.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
