"""train_v2_baselines.py — Section 4.5: threshold sweep companion, learned
baselines on the same features, the cascade, and the joint latency run.

This script produces every row of Table 5 and the seed-variance figures quoted
in the text. It replaces an earlier version that silently used the wrong data:
that version read `Proje_Kodlari/data/dataset`, which is the *video-frame*
folder built by `extract_training_frames.py` for the deep edge detector, and it
scored on the validation folder rather than the test split. The numbers it wrote
therefore had nothing to do with the experiment Section 4.5 describes.

Protocol, exactly as stated in Section 4.5
------------------------------------------
* Fit on the 1887-image training split (730 fire, 1157 no-fire).
* Select ONE hyperparameter per model family on the 402-image validation split
  by F1 alone (LogisticRegression: C; RandomForestClassifier: max_depth).
* Evaluate the selected configuration on the 410-image test split EXACTLY ONCE.
* Inputs are decoded at 1/8 scale and resized to 640x480, so the absolute and
  relative area criteria of Section 4.4 coincide and the comparison against the
  hand-set sweep of Table 4 is byte-identical in its inputs.
* Feature set A: 8x8x8 = 512-bin joint HSV histogram, normalised to unit sum.
  Feature set B: A + a 32-bin histogram of Sobel gradient magnitude.
  Feature set C: B + a 4x4x4 histogram for each of three equal horizontal bands.
* Balanced class weights, fixed seed 0 for the reported fit.
* Seed variance: each configuration is refitted under ten seeds with the splits
  and the selected hyperparameter held fixed, varying only estimator randomness.
* Cascade: the hand-set stage of Section 3 proposes boxes; a verifier fitted on
  training-split crops accepts or rejects each one; an image is positive if any
  proposal survives. NOTE: Section 4.5 does not state which feature set the
  verifier uses. This implementation uses feature set A on the crop and says so
  here rather than leaving it to be inferred; change VERIFIER_FEATURES if the
  manuscript is revised to specify otherwise.
* Latency: the hand-set pipeline and the two vertical-band models are timed
  together in ONE run on ONE machine, 30 frames repeated ten times each after a
  discarded warm-up, for 300 timed calls per configuration. Cross-machine
  latency comparisons are refused by construction.

Usage:
    python train_v2_baselines.py                # everything
    python train_v2_baselines.py --skip-cascade # skip the slow proposal stage
    python train_v2_baselines.py --skip-latency

Writes Proje_Kodlari/evaluation_results/v2_baselines/v2_baseline_metrics.json
and a per-image prediction CSV per reported configuration.
"""

import argparse
import csv
import io
import json
import pickle
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from v1_heuristic import run_heuristic_pipeline  # noqa: E402

# --------------------------------------------------------------------------
# Paths. The still-image cohort of Sections 4.1 to 4.5 — NOT the video frames.
# --------------------------------------------------------------------------
DATASET = Path("Proje_Kodlari/data/AR Souri Dataset")
TRAIN_DIR, VAL_DIR, TEST_DIR = DATASET / "train", DATASET / "val", DATASET / "test"
OUT_DIR = Path("Proje_Kodlari/evaluation_results/v2_baselines")

FIRE, NO_FIRE = 1, 0          # fire is the positive class throughout
WORK_SIZE = (640, 480)
SEED = 0
N_SEEDS = 10
VERIFIER_FEATURES = "A"

# Hyperparameter grids searched on the validation split, one parameter each.
LR_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
RF_GRID = [None, 8, 16, 32]


# --------------------------------------------------------------------------
# Feature extraction (Section 4.5)
# --------------------------------------------------------------------------
def _hsv_hist_512(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist


def _sobel_hist_32(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    hist = np.histogram(mag, bins=32, range=(0, 360))[0].astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist


def _band_hists_192(bgr):
    h = bgr.shape[0]
    third = h // 3
    parts = []
    for band in (bgr[:third], bgr[third:2 * third], bgr[2 * third:]):
        if band.size == 0:
            parts.append(np.zeros(64, dtype=np.float32))
            continue
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4],
                            [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
        s = hist.sum()
        parts.append(hist / s if s > 0 else hist)
    return np.concatenate(parts)


FEATURE_SETS = {
    "A": lambda b: _hsv_hist_512(b),
    "B": lambda b: np.concatenate([_hsv_hist_512(b), _sobel_hist_32(b)]),
    "C": lambda b: np.concatenate([_hsv_hist_512(b), _sobel_hist_32(b),
                                   _band_hists_192(b)]),
}
FEATURE_LABEL = {"A": "color", "B": "color + texture", "C": "+ vertical bands"}
FEATURE_SUFFIX = {"A": "color", "B": "color_texture", "C": "color_bands"}


# --------------------------------------------------------------------------
# Data loading. 1/8-scale decode then resize, per the Table 4 note.
# --------------------------------------------------------------------------
def _load_split(root):
    """Return [(path, label)], fire=1. Directory names are 'fire' / 'nofire'."""
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        label = FIRE if d.name.lower() == "fire" else NO_FIRE
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rows.append((p, label))
    return rows


def _read_frame(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:                      # 1/8 decode unsupported for this file
        bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.resize(bgr, WORK_SIZE, interpolation=cv2.INTER_LINEAR)


def _extract(rows, fs_key, cache):
    """Feature matrix for one split and one feature set, decoding each file once."""
    X, y, kept = [], [], []
    fn = FEATURE_SETS[fs_key]
    for path, label in rows:
        frame = cache.get(path)
        if frame is None:
            continue
        X.append(fn(frame))
        y.append(label)
        kept.append(path)
    return np.stack(X), np.array(y), kept


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == FIRE) & (y_pred == FIRE)))
    tn = int(np.sum((y_true == NO_FIRE) & (y_pred == NO_FIRE)))
    fp = int(np.sum((y_true == NO_FIRE) & (y_pred == FIRE)))
    fn = int(np.sum((y_true == FIRE) & (y_pred == NO_FIRE)))
    return tp, tn, fp, fn


def metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec and (prec + rec) else 0.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": round((tp + tn) / n * 100, 2) if n else 0.0,
        "Precision": round(prec * 100, 2) if prec is not None else None,
        "Recall": round(rec * 100, 2),
        "F1": round(f1 * 100, 2),
        "FPR": round(fp / (fp + tn) * 100, 2) if fp + tn else 0.0,
    }


def make_model(kind, param, seed):
    if kind == "lr":
        return LogisticRegression(C=param, class_weight="balanced",
                                  max_iter=1000, random_state=seed)
    return RandomForestClassifier(n_estimators=300, max_depth=param,
                                  class_weight="balanced", random_state=seed,
                                  n_jobs=-1)


def model_size_bytes(model):
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.tell()


# --------------------------------------------------------------------------
# Cascade: hand-set proposals verified by a learned classifier
# --------------------------------------------------------------------------
def _proposals(frame, h=50.0):
    return [b["rect"] for b in run_heuristic_pipeline(frame, None, h)]


def _crop_features(frame, rect):
    x, y, w, hh = rect
    x, y = max(0, x), max(0, y)
    crop = frame[y:y + hh, x:x + w]
    if crop.size == 0:
        return None
    return FEATURE_SETS[VERIFIER_FEATURES](crop)


def build_cascade(rows, cache, log):
    """Fit verifiers on training-split proposals; return fitted (lr, rf)."""
    X, y = [], []
    for path, label in rows:
        frame = cache.get(path)
        if frame is None:
            continue
        for rect in _proposals(frame):
            f = _crop_features(frame, rect)
            if f is not None:
                X.append(f)
                y.append(label)
    log(f"  cascade verifier training crops: {len(X)} "
        f"({int(np.sum(np.array(y) == FIRE))} from fire images)")
    if not X or len(set(y)) < 2:
        return None, None, len(X)
    X, y = np.stack(X), np.array(y)
    lr = make_model("lr", 1.0, SEED).fit(X, y)
    rf = make_model("rf", None, SEED).fit(X, y)
    return lr, rf, len(X)


def cascade_predict(rows, cache, verifier):
    """Image positive iff at least one proposal is accepted by the verifier."""
    y_true, y_pred, n_prop, n_kept = [], [], 0, 0
    for path, label in rows:
        frame = cache.get(path)
        if frame is None:
            continue
        rects = _proposals(frame)
        n_prop += len(rects)
        feats = [f for f in (_crop_features(frame, r) for r in rects) if f is not None]
        if feats:
            accepted = int(np.sum(verifier.predict(np.stack(feats)) == FIRE))
        else:
            accepted = 0
        n_kept += accepted
        hit = accepted > 0
        y_true.append(label)
        y_pred.append(FIRE if hit else NO_FIRE)
    return np.array(y_true), np.array(y_pred), n_prop, n_kept


# --------------------------------------------------------------------------
# Joint latency run (Section 4.5): one machine, one run, 300 timed calls each
# --------------------------------------------------------------------------
def joint_latency(frames, models, reps=10):
    def timed(fn):
        for f in frames:                       # warm-up, discarded
            fn(f)
        samples = []
        for _ in range(reps):
            for f in frames:
                t0 = time.perf_counter()
                fn(f)
                samples.append((time.perf_counter() - t0) * 1000.0)
        return {"median_ms": round(statistics.median(samples), 2),
                "mean_ms": round(statistics.fmean(samples), 2),
                "stdev_ms": round(statistics.stdev(samples), 2),
                "n": len(samples)}

    out = {"hand_set_pipeline": timed(lambda f: run_heuristic_pipeline(f, None, 50.0))}
    fe = FEATURE_SETS["C"]
    out["feature_extraction_C"] = timed(fe)
    for name, model in models.items():
        out[name] = timed(lambda f, m=model: m.predict(fe(f).reshape(1, -1)))
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cascade", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-seeds", action="store_true")
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not d.exists():
            sys.exit(f"missing split directory: {d}")

    tr, va, te = _load_split(TRAIN_DIR), _load_split(VAL_DIR), _load_split(TEST_DIR)
    log("=== train_v2_baselines.py (Section 4.5) ===")
    for name, rows in (("train", tr), ("val", va), ("test", te)):
        log(f"  {name}: {len(rows)} images "
            f"({sum(l == FIRE for _, l in rows)} fire, "
            f"{sum(l == NO_FIRE for _, l in rows)} no-fire)")

    log("\nDecoding at 1/8 scale and resizing to 640x480...")
    cache = {}
    for rows in (tr, va, te):
        for path, _ in rows:
            f = _read_frame(path)
            if f is not None:
                cache[path] = f
    log(f"  decoded {len(cache)} frames")

    feats = {}
    for fs in ("A", "B", "C"):
        log(f"Extracting feature set {fs}...")
        for split, rows in (("train", tr), ("val", va), ("test", te)):
            feats[(split, fs)] = _extract(rows, fs, cache)

    results = {
        "_protocol": {
            "fit_split": "train (1887 images)",
            "select_split": "val (402 images), one hyperparameter, F1 only",
            "report_split": "test (410 images), evaluated once",
            "input": "decoded at 1/8 scale, resized to 640x480",
            "seed": SEED, "n_seeds": N_SEEDS,
            "verifier_features": VERIFIER_FEATURES,
            "sklearn": sklearn.__version__,
            "opencv": cv2.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
        },
        "null_baselines": {}, "learned": {}, "cascade": {}, "latency": {},
    }

    # ---- null baselines on the test split -------------------------------
    _, y_te_A, _ = feats[("test", "A")]
    results["null_baselines"]["constant_positive"] = metrics(
        y_te_A, np.full_like(y_te_A, FIRE))
    results["null_baselines"]["constant_negative"] = metrics(
        y_te_A, np.full_like(y_te_A, NO_FIRE))

    # ---- learned baselines ----------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind, grid in (("lr", LR_GRID), ("rf", RF_GRID)):
        for fs in ("A", "B", "C"):
            name = kind + "_" + FEATURE_SUFFIX[fs]
            X_tr, y_tr, _ = feats[("train", fs)]
            X_va, y_va, _ = feats[("val", fs)]
            X_te, y_te, te_paths = feats[("test", fs)]

            # selection: one hyperparameter, validation F1 only
            scored = []
            for param in grid:
                m = make_model(kind, param, SEED).fit(X_tr, y_tr)
                scored.append((metrics(y_va, m.predict(X_va))["F1"], param))
            best_f1, best = max(scored, key=lambda t: t[0])

            model = make_model(kind, best, SEED).fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            row = metrics(y_te, y_pred)
            row.update({
                "feature_set": fs, "feature_label": FEATURE_LABEL[fs],
                "selected_hyperparameter": {"lr": "C", "rf": "max_depth"}[kind],
                "selected_value": best, "validation_F1": best_f1,
                "size_bytes": model_size_bytes(model),
            })

            if not args.skip_seeds:
                f1s = []
                for s in range(N_SEEDS):
                    ms = make_model(kind, best, s).fit(X_tr, y_tr)
                    f1s.append(metrics(y_te, ms.predict(X_te))["F1"])
                row["seed_sweep"] = {
                    "n": N_SEEDS, "mean_F1": round(statistics.fmean(f1s), 2),
                    "std_F1": round(statistics.pstdev(f1s), 2),
                    "min_F1": min(f1s), "max_F1": max(f1s), "all": f1s,
                }

            results["learned"][name] = row
            with open(OUT_DIR / f"test_predictions_{name}.csv", "w",
                      newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["path", "label", "prediction"])
                for p, t, q in zip(te_paths, y_te, y_pred):
                    w.writerow([str(p), int(t), int(q)])

            log(f"{name:18} | {FEATURE_LABEL[fs]:16} | "
                f"{row['selected_hyperparameter']}={best} | "
                f"F1 {row['F1']:5.2f}% | acc {row['Accuracy']:5.2f}% | "
                f"FPR {row['FPR']:5.2f}% | {row['size_bytes']/1024:8.1f} kB")

    # ---- cascade ---------------------------------------------------------
    if args.skip_cascade:
        results["cascade"]["_skipped"] = True
    else:
        log("\nFitting cascade verifiers on training-split proposals...")
        lr_v, rf_v, n_crops = build_cascade(tr, cache, log)
        if lr_v is None:
            results["cascade"]["_error"] = "no usable training proposals"
        else:
            results["cascade"]["_training_crops"] = n_crops
            for vname, v in (("logistic_verifier", lr_v), ("forest_verifier", rf_v)):
                yt, yp, n_prop, n_kept = cascade_predict(te, cache, v)
                row = metrics(yt, yp)
                row.update({"proposals_on_test": n_prop, "proposals_accepted": n_kept})
                results["cascade"][vname] = row
                log(f"cascade/{vname:18} | F1 {row['F1']:5.2f}% | "
                    f"{n_kept}/{n_prop} proposals accepted")

    # ---- joint latency ---------------------------------------------------
    if args.skip_latency:
        results["latency"]["_skipped"] = True
    else:
        log("\nJoint latency run (30 test frames x 10 reps, warm-up discarded)...")
        frames = [cache[p] for p, _ in te[:30] if p in cache]
        models = {}
        X_tr_C, y_tr_C, _ = feats[("train", "C")]
        for kind in ("lr", "rf"):
            key = f"{kind}_color_bands"
            if key in results["learned"]:
                models[key] = make_model(
                    kind, results["learned"][key]["selected_value"], SEED
                ).fit(X_tr_C, y_tr_C)
        results["latency"] = joint_latency(frames, models)
        results["latency"]["_note"] = (
            "Single machine, single run, so only ratios between these rows are "
            "quotable. The per-model rows are end-to-end (feature extraction "
            "plus prediction); feature_extraction_C is reported separately so "
            "the shared component can be quoted on its own, as Section 4.5 does.")
        for k, v in results["latency"].items():
            if isinstance(v, dict):
                log(f"  {k:24} median {v['median_ms']:7.2f} ms  "
                    f"(sd {v['stdev_ms']:.2f}, n={v['n']})")

    out = OUT_DIR / "v2_baseline_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"\nWrote {out}")
    log("Table 5 of the manuscript must be regenerated from this file; the "
        "values it currently prints came from the superseded script.")


if __name__ == "__main__":
    main()
