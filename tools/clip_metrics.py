"""Shared clip-level scoring for the V3 detector.

Both evaluators — `evaluate_v3_videos.py` (PyTorch) and
`tools/eval_clips_numpy.py` (NumPy, torch-free) — import this module, so the
two cannot drift apart in how they group clips, name metrics, or scale values.
Everything here is stored in percent.
"""

import math

# Decision-rule constants. Both evaluators must import these rather than
# redefining them; a divergence here silently changes every reported number.
FIRE_CLASS_INDEX = 0        # ImageFolder sorts alphabetically: fire=0, no_fire=1
SAMPLES_PER_SECOND = 2      # temporal sampling rate
CONSECUTIVE_FOR_ALARM = 3   # temporal smoothing filter
DEFAULT_FPS = 24.0          # fallback when the container reports no frame rate

# How each manifest split was consumed by `extract_training_frames.py`, which
# routes split=="train" to dataset/train and split=="test" to dataset/val, and
# never reads split=="validation". The released weights were produced under
# exactly this routing, so these roles describe the shipped model.
SPLIT_ROLE = {
    "train": "gradient_train",      # frames drove weight updates
    "test": "epoch_monitor",        # frames were the per-epoch monitor, no gradients
    "validation": "untouched",      # never read by any stage
}

# Which grouping is the headline, and why each of the others is reported.
GROUP_DEFINITIONS = {
    "unseen_by_gradient": ("epoch_monitor", "untouched"),
    "untouched": ("untouched",),
    "epoch_monitor": ("epoch_monitor",),
    "gradient_train": ("gradient_train",),
}
HEADLINE_GROUP = "unseen_by_gradient"


def wilson(successes, n, z=1.96):
    """95% Wilson score interval in percent; correct at k = 0 and k = n."""
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half) * 100, 2),
            round(min(1.0, centre + half) * 100, 2)]


def metrics(rows):
    """rows: iterable of dicts with 'label' ('fire'/'no_fire') and 'alarm' (bool)."""
    tp = sum(1 for r in rows if r["label"] == "fire" and r["alarm"])
    fn = sum(1 for r in rows if r["label"] == "fire" and not r["alarm"])
    fp = sum(1 for r in rows if r["label"] != "fire" and r["alarm"])
    tn = sum(1 for r in rows if r["label"] != "fire" and not r["alarm"])
    total = tp + tn + fp + fn
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "N_clips": total, "N_fire": tp + fn, "N_no_fire": tn + fp,
        "Accuracy": round(100 * (tp + tn) / total, 2) if total else 0.0,
        "Recall_TPR": round(100 * rec, 2),
        "Precision": round(100 * prec, 2),
        "F1": round(100 * f1, 2),
        "FPR": round(100 * fp / (fp + tn), 2) if (fp + tn) else 0.0,
        "Recall_TPR_wilson": wilson(tp, tp + fn),
        "FPR_wilson": wilson(fp, fp + tn),
    }


def group_and_score(rows):
    """rows: dicts with 'label', 'alarm' and the manifest 'split'."""
    for r in rows:
        r["role"] = SPLIT_ROLE.get(r.get("split"), "unknown")

    out = {}
    for name, roles in GROUP_DEFINITIONS.items():
        subset = [r for r in rows if r["role"] in roles]
        if subset:
            out[name] = metrics(subset)
    out["full_manifest_CONTAMINATED"] = metrics(rows)

    unknown = sorted({r.get("split") for r in rows if r["role"] == "unknown"})
    if unknown:
        out["_warning"] = (f"manifest contains split value(s) {unknown} that are not in "
                           f"SPLIT_ROLE; those clips are scored only in the contaminated total")
    return out


def print_summary(results):
    order = list(GROUP_DEFINITIONS) + ["full_manifest_CONTAMINATED"]
    print(f"\n{'group':<28}{'n':>5}{'fire':>6}{'TPR':>9}{'FPR':>8}{'F1':>8}{'Acc':>8}")
    print("-" * 72)
    for name in order:
        r = results.get(name)
        if not r:
            continue
        marker = " <- headline" if name == HEADLINE_GROUP else ""
        print(f"{name:<28}{r['N_clips']:>5}{r['N_fire']:>6}"
              f"{r['Recall_TPR']:>8.2f}%{r['FPR']:>7.2f}%{r['F1']:>7.2f}%{r['Accuracy']:>7.2f}%{marker}")
    if "_warning" in results:
        print(f"\nWARNING: {results['_warning']}")
