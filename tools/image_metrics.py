"""
Shared definitions for every image-level experiment (Sections 4.1 to 4.4).

This module plays the same role for the still-image cohort that
tools/clip_metrics.py plays for the clip cohort: the split walker, the
confusion-matrix arithmetic, the null baselines and the percentage
convention live here once, so that eval_image_level.py,
resolution_control.py and saturation_sweep.py cannot drift apart in how
they count an image or round a rate.
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset layout
# ---------------------------------------------------------------------------

# Folder-level class labels; the dataset carries no bounding-box annotation,
# so every experiment below is image-level binary classification.
FIRE_DIRS = ("fire",)
NO_FIRE_DIRS = ("nofire", "no_fire")

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

# The reference altitude passed by every still-image experiment. The cohorts
# carry no altitude metadata, so h is fixed here and A_min is therefore 600 px
# and d_merge 75 px throughout Sections 4.1 to 4.4.
REFERENCE_ALTITUDE_M = 50.0

DATASET_ROOT = Path("Proje_Kodlari/data/AR Souri Dataset")


def list_split(root: Path, split: str) -> list[tuple[Path, int]]:
    """Return [(path, label)] for one split, label 1 = fire, 0 = no fire.

    Anything whose suffix is not a supported image extension is skipped, which
    is how the stray desktop.ini entry in val/fire is excluded. The listing is
    sorted so that two runs visit the images in the same order.
    """
    rows: list[tuple[Path, int]] = []
    for dirnames, label in ((FIRE_DIRS, 1), (NO_FIRE_DIRS, 0)):
        for dirname in dirnames:
            d = root / split / dirname
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in SUPPORTED_SUFFIXES:
                    rows.append((p, label))
    return sorted(rows, key=lambda r: str(r[0]))


# ---------------------------------------------------------------------------
# Confusion-matrix arithmetic
# ---------------------------------------------------------------------------

def _pct(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 2)


def binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Confusion matrix -> the metric set reported in Tables 1 to 3.

    Precision is None (rendered as an em dash in the manuscript) when the rule
    emits no positive prediction, which is the case for the constant-negative
    baseline. F1 is likewise None when precision is undefined.

    F1 is formed from the unrounded precision and recall and rounded once, not
    from the rounded percentages. The distinction is not cosmetic: rounding
    twice moves the F1 of the headline clip cohort from 95.45 to 95.46, and it
    would put this module permanently one hundredth of a point away from
    clip_metrics.metrics, which computes the same quantity for the clip cohort.
    tests/test_metrics.py asserts that the two agree.
    """
    precision = _pct(tp, tp + fp)
    recall = _pct(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        p_raw = tp / (tp + fp) if (tp + fp) else 0.0
        r_raw = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = round(100.0 * 2.0 * p_raw * r_raw / (p_raw + r_raw), 2) \
            if (p_raw + r_raw) else 0.0
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": _pct(tp + tn, tp + tn + fp + fn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": _pct(tn, tn + fp),
        "FPR": _pct(fp, fp + tn),
    }


def null_baselines(n_fire: int, n_no_fire: int) -> dict:
    """The two constant predictors that never inspect the image."""
    return {
        "constant_positive": binary_metrics(
            tp=n_fire, tn=0, fp=n_no_fire, fn=0
        ),
        "constant_negative": binary_metrics(
            tp=0, tn=n_no_fire, fp=0, fn=n_fire
        ),
    }


def confusion_from_decisions(decisions: list[tuple[int, bool]]) -> dict:
    """[(label, predicted_positive)] -> metrics."""
    tp = sum(1 for lab, pred in decisions if lab == 1 and pred)
    fn = sum(1 for lab, pred in decisions if lab == 1 and not pred)
    fp = sum(1 for lab, pred in decisions if lab == 0 and pred)
    tn = sum(1 for lab, pred in decisions if lab == 0 and not pred)
    return binary_metrics(tp, tn, fp, fn)


# ---------------------------------------------------------------------------
# Paired significance test (Section 4.3)
# ---------------------------------------------------------------------------

def mcnemar(b: int, c: int) -> dict:
    """McNemar's test with Yates continuity correction on a paired 2x2.

    b = decisions the first condition got right and the second got wrong,
    c = decisions the second got right and the first got wrong.
    Returns chi-square on one degree of freedom and its p-value.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": None, "p": None}
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Survival function of chi-square with 1 dof, expressed through erfc so
    # that this module carries no SciPy dependency.
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": round(chi2, 2), "p": round(p, 4)}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def environment_record() -> dict:
    """The header written into every result file, so that which experiment ran
    in which environment is recorded rather than left to be inferred."""
    import platform
    import sys

    rec = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "argv": sys.argv,
    }
    try:
        import cv2

        rec["opencv"] = cv2.__version__
    except Exception:
        pass
    try:
        import numpy

        rec["numpy"] = numpy.__version__
    except Exception:
        pass
    return rec
