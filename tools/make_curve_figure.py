"""Draw the operating curves of Section 4.5 from v3_operating_curve.json.

Two panels, on the 194 clips that contributed no training gradient: the receiver
operating characteristic and the precision-recall curve. Both are drawn from the
points in the artifact rather than recomputed here, so the figure cannot drift
from the numbers the manuscript reports.

The reported operating point is marked on both. That point is not a choice made
when drawing the curve: it is where the deployed rule of Algorithm 2 sits, since
thresholding the clip score at one half reproduces its verdict on every clip.

Usage:  python tools/make_curve_figure.py
Writes: v3_operating_curve.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "Proje_Kodlari" / "evaluation_results" / "v3_deep_edge" / "v3_operating_curve.json"
OUT = ROOT / "v3_operating_curve.png"


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed:  python -m pip install matplotlib")
        return 1

    if not ART.exists():
        print(f"{ART} not found; run tools/clip_score_curve.py --report first")
        return 1
    blob = json.loads(ART.read_text(encoding="utf-8"))
    d = blob["held_out"]

    roc = d["roc_points"]
    pr = d["pr_points"]
    fpr = [p[0] * 100 for p in roc]
    tpr = [p[1] * 100 for p in roc]
    rec = [p[0] * 100 for p in pr]
    prec = [p[1] * 100 for p in pr]

    # the deployed operating point: the last precision-recall point whose
    # threshold is still above one half
    op = None
    for r, p, t in pr:
        if t > 0.5:
            op = (r * 100, p * 100)
    prior = 100.0 * d["n_positive"] / (d["n_positive"] + d["n_negative"])

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0), dpi=200)

    ax[0].plot([0, 100], [0, 100], color="#bbbbbb", lw=1.0, ls="--",
               label="chance")
    ax[0].plot(fpr, tpr, color="#1a1a1a", lw=1.8,
               label=f"deep edge (AUROC {d['AUROC']:.3f})")
    if op:
        # the ROC point matching the deployed rule
        fp_at = next((p[0] * 100 for p in roc if abs(p[1] * 100 - op[0]) < 1e-6), None)
        if fp_at is not None:
            ax[0].plot([fp_at], [op[0]], "o", ms=6, mfc="white", mec="#c62828", mew=1.8,
                       label="reported operating point")
    ax[0].set_xlabel("False positive rate (%)")
    ax[0].set_ylabel("True positive rate (%)")
    ax[0].set_title("Receiver operating characteristic", fontsize=10)

    ax[1].axhline(prior, color="#bbbbbb", lw=1.0, ls="--",
                  label=f"class prior ({prior:.1f}%)")
    ax[1].plot(rec, prec, color="#1a1a1a", lw=1.8,
               label=f"deep edge (AP {d['average_precision']:.3f})")
    if op:
        ax[1].plot([op[0]], [op[1]], "o", ms=6, mfc="white", mec="#c62828", mew=1.8,
                   label="reported operating point")
    ax[1].set_xlabel("Recall (%)")
    ax[1].set_ylabel("Precision (%)")
    ax[1].set_title("Precision-recall", fontsize=10)

    for a in ax:
        a.set_xlim(-2, 102)
        a.set_ylim(-2, 102)
        a.grid(True, lw=0.4, color="#e6e6e6")
        a.set_axisbelow(True)
        a.legend(loc="lower right", fontsize=8, framealpha=0.95)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"  held-out cohort: {d['n_positive']} fire, {d['n_negative']} no-fire")
    print(f"  AUROC {d['AUROC']}   average precision {d['average_precision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
