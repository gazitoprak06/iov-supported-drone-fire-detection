"""Render Figure 11: held-out vs contaminated confusion matrices for the V3 detector.

Reads the metrics produced by `tools/eval_clips_numpy.py --report` (or by
`evaluate_v3_videos.py`) so the figure can never drift from the reported table.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# CMC figure rules (author instructions, "Resolution and Format" and "Figure
# Labels and Captions"): maximum width 16.51 cm = 6.50 in, maximum height 20 cm,
# combo artwork at 600 dpi or better, labels no smaller than 8 pt and no larger
# than the 11 pt main text, standard sans fonts only, no alpha channel.
# The figure is therefore authored at its final printed size so that the point
# sizes below are the point sizes the reader sees; scaling a larger canvas down
# is what pushes labels under the 8 pt floor.
FIG_WIDTH_IN = 6.5
DPI = 600
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.facecolor": "white",
})

RESULTS = Path("Proje_Kodlari/evaluation_results/v3_deep_edge/v3_results_by_split.json")
OUT = Path("v3_confusion_matrix.png")

PANELS = [
    ("unseen_by_gradient", "(a) 194 unseen clips\nreported result", "#2e7d32"),
    ("gradient_train", "(b) 289 fitted clips\nmemorisation ceiling", "#ef6c00"),
    ("full_manifest_CONTAMINATED", "(c) all 483 clips\ncontaminated control", "#c62828"),
]


def panel(ax, res, title, colour):
    cm = np.array([[res["TP"], res["FN"]],
                   [res["FP"], res["TN"]]], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)

    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]*100:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")

    ax.set_xticks([0, 1], ["Alarm", "No alarm"])
    ax.set_yticks([0, 1], ["Fire", "No fire"])
    ax.set_xlabel("Predicted", labelpad=2)
    ax.set_ylabel("Actual", labelpad=2)
    ax.set_title(title, fontsize=8, color=colour, pad=4)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.tick_params(length=2, width=0.5, pad=2)

    sub = (f"TPR {res['Recall_TPR']:.2f}%  FPR {res['FPR']:.2f}%\n"
           f"F1 {res['F1']:.2f}%  Acc {res['Accuracy']:.2f}%")
    ax.text(0.5, -0.42, sub, transform=ax.transAxes, ha="center",
            va="top", fontsize=8, linespacing=1.4)


def main():
    results = json.loads(RESULTS.read_text())
    fig, axes = plt.subplots(1, len(PANELS),
                             figsize=(FIG_WIDTH_IN, 2.85), dpi=DPI)
    for ax, (key, title, colour) in zip(axes, PANELS):
        panel(ax, results[key], title, colour)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.84, bottom=0.30, wspace=0.60)
    fig.savefig(OUT, dpi=DPI, facecolor="white", transparent=False)

    from PIL import Image
    im = Image.open(OUT).convert("RGB")      # drop any alpha channel
    im.save(OUT)
    w, h = im.size
    print(f"Wrote {OUT}: {w}x{h} px = {w/FIG_WIDTH_IN:.0f} dpi at "
          f"{FIG_WIDTH_IN*2.54:.2f} cm wide, {h/DPI*2.54:.1f} cm tall")
    assert w / FIG_WIDTH_IN >= 600, "below the CMC 600 dpi combo requirement"
    assert h / DPI * 2.54 <= 20, "taller than the CMC 20 cm limit"


if __name__ == "__main__":
    main()
