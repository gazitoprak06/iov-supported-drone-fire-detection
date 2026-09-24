"""Draw Fig. 5, the deep edge decision path.

The manuscript carries this diagram as a Mermaid source block, which only a
browser renders. Converting the manuscript to Word or PDF therefore needs the
same diagram as a picture, and generating it from a description here keeps the
two from drifting: the boxes below are the nodes of that block and the arrows
are its edges, in the same order.

Only the shaded path from the camera feed to the alarm is evaluated in Section
4.5. The Grad-CAM branch is drawn dashed because it is an operator-facing
output that no metric in the paper scores.

Usage:  python tools/make_decision_path_figure.py
Writes: fig_decision_path.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fig_decision_path.png"

# label, centre, (width, height), face, edge, dashed
NODES = {
    "A": ("UAV camera feed,\nsampled at 2 frames/s", (5.0, 11.2), (3.5, 1.15),
          "#e1f5fe", "#03a9f4", False),
    "B": ("Resize to 224$\\times$224,\nnormalize", (5.0, 9.3), (3.1, 1.15),
          "#ffffff", "#9e9e9e", False),
    "C": ("MobileNetV3-Small\nedge CNN", (5.0, 7.4), (3.1, 1.15),
          "#fff3e0", "#ff9800", False),
    "E": ("Grad-CAM heatmap\nfor the operator", (8.4, 5.4), (3.1, 1.15),
          "#f5f5f5", "#9e9e9e", True),
    "F": ("Reset counter,\nno alarm", (1.5, 3.2), (2.7, 1.15),
          "#ffffff", "#9e9e9e", False),
    "G": ("Temporal smoothing:\n3 consecutive", (5.2, 3.2), (3.3, 1.15),
          "#ffffff", "#9e9e9e", False),
    "H": ("Clip-level fire alarm", (5.2, 1.2), (3.3, 1.0),
          "#e8f5e9", "#4caf50", False),
}
DIAMOND = ("argmax = fire?", (3.0, 5.4), (3.2, 1.3), "#ffffff", "#9e9e9e")

EDGES = [("A", "B", "", False), ("B", "C", "", False), ("C", "D", "", False),
         ("C", "E", "", True), ("D", "F", "No", False), ("D", "G", "Yes", False),
         ("G", "H", "", False)]


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
    except ImportError:
        print("matplotlib is not installed:  python -m pip install matplotlib")
        return 1

    fig, ax = plt.subplots(figsize=(7.0, 8.4), dpi=200)
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0.2, 12.2)
    ax.axis("off")

    boxes = {}
    for key, (label, (cx, cy), (w, h), face, edge, dashed) in NODES.items():
        p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.22",
                           linewidth=1.8, edgecolor=edge, facecolor=face,
                           linestyle="--" if dashed else "-", zorder=2)
        ax.add_patch(p)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5, zorder=3)
        boxes[key] = (cx, cy, w, h)

    label, (cx, cy), (w, h), face, edge = DIAMOND
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy),
                          (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, linewidth=1.8, edgecolor=edge,
                         facecolor=face, zorder=2))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5, zorder=3)
    boxes["D"] = (cx, cy, w, h)

    def anchor(a, b):
        """Leave each box from the side that faces the other one."""
        ax_, ay, aw, ah = boxes[a]
        bx, by, bw, bh = boxes[b]
        if abs(bx - ax_) > aw / 2 + 0.3 and by < ay:      # diagonal
            return (ax_ + (aw / 2) * (1 if bx > ax_ else -1), ay - ah / 4), \
                   (bx, by + bh / 2)
        return (ax_, ay - ah / 2), (bx, by + bh / 2)

    for a, b, text, dashed in EDGES:
        (x0, y0), (x1, y1) = anchor(a, b)
        col = "#9e9e9e" if dashed else "#555555"
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.4, color=col, zorder=1,
                                     linestyle="--" if dashed else "-",
                                     shrinkA=2, shrinkB=2))
        if text:
            ax.text((x0 + x1) / 2 + 0.28, (y0 + y1) / 2, text,
                    fontsize=9.5, ha="left", va="center", color="#333333")

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
