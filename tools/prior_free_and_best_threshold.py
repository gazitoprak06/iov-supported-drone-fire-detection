"""Two quantities the null-baseline argument of Section 4.1 needs.

The headline comparison is that the hand-set rule loses to a constant predictor
on F1 and on accuracy. Both of those metrics depend on the class prior, so a
reader is entitled to ask whether the finding is a property of the detector or
of a 38.78% positive rate. Informedness and the Matthews correlation do not
depend on the prior: both are zero for any rule whose predictions are
independent of the label, whatever the balance, and neither can be gamed by a
constant rule. Reporting them turns "loses to a constant predictor on one
metric" into "is indistinguishable from chance", which is both stronger and
harder to argue with.

The second quantity answers a different objection. Section 4.4 shows that a
model fitted on a colour histogram of the same images reaches a far higher F1,
and concludes that the failure is in the decision rule. A reviewer can reply
that the histogram is not the rule's evidence: it is HSV-only, it discards
connectivity, and it summarises the whole frame rather than the candidate
regions, so the fitted model may be reading scene context the rule never sees.

This program bounds what any rule over the pipeline's own output could achieve.
The released per-image record carries the surviving box count for every image
and every ablation. Sweeping a threshold over that count, and taking the best
F1 any threshold attains *with the test labels in hand*, gives an upper bound on
the performance of every possible decision rule that consumes only what the
pipeline passes forward. The bound is deliberately optimistic: it is chosen on
the data it is evaluated on, so it can only overstate. If even that bound sits
near the constant predictor, the pipeline's output statistic is close to
exhausted, and the gap to the fitted model cannot be explained by threshold
choice alone.

Usage:  python tools/prior_free_and_best_threshold.py
Writes: Proje_Kodlari/evaluation_results/v1_image_level/v1_prior_free.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from image_metrics import binary_metrics, environment_record  # noqa: E402

BASE = ROOT / "Proje_Kodlari" / "evaluation_results" / "v1_image_level"
PRED = BASE / "test_predictions_v1_ablation.csv"
OUT = BASE / "v1_prior_free.json"


def prior_free(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Informedness and the Matthews correlation.

    Both are zero when the prediction carries no information about the label,
    for any class balance, which is exactly the property the constant-predictor
    comparison is reaching for. Informedness is recall + specificity - 1.
    """
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {"informedness": round(tpr + tnr - 1.0, 4),
            "mcc": round(mcc, 4)}


def best_threshold(counts, labels) -> dict:
    """Best F1 attainable by thresholding the box count, chosen on these labels.

    The rule is "positive iff count >= t". Every distinct count is tried, plus
    the degenerate thresholds that call everything positive or nothing positive,
    so the sweep covers the whole family.
    """
    best = None
    for t in sorted(set(counts)) + [max(counts) + 1]:
        tp = sum(1 for c, y in zip(counts, labels) if y and c >= t)
        fn = sum(1 for c, y in zip(counts, labels) if y and c < t)
        fp = sum(1 for c, y in zip(counts, labels) if not y and c >= t)
        tn = sum(1 for c, y in zip(counts, labels) if not y and c < t)
        m = binary_metrics(tp, tn, fp, fn)
        f1 = m["F1"] or 0.0
        if best is None or f1 > best["F1"]:
            best = dict(m, F1=f1, threshold=t, TP=tp, TN=tn, FP=fp, FN=fn,
                        **prior_free(tp, tn, fp, fn))
    return best


def main() -> int:
    if not PRED.exists():
        print(f"{PRED} not found; run tools/eval_image_level.py first")
        return 1

    rows = list(csv.DictReader(PRED.open(encoding="utf-8")))
    configs = [k[len("pred::"):] for k in rows[0] if k.startswith("pred::")]
    labels = [int(r["label"]) for r in rows]

    out = {"_protocol": {
        "experiment": "Section 4.1 - prior-free scores for Table 1, and an "
                      "optimistic bound on any rule over the pipeline's own "
                      "box count",
        "source": "test_predictions_v1_ablation.csv; no image is re-processed",
        "bound_is_optimistic": "the threshold is chosen on the same labels it "
                               "is scored against, so it can only overstate",
        "n_images": len(rows),
        "n_fire": sum(labels), "n_no_fire": len(labels) - sum(labels),
        "environment": environment_record(),
    }, "as_released": {}, "best_threshold_on_box_count": {}}

    for cfg in configs:
        pred = [int(r[f"pred::{cfg}"]) for r in rows]
        tp = sum(1 for p, y in zip(pred, labels) if p and y)
        fn = sum(1 for p, y in zip(pred, labels) if not p and y)
        fp = sum(1 for p, y in zip(pred, labels) if p and not y)
        tn = sum(1 for p, y in zip(pred, labels) if not p and not y)
        out["as_released"][cfg] = dict(binary_metrics(tp, tn, fp, fn),
                                       **prior_free(tp, tn, fp, fn))

        counts = [int(r[f"nbox::{cfg}"]) for r in rows]
        out["best_threshold_on_box_count"][cfg] = best_threshold(counts, labels)

    n_fire, n_no = sum(labels), len(labels) - sum(labels)
    out["null_baselines"] = {
        "constant_positive": dict(binary_metrics(n_fire, 0, n_no, 0),
                                  **prior_free(n_fire, 0, n_no, 0)),
        "constant_negative": dict(binary_metrics(0, n_no, 0, n_fire),
                                  **prior_free(0, n_no, 0, n_fire)),
    }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}\n")
    print(f"{'configuration':34s} {'F1':>7s} {'MCC':>8s} {'informed':>9s}"
          f" | {'best-t F1':>9s} {'best-t MCC':>10s}")
    for cfg in configs:
        a = out["as_released"][cfg]
        b = out["best_threshold_on_box_count"][cfg]
        print(f"  {cfg:32s} {a['F1']:7.2f} {a['mcc']:8.4f} {a['informedness']:9.4f}"
              f" | {b['F1']:9.2f} {b['mcc']:10.4f}")
    nb = out["null_baselines"]
    print(f"\n  {'constant positive':32s} {nb['constant_positive']['F1']:7.2f} "
          f"{nb['constant_positive']['mcc']:8.4f} "
          f"{nb['constant_positive']['informedness']:9.4f}")
    print(f"  {'constant negative':32s} {'—':>7s} "
          f"{nb['constant_negative']['mcc']:8.4f} "
          f"{nb['constant_negative']['informedness']:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
