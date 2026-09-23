"""
The deep edge detector scored on the still-image test split of Sections 4.1 to
4.4, so that all three architectures have one cohort in common.

Table 5 reports a progression rather than a comparison, because V1 and V2 were
scored on still images and the deep edge detector on clips. The obstacle was
never symmetric. Running V1 and V2 over the 483-clip corpus means decoding it;
running the deep edge detector over 410 still images means 410 forward passes,
which is minutes. This program does the cheap direction, and the result is a
like-for-like row for the deep edge detector in the cohort the other two were
measured in.

Two properties of the measurement have to be stated with it or it will be
misread. The clip-level alarm rule of Algorithm 2 requires three consecutive
positive samples and is undefined on a single image, so the verdict here is the
bare argmax of the same weights; this is the detector's per-frame classifier and
not the deployed alarm. And the weights were fitted on frames of low-altitude
aerial video, whereas this cohort mixes ground-level and aerial photography
across three orders of magnitude of resolution. The row is therefore a transfer
measurement. It is reported whichever way it falls, under the standard the paper
applies to the method it replaces.

Usage:  python tools/v3_on_still_images.py [--budget SECONDS]
Writes: Proje_Kodlari/evaluation_results/v3_deep_edge/v3_on_still_images.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from clip_metrics import FIRE_CLASS_INDEX  # noqa: E402
from image_metrics import (  # noqa: E402
    DATASET_ROOT, binary_metrics, environment_record, list_split, null_baselines,
)
from torch_free_mobilenetv3 import (  # noqa: E402
    MobileNetV3SmallNumpy, load_state_dict, preprocess_bgr,
)

BASE = ROOT / "Proje_Kodlari"
WEIGHTS = BASE / "evaluation_results" / "v3_deep_edge" / "v3_mobilenet.pth"
OUT = BASE / "evaluation_results" / "v3_deep_edge" / "v3_on_still_images.json"
CACHE = BASE / "evaluation_results" / "v3_deep_edge" / "_v3_still_predictions.jsonl"
PRED = BASE / "evaluation_results" / "v3_deep_edge" / "test_predictions_v3_still.csv"


def load_done() -> dict:
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["path"]] = r
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=150.0)
    args = ap.parse_args()

    images = list_split(ROOT / DATASET_ROOT, "test")   # [(path, label 1=fire)]
    done = load_done()
    todo = [(p, y) for p, y in images if str(p) not in done]

    if todo:
        model = MobileNetV3SmallNumpy(load_state_dict(WEIGHTS))
        started = time.time()
        with CACHE.open("a", encoding="utf-8") as fh:
            for p, y in todo:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                logits = model.forward(preprocess_bgr(img))
                pred = 1 if int(np.argmax(logits)) == FIRE_CLASS_INDEX else 0
                fh.write(json.dumps({"path": str(p), "label": y, "pred": pred}) + "\n")
                fh.flush()
                if time.time() - started > args.budget:
                    break
        done = load_done()

    left = len(images) - len(done)
    if left:
        print(f"{len(done)}/{len(images)} scored, {left} left; rerun to continue")
        return 0

    rows = [done[str(p)] for p, _ in images]
    tp = sum(1 for r in rows if r["label"] == 1 and r["pred"] == 1)
    fn = sum(1 for r in rows if r["label"] == 1 and r["pred"] == 0)
    fp = sum(1 for r in rows if r["label"] == 0 and r["pred"] == 1)
    tn = sum(1 for r in rows if r["label"] == 0 and r["pred"] == 0)

    out = {"_protocol": {
        "experiment": "Section 4.5 - the deep edge classifier on the still-image "
                      "test split, giving all three architectures one cohort in common",
        "decision_rule": "bare argmax of the two-class head; the three-consecutive"
                         "-sample alarm of Algorithm 2 is undefined on a single image",
        "domain": "weights fitted on low-altitude aerial video frames, applied to a "
                  "mixed ground and aerial photographic corpus: a transfer measurement",
        "n_images": len(rows), "n_fire": tp + fn, "n_no_fire": fp + tn,
        "environment": environment_record(),
    },
        "deep_edge_still": dict(binary_metrics(tp, tn, fp, fn), TP=tp, TN=tn, FP=fp, FN=fn),
        "null_baselines": null_baselines(tp + fn, fp + tn),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    PRED.write_text("path,label,prediction\n" + "".join(
        f"{r['path']},{r['label']},{r['pred']}\n" for r in rows), encoding="utf-8")

    m = out["deep_edge_still"]
    print(f"Wrote {OUT}\n")
    print(f"deep edge on {len(rows)} still images "
          f"({tp + fn} fire, {fp + tn} no-fire)")
    print(f"   Recall={m['Recall']}%  Precision={m['Precision']}%  "
          f"F1={m['F1']}%  Accuracy={m['Accuracy']}%  FPR={m['FPR']}%")
    nb = out["null_baselines"]
    print(f"   null constant-positive F1={nb['constant_positive']['F1']}%, "
          f"constant-negative Accuracy={nb['constant_negative']['Accuracy']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
