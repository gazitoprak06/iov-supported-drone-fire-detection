"""
Sections 4.1 and 4.2: image-level evaluation against null baselines, and the
colour-space ablation.

Produces, in one pass over the test split and at native resolution:
  * the confusion matrix and metric set of the released configuration (Sec 4.1);
  * the two constant-predictor null baselines on the same split;
  * the seven colour-space configurations of Table 2 (Sec 4.2);
  * a per-image record, so that any cell of any row can be traced to a file.

The temporal layer is inactive throughout: the inputs are independent still
images, prev_frame is None, and by the argument following Eq. (1) the detector
reduces to its purely chromatic form.

The cohort spans three orders of magnitude in resolution and the morphological
chain runs at native size, so a full pass is long. Per-image records are
therefore appended to a JSONL checkpoint as they are produced and a rerun skips
whatever is already recorded; the aggregate files are written only once every
image of the split is present. Use --max-seconds to bound a single invocation.

Usage:  python tools/eval_image_level.py [--split test] [--max-seconds S]
Writes: Proje_Kodlari/evaluation_results/v1_image_level/
          _v1_image_level_perimage.jsonl   (checkpoint, one record per image)
          v1_image_level_metrics.json      (aggregate, written when complete)
          test_predictions_v1_ablation.csv (per-image table, written when complete)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from image_metrics import (  # noqa: E402
    DATASET_ROOT,
    REFERENCE_ALTITUDE_M,
    binary_metrics,
    environment_record,
    list_split,
    null_baselines,
)
from v1_configurable import ABLATION_CONFIGS, run_configurable  # noqa: E402

OUT_DIR = ROOT / "Proje_Kodlari" / "evaluation_results" / "v1_image_level"
CHECKPOINT_FMT = "_v1_image_level_perimage{suffix}.jsonl"
CHECKPOINT = OUT_DIR / CHECKPOINT_FMT.format(suffix="")


def load_checkpoint(path=None) -> dict[str, dict]:
    path = path or CHECKPOINT
    done: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["file"]] = rec
    return done


def aggregate(records: list[dict], split: str, n_images: int) -> dict:
    n_fire = sum(1 for r in records if r["label"] == 1)
    n_no_fire = len(records) - n_fire

    counts = {name: [0, 0, 0, 0] for name, _ in ABLATION_CONFIGS}
    spurious = {name: 0 for name, _ in ABLATION_CONFIGS}
    for r in records:
        for name, _ in ABLATION_CONFIGS:
            pred = bool(r[f"pred::{name}"])
            tp, tn, fp, fn = counts[name]
            if r["label"] == 1:
                tp, fn = (tp + 1, fn) if pred else (tp, fn + 1)
            else:
                if pred:
                    fp += 1
                else:
                    tn += 1
                spurious[name] += r[f"nbox::{name}"]
            counts[name] = [tp, tn, fp, fn]

    results = {
        "_protocol": {
            "experiment": "Sections 4.1 and 4.2 - image-level null baselines "
            "and colour-space ablation",
            "split": split,
            "n_images": n_images,
            "n_fire": n_fire,
            "n_no_fire": n_no_fire,
            "input": "decoded at native resolution, no normalization or augmentation",
            "altitude_m": REFERENCE_ALTITUDE_M,
            "temporal_layer": "disabled (still images, prev_frame=None)",
            "decision_rule": "image is positive iff the detector emits at least one box",
            "disabled_mask_convention": "a disabled colour space contributes an "
            "all-pass mask to the intersection. The white-hot bypass is itself an "
            "RGB-channel rule, so the default convention (follows_rgb) disables it "
            "together with the RGB mask; --whitehot always keeps it on in every "
            "row instead, which changes the three rows where RGB is off.",
            "environment": environment_record(),
        },
        "null_baselines": null_baselines(n_fire, n_no_fire),
        "ablation": {
            name: binary_metrics(*counts[name]) for name, _ in ABLATION_CONFIGS
        },
        "spurious_boxes_on_no_fire": spurious,
    }
    results["released_configuration"] = results["ablation"][
        "HSV + RGB + YCbCr (proposed)"
    ]
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default=str(ROOT / DATASET_ROOT))
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop this invocation gracefully after S seconds (0 = no bound)",
    )
    ap.add_argument("--restart", action="store_true", help="discard the checkpoint")
    ap.add_argument("--whitehot", choices=("follows_rgb", "always"), default="follows_rgb",
                    help="how the overexposure bypass is treated in ablation rows")
    args = ap.parse_args()

    rows = list_split(Path(args.dataset), args.split)
    if not rows:
        print(f"ERROR: no images found under {args.dataset}/{args.split}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.whitehot == "follows_rgb" else "_" + args.whitehot
    checkpoint = OUT_DIR / CHECKPOINT_FMT.format(suffix=suffix)
    if args.restart and checkpoint.exists():
        checkpoint.unlink()

    done = load_checkpoint(checkpoint)
    n_fire = sum(1 for _, lab in rows if lab == 1)
    print(
        f"{len(rows)} images ({n_fire} fire, {len(rows) - n_fire} no-fire), "
        f"native resolution; {len(done)} already recorded"
    )

    todo = [
        (p, lab)
        for p, lab in rows
        if str(p.relative_to(Path(args.dataset))) not in done
    ]

    t0 = time.time()
    with checkpoint.open("a", encoding="utf-8") as fh:
        for i, (path, label) in enumerate(todo, 1):
            if args.max_seconds and (time.time() - t0) > args.max_seconds:
                print(
                    f"\nTime bound reached; {len(done)}/{len(rows)} recorded. "
                    f"Rerun to continue."
                )
                break
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"ERROR: could not decode {path}")
                return 1
            h_px, w_px = frame.shape[:2]
            rec = {
                "file": str(path.relative_to(Path(args.dataset))),
                "label": label,
                "width": w_px,
                "height": h_px,
                "megapixels": round(w_px * h_px / 1e6, 4),
            }
            for name, active in ABLATION_CONFIGS:
                boxes = run_configurable(frame, None, REFERENCE_ALTITUDE_M,
                                         active=active, whitehot=args.whitehot)
                rec[f"pred::{name}"] = int(len(boxes) > 0)
                rec[f"nbox::{name}"] = len(boxes)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[rec["file"]] = rec
            if i % 10 == 0:
                el = time.time() - t0
                print(
                    f"  {len(done)}/{len(rows)}  ({el:.0f}s this run, "
                    f"{el / i:.2f}s/image)",
                    flush=True,
                )

    if len(done) < len(rows):
        print(f"\nIncomplete: {len(done)}/{len(rows)}. Aggregate not written.")
        return 2

    records = [done[str(p.relative_to(Path(args.dataset)))] for p, _ in rows]
    results = aggregate(records, args.split, len(rows))
    results["_protocol"]["whitehot_convention"] = args.whitehot

    out_json = OUT_DIR / f"v1_image_level_metrics{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    out_csv = OUT_DIR / f"test_predictions_v1_ablation{suffix}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    print("\n--- Section 4.1, released configuration ---")
    print(json.dumps(results["released_configuration"], indent=2))
    print("\n--- Section 4.2, Table 2 ---")
    for name, _ in ABLATION_CONFIGS:
        m = results["ablation"][name]
        print(
            f"{name:32s} TP{m['TP']:4d} TN{m['TN']:4d} FP{m['FP']:4d} FN{m['FN']:4d}  "
            f"P {m['Precision']}  R {m['Recall']}  F1 {m['F1']}  FPR {m['FPR']}"
        )
    for key in ("constant_positive", "constant_negative"):
        m = results["null_baselines"][key]
        print(
            f"{('Null: ' + key):32s} TP{m['TP']:4d} TN{m['TN']:4d} FP{m['FP']:4d} "
            f"FN{m['FN']:4d}  P {m['Precision']}  R {m['Recall']}  F1 {m['F1']}  "
            f"Acc {m['Accuracy']}"
        )
    print("\nSpurious boxes on the no-fire images:")
    for name, _ in ABLATION_CONFIGS:
        print(f"  {name:32s} {results['spurious_boxes_on_no_fire'][name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
