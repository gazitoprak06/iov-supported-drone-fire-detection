"""
Section 4.3: resolution sensitivity of the fixed area threshold.

The minimum cluster area of Eq. (2) is an absolute pixel count, 600 px at the
reference altitude, while the still-image dataset spans orders of magnitude in
resolution. On a large image the criterion is close to inoperative and any
sufficiently small chromatic speck survives it, which would inflate the
false-alarm rate of Sections 4.1 and 4.2 for a reason that has nothing to do
with colour-space thresholding.

This script separates the scale effect from the chromatic effect. The test
split is evaluated twice with a single change between the conditions: in the
resized condition each image is decoded at full resolution and then resized to
640 x 480 by bilinear interpolation, the resolution at which the video
experiments operate, before being passed to an otherwise identical detector.

It also produces the supporting statistics quoted in the section: the
megapixel range and median of the cohort, the spurious box count per resolution
band, the Pearson correlation between image area and spurious box count, and
the paired McNemar tests over the two decision sets.

Usage:  python tools/resolution_control.py [--split test]
Writes: Proje_Kodlari/evaluation_results/v1_image_level/v1_resolution_control.json
"""

from __future__ import annotations

import argparse
import json
import statistics
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
    mcnemar,
)
from v1_heuristic import run_heuristic_pipeline  # noqa: E402

OUT_DIR = ROOT / "Proje_Kodlari" / "evaluation_results" / "v1_image_level"

WORK_SIZE = (640, 480)

# Resolution bands, in megapixels, used for the spurious-box breakdown.
BANDS = [
    ("< 2 MP", 0.0, 2.0),
    ("2 - 5 MP", 2.0, 5.0),
    ("5 - 10 MP", 5.0, 10.0),
    ("10 - 25 MP", 10.0, 25.0),
    ("> 25 MP", 25.0, float("inf")),
]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=str(ROOT / DATASET_ROOT))
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop this invocation gracefully after S seconds (0 = no bound)",
    )
    ap.add_argument("--restart", action="store_true", help="discard the checkpoint")
    args = ap.parse_args()

    rows = list_split(Path(args.dataset), args.split)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"ERROR: no images found under {args.dataset}/{args.split}")
        return 1

    n_fire = sum(1 for _, lab in rows if lab == 1)

    # Both conditions run at native decode, so a full pass is long; per-image
    # records are checkpointed and a rerun resumes where the last one stopped.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT_DIR / "_v1_resolution_perimage.jsonl"
    if args.restart and checkpoint.exists():
        checkpoint.unlink()

    done: dict[str, dict] = {}
    if checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["file"]] = rec

    print(
        f"{len(rows)} images ({n_fire} fire, {len(rows) - n_fire} no-fire); "
        f"{len(done)} already recorded"
    )

    todo = [
        (p, lab)
        for p, lab in rows
        if str(p.relative_to(Path(args.dataset))) not in done
    ]

    t0 = time.time()
    with checkpoint.open("a", encoding="utf-8") as fh:
        for idx, (path, label) in enumerate(todo, 1):
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

            b_nat = run_heuristic_pipeline(frame, None, REFERENCE_ALTITUDE_M)
            small = cv2.resize(frame, WORK_SIZE, interpolation=cv2.INTER_LINEAR)
            b_res = run_heuristic_pipeline(small, None, REFERENCE_ALTITUDE_M)

            rec = {
                "file": str(path.relative_to(Path(args.dataset))),
                "label": label,
                "megapixels": round(w_px * h_px / 1e6, 4),
                "nbox_native": len(b_nat),
                "nbox_resized": len(b_res),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[rec["file"]] = rec
            if idx % 25 == 0:
                print(
                    f"  {len(done)}/{len(rows)}  ({time.time() - t0:.0f}s this run)",
                    flush=True,
                )

    if len(done) < len(rows):
        print(f"\nIncomplete: {len(done)}/{len(rows)}. Aggregate not written.")
        return 2

    native: list[tuple[int, bool]] = []
    resized: list[tuple[int, bool]] = []
    megapixels: list[float] = []
    nofire_mp: list[float] = []
    nofire_boxes_native: list[int] = []
    boxes_native_total = 0
    boxes_resized_total = 0
    band_rows = {name: {"n": 0, "boxes": 0, "fp": 0} for name, _, _ in BANDS}

    for path, label in rows:
        rec = done[str(path.relative_to(Path(args.dataset)))]
        mp = rec["megapixels"]
        megapixels.append(mp)
        native.append((label, rec["nbox_native"] > 0))
        resized.append((label, rec["nbox_resized"] > 0))
        if label == 0:
            boxes_native_total += rec["nbox_native"]
            boxes_resized_total += rec["nbox_resized"]
            nofire_mp.append(mp)
            nofire_boxes_native.append(rec["nbox_native"])
            for name, lo, hi in BANDS:
                if lo <= mp < hi:
                    band_rows[name]["n"] += 1
                    band_rows[name]["boxes"] += rec["nbox_native"]
                    band_rows[name]["fp"] += 1 if rec["nbox_native"] > 0 else 0
                    break

    m_nat = binary_metrics(
        sum(1 for l, p in native if l == 1 and p),
        sum(1 for l, p in native if l == 0 and not p),
        sum(1 for l, p in native if l == 0 and p),
        sum(1 for l, p in native if l == 1 and not p),
    )
    m_res = binary_metrics(
        sum(1 for l, p in resized if l == 1 and p),
        sum(1 for l, p in resized if l == 0 and not p),
        sum(1 for l, p in resized if l == 0 and p),
        sum(1 for l, p in resized if l == 1 and not p),
    )

    # Paired McNemar tests. b = native correct & resized wrong,
    # c = resized correct & native wrong.
    def paired(subset_label: int | None) -> dict:
        b = c = 0
        for (lab, pn), (_, pr) in zip(native, resized):
            if subset_label is not None and lab != subset_label:
                continue
            cn = pn == (lab == 1)
            cr = pr == (lab == 1)
            if cn and not cr:
                b += 1
            elif cr and not cn:
                c += 1
        return mcnemar(b, c)

    for name, _, _ in BANDS:
        r = band_rows[name]
        r["mean_boxes_per_image"] = (
            round(r["boxes"] / r["n"], 2) if r["n"] else None
        )
        r["false_alarm_rate"] = (
            round(100.0 * r["fp"] / r["n"], 2) if r["n"] else None
        )

    removed = boxes_native_total - boxes_resized_total
    results = {
        "_protocol": {
            "experiment": "Section 4.3 - resolution sensitivity of the fixed area threshold",
            "split": args.split,
            "n_images": len(rows),
            "conditions": {
                "native": "decoded at native resolution (the evaluation of Section 4.1)",
                "resized": f"decoded at full resolution then resized to {WORK_SIZE[0]}x{WORK_SIZE[1]} bilinearly",
            },
            "difference_between_conditions": "one cv2.resize call; the detector is otherwise identical",
            "altitude_m": REFERENCE_ALTITUDE_M,
            "environment": environment_record(),
        },
        "cohort_resolution": {
            "min_megapixels": round(min(megapixels), 2),
            "max_megapixels": round(max(megapixels), 2),
            "median_megapixels": round(statistics.median(megapixels), 2),
            "A_min_px": 600,
            "A_min_as_pct_of_median_image": round(
                100.0 * 600 / (statistics.median(megapixels) * 1e6), 4
            ),
            "A_min_as_pct_of_work_frame": round(
                100.0 * 600 / (WORK_SIZE[0] * WORK_SIZE[1]), 4
            ),
        },
        "native": m_nat,
        "resized": m_res,
        "spurious_boxes_on_no_fire": {
            "native": boxes_native_total,
            "resized": boxes_resized_total,
            "removed": removed,
            "removed_pct": round(100.0 * removed / boxes_native_total, 2)
            if boxes_native_total
            else None,
            "max_boxes_on_a_single_image": max(nofire_boxes_native)
            if nofire_boxes_native
            else 0,
            "megapixels_of_that_image": round(
                nofire_mp[nofire_boxes_native.index(max(nofire_boxes_native))], 2
            )
            if nofire_boxes_native
            else None,
        },
        "pearson_area_vs_spurious_boxes": pearson(nofire_mp, [float(b) for b in nofire_boxes_native]),
        "resolution_bands_no_fire_native": band_rows,
        "mcnemar": {
            "no_fire_images": paired(0),
            "fire_images": paired(1),
            "all_images": paired(None),
        },
    }
    results["changes"] = {
        "Accuracy_pp": round(m_res["Accuracy"] - m_nat["Accuracy"], 2),
        "Precision_pp": round(m_res["Precision"] - m_nat["Precision"], 2),
        "Recall_pp": round(m_res["Recall"] - m_nat["Recall"], 2),
        "F1_pp": round(m_res["F1"] - m_nat["F1"], 2),
        "Specificity_pp": round(m_res["Specificity"] - m_nat["Specificity"], 2),
        "FPR_pp": round(m_res["FPR"] - m_nat["FPR"], 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "v1_resolution_control.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}\n")
    print(json.dumps(
        {k: results[k] for k in ("cohort_resolution", "native", "resized", "changes",
                                 "spurious_boxes_on_no_fire",
                                 "pearson_area_vs_spurious_boxes", "mcnemar")},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
