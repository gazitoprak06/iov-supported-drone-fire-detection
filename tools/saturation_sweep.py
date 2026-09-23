"""
Section 4.4, the saturation sweep table.

The false-positive analysis of Section 4.1 indicates which parameter to sweep:
the surviving spurious detections sit on bright but weakly saturated regions,
principally cloud, mist, sunlit snow and pale rock, whereas flame is strongly
saturated. The saturation floor of the three hue bands of Section 3.1 is
therefore the best-motivated single parameter.

This script sweeps that floor from the released value of 60 to 220 with every
other setting held fixed, and reports precision against the class prior at each
setting. If precision does not move off the prior as recall is driven down, the
positive predictions are close to uncorrelated with the label and no setting
within the threshold family is a good one.

Inputs follow the path used by the learned baselines, so the two are byte
identical: the reduced-scale decoder is asked for one eighth of the stored
dimensions and the result is resized bilinearly to 640 x 480. At that size the
absolute and relative area criteria of Section 4.3 coincide, so the sweep
isolates the saturation effect rather than re-measuring the scale confound.

Convention: the released floors are (60, 80, 60) for the flame-core, halo and
red-edge bands. A sweep value S sets all three floors to S, so the S = 60 row
is the released configuration up to the halo band, whose released floor of 80
is the one value the sweep does not preserve; this is recorded in the result
file as `sweep_sets_all_three_floors` so that the row is not misread.

Usage:  python tools/saturation_sweep.py [--split test]
Writes: Proje_Kodlari/evaluation_results/v1_image_level/v1_saturation_sweep.json
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

from image_metrics import (  # noqa: E402
    DATASET_ROOT,
    REFERENCE_ALTITUDE_M,
    binary_metrics,
    environment_record,
    list_split,
)
from v1_configurable import ALL_SPACES, DEFAULT_SAT_FLOORS, run_configurable  # noqa: E402

OUT_DIR = ROOT / "Proje_Kodlari" / "evaluation_results" / "v1_image_level"

WORK_SIZE = (640, 480)
SWEEP_VALUES = [60, 80, 100, 120, 140, 160, 180, 200, 220]


def load_reduced(path: Path):
    """The Table 3 input path: 1/8 decode, then bilinear resize to 640x480."""
    bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:
        bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.resize(bgr, WORK_SIZE, interpolation=cv2.INTER_LINEAR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=str(ROOT / DATASET_ROOT))
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument(
        "--only",
        type=int,
        nargs="*",
        default=None,
        help="sweep only these S values (results merge into the existing file)",
    )
    args = ap.parse_args()

    rows = list_split(Path(args.dataset), args.split)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"ERROR: no images found under {args.dataset}/{args.split}")
        return 1

    n_fire = sum(1 for _, lab in rows if lab == 1)
    n_no_fire = len(rows) - n_fire
    class_prior = round(100.0 * n_fire / len(rows), 2)
    print(
        f"{len(rows)} images ({n_fire} fire, {n_no_fire} no-fire), "
        f"class prior {class_prior}%"
    )

    # Decode once, sweep in memory. Decoding dominates the cost of this
    # experiment (the sweep itself is a few seconds per row), so the decoded
    # stack is cached to disk and reused by a rerun.
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"_sweep_frames_{args.split}.npy"
    progress = OUT_DIR / f"_sweep_frames_{args.split}.progress"
    labels = np.array([lab for _, lab in rows], dtype=np.int8)
    shape = (len(rows), WORK_SIZE[1], WORK_SIZE[0], 3)

    if args.rebuild_cache:
        cache.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)

    # The cache is written straight to disk as a memmap and a sidecar records
    # how many frames are valid, so decoding can itself be spread over several
    # invocations on a host that bounds wall time.
    if cache.exists():
        stack = np.lib.format.open_memmap(cache, mode="r+")
        if stack.shape != shape:
            print("Cache is stale; rebuilding.")
            del stack
            cache.unlink()
            progress.unlink(missing_ok=True)
    if not cache.exists():
        stack = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=shape)

    n_cached = int(progress.read_text()) if progress.exists() else 0
    if n_cached < len(rows):
        t_dec = time.time()
        for i in range(n_cached, len(rows)):
            if args.max_seconds and (time.time() - t_dec) > args.max_seconds:
                break
            img = load_reduced(rows[i][0])
            if img is None:
                print(f"ERROR: could not decode {rows[i][0]}")
                return 1
            stack[i] = img
            n_cached = i + 1
            if n_cached % 50 == 0:
                stack.flush()
                progress.write_text(str(n_cached))
                print(f"  decoded {n_cached}/{len(rows)}", flush=True)
        stack.flush()
        progress.write_text(str(n_cached))
        if n_cached < len(rows):
            print(
                f"\nFrame cache incomplete: {n_cached}/{len(rows)} decoded. "
                f"Rerun to continue."
            )
            return 2
        print(
            f"Decoded {len(rows)} images at 1/8 scale -> "
            f"{WORK_SIZE[0]}x{WORK_SIZE[1]}, cached to {cache.name}"
        )
    else:
        print(f"Loaded {len(rows)} decoded frames from {cache.name}")

    frames = [(stack[i], int(labels[i])) for i in range(len(rows))]

    # Rows already computed by an earlier invocation are reused, so the sweep
    # can be completed across several runs.
    out_path = OUT_DIR / "v1_saturation_sweep.json"
    existing: dict[int, dict] = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            existing = {int(m["S"]): m for m in prev.get("sweep", [])}
            if existing:
                print(f"Reusing {len(existing)} rows already computed: "
                      f"{sorted(existing)}")
        except Exception:
            existing = {}

    wanted = args.only if args.only else SWEEP_VALUES

    sweep = []
    t0 = time.time()
    for s in SWEEP_VALUES:
        if s in existing and s not in (args.only or []):
            sweep.append(existing[s])
            continue
        if s not in wanted:
            continue
        counts = [0, 0, 0, 0]  # tp tn fp fn
        for img, label in frames:
            boxes = run_configurable(
                img,
                None,
                REFERENCE_ALTITUDE_M,
                active=ALL_SPACES,
                sat_floors=(s, s, s),
            )
            pred = len(boxes) > 0
            if label == 1:
                counts[0 if pred else 3] += 1
            else:
                counts[2 if pred else 1] += 1
        m = binary_metrics(*counts)
        m["S"] = s
        m["precision_minus_prior_pp"] = (
            round(m["Precision"] - class_prior, 2) if m["Precision"] is not None else None
        )
        m["is_released_configuration"] = s == DEFAULT_SAT_FLOORS[0]
        sweep.append(m)
        print(
            f"  S={s:3d}  TP{m['TP']:4d} TN{m['TN']:4d} FP{m['FP']:4d} FN{m['FN']:4d}  "
            f"P {m['Precision']}  R {m['Recall']}  F1 {m['F1']}  FPR {m['FPR']}  "
            f"(P-prior {m['precision_minus_prior_pp']} pp)   [{time.time() - t0:.0f}s]",
            flush=True,
        )

    sweep.sort(key=lambda m: m["S"])
    complete = len(sweep) == len(SWEEP_VALUES)

    recalls = [m["Recall"] for m in sweep if m["Recall"] is not None]
    fprs = [m["FPR"] for m in sweep if m["FPR"] is not None]
    deltas = [
        abs(m["precision_minus_prior_pp"])
        for m in sweep
        if m["precision_minus_prior_pp"] is not None
    ]

    results = {
        "_protocol": {
            "experiment": "Section 4.4 - saturation floor sweep of the three hue bands",
            "split": args.split,
            "n_images": len(rows),
            "n_fire": n_fire,
            "n_no_fire": n_no_fire,
            "class_prior_pct": class_prior,
            "input": "decoded at 1/8 scale, resized to 640x480 bilinearly "
            "(byte identical to the learned-baseline inputs)",
            "swept_values": SWEEP_VALUES,
            "released_floors": list(DEFAULT_SAT_FLOORS),
            "sweep_sets_all_three_floors": True,
            "all_other_settings": "held fixed at Section 3",
            "temporal_layer": "disabled (still images)",
            "environment": environment_record(),
        },
        "sweep": sweep,
        "_complete": complete,
        "summary": {
            "recall_range_pp": round(max(recalls) - min(recalls), 2) if recalls else None,
            "recall_max_pct": max(recalls) if recalls else None,
            "recall_min_pct": min(recalls) if recalls else None,
            "fpr_range_pp": round(max(fprs) - min(fprs), 2) if fprs else None,
            "max_abs_precision_deviation_from_prior_pp": max(deltas) if deltas else None,
            "n_settings_with_precision_below_prior": sum(
                1
                for m in sweep
                if m["precision_minus_prior_pp"] is not None
                and m["precision_minus_prior_pp"] < 0
            ),
        },
    }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps(results["summary"], indent=2))
    if not complete:
        missing = sorted(set(SWEEP_VALUES) - {m["S"] for m in sweep})
        print(f"\nIncomplete: still missing S = {missing}. Rerun to continue.")
        return 2
    print("\nSweep complete: all rows present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
