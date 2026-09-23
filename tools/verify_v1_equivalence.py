"""
Proves that tools/v1_configurable.py reproduces the released pipeline exactly.

The ablation of Section 4.2 and the saturation sweep of Section 4.4 are run by
a parameterized reimplementation rather than by v1_heuristic.py itself, because
the reference function does not expose the colour-space set or the saturation
floors. That is only defensible if the parameterized version reduces to the
reference one at the released configuration, so this script checks it: over
every image of the test split, at active=all and the released floors, the two
must emit the same boxes in the same order, with identical rectangles, scores
and areas.

Exits non-zero on the first divergence, so it can gate a commit. This is the
still-image counterpart of tools/verify_numpy_model.py.

Usage:  python tools/verify_v1_equivalence.py [--split test] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from image_metrics import DATASET_ROOT, REFERENCE_ALTITUDE_M, list_split  # noqa: E402
from v1_configurable import ALL_SPACES, DEFAULT_SAT_FLOORS, run_configurable  # noqa: E402
from v1_heuristic import run_heuristic_pipeline  # noqa: E402

TOL = 1e-9


def boxes_equal(a: list[dict], b: list[dict]) -> tuple[bool, str]:
    if len(a) != len(b):
        return False, f"box count {len(a)} vs {len(b)}"
    for k, (ba, bb) in enumerate(zip(a, b)):
        if tuple(ba["rect"]) != tuple(bb["rect"]):
            return False, f"box {k} rect {ba['rect']} vs {bb['rect']}"
        if abs(ba["score"] - bb["score"]) > TOL:
            return False, f"box {k} score {ba['score']} vs {bb['score']}"
        if abs(ba["area"] - bb["area"]) > TOL:
            return False, f"box {k} area {ba['area']} vs {bb['area']}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=str(ROOT / DATASET_ROOT))
    args = ap.parse_args()

    rows = list_split(Path(args.dataset), args.split)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"ERROR: no images found under {args.dataset}/{args.split}")
        return 1

    print(f"Comparing {len(rows)} images of the {args.split} split ...")
    checked = 0
    for path, _label in rows:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"ERROR: could not decode {path}")
            return 1
        ref = run_heuristic_pipeline(frame, None, REFERENCE_ALTITUDE_M)
        cfg = run_configurable(
            frame,
            None,
            REFERENCE_ALTITUDE_M,
            active=ALL_SPACES,
            sat_floors=DEFAULT_SAT_FLOORS,
        )
        ok, why = boxes_equal(ref, cfg)
        if not ok:
            print(f"\nDIVERGENCE on {path.name}: {why}")
            return 1
        checked += 1
        if checked % 50 == 0:
            print(f"  {checked}/{len(rows)} identical")

    print(f"\n{checked}/{checked} images identical.")
    print("v1_configurable.py reduces to v1_heuristic.py at the released configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
