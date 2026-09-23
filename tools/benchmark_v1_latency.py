"""
Run-to-run variability of the hand-set pipeline's per-frame cost.

Section 4.4 quotes a single median for the hand-set pipeline, measured in the
same run as the learned models so that the ratio between them is meaningful.
The Conclusion makes a separate and weaker claim: that the absolute figure is
strongly host-dependent and that repeating the identical measurement over the
identical frames does not reproduce it. That claim needs its own record,
because it is a statement about the spread across executions rather than about
any one execution.

This script therefore performs several independent executions. Each is a fresh
warm-up followed by its own timed block, so the between-execution spread it
reports is the quantity the Conclusion refers to, not the within-execution
spread that Section 4.4 already reports as a standard deviation.

The detector contains no stochastic component, so every execution processes
identical pixels and returns identical boxes; the script asserts that, and any
variation it reports is therefore timing and not behaviour.

Usage:  python tools/benchmark_v1_latency.py [--executions 5] [--frames 30] [--reps 10]
Writes: Proje_Kodlari/evaluation_results/v1_image_level/v1_latency_variability.json
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
    environment_record,
    list_split,
)
from v1_heuristic import run_heuristic_pipeline  # noqa: E402

OUT_DIR = ROOT / "Proje_Kodlari" / "evaluation_results" / "v1_image_level"
WORK_SIZE = (640, 480)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default=str(ROOT / DATASET_ROOT))
    ap.add_argument("--executions", type=int, default=5)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    rows = list_split(Path(args.dataset), args.split)[: args.frames]
    if not rows:
        print(f"ERROR: no images under {args.dataset}/{args.split}")
        return 1

    frames = []
    for path, _ in rows:
        bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
        if bgr is None:
            bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"ERROR: could not decode {path}")
            return 1
        frames.append(cv2.resize(bgr, WORK_SIZE, interpolation=cv2.INTER_LINEAR))
    print(f"{len(frames)} frames at {WORK_SIZE[0]}x{WORK_SIZE[1]}")

    # The pipeline is deterministic, so the box set must not vary between
    # executions. Recorded once here and asserted after every execution.
    reference = [len(run_heuristic_pipeline(f, None, REFERENCE_ALTITUDE_M)) for f in frames]

    executions = []
    for e in range(args.executions):
        for f in frames:                      # warm-up, discarded
            run_heuristic_pipeline(f, None, REFERENCE_ALTITUDE_M)
        samples = []
        counts = []
        for _ in range(args.reps):
            for f in frames:
                t = time.perf_counter()
                boxes = run_heuristic_pipeline(f, None, REFERENCE_ALTITUDE_M)
                samples.append((time.perf_counter() - t) * 1000.0)
                counts.append(len(boxes))
        if counts[: len(frames)] != reference:
            print("ERROR: the pipeline returned a different box set between executions")
            return 1
        rec = {
            "execution": e + 1,
            "n_samples": len(samples),
            "mean_ms": round(statistics.fmean(samples), 2),
            "median_ms": round(statistics.median(samples), 2),
            "stdev_ms": round(statistics.stdev(samples), 2),
            "min_ms": round(min(samples), 2),
            "max_ms": round(max(samples), 2),
        }
        executions.append(rec)
        print(f"  execution {e+1}: mean {rec['mean_ms']} ms, median {rec['median_ms']} ms, "
              f"sd {rec['stdev_ms']} ms", flush=True)

    means = [r["mean_ms"] for r in executions]
    medians = [r["median_ms"] for r in executions]
    out = {
        "_protocol": {
            "experiment": "run-to-run variability of the hand-set pipeline "
                          "(Conclusion, engineering observation)",
            "frames": len(frames),
            "input": "decoded at 1/8 scale, resized to 640x480",
            "executions": args.executions,
            "reps_per_execution": args.reps,
            "samples_per_execution": len(frames) * args.reps,
            "protocol": "each execution is an independent warm-up pass over the "
                        "frames, discarded, followed by its own timed block; the "
                        "spread reported here is between executions, not within one",
            "determinism_check": "the emitted box set was identical in every "
                                 "execution, so the spread is timing and not behaviour",
            "single_thread": True,
            "environment": environment_record(),
        },
        "executions": executions,
        "summary": {
            "min_execution_mean_ms": min(means),
            "max_execution_mean_ms": max(means),
            "ratio_max_over_min": round(max(means) / min(means), 2),
            "median_of_execution_means_ms": round(statistics.median(means), 2),
            "min_execution_median_ms": min(medians),
            "max_execution_median_ms": max(medians),
            "fps_at_slowest_execution": round(1000.0 / max(means), 1),
            "fps_at_fastest_execution": round(1000.0 / min(means), 1),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "v1_latency_variability.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {p}")
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
