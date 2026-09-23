"""
Seed variance of the deep edge detector.

Section 4.5 reports one fit. Section 5 lists the single seed as a limitation,
and it is the limitation a reviewer is most likely to press, because a three
epoch fine-tune of a small backbone on 2906 frames is exactly the regime in
which the seed can move a reported figure by more than the margins this
literature competes over.

This script closes that gap. It refits the backbone under several seeds, scores
each fit over the full clip manifest with the same torch-free evaluator that
produced the reported figures, and reports the mean and the population standard
deviation of every headline quantity across seeds, on each of the partitions
the paper reports separately.

Nothing else varies. The splits, the epoch count, the learning rate, the batch
size, the augmentation and the alarm rule are all held at the values of Section
4.5; the only varying quantity is the seed that drives the loader shuffle, the
brightness and contrast jitter and the initialisation of the two-class head.

Cost. Each seed is a full three-epoch CPU fine-tune followed by a pass over 483
clips, so budget roughly an hour per seed on a mobile-class CPU. The script
checkpoints after every seed and skips seeds already recorded, so it can be run
across several sessions.

Usage:  python tools/multiseed_v3.py --seeds 0 1 2 3 4
        python tools/multiseed_v3.py --report        # aggregate what exists
Writes: Proje_Kodlari/evaluation_results/v3_deep_edge/v3_seed_variance.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from image_metrics import environment_record  # noqa: E402

OUT_DIR = ROOT / "Proje_Kodlari" / "evaluation_results" / "v3_deep_edge"
OUT = OUT_DIR / "v3_seed_variance.json"
PER_SEED = OUT_DIR / "_seed_runs.json"

# The groups the paper reports separately, and the quantities quoted for each.
GROUPS = ("unseen_by_gradient", "untouched", "epoch_monitor",
          "gradient_train", "full_manifest_CONTAMINATED")
QUANTITIES = ("Recall_TPR", "FPR", "F1", "Accuracy")


def run_one_seed(seed: int) -> dict:
    """Refit under one seed and score the whole manifest with that fit."""
    weights = f"v3_mobilenet_seed{seed}.pth"
    verdict_cache = OUT_DIR / f"_clip_verdicts_seed{seed}.json"

    print(f"\n=== seed {seed}: training ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "train_v3.py"),
                    "--seed", str(seed), "--out", weights],
                   cwd=ROOT, check=True)

    print(f"=== seed {seed}: scoring the manifest ===", flush=True)
    # The evaluator takes its weights and its cache from the environment so
    # that the reported run's own artifacts are never overwritten.
    env_args = ["--weights", str(OUT_DIR / weights),
                "--cache", str(verdict_cache),
                "--out", str(OUT_DIR / f"_v3_results_seed{seed}.json")]
    subprocess.run([sys.executable, str(ROOT / "tools" / "eval_clips_numpy.py"),
                    "--budget", "1e9", *env_args], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "eval_clips_numpy.py"),
                    "--report", *env_args], cwd=ROOT, check=True)

    return json.loads((OUT_DIR / f"_v3_results_seed{seed}.json").read_text(encoding="utf-8"))


def aggregate(runs: dict) -> dict:
    """Mean and population standard deviation of each quantity across seeds."""
    seeds = sorted(runs, key=int)
    summary = {}
    for group in GROUPS:
        present = [runs[s][group] for s in seeds if group in runs[s]]
        if len(present) < 2:
            continue
        block = {}
        for q in QUANTITIES:
            vals = [p[q] for p in present if p.get(q) is not None]
            if not vals:
                continue
            block[q] = {
                "mean": round(statistics.fmean(vals), 2),
                "sd": round(statistics.pstdev(vals), 2),
                "min": min(vals),
                "max": max(vals),
                "range": round(max(vals) - min(vals), 2),
                "values": vals,
            }
        block["n_seeds"] = len(present)
        summary[group] = block
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--report", action="store_true",
                    help="aggregate the seeds already recorded and stop")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = json.loads(PER_SEED.read_text(encoding="utf-8")) if PER_SEED.exists() else {}

    if not args.report:
        for seed in args.seeds:
            if str(seed) in runs:
                print(f"seed {seed} already recorded, skipping")
                continue
            runs[str(seed)] = run_one_seed(seed)
            PER_SEED.write_text(json.dumps(runs, indent=2), encoding="utf-8")
            print(f"seed {seed} recorded ({len(runs)} total)")

    if len(runs) < 2:
        print(f"\nOnly {len(runs)} seed(s) recorded; at least two are needed "
              f"before a spread can be reported.")
        return 2

    summary = aggregate(runs)
    out = {
        "_protocol": {
            "experiment": "Section 4.5 - seed variance of the deep edge detector",
            "seeds": sorted(int(s) for s in runs),
            "varying": "loader shuffle, brightness and contrast jitter, "
                       "and initialisation of the two-class head",
            "held_fixed": "splits, three epochs, Adam at 1e-4, batch size 32, "
                          "and the alarm rule of Section 4.5",
            "deviation": "population standard deviation over the seeds",
            "environment": environment_record(),
        },
        "per_seed": runs,
        "summary": summary,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}\n")
    for group, block in summary.items():
        mark = "  <- headline" if group == "unseen_by_gradient" else ""
        print(f"{group} (n={block['n_seeds']} seeds){mark}")
        for q in QUANTITIES:
            if q in block:
                b = block[q]
                print(f"   {q:12s} {b['mean']:6.2f} +/- {b['sd']:.2f}   "
                      f"[{b['min']:.2f}, {b['max']:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
