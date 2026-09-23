"""
Diagnostic: at which stage of the localization chain were the spurious-box
totals of Section 4.3 counted?

Section 4.3 reports a spurious-box total over the 251 no-fire test images. That
total depends on where in Sections 3.1 to 3.3 the boxes are counted, because
the asymmetric suppression and the proximity merge both reduce the count, and
the manuscript does not say which stage it used. This script counts all three
so that the reported total can be attributed rather than guessed.

Usage:  python tools/box_count_stages.py [--max-seconds S]
Writes: Proje_Kodlari/evaluation_results/v1_image_level/v1_box_count_stages.json
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

from image_metrics import DATASET_ROOT, REFERENCE_ALTITUDE_M, environment_record, list_split
from v1_configurable import run_configurable

OUT_DIR = ROOT / "Proje_Kodlari" / "evaluation_results" / "v1_image_level"
CHECKPOINT = OUT_DIR / "_v1_box_stages.jsonl"
WORK_SIZE = (640, 480)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default=str(ROOT / DATASET_ROOT))
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args()

    rows = [(p, l) for p, l in list_split(Path(args.dataset), args.split) if l == 0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); done[r["file"]] = r

    print(f"{len(rows)} no-fire images; {len(done)} already recorded")
    todo = [(p, l) for p, l in rows if str(p.relative_to(Path(args.dataset))) not in done]

    t0 = time.time()
    with CHECKPOINT.open("a", encoding="utf-8") as fh:
        for i, (path, _l) in enumerate(todo, 1):
            if args.max_seconds and (time.time() - t0) > args.max_seconds:
                print(f"\nTime bound; {len(done)}/{len(rows)}. Rerun to continue.")
                break
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"ERROR: {path}"); return 1
            _, nat = run_configurable(frame, None, REFERENCE_ALTITUDE_M, stages=True)
            small = cv2.resize(frame, WORK_SIZE, interpolation=cv2.INTER_LINEAR)
            _, res = run_configurable(small, None, REFERENCE_ALTITUDE_M, stages=True)
            rec = {"file": str(path.relative_to(Path(args.dataset))),
                   "native": nat, "resized": res}
            fh.write(json.dumps(rec) + "\n"); fh.flush(); done[rec["file"]] = rec
            if i % 25 == 0:
                print(f"  {len(done)}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    if len(done) < len(rows):
        print(f"\nIncomplete: {len(done)}/{len(rows)}."); return 2

    out = {"_protocol": {"experiment": "Section 4.3 spurious-box totals by pipeline stage",
                         "split": args.split, "n_no_fire_images": len(rows),
                         "environment": environment_record()}}
    for cond in ("native", "resized"):
        out[cond] = {k: sum(done[str(p.relative_to(Path(args.dataset)))][cond][k]
                            for p, _ in rows)
                     for k in ("pre_nms", "post_nms", "post_merge")}
    n, r = out["native"], out["resized"]
    out["removed_by_resize"] = {k: n[k] - r[k] for k in n}
    out["removed_pct"] = {k: round(100.0 * (n[k] - r[k]) / n[k], 2) if n[k] else None
                          for k in n}
    (OUT_DIR / "v1_box_count_stages.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: out[k] for k in ("native", "resized", "removed_by_resize", "removed_pct")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
