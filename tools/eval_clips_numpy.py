"""Resumable, torch-free clip-level evaluation of the V3 MobileNetV3 detector.

Mirrors the decision logic of `evaluate_v3_videos.py` exactly, but runs on
NumPy so it needs no PyTorch install. Grouping and metrics come from
`tools/clip_metrics.py`, which both evaluators share. Per-clip verdicts are
cached, so the run can be interrupted and resumed.

    python tools/eval_clips_numpy.py --budget 150   # work for ~150 s, then checkpoint
    python tools/eval_clips_numpy.py --report       # summarise what is cached

Before trusting the output, run `tools/verify_numpy_model.py` on a machine that
has PyTorch: it checks this NumPy forward pass against torchvision's on real
frames. The numbers this script produces are only as good as that agreement.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from clip_metrics import (  # noqa: E402
    CONSECUTIVE_FOR_ALARM, DEFAULT_FPS, FIRE_CLASS_INDEX, SAMPLES_PER_SECOND,
    group_and_score, print_summary,
)
from torch_free_mobilenetv3 import (  # noqa: E402
    MobileNetV3SmallNumpy, load_state_dict, preprocess_bgr,
)

# Anchored to this file rather than to the caller's working directory, so the
# program runs from anywhere. Every tool written later in the project does the
# same; these four were the holdouts and required being run from the repo root.
ROOT = Path(__file__).resolve().parent.parent / "Proje_Kodlari"
MANIFEST = ROOT / "annotations" / "video_evaluation_manifest.json"
WEIGHTS = ROOT / "evaluation_results" / "v3_deep_edge" / "v3_mobilenet.pth"
CACHE = ROOT / "evaluation_results" / "v3_deep_edge" / "_clip_verdicts.json"
OUT = ROOT / "evaluation_results" / "v3_deep_edge" / "v3_results_by_split.json"


def clip_verdict(model, path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
    step = max(1, int(fps / SAMPLES_PER_SECOND))

    is_alarm = False
    consecutive = 0
    frame_idx = 0
    n_inf = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            logits = model.forward(preprocess_bgr(frame))
            n_inf += 1
            if int(np.argmax(logits)) == FIRE_CLASS_INDEX:
                consecutive += 1
                if consecutive >= CONSECUTIVE_FOR_ALARM:
                    is_alarm = True
                    break
            else:
                consecutive = 0
        frame_idx += 1

    cap.release()
    return is_alarm, n_inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=150.0,
                    help="seconds of work before checkpointing")
    ap.add_argument("--report", action="store_true")
    # The weight file, the verdict cache and the results file are overridable so
    # that a seed sweep (tools/multiseed_v3.py) can score a refitted network
    # without overwriting the artifacts of the run reported in the paper.
    ap.add_argument("--weights", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    global WEIGHTS, CACHE, OUT
    if args.weights: WEIGHTS = Path(args.weights)
    if args.cache:   CACHE = Path(args.cache)
    if args.out:     OUT = Path(args.out)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    videos = manifest["videos"]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    if args.report:
        missing = [v["id"] for v in videos if v["id"] not in cache]
        if missing:
            sys.exit(f"{len(missing)} clips not yet evaluated; run without --report first")
        dropped = [v["id"] for v in videos if cache[v["id"]].get("missing")]
        if dropped:
            sys.exit(f"{len(dropped)} clip files were missing on disk "
                     f"({dropped[:5]}...); refusing to report a partial corpus")
        rows = [dict(v, alarm=cache[v["id"]]["alarm"]) for v in videos]
        results = group_and_score(rows)
        print(json.dumps(results, indent=2))
        print_summary(results)
        OUT.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {OUT}")
        return

    model = MobileNetV3SmallNumpy(load_state_dict(WEIGHTS))
    todo = [v for v in videos if v["id"] not in cache]
    print(f"{len(cache)}/{len(videos)} cached; {len(todo)} remaining", flush=True)

    def checkpoint():
        """Write the cache through a temporary file.

        The docstring promises the run can be interrupted and resumed, but a
        single write after the loop loses everything to a Ctrl-C, and an
        interrupt during the write truncates a file the paper depends on. The
        rename is atomic on both platforms, so the cache is either the previous
        state or the new one and never a half-written mixture.
        """
        tmp = CACHE.with_suffix(CACHE.suffix + ".tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, CACHE)

    t_start = time.time()
    done = 0
    try:
        for v in todo:
            if time.time() - t_start > args.budget:
                break
            path = ROOT / v["path"]
            if not path.exists():
                cache[v["id"]] = {"alarm": False, "missing": True}
                continue
            alarm, n_inf = clip_verdict(model, path)
            cache[v["id"]] = {"alarm": bool(alarm), "n_inferences": n_inf}
            done += 1
            if done % 20 == 0:
                checkpoint()
    except KeyboardInterrupt:
        checkpoint()
        print(f"\ninterrupted; {len(cache)}/{len(videos)} clips cached", flush=True)
        return 130

    checkpoint()
    print(f"processed {done} clips in {time.time()-t_start:.0f}s; "
          f"{len(cache)}/{len(videos)} total", flush=True)


if __name__ == "__main__":
    main()
