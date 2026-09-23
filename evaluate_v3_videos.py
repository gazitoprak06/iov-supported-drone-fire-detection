"""V3 Deep Edge (MobileNetV3-Small) clip-level evaluation, PyTorch reference path.

Methodology
-----------
`extract_training_frames.py` routes manifest clips to the image folders by their
`split` field: split=="train" supplies the training images, split=="test"
supplies the per-epoch validation images, and split=="validation" is not read at
all. Scoring the detector on the whole manifest therefore scores it on clips
whose own frames produced its weights.

This script groups the corpus by how each clip was actually consumed, via
`tools/clip_metrics.py`, which is shared with the torch-free evaluator so the
two cannot disagree about grouping or metric scaling:

    gradient_train      weights were fitted on frames from these clips
    epoch_monitor       no gradients, but watched during training
    untouched           never read by any stage - the strictest evidence
    unseen_by_gradient  epoch_monitor + untouched - the headline result
    full_manifest_...   all clips, partition ignored - the contaminated control

Usage:
    python evaluate_v3_videos.py
    python evaluate_v3_videos.py --roles untouched
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parent / "tools"))
from clip_metrics import (  # noqa: E402
    CONSECUTIVE_FOR_ALARM, DEFAULT_FPS, FIRE_CLASS_INDEX, SAMPLES_PER_SECOND,
    SPLIT_ROLE, group_and_score, print_summary,
)

ROOT = Path("Proje_Kodlari")
MANIFEST = ROOT / "annotations" / "video_evaluation_manifest.json"
WEIGHTS = ROOT / "evaluation_results" / "v3_deep_edge" / "v3_mobilenet.pth"
OUT = ROOT / "evaluation_results" / "v3_deep_edge" / "v3_results_by_split.json"
VERDICTS = ROOT / "evaluation_results" / "v3_deep_edge" / "_clip_verdicts_torch.json"


def build_model(device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    return model


def clip_verdict(model, transform, device, path, timings):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
    step = max(1, int(fps / SAMPLES_PER_SECOND))

    is_alarm = False
    consecutive = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                t0 = time.perf_counter()
                out = model(tensor)
                timings.append((time.perf_counter() - t0) * 1000.0)
            if int(torch.argmax(out, 1).item()) == FIRE_CLASS_INDEX:
                consecutive += 1
                if consecutive >= CONSECUTIVE_FOR_ALARM:
                    is_alarm = True
                    break
            else:
                consecutive = 0
        frame_idx += 1

    cap.release()
    return is_alarm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="*", default=None,
                    choices=sorted(set(SPLIT_ROLE.values())),
                    help="evaluate only clips in these roles (default: all)")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    videos = manifest["videos"]

    unknown = sorted({v.get("split") for v in videos if v.get("split") not in SPLIT_ROLE})
    if unknown:
        print(f"WARNING: manifest split value(s) {unknown} are not described in "
              f"clip_metrics.SPLIT_ROLE; update that table before reporting.")

    if args.roles:
        videos = [v for v in videos if SPLIT_ROLE.get(v.get("split")) in args.roles]

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model = build_model(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []
    verdicts = {}
    timings = []
    missing = []

    print(f"Evaluating {len(videos)} clips on {args.threads} thread(s)...", flush=True)
    for i, v in enumerate(videos):
        path = ROOT / v["path"]
        if not path.exists():
            missing.append(v["id"])
            continue
        alarm = clip_verdict(model, transform, device, path, timings)
        verdicts[v["id"]] = {"alarm": bool(alarm), "evaluator": "torch"}
        rows.append(dict(v, alarm=alarm))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(videos)}]", flush=True)

    if missing:
        sys.exit(f"{len(missing)} clip files are missing on disk ({missing[:5]}...); "
                 f"refusing to report metrics over a partial corpus")

    results = group_and_score(rows)

    # Kept out of the metrics artifact deliberately: that file is consumed by
    # the manuscript checker and the figure script, and both must see the same
    # top-level keys no matter which evaluator produced it.
    if timings:
        timings.sort()
        (ROOT / "evaluation_results" / "v3_deep_edge" / "_loop_latency_ms.json").write_text(
            json.dumps({
                "mean": round(sum(timings) / len(timings), 2),
                "median": round(timings[len(timings) // 2], 2),
                "p95": round(timings[int(0.95 * len(timings))], 2),
                "n_inferences": len(timings),
                "threads": args.threads,
                "note": "forward pass only, measured inside the clip loop and not "
                        "warm-up corrected; use tools/benchmark_latency.py for the "
                        "figure quoted in the paper",
            }, indent=2))

    print(json.dumps(results, indent=2))
    print_summary(results)

    VERDICTS.write_text(json.dumps(
        {"_partial_roles": args.roles, "_evaluator": "torch", "verdicts": verdicts}, indent=2))
    if not args.roles:
        OUT.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {OUT}")
    else:
        print(f"\nPartial run (--roles {args.roles}); {OUT} left untouched.")
    print(f"Wrote {VERDICTS}")


if __name__ == "__main__":
    main()
