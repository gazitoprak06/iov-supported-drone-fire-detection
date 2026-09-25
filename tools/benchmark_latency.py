"""Per-frame CPU inference latency for the V3 MobileNetV3 fire detector.

The paper reports a per-frame latency figure for the CPU-only edge budget.
That figure must come from this script rather than from an estimate. Run it on
the machine whose CPU you name in the paper:

    python tools/benchmark_latency.py --threads 1 --runs 300

It prints, and writes to
`Proje_Kodlari/evaluation_results/v3_deep_edge/v3_latency.json`:

    * mean / median / p95 per-frame latency, warm-up excluded
    * the full end-to-end cost (decode + preprocess + forward), which is the
      number that actually bounds the achievable frame rate
    * the host CPU string, thread count and torch version, so the measurement
      is attributable

Reporting note: state the thread count. A single thread is the honest setting
for a shared, thermally constrained companion computer; multi-thread numbers
are not comparable to it and should not be mixed in the same table.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent / "Proje_Kodlari"
WEIGHTS = ROOT / "evaluation_results" / "v3_deep_edge" / "v3_mobilenet.pth"
OUT = ROOT / "evaluation_results" / "v3_deep_edge" / "v3_latency.json"


def cpu_name():
    try:
        if platform.system() == "Windows":
            return platform.processor()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def summarise(samples_ms):
    samples_ms = sorted(samples_ms)
    return {
        "mean_ms": round(statistics.fmean(samples_ms), 2),
        "median_ms": round(statistics.median(samples_ms), 2),
        "p95_ms": round(samples_ms[int(0.95 * len(samples_ms)) - 1], 2),
        "min_ms": round(samples_ms[0], 2),
        "max_ms": round(samples_ms[-1], 2),
        "n": len(samples_ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--video", default=None, help="optional clip for end-to-end decode+infer timing")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    pil = Image.fromarray(frame)
    tensor = transform(pil).unsqueeze(0)

    # --- forward pass only -------------------------------------------------
    with torch.no_grad():
        for _ in range(args.warmup):
            model(tensor)
        forward_ms = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model(tensor)
            forward_ms.append((time.perf_counter() - t0) * 1000.0)

    # --- preprocess + forward ---------------------------------------------
    with torch.no_grad():
        for _ in range(args.warmup):
            model(transform(pil).unsqueeze(0))
        pipeline_ms = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model(transform(pil).unsqueeze(0))
            pipeline_ms.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "host_cpu": cpu_name(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "threads": args.threads,
        "input_resolution": "640x480 source, resized to 224x224",
        "forward_only": summarise(forward_ms),
        "preprocess_plus_forward": summarise(pipeline_ms),
    }

    # --- optional: full decode + preprocess + forward on a real clip -------
    if args.video:
        import cv2
        cap = cv2.VideoCapture(args.video)
        e2e_ms = []
        while len(e2e_ms) < args.runs:
            t0 = time.perf_counter()
            ret, f = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                model(transform(img).unsqueeze(0))
            e2e_ms.append((time.perf_counter() - t0) * 1000.0)
        cap.release()
        if e2e_ms:
            result["decode_preprocess_forward"] = summarise(e2e_ms)
            result["source_clip"] = args.video

    print(json.dumps(result, indent=2))
    fwd = result["forward_only"]
    pipe = result["preprocess_plus_forward"]
    print(f"\nQuote in the paper: {pipe['mean_ms']:.2f} ms per frame "
          f"(preprocess + forward, mean of {pipe['n']}, {args.threads} thread(s), {result['host_cpu']}).")
    print(f"Forward pass alone: {fwd['mean_ms']:.2f} ms. "
          f"Sustainable frame rate at the pipeline figure: {1000.0/pipe['mean_ms']:.1f} FPS.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
