"""Cross-check the torch-free NumPy forward pass against torchvision.

`tools/torch_free_mobilenetv3.py` re-implements MobileNetV3-Small by hand so
the evaluation can run without PyTorch. A hand-written forward pass is only
trustworthy if it agrees with the reference implementation, so this script
compares the two on real frames and reports the disagreement.

Run it on a machine that HAS PyTorch:

    python tools/verify_numpy_model.py --n 200

It fails (non-zero exit) if any sampled frame receives a different predicted
class, or if the maximum absolute logit difference exceeds --tol. A small
non-zero logit difference is expected and harmless; a class disagreement is
not, because clip verdicts are argmax decisions.
"""

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parent))
from torch_free_mobilenetv3 import (  # noqa: E402
    MobileNetV3SmallNumpy, load_state_dict, preprocess_bgr,
)

ROOT = Path("Proje_Kodlari")
WEIGHTS = ROOT / "evaluation_results" / "v3_deep_edge" / "v3_mobilenet.pth"
IMAGE_GLOB = str(ROOT / "data" / "dataset" / "*" / "*" / "*.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="frames to compare")
    ap.add_argument("--tol", type=float, default=1e-3, help="max allowed absolute logit delta")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(IMAGE_GLOB))
    if not paths:
        sys.exit(f"no images found under {IMAGE_GLOB}; run extract_training_frames.py first")
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.n]

    torch_model = models.mobilenet_v3_small(weights=None)
    torch_model.classifier[3] = nn.Linear(torch_model.classifier[3].in_features, 2)
    torch_model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    torch_model.eval()

    numpy_model = MobileNetV3SmallNumpy(load_state_dict(WEIGHTS))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    max_delta = 0.0
    disagreements = []

    for path in paths:
        pil = Image.open(path).convert("RGB")
        with torch.no_grad():
            ref = torch_model(transform(pil).unsqueeze(0)).numpy()[0]

        bgr = np.asarray(pil, dtype=np.uint8)[:, :, ::-1]
        got = numpy_model.forward(preprocess_bgr(np.ascontiguousarray(bgr)))

        delta = float(np.max(np.abs(ref - got)))
        max_delta = max(max_delta, delta)
        if int(np.argmax(ref)) != int(np.argmax(got)):
            disagreements.append((path, ref.tolist(), got.tolist()))

    print(f"compared {len(paths)} frames")
    print(f"max absolute logit difference: {max_delta:.3e} (tolerance {args.tol:.1e})")
    print(f"predicted-class disagreements: {len(disagreements)}")

    ok = True
    if disagreements:
        ok = False
        print("\nframes where the two implementations disagree:")
        for path, ref, got in disagreements[:10]:
            print(f"  {path}\n    torch={ref}\n    numpy={got}")
    if max_delta > args.tol:
        ok = False
        print(f"\nFAIL: logit difference {max_delta:.3e} exceeds tolerance {args.tol:.1e}")

    if not ok:
        sys.exit(1)
    print("\nPASS: the NumPy implementation reproduces torchvision on these frames.")


if __name__ == "__main__":
    main()
