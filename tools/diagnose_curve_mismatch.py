"""Explain why a clip's curve score disagrees with its published verdict.

tools/clip_score_curve.py refuses to emit a curve unless thresholding its score
at 0.5 reproduces the verdict recorded by tools/eval_clips_numpy.py for every
clip, because a curve that does not pass through the reported operating point is
not a curve of the reported system. When that check fires, this program isolates
the cause on the named clips.

It re-samples each clip twice: once the way the released evaluator does, with
cap.read() on every frame and the sample taken every step-th one, and once the
way the curve script does, with cap.grab() skipping the frames between samples
and cap.retrieve() decoding only those kept. If the two disagree on the number
of samples or on any probability, the sampler is the cause and the curve script
has to adopt read(). If they agree, the cause is the decision boundary itself:
argmax over two logits breaks a tie toward the first class, whereas p > 0.5 does
not, and the stored probabilities are rounded to five places, so a probability
just above a half can be stored as exactly a half.

Usage: python tools/diagnose_curve_mismatch.py VID048 VID211
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from clip_metrics import (  # noqa: E402
    CONSECUTIVE_FOR_ALARM, DEFAULT_FPS, FIRE_CLASS_INDEX, SAMPLES_PER_SECOND,
)
from torch_free_mobilenetv3 import (  # noqa: E402
    MobileNetV3SmallNumpy, load_state_dict, preprocess_bgr,
)

BASE = ROOT / "Proje_Kodlari"
MANIFEST = BASE / "annotations" / "video_evaluation_manifest.json"
WEIGHTS = BASE / "evaluation_results" / "v3_deep_edge" / "v3_mobilenet.pth"
VERDICTS = BASE / "evaluation_results" / "v3_deep_edge" / "_clip_verdicts.json"
PROBS = BASE / "evaluation_results" / "v3_deep_edge" / "_clip_frame_probs.jsonl"


def softmax_fire(logits):
    z = np.asarray(logits, dtype=np.float64).ravel()
    z = z - z.max()
    e = np.exp(z)
    return float(e[FIRE_CLASS_INDEX] / e.sum())


def step_for(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
    return max(1, int(fps / SAMPLES_PER_SECOND))


def sample_read(model, path):
    """The released evaluator's traversal: read() every frame, keep every step-th."""
    cap = cv2.VideoCapture(str(path))
    step = step_for(cap)
    out, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            logits = model.forward(preprocess_bgr(frame))
            out.append((softmax_fire(logits), int(np.argmax(logits)) == FIRE_CLASS_INDEX))
        idx += 1
    cap.release()
    return out, step


def sample_grab(model, path):
    """The curve script's traversal: grab() past the frames between samples."""
    cap = cv2.VideoCapture(str(path))
    step = step_for(cap)
    out, idx = [], 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            logits = model.forward(preprocess_bgr(frame))
            out.append((softmax_fire(logits), int(np.argmax(logits)) == FIRE_CLASS_INDEX))
        idx += 1
    cap.release()
    return out, step


def alarm_from_flags(flags):
    k = 0
    for f in flags:
        k = k + 1 if f else 0
        if k >= CONSECUTIVE_FOR_ALARM:
            return True
    return False


def score_from_probs(ps):
    k = CONSECUTIVE_FOR_ALARM
    if len(ps) < k:
        return 0.0
    return max(min(ps[i:i + k]) for i in range(len(ps) - k + 1))


def main(ids) -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = man["videos"] if isinstance(man, dict) and "videos" in man else man
    by_id = {c["id"]: c for c in clips}
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
    stored = {}
    if PROBS.exists():
        for line in PROBS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                stored[r["id"]] = r

    model = MobileNetV3SmallNumpy(load_state_dict(WEIGHTS))

    for cid in ids:
        c = by_id.get(cid)
        if c is None:
            print(f"{cid}: not in the manifest")
            continue
        path = ROOT / c["path"]
        if not path.exists():
            path = BASE / c["path"]
        print(f"\n=== {cid}  ({c['label']}, split {c['split']})")
        print(f"    {path.name}")

        r, step_r = sample_read(model, path)
        g, step_g = sample_grab(model, path)
        pr = [p for p, _ in r]
        pg = [p for p, _ in g]

        print(f"    stride            read={step_r}  grab={step_g}")
        print(f"    samples           read={len(pr)}  grab={len(pg)}"
              f"   stored={stored.get(cid, {}).get('n_samples', '-')}")
        print(f"    alarm, argmax     read={alarm_from_flags([f for _, f in r])}"
              f"  grab={alarm_from_flags([f for _, f in g])}")
        print(f"    published verdict {bool(verdicts[cid]['alarm'])}"
              f"   (inferences {verdicts[cid].get('n_inferences')})")
        print(f"    score>0.5         read={score_from_probs(pr) > 0.5}"
              f"  grab={score_from_probs(pg) > 0.5}"
              f"  stored={stored.get(cid, {}).get('score', float('nan')) > 0.5}")
        print(f"    score             read={score_from_probs(pr):.6f}"
              f"  grab={score_from_probs(pg):.6f}"
              f"  stored={stored.get(cid, {}).get('score', float('nan'))}")

        n = min(len(pr), len(pg))
        diff = [i for i in range(n) if abs(pr[i] - pg[i]) > 1e-9]
        print(f"    probabilities differ at {len(diff)} of {n} shared samples")
        if diff:
            for i in diff[:5]:
                print(f"       sample {i}: read={pr[i]:.6f}  grab={pg[i]:.6f}")
        near = [i for i, p in enumerate(pr) if abs(p - 0.5) < 1e-4]
        if near:
            print(f"    samples within 1e-4 of the boundary: {near[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["VID048", "VID211"]))
