"""
Per-clip continuous scores, so that Section 4.5 can report a curve rather than
a single operating point.

The paper reports one operating point because the alarm rule is a plain argmax
followed by a three-consecutive-sample requirement, and that rule exposes no
knob to sweep. It does expose one implicitly. Write p_i for the fire-class
probability of the i-th sample of a clip. The clip alarms at decision threshold
t exactly when some window of CONSECUTIVE_FOR_ALARM successive samples has
every p_i above t, so the largest threshold at which the clip still alarms is

    score = max over windows w of ( min over i in w of p_i ).

Sweeping t over that score therefore reproduces the deployed rule at every
operating point rather than approximating it, and at t = 0.5 it must reproduce
the recorded verdicts exactly, since the deployed test is an argmax over two
logits with the fire class at index zero, which numpy resolves in favour of the
first index on a tie and which is therefore p >= 0.5 rather than p > 0.5. This
program asserts that agreement rather than assuming it, and refuses to emit a
curve without it.

Two differences from tools/eval_clips_numpy.py are deliberate. That program
stops at the first alarm, because the verdict cannot change afterwards; a curve
needs every sample, so the scan here runs to the end of the clip. And frames
between samples are skipped with grab() rather than decoded and discarded with
read(), which is what keeps the full scan affordable.

Usage:  python tools/clip_score_curve.py [--budget SECONDS]
        python tools/clip_score_curve.py --report
Writes: Proje_Kodlari/evaluation_results/v3_deep_edge/_clip_frame_probs.jsonl
        Proje_Kodlari/evaluation_results/v3_deep_edge/v3_operating_curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from clip_metrics import (  # noqa: E402
    CONSECUTIVE_FOR_ALARM, DEFAULT_FPS, FIRE_CLASS_INDEX, SAMPLES_PER_SECOND,
    wilson,
)
from image_metrics import environment_record  # noqa: E402
from torch_free_mobilenetv3 import (  # noqa: E402
    MobileNetV3SmallNumpy, load_state_dict, preprocess_bgr,
)

BASE = ROOT / "Proje_Kodlari"
MANIFEST = BASE / "annotations" / "video_evaluation_manifest.json"
WEIGHTS = BASE / "evaluation_results" / "v3_deep_edge" / "v3_mobilenet.pth"
VERDICTS = BASE / "evaluation_results" / "v3_deep_edge" / "_clip_verdicts.json"
PROBS = BASE / "evaluation_results" / "v3_deep_edge" / "_clip_frame_probs.jsonl"
OUT = BASE / "evaluation_results" / "v3_deep_edge" / "v3_operating_curve.json"

HELD_OUT_SPLITS = {"test", "validation"}


def softmax_fire(logits) -> float:
    z = np.asarray(logits, dtype=np.float64).ravel()
    z = z - z.max()
    e = np.exp(z)
    return float(e[FIRE_CLASS_INDEX] / e.sum())


def clip_probs(model, path) -> list[float]:
    """Fire-class probability of every sample of a clip, to the end of the clip.

    Frames that are not sampled are advanced with grab(), which reads the packet
    without decoding it. At the 24 fps of most clips in this corpus that skips
    eleven decodes out of every twelve.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
    step = max(1, int(fps / SAMPLES_PER_SECOND))

    probs: list[float] = []
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            probs.append(softmax_fire(model.forward(preprocess_bgr(frame))))
        idx += 1
    cap.release()
    return probs


def clip_score(probs: list[float]) -> float:
    """Largest threshold at which the clip still raises an alarm.

    Zero when the clip is shorter than the consecutive-sample requirement, which
    is the correct answer: such a clip cannot alarm at any threshold.
    """
    k = CONSECUTIVE_FOR_ALARM
    if len(probs) < k:
        return 0.0
    return max(min(probs[i:i + k]) for i in range(len(probs) - k + 1))


def load_done() -> dict:
    done = {}
    if PROBS.exists():
        for line in PROBS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["id"]] = r
    return done


# --------------------------------------------------------------------------
# curve
# --------------------------------------------------------------------------

def curve(scores, labels) -> dict:
    """ROC and precision-recall points, plus the areas under them.

    Thresholds are the distinct observed scores, so every attainable operating
    point appears and no point is invented by interpolation. The ROC area is the
    trapezoid rule over the resulting points; the average precision is the
    step-wise sum used for precision-recall, which does not interpolate.
    """
    pos = sum(labels)
    neg = len(labels) - pos
    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    roc, pr = [(0.0, 0.0)], []
    tp = fp = 0
    i = 0
    while i < len(order):
        t = scores[order[i]]
        while i < len(order) and scores[order[i]] == t:
            if labels[order[i]]:
                tp += 1
            else:
                fp += 1
            i += 1
        roc.append((fp / neg if neg else 0.0, tp / pos if pos else 0.0))
        pr.append((tp / pos if pos else 0.0, tp / (tp + fp), t))

    auroc = sum((roc[j][0] - roc[j - 1][0]) * (roc[j][1] + roc[j - 1][1]) / 2.0
                for j in range(1, len(roc)))
    ap, prev_r = 0.0, 0.0
    for r, p, _ in pr:
        ap += (r - prev_r) * p
        prev_r = r

    return {
        "AUROC": round(auroc, 4),
        "average_precision": round(ap, 4),
        "n_positive": pos,
        "n_negative": neg,
        "roc_points": [[round(x, 6), round(y, 6)] for x, y in roc],
        "pr_points": [[round(r, 6), round(p, 6), round(t, 6)] for r, p, t in pr],
    }


def operating_point(scores, labels, t) -> dict:
    # >= t, matching the argmax tie-break of the deployed rule; see the
    # agreement check in report().
    tp = sum(1 for s, y in zip(scores, labels) if y and s >= t)
    fn = sum(1 for s, y in zip(scores, labels) if y and s < t)
    fp = sum(1 for s, y in zip(scores, labels) if not y and s >= t)
    tn = sum(1 for s, y in zip(scores, labels) if not y and s < t)
    rec = 100.0 * tp / (tp + fn) if tp + fn else 0.0
    prec = 100.0 * tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lo, hi = wilson(tp, tp + fn) if tp + fn else (0.0, 0.0)
    return {"threshold": round(t, 4), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Recall_TPR": round(rec, 2), "Precision": round(prec, 2),
            "F1": round(f1, 2),
            "FPR": round(100.0 * fp / (fp + tn), 2) if fp + tn else 0.0,
            "Recall_TPR_wilson": [lo, hi]}


def report(verdict_path=None) -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = man["videos"] if isinstance(man, dict) and "videos" in man else man
    done = load_done()
    vpath = Path(verdict_path) if verdict_path else VERDICTS
    verdicts = json.loads(vpath.read_text(encoding="utf-8"))

    missing = [c["id"] for c in clips if c["id"] not in done]
    if missing:
        print(f"{len(missing)} clips still unscored; rerun without --report")
        return 1

    absent = [c["id"] for c in clips if done[c["id"]].get("missing")]
    if absent:
        print(f"{len(absent)} clip files were not on disk when scored "
              f"({absent[:5]}...); refusing to draw a curve over a partial corpus")
        return 1

    # A curve that does not pass through the reported operating point is not a
    # curve of the reported system, so the agreement is required rather than
    # assumed. It has to be checked against verdicts produced by the same video
    # decoder: the released verdicts and an OpenCV 5 run differ on two of the
    # 483 clips, and comparing across that boundary would mix two questions.
    # The deployed test is argmax over two logits with the fire class at index
    # zero, and numpy.argmax breaks a tie toward the first index, so the rule is
    # p >= 0.5 and not p > 0.5. A strict comparison here would disagree with the
    # evaluator on any exact tie.
    disagree = [c["id"] for c in clips
                if (done[c["id"]]["score"] >= 0.5) != bool(verdicts[c["id"]]["alarm"])]
    if disagree:
        print(f"ERROR: {len(disagree)} clips disagree at t=0.5 with {vpath.name}: "
              f"{disagree[:8]}")
        print("If those are the clips the decoder-dependence note names, pass the "
              "verdict file produced in this environment with --verdicts.")
        return 1

    out = {"_protocol": {
        "experiment": "Section 4.5 - operating curve over the deployed alarm rule",
        "score": "max over windows of CONSECUTIVE_FOR_ALARM successive samples "
                 "of the minimum fire probability in the window; the largest "
                 "threshold at which the clip still alarms",
        "consecutive_for_alarm": CONSECUTIVE_FOR_ALARM,
        "samples_per_second": SAMPLES_PER_SECOND,
        "reproduces_verdicts_at_threshold": 0.5,
        "verdicts_checked_against": str(vpath),
        "n_clips_checked": len(clips),
        "environment": environment_record(),
    }}

    for name, sel in (("held_out", lambda c: c["split"] in HELD_OUT_SPLITS),
                      ("full_manifest", lambda c: True)):
        rows = [c for c in clips if sel(c)]
        sc = [done[c["id"]]["score"] for c in rows]
        ly = [1 if c["label"] == "fire" else 0 for c in rows]
        blk = curve(sc, ly)
        blk["n_clips"] = len(rows)
        blk["operating_point_as_reported"] = operating_point(sc, ly, 0.5)
        out[name] = blk

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}\n")
    for name in ("held_out", "full_manifest"):
        b = out[name]
        print(f"{name:14s} n={b['n_clips']:3d}  AUROC={b['AUROC']}  "
              f"AP={b['average_precision']}  "
              f"(at t=0.5: TPR={b['operating_point_as_reported']['Recall_TPR']}%, "
              f"FPR={b['operating_point_as_reported']['FPR']}%)")
    print(f"\nVerdicts at t=0.5 agree with {vpath.name} on all {len(clips)} clips.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=150.0,
                    help="seconds of work before checkpointing and exiting")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verdicts", default=None,
                    help="verdict file to check the operating point against; "
                         "defaults to the released one")
    args = ap.parse_args()

    if args.report:
        return report(args.verdicts)

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = man["videos"] if isinstance(man, dict) and "videos" in man else man
    done = load_done()
    todo = [c for c in clips if c["id"] not in done]
    if not todo:
        print(f"All {len(clips)} clips scored; run with --report")
        return 0

    model = MobileNetV3SmallNumpy(load_state_dict(WEIGHTS))

    started = time.time()
    n = 0
    with PROBS.open("a", encoding="utf-8") as fh:
        for c in todo:
            path = ROOT / c["path"] if not Path(c["path"]).is_absolute() else Path(c["path"])
            if not path.exists():
                path = BASE / c["path"]
            # A clip that is not on disk must not be scored zero and admitted to
            # the curve: it would pass the t = 0.5 check against a verdict file
            # that also records no alarm for it, and the curve would silently
            # cover a partial corpus. Record the absence and refuse at report
            # time, as tools/eval_clips_numpy.py does.
            missing = not path.exists()
            probs = [] if missing else clip_probs(model, path)
            rec = {"id": c["id"], "n_samples": len(probs), "missing": missing,
                   "score": round(clip_score(probs), 6),
                   "probs": [round(p, 5) for p in probs]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            if time.time() - started > args.budget:
                break

    left = len(clips) - len(done) - n
    rate = n / max(time.time() - started, 1e-9)
    print(f"scored {n} clips this run, {left} left "
          f"({rate:.2f} clips/s, about {left / rate / 60:.1f} min remaining)"
          if rate > 0 else f"scored {n}, {left} left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
