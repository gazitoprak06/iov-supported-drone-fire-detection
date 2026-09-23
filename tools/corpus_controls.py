"""
Two controls on the clip corpus that the headline result of Section 4.5 needs.

The corpus is assembled from two sources, and its clips are cut from a smaller
number of source events. Both facts give a classifier a route to the right
answer that has nothing to do with fire, and neither is neutralised by the
clip-level partition the paper already applies. This script measures both.

Control 1, the source confound. Every negative clip comes from the ERA corpus,
while the positives are split between ERA and the Boreal wildfire corpus. A
classifier that learned to recognise the Boreal source rather than fire would
score perfectly on the Boreal positives and would still post a high recall
overall. The control therefore reports recall by source, and then scores the
held-out set restricted to ERA alone, where the source is constant and cannot
carry any signal.

Control 2, event-level leakage. The clips are cut from source events, and
several events contributed more than one clip. Partitioning at the clip level
does not stop two clips of the same fire, filmed in the same flight by the same
camera, from landing one in training and one in the held-out set. The control
reports how many events span partitions and scores the held-out clips whose
event contributed nothing to training, which is the stricter cohort.

Usage:  python tools/corpus_controls.py
Writes: Proje_Kodlari/evaluation_results/v3_deep_edge/v3_corpus_controls.json
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from clip_metrics import wilson  # noqa: E402
from image_metrics import binary_metrics, environment_record  # noqa: E402

MANIFEST = ROOT / "Proje_Kodlari" / "annotations" / "video_evaluation_manifest.json"
VERDICTS = ROOT / "Proje_Kodlari" / "evaluation_results" / "v3_deep_edge" / "_clip_verdicts.json"
OUT = ROOT / "Proje_Kodlari" / "evaluation_results" / "v3_deep_edge" / "v3_corpus_controls.json"

# Boreal clips are named after the Finnish municipality they were flown over;
# every other prefix is an ERA event category.
BOREAL_PREFIXES = {"evo", "heinola", "karkkila", "ruokolahti"}

# The clip-level partition names used by the manifest, mapped to the roles the
# paper reports. Held-out means "contributed no training gradient".
HELD_OUT_SPLITS = {"test", "validation"}


def prefix(path: str) -> str:
    return re.split(r"[_ ]", os.path.basename(path))[0]


def source(path: str) -> str:
    return "Boreal" if prefix(path) in BOREAL_PREFIXES else "ERA"


def event(path: str) -> str:
    """The source event a clip was cut from.

    Clip filenames end in a global clip index, so stripping the trailing
    underscore-number leaves the event: 'Fire_004 _1624.mp4' -> 'Fire_004',
    'karkkila_66_3082.mp4' -> 'karkkila_66'.
    """
    return re.sub(r"\s*_\d+\.mp4$", "", os.path.basename(path))


def score(rows, verdicts) -> dict:
    """Confusion matrix for a subset of clips.

    The arithmetic comes from image_metrics.binary_metrics rather than being
    restated here, so that these controls cannot round differently from the
    tables they are compared against.
    """
    tp = sum(1 for v in rows if v["label"] == "fire" and verdicts[v["id"]]["alarm"])
    fn = sum(1 for v in rows if v["label"] == "fire" and not verdicts[v["id"]]["alarm"])
    fp = sum(1 for v in rows if v["label"] == "no_fire" and verdicts[v["id"]]["alarm"])
    tn = sum(1 for v in rows if v["label"] == "no_fire" and not verdicts[v["id"]]["alarm"])
    m = binary_metrics(tp, tn, fp, fn)
    out = {
        "N_clips": len(rows), "N_fire": tp + fn, "N_no_fire": fp + tn,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": m["Accuracy"], "Recall_TPR": m["Recall"],
        "Precision": m["Precision"], "F1": m["F1"], "FPR": m["FPR"],
    }
    if tp + fn:
        lo, hi = wilson(tp, tp + fn)
        out["Recall_TPR_wilson"] = [lo, hi]
    return out


def main() -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = man["videos"] if isinstance(man, dict) and "videos" in man else man
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))

    missing = [c["id"] for c in clips if c["id"] not in verdicts]
    if missing:
        print(f"ERROR: {len(missing)} clips have no verdict; run tools/eval_clips_numpy.py first")
        return 1

    held = [c for c in clips if c["split"] in HELD_OUT_SPLITS]
    train_events = {event(c["path"]) for c in clips if c["split"] == "train"}

    composition = collections.Counter((c["label"], source(c["path"])) for c in clips)

    # --- Control 1: source ---------------------------------------------------
    by_source = {}
    for s in ("Boreal", "ERA"):
        rows = [c for c in held if c["label"] == "fire" and source(c["path"]) == s]
        if rows:
            by_source[s] = score(rows, verdicts)
    era_only = [c for c in held if source(c["path"]) == "ERA"]

    # --- Control 2: event-level leakage --------------------------------------
    spanning = {
        e: sorted({c["split"] for c in clips if event(c["path"]) == e})
        for e in {event(c["path"]) for c in clips}
    }
    spanning = {e: sp for e, sp in spanning.items() if len(sp) > 1}
    clean = [c for c in held if event(c["path"]) not in train_events]
    leaky = [c for c in held if event(c["path"]) in train_events]

    out = {
        "_protocol": {
            "experiment": "Section 4.5 - source and event-level controls on the clip corpus",
            "n_clips": len(clips),
            "n_held_out": len(held),
            "boreal_prefixes": sorted(BOREAL_PREFIXES),
            "event_definition": "filename with the trailing clip index removed",
            "environment": environment_record(),
        },
        "composition": {f"{lab}/{src}": n for (lab, src), n in sorted(composition.items())},
        "n_distinct_events": len({event(c["path"]) for c in clips}),
        "source_control": {
            "held_out_fire_by_source": by_source,
            "era_only_held_out": score(era_only, verdicts),
            "note": "every negative is ERA, so the ERA-only row holds the source "
                    "constant and no source signal can contribute to it",
        },
        "event_control": {
            "n_events_spanning_partitions": len(spanning),
            "held_out_all": score(held, verdicts),
            "held_out_event_unseen_in_training": score(clean, verdicts),
            "held_out_event_also_in_training": score(leaky, verdicts),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}\n")
    print("composition:", out["composition"])
    print(f"distinct events: {out['n_distinct_events']}, "
          f"spanning partitions: {len(spanning)}\n")
    for lab, blk in (("held-out, all", out["event_control"]["held_out_all"]),
                     ("held-out, event unseen in training",
                      out["event_control"]["held_out_event_unseen_in_training"]),
                     ("held-out, ERA only (source constant)",
                      out["source_control"]["era_only_held_out"])):
        print(f"  {lab:38s} n={blk['N_clips']:3d}  TPR={blk['Recall_TPR']}%  "
              f"FPR={blk['FPR']}%  F1={blk['F1']}%")
    for s, blk in by_source.items():
        print(f"  held-out fire, {s:26s} n={blk['N_clips']:3d}  TPR={blk['Recall_TPR']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
