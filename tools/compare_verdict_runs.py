"""Compare two clip-verdict runs and report what the difference costs.

The released verdicts in _clip_verdicts.json were produced in one environment.
Re-running tools/eval_clips_numpy.py elsewhere should reproduce them exactly,
since the weights are fixed, the sampling is deterministic and the decision rule
is a plain argmax. Anything that does not reproduce is either a defect or a
property of the video decoder, and the paper needs to know which.

This program names the clips that disagree, says which direction each moved, and
recomputes the reported cohort figures both ways so that the size of the effect
is stated rather than guessed.

Usage: python tools/compare_verdict_runs.py _verdicts_winA.json
       python tools/compare_verdict_runs.py A.json B.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from image_metrics import binary_metrics, environment_record  # noqa: E402

BASE = ROOT / "Proje_Kodlari"
MANIFEST = BASE / "annotations" / "video_evaluation_manifest.json"
PUBLISHED = BASE / "evaluation_results" / "v3_deep_edge" / "_clip_verdicts.json"
OUT = BASE / "evaluation_results" / "v3_deep_edge" / "v3_decoder_dependence.json"
HELD_OUT = {"test", "validation"}


def score(rows, verdicts):
    tp = sum(1 for v in rows if v["label"] == "fire" and verdicts[v["id"]]["alarm"])
    fn = sum(1 for v in rows if v["label"] == "fire" and not verdicts[v["id"]]["alarm"])
    fp = sum(1 for v in rows if v["label"] == "no_fire" and verdicts[v["id"]]["alarm"])
    tn = sum(1 for v in rows if v["label"] == "no_fire" and not verdicts[v["id"]]["alarm"])
    m = binary_metrics(tp, tn, fp, fn)
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, Recall=m["Recall"], FPR=m["FPR"],
                F1=m["F1"], Accuracy=m["Accuracy"])


def main(argv) -> int:
    # With no argument both runs would be the released file, and the program
    # would happily overwrite the artifact with a vacuous zero-disagreement
    # comparison of that file against itself.
    if not argv:
        print(__doc__.strip().splitlines()[-2].strip())
        print("error: name at least one verdict file to compare against the released one")
        return 2
    a_path, b_path = PUBLISHED, Path(argv[0])
    if len(argv) > 1:
        a_path, b_path = Path(argv[0]), Path(argv[1])

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = man["videos"] if isinstance(man, dict) and "videos" in man else man
    A = json.loads(a_path.read_text(encoding="utf-8"))
    B = json.loads(b_path.read_text(encoding="utf-8"))

    print(f"A = {a_path}")
    print(f"B = {b_path}\n")

    # Scoring below indexes every held-out clip in both runs, so a clip absent
    # from either would raise rather than be reported. Refuse first.
    absent = [c["id"] for c in clips if c["id"] not in A or c["id"] not in B]
    if absent:
        print(f"{len(absent)} clips are missing from one of the two runs, "
              f"e.g. {absent[:5]}; refusing to compare a partial corpus")
        return 1

    diff = [c for c in clips if bool(A[c["id"]]["alarm"]) != bool(B[c["id"]]["alarm"])]

    print(f"{len(clips)} clips, {len(diff)} disagree "
          f"({100.0 * len(diff) / len(clips):.2f}%)\n")
    for c in diff:
        cid = c["id"]
        print(f"  {cid}  {c['label']:8s} {c['split']:11s} "
              f"A alarm={bool(A[cid]['alarm'])} (n={A[cid].get('n_inferences')})  "
              f"B alarm={bool(B[cid]['alarm'])} (n={B[cid].get('n_inferences')})  "
              f"{Path(c['path']).name}")

    held = [c for c in clips if c["split"] in HELD_OUT]
    print("\nreported cohort, 194 clips that contributed no training gradient:")
    for name, v in (("A", A), ("B", B)):
        s = score(held, v)
        print(f"  {name}: TP={s['TP']:3d} TN={s['TN']:3d} FP={s['FP']:2d} FN={s['FN']:2d}"
              f"   TPR={s['Recall']}%  FPR={s['FPR']}%  F1={s['F1']}%  Acc={s['Accuracy']}%")

    def shown(p: Path) -> str:
        """Name a run by its path relative to the project, never absolutely.

        An absolute path in a released artifact records the machine it was made
        on rather than the file it refers to, and is meaningless to a reader.
        """
        try:
            return p.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return p.name

    out = {"_protocol": {
        "experiment": "Section 4.5 - decoder dependence of the clip verdicts",
        "run_A": shown(a_path), "run_B": shown(b_path),
        "note": "same weights, same sampling policy, same decision rule; the two "
                "runs differ only in the video decoder that produced the frames",
        "environment_of_this_comparison": environment_record(),
    },
        "n_clips": len(clips),
        "n_disagree": len(diff),
        "pct_disagree": round(100.0 * len(diff) / len(clips), 2),
        "disagreeing_clips": [
            {"id": c["id"], "label": c["label"], "split": c["split"],
             "file": Path(c["path"]).name,
             "A_alarm": bool(A[c["id"]]["alarm"]), "B_alarm": bool(B[c["id"]]["alarm"])}
            for c in diff],
        "held_out_run_A": score(held, A),
        "held_out_run_B": score(held, B),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
