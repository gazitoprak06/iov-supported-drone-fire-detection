"""
Constant-predictor null baselines for the clip cohorts of Section 4.5.

Section 4.1 audits the hand-set detector against the two constant predictors
because on an imbalanced split a constant rule is the floor a detector has to
clear before any of its other numbers mean anything. The replacement proposed
in Section 4.5 is owed the same audit on its own corpus, and this program
computes it so that the comparison is a released artifact rather than an
arithmetic claim in prose.

Nothing here re-runs the detector. The cohort composition and the per-clip
verdicts already exist; the constant predictors are determined entirely by the
class counts, and the detector's row is read back from the recorded confusion
matrix. The program therefore also serves as a cross-check: it recomputes the
detector's accuracy and F1 from the stored cells with the same arithmetic the
still-image cohort uses, and fails if they disagree with the recorded values.

Usage:  python tools/clip_null_baselines.py
Writes: Proje_Kodlari/evaluation_results/v3_deep_edge/v3_null_baselines.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from image_metrics import binary_metrics, environment_record  # noqa: E402

BASE = ROOT / "Proje_Kodlari" / "evaluation_results" / "v3_deep_edge"
SPLITS = BASE / "v3_results_by_split.json"
OUT = BASE / "v3_null_baselines.json"

# The cohorts the manuscript quotes a null baseline against.
COHORTS = ("unseen_by_gradient", "full_manifest_CONTAMINATED")


def main() -> int:
    blob = json.loads(SPLITS.read_text(encoding="utf-8"))
    out = {"_protocol": {
        "experiment": "Section 4.5 and Table 5 - constant-predictor null "
                      "baselines on the clip cohorts",
        "definition": "the constant-positive rule alarms on every clip; the "
                      "constant-negative rule alarms on none. Both are fixed by "
                      "the class counts alone and read no pixel.",
        "source": "cohort composition and confusion matrices of "
                  "v3_results_by_split.json; no clip is re-evaluated",
        "environment": environment_record(),
    }}

    for name in COHORTS:
        blk = blob[name]
        n_fire, n_no_fire = blk["N_fire"], blk["N_no_fire"]

        # The detector, recomputed from the stored cells rather than copied.
        det = binary_metrics(blk["TP"], blk["TN"], blk["FP"], blk["FN"])
        for key in ("Recall", "Precision", "F1", "Accuracy", "FPR"):
            recorded = blk.get(key if key != "Recall" else "Recall_TPR")
            if recorded is not None and det[key] is not None:
                if abs(det[key] - recorded) > 0.005:
                    print(f"ERROR: {name}.{key} recomputes to {det[key]}, "
                          f"artifact records {recorded}")
                    return 1

        out[name] = {
            "N_clips": blk["N_clips"], "N_fire": n_fire, "N_no_fire": n_no_fire,
            "prevalence": round(100.0 * n_fire / (n_fire + n_no_fire), 2),
            # every clip called fire: all positives found, every negative wrong
            "constant_positive": binary_metrics(n_fire, 0, n_no_fire, 0),
            # no clip called fire: no positive found, every negative right
            "constant_negative": binary_metrics(0, n_no_fire, 0, n_fire),
            "deep_edge": det,
        }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}\n")
    for name in COHORTS:
        b = out[name]
        print(f"{name}  (n={b['N_clips']}, {b['prevalence']}% fire)")
        print(f"   constant-positive   F1={b['constant_positive']['F1']}%  "
              f"Acc={b['constant_positive']['Accuracy']}%")
        print(f"   constant-negative   F1={b['constant_negative']['F1']}  "
              f"Acc={b['constant_negative']['Accuracy']}%")
        print(f"   deep edge           F1={b['deep_edge']['F1']}%  "
              f"Acc={b['deep_edge']['Accuracy']}%\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
