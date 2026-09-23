"""Pre-submission check: every V3 number in makale.html must match the artifacts.

Run before exporting to .docx:

    python tools/check_paper_numbers.py

Checks performed
----------------
1. Every <span data-v3="split.metric">value</span> in the manuscript matches
   `v3_results_by_split.json` to the displayed precision.
2. The derived gaps quoted in the abstract, Section 4.8 and the conclusion are
   consistent with those same numbers.
3. The Wilson confidence intervals quoted in Section 4.8 and Section 5 are
   recomputed from the confusion matrix.
4. No `.needs-measurement` placeholder is left unfilled.
5. The clip counts and class balances quoted in prose match the manifest.
6. None of the superseded figures or overclaims from the pre-correction draft
   has survived anywhere in the file.

Exit status is non-zero if any check fails, so this can gate a commit.
"""

import json
import math
import re
import sys
from pathlib import Path

PAPER = Path("makale.html")
RESULTS = Path("Proje_Kodlari/evaluation_results/v3_deep_edge/v3_results_by_split.json")
MANIFEST = Path("Proje_Kodlari/annotations/video_evaluation_manifest.json")

# Namespaces for the data-num tags that cover Tables 1 to 4. data-v3 tags,
# handled separately below, predate these and cover the deep edge section only.
ARTIFACTS = {
    "v1":    Path("Proje_Kodlari/evaluation_results/v1_image_level/v1_image_level_metrics.json"),
    "res":   Path("Proje_Kodlari/evaluation_results/v1_image_level/v1_resolution_control.json"),
    "sweep": Path("Proje_Kodlari/evaluation_results/v1_image_level/v1_saturation_sweep.json"),
    "v2":    Path("Proje_Kodlari/evaluation_results/v2_baselines/v2_baseline_metrics.json"),
    "v1lat": Path("Proje_Kodlari/evaluation_results/v1_image_level/v1_latency_variability.json"),
    "ctrl":  Path("Proje_Kodlari/evaluation_results/v3_deep_edge/v3_corpus_controls.json"),
    "seed":  Path("Proje_Kodlari/evaluation_results/v3_deep_edge/v3_seed_variance.json"),
}


def resolve(blob, path):
    """Walk a dotted path, treating an integer step as a list index.

    Keys may themselves contain dots and spaces (the ablation rows are named
    'HSV + RGB + YCbCr (proposed)'), so the longest matching key wins at each
    step rather than splitting naively on every dot.
    """
    node = blob
    rest = path
    while rest:
        if isinstance(node, list):
            head, _, rest = rest.partition(".")
            node = node[int(head)]
            continue
        if not isinstance(node, dict):
            return None
        for key in sorted(node, key=len, reverse=True):
            if rest == key:
                return node[key]
            if rest.startswith(key + "."):
                node = node[key]
                rest = rest[len(key) + 1:]
                break
        else:
            return None
    return node


def check_tagged_numbers(html, check):
    """Every data-num span must equal its artifact value to displayed precision."""
    cache = {}
    n = 0
    for ns, path, shown in re.findall(r'<span data-num="([^:]+):([^"]+)">([^<]+)</span>', html):
        n += 1
        if ns not in ARTIFACTS:
            check(False, f"unknown artifact namespace '{ns}' in tag {ns}:{path}")
            continue
        if ns not in cache:
            if not ARTIFACTS[ns].exists():
                check(False, f"artifact {ARTIFACTS[ns]} not found for namespace '{ns}'")
                cache[ns] = {}
            else:
                cache[ns] = json.loads(ARTIFACTS[ns].read_text(encoding="utf-8"))
        expected = resolve(cache[ns], path)
        if expected is None:
            check(False, f"tag {ns}:{path} does not resolve in {ARTIFACTS[ns].name}")
            continue
        try:
            got = float(shown.replace("%", "").replace(",", "").strip())
        except ValueError:
            check(False, f"tag {ns}:{path} shows non-numeric text {shown!r}")
            continue
        check(abs(got - float(expected)) < 0.005,
              f"tag {ns}:{path} shows {shown} but the artifact holds {expected}")
    return n

failures = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def wilson(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half) * 100, 2), round(min(1.0, centre + half) * 100, 2))


def main():
    html = PAPER.read_text(encoding="utf-8")
    results = json.loads(RESULTS.read_text())

    # -- 1. data-v3 spans ---------------------------------------------------
    spans = re.findall(r'<span data-v3="([^"]+)">([^<]+)</span>', html)
    print(f"Found {len(spans)} tagged numbers in {PAPER}")
    for ref, shown in spans:
        split, metric = ref.split(".")
        check(split in results, f"unknown split '{split}' referenced by data-v3")
        if split not in results:
            continue
        check(metric in results[split], f"unknown metric '{metric}' in split '{split}'")
        if metric not in results[split]:
            continue
        expected = results[split][metric]
        shown_clean = shown.strip().rstrip("%")
        try:
            shown_val = float(shown_clean)
        except ValueError:
            check(False, f"{ref}: '{shown}' is not numeric")
            continue
        check(abs(shown_val - float(expected)) < 0.005,
              f"{ref}: manuscript says {shown}, artifact says {expected}")

    unseen = results["unseen_by_gradient"]
    seen = results["gradient_train"]
    full = results["full_manifest_CONTAMINATED"]

    # -- 2. internal consistency of the artifact itself ---------------------
    parts = ["gradient_train", "epoch_monitor", "untouched"]
    for field in ("TP", "TN", "FP", "FN", "N_clips"):
        check(sum(results[p][field] for p in parts) == full[field],
              f"the three partitions do not sum to the full corpus on {field}")
    for field in ("TP", "TN", "FP", "FN", "N_clips"):
        check(results["epoch_monitor"][field] + results["untouched"][field]
              == unseen[field],
              f"epoch_monitor + untouched != unseen_by_gradient on {field}")

    # -- 3. derived gaps ----------------------------------------------------
    gap_seen = round(seen["Recall_TPR"] - unseen["Recall_TPR"], 2)
    check(f"{gap_seen:.2f} points of recall" in html,
          f"Section 4.8 should quote a memorized/unseen recall gap of {gap_seen:.2f} points")

    gap_full = round(full["Recall_TPR"] - unseen["Recall_TPR"], 2)
    check(f"{gap_full:.2f} points" in html and f"{gap_full:.2f}-point gap" in html,
          f"the contaminated/unseen gap of {gap_full:.2f} points should appear in both "
          f"Section 4.8 and the abstract")

    # -- 4. Wilson intervals ------------------------------------------------
    lo, hi = wilson(unseen["TP"], unseen["N_fire"])
    check([lo, hi] == unseen["Recall_TPR_wilson"],
          "artifact recall Wilson interval disagrees with an independent recomputation")
    check(f"[{lo:.2f}%, {hi:.2f}%]" in html,
          f"recall Wilson interval should be [{lo:.2f}%, {hi:.2f}%]")

    flo, fhi = wilson(unseen["FP"], unseen["N_no_fire"])
    check([flo, fhi] == unseen["FPR_wilson"],
          "artifact FPR Wilson interval disagrees with an independent recomputation")
    check(f"[{flo:.2f}%, {fhi:.2f}%]" in html,
          f"false-positive-rate Wilson interval should be [{flo:.2f}%, {fhi:.2f}%]")

    # -- 4. unfilled placeholders ------------------------------------------
    placeholders = re.findall(r'<span class="needs-measurement">([^<]*)</span>', html)
    check(not placeholders,
          f"{len(placeholders)} measurement placeholder(s) still unfilled: "
          f"{'; '.join(p[:60] for p in placeholders)}")

    # -- 5. clip counts and compositions cited in prose ---------------------
    check(f"{unseen['N_clips']}</span> clips that contributed no training gradient" in html,
          f"the unseen set should be described as {unseen['N_clips']} clips")
    flat = re.sub(r"<[^>]+>", "", html)
    check(f"({unseen['N_fire']} fire, {unseen['N_no_fire']} no-fire)" in flat,
          f"unseen composition should be stated as ({unseen['N_fire']} fire, "
          f"{unseen['N_no_fire']} no-fire)")
    check(f"{seen['TP']}</span> of {seen['N_fire']} fires" in html,
          f"the memorized split should be stated as {seen['TP']} of {seen['N_fire']} fires")
    check(str(full["N_clips"]) in html, "the full corpus size should appear")

    # -- 5b. every tagged number of Tables 1 to 4 against its artifact -------
    n_tagged = check_tagged_numbers(html, check)
    print(f"Found {n_tagged} data-num tagged numbers across Tables 1 to 4")

    # -- 6. the results artifact itself must agree with the clip manifest ----
    # Checks 1 to 5 compare the manuscript against the results artifact. If the
    # artifact were itself built over the wrong set of clips every one of them
    # would still pass, so the manifest is reconciled here independently.
    if not MANIFEST.exists():
        check(False, f"clip manifest not found at {MANIFEST}")
    else:
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        clips = man["videos"] if isinstance(man, dict) and "videos" in man else man
        n_total = len(clips)
        n_fire = sum(1 for c in clips if c.get("label") == "fire")
        by_split = {}
        for c in clips:
            key = (c.get("split"), c.get("label"))
            by_split[key] = by_split.get(key, 0) + 1

        check(n_total == full["N_clips"],
              f"manifest holds {n_total} clips but the artifact reports "
              f"{full['N_clips']}")
        check(n_fire == full["N_fire"],
              f"manifest holds {n_fire} fire clips but the artifact reports "
              f"{full['N_fire']}")
        check(str(n_fire) in flat and str(n_total - n_fire) in flat,
              f"the corpus composition ({n_fire} fire, {n_total - n_fire} no-fire) "
              f"should appear in the prose")

        # SPLIT_ROLE in clip_metrics maps the manifest's split names onto the
        # three roles the paper reports; the counts must line up group by group.
        roles = {
            "gradient_train": "train",
            "epoch_monitor": "test",
            "untouched": "validation",
        }
        for group, split_name in roles.items():
            g = results[group]
            m_fire = by_split.get((split_name, "fire"), 0)
            m_no = by_split.get((split_name, "no_fire"), 0)
            check(m_fire == g["N_fire"] and m_no == g["N_no_fire"],
                  f"manifest split '{split_name}' holds {m_fire} fire / {m_no} "
                  f"no-fire but the artifact's {group} reports {g['N_fire']} / "
                  f"{g['N_no_fire']}")

    # -- 6. stale numbers and claims from the pre-correction draft ----------
    for stale, why in [
        ("98.55% True Positive Rate", "accuracy was mislabelled as TPR"),
        ("168/170", "TP was overstated"),
        ("34.50 ms", "unsourced latency figure"),
        ("19.64", "unsourced V1 latency"),
        ("9.82", "unsourced V2 latency"),
        ("385 training clips", "wrong training-clip count; 96 validation clips were never read"),
        ("test_heldout", "superseded artifact key"),
        ("train_seen", "superseded artifact key"),
        ("98 held-out clips", "superseded two-way partition wording"),
        ("held-out clips</b>", "superseded two-way partition wording"),
        ("no stage of the pipeline", "the 96 clips are read at evaluation time"),
        ("rigorously benchmarked", "unsupported backbone benchmark claim"),
        ("definitively proves", "overclaim"),
        ("completely eliminates", "overclaim"),
        ("fundamentally fail", "overclaim on a 32-clip control"),
        ("in Section 4.9 for V3", "Section 4.9 contains no V3 measurement"),
    ]:
        check(stale not in html, f"stale text still present ({why}): '{stale}'")

    # -- 7. every split named in the artifact is described in the paper -----
    for group in ("unseen_by_gradient", "untouched", "epoch_monitor",
                  "gradient_train", "full_manifest_CONTAMINATED"):
        check(group in results, f"artifact is missing the '{group}' grouping")
    check("_warning" not in results,
          f"the evaluator warned about the manifest: {results.get('_warning')}")

    # -- report -------------------------------------------------------------
    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All numbers in the manuscript trace to the artifacts.")


if __name__ == "__main__":
    main()
