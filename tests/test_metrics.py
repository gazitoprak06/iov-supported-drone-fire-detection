"""
Unit tests for the two shared definition modules.

Every number in the manuscript is computed by tools/image_metrics.py or
tools/clip_metrics.py, so a silent change in either would move results without
changing any visible line of an experiment. tools/check_paper_numbers.py checks
that the manuscript agrees with the artifacts; these tests check that the
arithmetic producing those artifacts is itself correct, against values worked
out by hand rather than against the modules' own output.

Run:  python -m unittest discover -s tests
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from clip_metrics import (  # noqa: E402
    CONSECUTIVE_FOR_ALARM,
    DEFAULT_FPS,
    FIRE_CLASS_INDEX,
    SAMPLES_PER_SECOND,
    wilson,
)
from image_metrics import (  # noqa: E402
    binary_metrics,
    confusion_from_decisions,
    mcnemar,
    null_baselines,
)


class TestBinaryMetrics(unittest.TestCase):
    def test_reported_section_4_1_confusion(self):
        """The released configuration of Section 4.1, computed by hand.

        TP=129 TN=49 FP=202 FN=30 over 410 images.
        accuracy    = 178/410 = 43.41%
        precision   = 129/331 = 38.97%
        recall      = 129/159 = 81.13%
        specificity =  49/251 = 19.52%
        FPR         = 202/251 = 80.48%
        """
        m = binary_metrics(tp=129, tn=49, fp=202, fn=30)
        self.assertEqual(m["Accuracy"], 43.41)
        self.assertEqual(m["Precision"], 38.97)
        self.assertEqual(m["Recall"], 81.13)
        self.assertEqual(m["Specificity"], 19.52)
        self.assertEqual(m["FPR"], 80.48)
        # F1 = 2PR/(P+R) on the rounded percentages, as the module defines it
        self.assertEqual(m["F1"], 52.65)

    def test_perfect_classifier(self):
        m = binary_metrics(tp=10, tn=10, fp=0, fn=0)
        for key in ("Accuracy", "Precision", "Recall", "Specificity", "F1"):
            self.assertEqual(m[key], 100.0)
        self.assertEqual(m["FPR"], 0.0)

    def test_precision_undefined_when_no_positive_prediction(self):
        """The constant-negative rule emits no positive, so precision is None.

        The manuscript renders this as an em dash rather than as 0%, and the
        distinction matters: a precision of zero would be a measurement, an
        undefined precision is the absence of one.
        """
        m = binary_metrics(tp=0, tn=251, fp=0, fn=159)
        self.assertIsNone(m["Precision"])
        self.assertEqual(m["Recall"], 0.0)
        self.assertEqual(m["Accuracy"], 61.22)

    def test_fpr_undefined_without_negatives(self):
        m = binary_metrics(tp=5, tn=0, fp=0, fn=1)
        self.assertIsNone(m["FPR"])
        self.assertIsNone(m["Specificity"])

    def test_confusion_from_decisions_matches_direct_counts(self):
        decisions = [(1, True)] * 129 + [(1, False)] * 30 \
                  + [(0, True)] * 202 + [(0, False)] * 49
        self.assertEqual(confusion_from_decisions(decisions),
                         binary_metrics(tp=129, tn=49, fp=202, fn=30))


class TestNullBaselines(unittest.TestCase):
    def test_constant_predictors_on_the_test_split(self):
        """159 fire and 251 no-fire, the split of Sections 4.1 to 4.4."""
        nb = null_baselines(n_fire=159, n_no_fire=251)
        pos, neg = nb["constant_positive"], nb["constant_negative"]
        self.assertEqual(pos["Recall"], 100.0)
        self.assertEqual(pos["Precision"], 38.78)
        self.assertEqual(pos["F1"], 55.89)
        self.assertEqual(pos["FPR"], 100.0)
        self.assertEqual(neg["Accuracy"], 61.22)
        self.assertEqual(neg["Recall"], 0.0)
        self.assertIsNone(neg["Precision"])

    def test_the_paper_s_central_claim_holds_arithmetically(self):
        """The hand-set rule loses to both null baselines on this split."""
        nb = null_baselines(159, 251)
        rule = binary_metrics(tp=129, tn=49, fp=202, fn=30)
        self.assertLess(rule["F1"], nb["constant_positive"]["F1"])
        self.assertLess(rule["Accuracy"], nb["constant_negative"]["Accuracy"])


class TestMcNemar(unittest.TestCase):
    def test_section_4_3_no_fire_images(self):
        """Resizing corrects 32 no-fire decisions and breaks none."""
        r = mcnemar(b=0, c=32)
        self.assertEqual(r["chi2"], 30.03)
        self.assertLess(r["p"], 0.0001)

    def test_section_4_3_fire_images(self):
        """Resizing breaks 18 fire decisions and corrects none."""
        r = mcnemar(b=18, c=0)
        self.assertEqual(r["chi2"], 16.06)
        self.assertAlmostEqual(r["p"], 0.0001, places=4)

    def test_section_4_3_all_images_is_not_significant(self):
        r = mcnemar(b=18, c=32)
        self.assertEqual(r["chi2"], 3.38)
        self.assertGreater(r["p"], 0.05)

    def test_symmetric_under_swap(self):
        self.assertEqual(mcnemar(7, 19)["chi2"], mcnemar(19, 7)["chi2"])

    def test_no_discordant_pairs(self):
        r = mcnemar(0, 0)
        self.assertIsNone(r["chi2"])


class TestWilson(unittest.TestCase):
    def test_reported_recall_interval(self):
        """63 of 68 fires alarmed, the headline cohort of Section 4.5."""
        lo, hi = wilson(63, 68)
        self.assertEqual([lo, hi], [83.91, 96.82])

    def test_reported_fpr_interval(self):
        """1 false alarm in 126 negative clips."""
        lo, hi = wilson(1, 126)
        self.assertEqual([lo, hi], [0.14, 4.36])

    def test_interval_contains_the_point_estimate(self):
        for k, n in ((63, 68), (1, 126), (32, 34), (102, 102), (0, 50)):
            lo, hi = wilson(k, n)
            point = 100.0 * k / n
            self.assertLessEqual(lo, point + 1e-9)
            self.assertGreaterEqual(hi, point - 1e-9)

    def test_interval_narrows_as_the_sample_grows(self):
        narrow = wilson(920, 1000)
        wide = wilson(92, 100)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_boundaries_stay_inside_zero_and_one_hundred(self):
        for k, n in ((0, 10), (10, 10), (1, 3)):
            lo, hi = wilson(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 100.0)


class TestDecisionConstants(unittest.TestCase):
    """These four constants define the alarm reported in Section 4.5.

    They are asserted here because a change to any of them would alter every
    clip-level number in the paper without altering a visible line of either
    evaluator, both of which import rather than restate them.
    """

    def test_fire_class_index_follows_alphabetical_imagefolder_order(self):
        self.assertEqual(FIRE_CLASS_INDEX, 0)
        self.assertEqual(sorted(["fire", "no_fire"]).index("fire"), FIRE_CLASS_INDEX)

    def test_sampling_and_smoothing(self):
        self.assertEqual(SAMPLES_PER_SECOND, 2)
        self.assertEqual(CONSECUTIVE_FOR_ALARM, 3)
        self.assertEqual(DEFAULT_FPS, 24.0)

    def test_alarm_latency_implied_by_the_sampling_policy(self):
        """Three consecutive samples at two per second is about 1.5 s."""
        self.assertAlmostEqual(
            CONSECUTIVE_FOR_ALARM / SAMPLES_PER_SECOND, 1.5, places=6)

    def test_stride_is_integer_division_of_the_frame_rate(self):
        """At 24 fps the evaluator sees every twelfth frame."""
        self.assertEqual(max(1, int(DEFAULT_FPS / SAMPLES_PER_SECOND)), 12)
        # The nominal two samples per second is exact only for an even rate.
        self.assertEqual(max(1, int(25.0 / SAMPLES_PER_SECOND)), 12)


class TestSharedModulesAgree(unittest.TestCase):
    def test_both_modules_compute_the_same_confusion_matrix(self):
        """image_metrics and clip_metrics must not diverge on shared cells."""
        from clip_metrics import metrics as clip_metrics_fn

        rows = [{"label": "fire", "alarm": True}] * 63 \
             + [{"label": "fire", "alarm": False}] * 5 \
             + [{"label": "no_fire", "alarm": True}] * 1 \
             + [{"label": "no_fire", "alarm": False}] * 125
        clip = clip_metrics_fn(rows)
        image = binary_metrics(tp=63, tn=125, fp=1, fn=5)
        for a, b in (("Recall_TPR", "Recall"), ("FPR", "FPR"),
                     ("Accuracy", "Accuracy"), ("F1", "F1")):
            self.assertEqual(clip[a], image[b], f"{a} vs {b}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
