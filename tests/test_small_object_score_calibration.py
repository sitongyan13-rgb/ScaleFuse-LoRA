import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/search_small_object_score_calibration.py"
SPEC = importlib.util.spec_from_file_location("small_object_score_calibration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SmallObjectScoreCalibrationTest(unittest.TestCase):

    def test_zero_bias_preserves_finite_scores_inside_open_interval(self):
        boxes = np.asarray([[0, 0, 10, 10], [0, 0, 50, 50]], dtype=float)
        scores = np.asarray([0.2, 0.8], dtype=float)
        result = MODULE.calibrate_scores(boxes, scores, 256, 0.0)
        np.testing.assert_array_equal(result, scores)

    def test_positive_bias_boosts_only_small_boxes(self):
        boxes = np.asarray([[0, 0, 10, 10], [0, 0, 50, 50]], dtype=float)
        scores = np.asarray([0.2, 0.8], dtype=float)
        result = MODULE.calibrate_scores(boxes, scores, 256, 0.5)
        self.assertGreater(result[0], scores[0])
        self.assertEqual(result[1], scores[1])

    def test_area_threshold_is_inclusive(self):
        boxes = np.asarray([[0, 0, 16, 16]], dtype=float)
        scores = np.asarray([0.5], dtype=float)
        result = MODULE.calibrate_scores(boxes, scores, 256, 0.1)
        self.assertGreater(result[0], 0.5)

    def test_stable_top_k_retains_original_order_for_ties(self):
        scores = np.asarray([0.9, 0.9, 0.7], dtype=float)
        valid = np.asarray([True, True, True])
        np.testing.assert_array_equal(MODULE.rank_top_k(scores, valid, 2), [0, 1])

    def test_invalid_calibration_inputs_raise(self):
        cases = [
            (np.zeros((2, 3)), np.zeros(2), 256, 0.1),
            (np.zeros((2, 4)), np.zeros(3), 256, 0.1),
            (np.zeros((2, 4)), np.asarray([0.1, np.nan]), 256, 0.1),
            (np.zeros((2, 4)), np.zeros(2), 0, 0.1),
            (np.zeros((2, 4)), np.zeros(2), 256, -0.1),
        ]
        for boxes, scores, threshold, bias in cases:
            with self.subTest(threshold=threshold, bias=bias):
                with self.assertRaises(ValueError):
                    MODULE.calibrate_scores(boxes, scores, threshold, bias)


if __name__ == "__main__":
    unittest.main()
