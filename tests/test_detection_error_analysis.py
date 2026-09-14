import unittest

import numpy as np

from scripts.analyze_detection_errors import (
    _average_ranks,
    pairwise_iou,
    ranked_error_counts,
    score_iou_alignment,
    size_name,
)


class DetectionErrorAnalysisTest(unittest.TestCase):
    def test_pairwise_iou(self):
        result = pairwise_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15]])
        np.testing.assert_allclose(result, [[1.0, 25.0 / 175.0]])

    def test_size_boundaries(self):
        self.assertEqual(size_name(32 ** 2 - 1), "small")
        self.assertEqual(size_name(32 ** 2), "medium")
        self.assertEqual(size_name(96 ** 2), "large")

    def test_ranked_error_taxonomy(self):
        predictions = {1: {
            "boxes": np.asarray([[0, 0, 10, 10], [0, 0, 10, 10],
                                 [16, 16, 26, 26], [50, 50, 60, 60]]),
            "scores": np.asarray([0.9, 0.8, 0.7, 0.6]),
            "labels": np.zeros(4, dtype=np.int64),
        }}
        ground_truth = {1: [
            {"id": 1, "box": [0, 0, 10, 10], "area": 100, "size": "small"},
            {"id": 2, "box": [20, 20, 30, 30], "area": 100, "size": "small"},
        ]}
        counts, _ = ranked_error_counts(predictions, ground_truth, 0.25, 0.5)
        self.assertEqual(counts["true_positive"], 1)
        self.assertEqual(counts["duplicate"], 1)
        self.assertEqual(counts["localization"], 1)
        self.assertEqual(counts["background"], 1)
        self.assertEqual(counts["false_negative"], 1)

    def test_average_ranks_with_ties(self):
        np.testing.assert_allclose(_average_ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])

    def test_score_iou_alignment(self):
        predictions = {1: {
            "boxes": np.asarray([[0, 0, 10, 10], [0, 0, 8, 8], [50, 50, 60, 60]]),
            "scores": np.asarray([0.9, 0.6, 0.1]),
            "labels": np.zeros(3, dtype=np.int64),
        }}
        ground_truth = {1: [
            {"id": 1, "box": [0, 0, 10, 10], "area": 100, "size": "small"},
        ]}
        result = score_iou_alignment(predictions, ground_truth)
        self.assertEqual(result["predictions"], 3)
        self.assertAlmostEqual(result["pearson_score_vs_best_iou"], 1.0, places=2)
        self.assertAlmostEqual(result["spearman_score_vs_best_iou"], 1.0)
        self.assertAlmostEqual(result["mean_best_iou_all"], (1 + 0.64) / 3)


if __name__ == "__main__":
    unittest.main()
