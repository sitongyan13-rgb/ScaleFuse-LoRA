import unittest

import numpy as np

from scripts.search_prediction_tricks import fast_nms, greedy_nms, pairwise_iou


class PredictionTricksTest(unittest.TestCase):

    def test_pairwise_iou(self):
        box = np.array([0, 0, 10, 10], dtype=np.float64)
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
        values = pairwise_iou(box, boxes)
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(values[1], 25.0 / 175.0)
        self.assertEqual(values[2], 0.0)

    def test_greedy_nms_removes_lower_scored_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]])
        scores = np.array([0.8, 0.9, 0.7])
        keep = greedy_nms(boxes, scores, 0.5)
        self.assertEqual(keep.tolist(), [1, 2])

    def test_stable_tie_order_and_validation(self):
        boxes = np.array([[0, 0, 2, 2], [4, 4, 6, 6]])
        scores = np.array([0.5, 0.5])
        self.assertEqual(greedy_nms(boxes, scores, 0.5).tolist(), [0, 1])
        with self.assertRaises(ValueError):
            greedy_nms(boxes, np.array([0.5]), 0.5)
        with self.assertRaises(ValueError):
            greedy_nms(boxes, scores, 1.1)

    def test_fast_nms_matches_reference_for_unique_scores(self):
        boxes = np.array([
            [0, 0, 10, 10], [1, 1, 11, 11], [5, 5, 15, 15],
            [20, 20, 30, 30], [22, 22, 32, 32],
        ], dtype=np.float32)
        scores = np.array([0.91, 0.82, 0.73, 0.64, 0.55], dtype=np.float32)
        for threshold in (0.3, 0.5, 0.8, 1.0):
            with self.subTest(threshold=threshold):
                self.assertEqual(
                    fast_nms(boxes, scores, threshold).tolist(),
                    greedy_nms(boxes, scores, threshold).tolist())


if __name__ == "__main__":
    unittest.main()
