import unittest

import torch

from research.mmdet_plugins import scale_adaptive_regression_weights


class ScaleAdaptiveRegressionWeightsTest(unittest.TestCase):

    def test_small_positive_boxes_receive_continuous_extra_weight(self):
        targets = torch.tensor([
            [0.5, 0.5, 0.05, 0.05],
            [0.5, 0.5, 0.10, 0.10],
            [0.5, 0.5, 0.20, 0.20],
        ])
        weights = torch.ones_like(targets)
        result = scale_adaptive_regression_weights(
            targets, weights, reference_area=0.01, max_weight=2.0)
        self.assertTrue(torch.allclose(result[0], torch.full((4,), 1.75)))
        self.assertTrue(torch.equal(result[1], torch.ones(4)))
        self.assertTrue(torch.equal(result[2], torch.ones(4)))

    def test_negative_rows_remain_exactly_zero(self):
        targets = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.4, 0.6, 0.02, 0.03],
        ])
        weights = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        result = scale_adaptive_regression_weights(targets, weights)
        self.assertTrue(torch.equal(result[0], torch.zeros(4)))
        self.assertAlmostEqual(result[1, 0].item(), 1.94, places=6)

    def test_invalid_hyperparameters_fail_loudly(self):
        targets = torch.zeros((1, 4))
        weights = torch.zeros_like(targets)
        for reference_area, max_weight in (
            (0.0, 2.0), (-0.1, 2.0), (0.01, 0.9)
        ):
            with self.subTest(reference_area=reference_area, max_weight=max_weight):
                with self.assertRaises(ValueError):
                    scale_adaptive_regression_weights(
                        targets, weights, reference_area=reference_area,
                        max_weight=max_weight)

    def test_shape_and_negative_size_validation(self):
        with self.assertRaises(ValueError):
            scale_adaptive_regression_weights(torch.zeros(4), torch.zeros(4))
        with self.assertRaises(ValueError):
            scale_adaptive_regression_weights(
                torch.zeros((2, 4)), torch.zeros((2, 3)))
        targets = torch.tensor([[0.5, 0.5, -0.1, 0.1]])
        with self.assertRaises(ValueError):
            scale_adaptive_regression_weights(targets, torch.ones_like(targets))


if __name__ == "__main__":
    unittest.main()
