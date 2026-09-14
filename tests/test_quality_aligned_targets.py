import unittest

import torch

from research.mmdet_plugins import localization_quality_token_targets


class QualityAlignedTargetsTest(unittest.TestCase):
    def test_positive_targets_are_scaled_by_aligned_iou(self):
        tokens = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        predictions = torch.tensor([
            [0.5, 0.5, 0.4, 0.4],
            [0.6, 0.5, 0.4, 0.4],
            [0.1, 0.1, 0.1, 0.1],
        ], requires_grad=True)
        targets = torch.tensor([
            [0.5, 0.5, 0.4, 0.4],
            [0.5, 0.5, 0.4, 0.4],
            [0.0, 0.0, 0.0, 0.0],
        ])
        weights = torch.tensor([[1.0] * 4, [1.0] * 4, [0.0] * 4])
        result = localization_quality_token_targets(
            tokens, predictions, targets, weights
        )
        self.assertAlmostEqual(float(result[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(result[1, 0]), 0.6, places=6)
        self.assertTrue(torch.equal(result[2], tokens[2]))
        self.assertFalse(result.requires_grad)

    def test_power_is_applied_and_inputs_are_not_modified(self):
        tokens = torch.tensor([[1.0, 1.0]])
        predictions = torch.tensor([[0.6, 0.5, 0.4, 0.4]])
        targets = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        weights = torch.ones_like(targets)
        result = localization_quality_token_targets(
            tokens, predictions, targets, weights, power=2.0
        )
        self.assertAlmostEqual(float(result[0, 0]), 0.36, places=6)
        self.assertTrue(torch.equal(tokens, torch.ones_like(tokens)))

    def test_invalid_shapes_and_power_fail_loudly(self):
        tokens = torch.ones(1, 2)
        boxes = torch.ones(1, 4)
        with self.assertRaises(ValueError):
            localization_quality_token_targets(tokens, boxes, boxes, boxes, 0)
        with self.assertRaises(ValueError):
            localization_quality_token_targets(tokens[0], boxes, boxes, boxes)
        with self.assertRaises(ValueError):
            localization_quality_token_targets(tokens, boxes[:, :3], boxes, boxes)


if __name__ == "__main__":
    unittest.main()
