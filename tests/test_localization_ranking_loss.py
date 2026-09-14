import unittest

import torch

from research.mmdet_plugins import localization_aware_ranking_loss


class LocalizationAwareRankingLossTest(unittest.TestCase):
    def inputs(self, logits):
        token_targets = torch.zeros(1, 4, 3)
        token_targets[0, :2, 0] = 1
        predictions = torch.tensor([[[0.5, 0.5, 0.4, 0.4],
                                     [0.65, 0.5, 0.4, 0.4],
                                     [0.1, 0.1, 0.1, 0.1],
                                     [0.9, 0.9, 0.1, 0.1]]], requires_grad=True)
        targets = torch.tensor([[[0.5, 0.5, 0.4, 0.4],
                                 [0.5, 0.5, 0.4, 0.4],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]]])
        weights = torch.zeros_like(targets)
        weights[0, :2] = 1
        mask = torch.tensor([[True, True, False]])
        return logits, token_targets, predictions, targets, weights, mask

    def test_better_quality_order_has_lower_loss(self):
        ordered = torch.tensor([[[3.0, -3.0, -9.0], [1.0, -3.0, -9.0],
                                 [0.0, -2.0, -9.0], [-1.0, -2.0, -9.0]]])
        reversed_logits = ordered.clone()
        reversed_logits[0, 0, 0], reversed_logits[0, 1, 0] = 1.0, 3.0
        ordered_loss = localization_aware_ranking_loss(*self.inputs(ordered))
        reversed_loss = localization_aware_ranking_loss(
            *self.inputs(reversed_logits))
        self.assertLess(float(ordered_loss), float(reversed_loss))

    def test_gradients_only_flow_to_logits(self):
        logits = torch.tensor([[[2.0, -3.0, -9.0], [1.0, -3.0, -9.0],
                                [0.5, -2.0, -9.0], [-1.0, -2.0, -9.0]]],
                              requires_grad=True)
        inputs = self.inputs(logits)
        loss = localization_aware_ranking_loss(*inputs)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertIsNone(inputs[2].grad)

    def test_no_positive_is_finite_zero(self):
        inputs = list(self.inputs(torch.zeros(1, 4, 3, requires_grad=True)))
        inputs[4].zero_()
        loss = localization_aware_ranking_loss(*inputs)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(inputs[0].grad).all())

    def test_invalid_settings_fail_loudly(self):
        inputs = self.inputs(torch.zeros(1, 4, 3))
        with self.assertRaises(ValueError):
            localization_aware_ranking_loss(*inputs, temperature=0)
        with self.assertRaises(ValueError):
            localization_aware_ranking_loss(*inputs[:-1], torch.ones(1, 2))


if __name__ == "__main__":
    unittest.main()
