"""Unit tests for scale-aware cross-level visual feature fusion."""

import unittest

import torch

from research.mmdet_plugins import ScaleAwareCrossLevelFusion


class ScaleAwareCrossLevelFusionTest(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.module = ScaleAwareCrossLevelFusion(
            dim=32, num_levels=4, gate_hidden_dim=16
        )
        self.features = (
            torch.randn(2, 32, 16, 20),
            torch.randn(2, 32, 8, 10),
            torch.randn(2, 32, 4, 5),
            torch.randn(2, 32, 2, 3),
        )

    def test_zero_initialization_is_exact_identity(self) -> None:
        outputs, gates = self.module(self.features)
        self.assertEqual(tuple(gates.shape), (2, 3))
        for output, original in zip(outputs, self.features):
            self.assertTrue(torch.equal(output, original))
        self.assertTrue(bool(((gates > 0) & (gates < 1)).all()))

    def test_active_residual_preserves_shapes_and_has_finite_gradients(self) -> None:
        with torch.no_grad():
            for projection in self.module.residual_projections:
                projection.weight.copy_(
                    torch.eye(32).reshape(32, 32, 1, 1)
                )
        outputs, gates = self.module(self.features)
        self.assertEqual([tuple(x.shape) for x in outputs],
                         [tuple(x.shape) for x in self.features])
        self.assertGreater(float((outputs[0] - self.features[0]).abs().max()), 0)
        loss = sum(output.square().mean() for output in outputs) + gates.mean()
        loss.backward()
        for name, parameter in self.module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_top_down_path_changes_all_but_coarsest_level(self) -> None:
        with torch.no_grad():
            for projection in self.module.residual_projections:
                projection.weight.copy_(
                    torch.eye(32).reshape(32, 32, 1, 1)
                )
        outputs, _ = self.module(self.features)
        for index in range(3):
            self.assertFalse(torch.equal(outputs[index], self.features[index]))
        self.assertTrue(torch.equal(outputs[3], self.features[3]))

    def test_invalid_feature_count_and_shape_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            self.module(self.features[:3])
        malformed = list(self.features)
        malformed[1] = torch.randn(2, 31, 8, 10)
        with self.assertRaises(ValueError):
            self.module(tuple(malformed))

    def test_invalid_constructor_arguments_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            ScaleAwareCrossLevelFusion(dim=32, num_levels=1)
        with self.assertRaises(ValueError):
            ScaleAwareCrossLevelFusion(dim=32, interpolation_mode="bicubic")
        with self.assertRaises(ValueError):
            ScaleAwareCrossLevelFusion(dim=32, gate_mode="unsupported")

    def test_constant_one_gate_is_parameter_free_and_exact(self) -> None:
        module = ScaleAwareCrossLevelFusion(
            dim=32, num_levels=4, gate_mode="constant_one"
        )
        outputs, gates = module(self.features)
        self.assertIsNone(module.gates)
        self.assertTrue(torch.equal(gates, torch.ones_like(gates)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs, self.features)))
        self.assertFalse(any("gates" in name for name, _ in module.named_parameters()))


if __name__ == "__main__":
    unittest.main()
