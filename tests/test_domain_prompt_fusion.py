"""Unit tests for visual-conditioned domain prompt fusion."""

import unittest

import torch

from research.mmdet_plugins import VisualConditionedPromptFusion


class VisualConditionedPromptFusionTest(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.module = VisualConditionedPromptFusion(
            dim=16, num_prompts=4, gate_hidden_dim=8
        )
        self.features = (
            torch.randn(2, 16, 5, 7),
            torch.randn(2, 16, 3, 4),
        )
        self.embeddings = torch.randn(2, 6, 16)
        self.mask = torch.tensor(
            [[True, True, True, False, False, False],
             [True, True, True, True, False, False]]
        )

    def test_zero_initialization_is_exact_identity(self) -> None:
        output, weights = self.module(
            self.features, self.embeddings, self.mask
        )
        self.assertTrue(torch.equal(output, self.embeddings))
        self.assertEqual(tuple(weights.shape), (2, 4))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2)))
        self.assertEqual(int(torch.count_nonzero(self.module.projection.weight)), 0)

    def test_masked_padding_is_never_modified(self) -> None:
        with torch.no_grad():
            self.module.projection.weight.copy_(torch.eye(16))
        output, _ = self.module(self.features, self.embeddings, self.mask)
        padding = ~self.mask
        self.assertTrue(torch.equal(output[padding], self.embeddings[padding]))
        self.assertGreater(
            float((output[self.mask] - self.embeddings[self.mask]).abs().max()),
            0.0,
        )

    def test_gradients_are_finite_after_residual_is_active(self) -> None:
        with torch.no_grad():
            self.module.projection.weight.copy_(torch.eye(16))
        output, weights = self.module(self.features, self.embeddings, self.mask)
        loss = output.square().mean() + weights.square().mean()
        loss.backward()
        for name, parameter in self.module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        self.assertGreater(
            sum(
                int(torch.count_nonzero(parameter.grad))
                for parameter in self.module.parameters()
            ),
            0,
        )

    def test_invalid_shapes_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            self.module((), self.embeddings, self.mask)
        with self.assertRaises(ValueError):
            self.module(self.features, self.embeddings, self.mask[:, :-1])

    def test_uniform_gate_is_static_and_parameter_free(self) -> None:
        module = VisualConditionedPromptFusion(
            dim=16,
            num_prompts=4,
            gate_hidden_dim=8,
            gate_mode="uniform",
        )
        output, weights = module(self.features, self.embeddings, self.mask)
        self.assertTrue(torch.equal(output, self.embeddings))
        self.assertTrue(
            torch.equal(weights, torch.full_like(weights, 0.25))
        )
        self.assertIsNone(module.gate)
        self.assertEqual(
            {name for name, _ in module.named_parameters()},
            {"prompt_bank", "projection.weight"},
        )

    def test_invalid_gate_mode_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            VisualConditionedPromptFusion(
                dim=16,
                num_prompts=4,
                gate_hidden_dim=8,
                gate_mode="unsupported",
            )

    def test_uniform_prior_mix_bounds_routing_and_keeps_gradients(self) -> None:
        module = VisualConditionedPromptFusion(
            dim=16,
            num_prompts=4,
            gate_hidden_dim=8,
            uniform_prior_mix=0.5,
        )
        with torch.no_grad():
            module.gate[-1].bias.copy_(torch.tensor([20.0, -20.0, -20.0, -20.0]))
            module.projection.weight.copy_(torch.eye(16))
        output, weights = module(self.features, self.embeddings, self.mask)
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2)))
        self.assertGreaterEqual(float(weights.min()), 0.5 / 4 - 1e-7)
        self.assertLessEqual(float(weights.max()), 0.5 + 0.5 / 4 + 1e-7)
        output.square().mean().backward()
        gate_grad = module.gate[-1].weight.grad
        self.assertIsNotNone(gate_grad)
        self.assertTrue(bool(torch.isfinite(gate_grad).all()))

    def test_invalid_uniform_prior_mix_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            VisualConditionedPromptFusion(
                dim=16, num_prompts=4, gate_hidden_dim=8, uniform_prior_mix=1.1
            )


if __name__ == "__main__":
    unittest.main()
