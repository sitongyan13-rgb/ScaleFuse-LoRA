#!/usr/bin/env python
"""Audit learned prompt-routing behavior on the fixed Pothole validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "third_party" / "mmdetection-3.3.0"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().float()).cpu())


def summary(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "sample_sd": float(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0,
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def main():
    args = parse_args()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    cfg = Config.fromfile(str(config))
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.launcher = "none"
    cfg.randomness.seed = args.seed
    cfg.default_hooks.visualization.draw = False
    runner = Runner.from_cfg(cfg)
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    module = model.domain_prompt_fusion
    if module.gate_mode != "learned":
        raise RuntimeError("Routing audit requires a learned gate")

    weights = []
    relative_residuals = []
    residual_rms = []

    def capture(_module, inputs, kwargs, outputs):
        text_embeddings = kwargs["text_embeddings"].detach().float()
        text_token_mask = kwargs["text_token_mask"].detach().bool()
        adapted, prompt_weights = outputs
        adapted = adapted.detach().float()
        prompt_weights = prompt_weights.detach().float()
        for index in range(prompt_weights.shape[0]):
            valid = text_token_mask[index]
            before = text_embeddings[index, valid]
            delta = adapted[index, valid] - before
            before_rms = torch.sqrt(torch.mean(before.square())).clamp_min(1e-12)
            delta_rms = torch.sqrt(torch.mean(delta.square()))
            weights.append(prompt_weights[index].cpu())
            residual_rms.append(float(delta_rms.cpu()))
            relative_residuals.append(float((delta_rms / before_rms).cpu()))

    handle = module.register_forward_hook(capture, with_kwargs=True)
    try:
        metrics = runner.test()
    finally:
        handle.remove()
    if len(weights) != 133:
        raise RuntimeError("Expected 133 validation images, captured {}".format(len(weights)))

    matrix = torch.stack(weights).double()
    if not torch.isfinite(matrix).all():
        raise RuntimeError("Non-finite prompt weight")
    row_sums = matrix.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        raise RuntimeError("Prompt weights do not form a simplex")
    count = matrix.shape[1]
    uniform = torch.full_like(matrix, 1.0 / count)
    entropy = -(matrix.clamp_min(1e-12).log() * matrix).sum(dim=1)
    normalized_entropy = entropy / math.log(count)
    l1_uniform = (matrix - uniform).abs().sum(dim=1)
    max_weight = matrix.max(dim=1).values
    dominant = matrix.argmax(dim=1)
    dominant_counts = [int((dominant == index).sum()) for index in range(count)]

    projected = module.projection(module.prompt_bank).detach().float()
    projected_norms = torch.linalg.vector_norm(projected, dim=1)
    normalized = torch.nn.functional.normalize(projected, dim=1, eps=1e-12)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~torch.eye(count, dtype=torch.bool, device=cosine.device)]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "ODinW Pothole",
            "split": "validation",
            "images": len(weights),
            "seed": args.seed,
            "config": str(config),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            "held_out_test_evaluated": False,
        },
        "validation_metrics": {key: float(value) for key, value in metrics.items()},
        "parameters": {
            "prompt_count": count,
            "temperature": module.temperature,
            "prompt_bank_frobenius_norm": tensor_norm(module.prompt_bank),
            "projection_frobenius_norm": tensor_norm(module.projection.weight),
            "gate_parameter_frobenius_norm": math.sqrt(
                sum(tensor_norm(parameter) ** 2 for parameter in module.gate.parameters())
            ),
            "projected_prompt_norms": projected_norms.cpu().tolist(),
            "projected_prompt_pairwise_cosine_offdiagonal": summary(off_diagonal.cpu().tolist()),
        },
        "routing_summary": {
            "mean_weights": matrix.mean(dim=0).tolist(),
            "per_prompt_sample_sd": matrix.std(dim=0, unbiased=True).tolist(),
            "dominant_prompt_counts": dominant_counts,
            "normalized_entropy": summary(normalized_entropy.tolist()),
            "max_prompt_weight": summary(max_weight.tolist()),
            "l1_distance_from_uniform": summary(l1_uniform.tolist()),
            "text_residual_rms": summary(residual_rms),
            "text_residual_relative_rms": summary(relative_residuals),
        },
        "per_image": [
            {
                "index": index,
                "weights": matrix[index].tolist(),
                "normalized_entropy": float(normalized_entropy[index]),
                "max_prompt_weight": float(max_weight[index]),
                "l1_distance_from_uniform": float(l1_uniform[index]),
                "text_residual_rms": residual_rms[index],
                "text_residual_relative_rms": relative_residuals[index],
            }
            for index in range(len(weights))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["routing_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
