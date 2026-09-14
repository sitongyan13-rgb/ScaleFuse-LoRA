#!/usr/bin/env python
"""Run an official-checkpoint one-batch forward/backward audit."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "third_party" / "mmdetection"):
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint

from mmdet.registry import DATASETS, MODELS
from torch.utils.data import DataLoader

import research.mmdet_plugins  # noqa: F401


def collate_one(batch):
    return {"inputs": [batch[0]["inputs"]], "data_samples": [batch[0]["data_samples"]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/proposed/odinw_pothole_domain_prompt_fusion_lora_r16_smoke.py",
    )
    parser.add_argument(
        "--output",
        default="experiments/raw_metrics/domain_prompt_fusion_one_batch.json",
    )
    args = parser.parse_args()

    init_default_scope("mmdet")
    cfg = Config.fromfile(str(ROOT / args.config))
    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint = _load_checkpoint(str(ROOT / cfg.load_from), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    invalid_missing = [
        key
        for key in missing_keys
        if "lora_down" not in key
        and "lora_up" not in key
        and "domain_prompt_fusion" not in key
        and "scale_aware_fusion" not in key
        and not key.startswith("bbox_head.cls_branches.")
        and key != "dn_query_generator.label_embedding.weight"
    ]
    allowed_unexpected_suffixes = (
        "pooler.dense.weight",
        "pooler.dense.bias",
        "embeddings.position_ids",
    )
    invalid_unexpected = [
        key
        for key in unexpected_keys
        if not key.endswith(allowed_unexpected_suffixes)
    ]
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(
            "Unexpected checkpoint mismatch: missing={} unexpected={}".format(
                invalid_missing, invalid_unexpected
            )
        )
    trainable_hook = next(
        hook
        for hook in cfg.custom_hooks
        if hook.get("type") == "SetTrainableModulesHook"
    )
    trainable_prefixes = tuple(trainable_hook.get("train_only_prefixes", ()))
    trainable_substrings = tuple(trainable_hook.get("train_only_substrings", ()))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(trainable_prefixes) or any(
            item in name for item in trainable_substrings
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataset_cfg = dict(cfg.train_dataloader.dataset)
    dataset = DATASETS.build(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_one,
    )
    raw_batch = next(iter(loader))
    batch = model.data_preprocessor(raw_batch, training=True)
    losses = model.loss(batch["inputs"], batch["data_samples"])
    loss, log_vars = model.parse_losses(losses)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite one-batch loss")
    loss.backward()

    gradients = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        gradients.append(
            {
                "name": name,
                "numel": parameter.numel(),
                "gradient_present": gradient is not None,
                "gradient_nonzero": (
                    int(torch.count_nonzero(gradient)) if gradient is not None else 0
                ),
                "gradient_finite": (
                    bool(torch.isfinite(gradient).all())
                    if gradient is not None
                    else False
                ),
            }
        )
    if not gradients or not all(row["gradient_present"] for row in gradients):
        missing = [row["name"] for row in gradients if not row["gradient_present"]]
        raise RuntimeError("Missing trainable gradients: {}".format(missing))
    if not all(row["gradient_finite"] for row in gradients):
        raise RuntimeError("Non-finite trainable gradient")

    prompt_gradients = [
        row for row in gradients if "domain_prompt_fusion" in row["name"]
    ]
    scale_gradients = [
        row for row in gradients if "scale_aware_fusion" in row["name"]
    ]
    lora_gradients = [
        row
        for row in gradients
        if "lora_down" in row["name"] or "lora_up" in row["name"]
    ]
    result = {
        "created_local": datetime.now().astimezone().isoformat(),
        "config": args.config,
        "checkpoint": cfg.load_from,
        "device": str(device),
        "dataset_length": len(dataset),
        "sample_img_id": batch["data_samples"][0].img_id,
        "loss": float(loss.detach()),
        "log_vars": {key: float(value) for key, value in log_vars.items()},
        "trainable_parameter_tensors": len(gradients),
        "trainable_parameters": sum(row["numel"] for row in gradients),
        "gradient_nonzero_tensors": sum(row["gradient_nonzero"] > 0 for row in gradients),
        "gradient_nonfinite_tensors": sum(not row["gradient_finite"] for row in gradients),
        "prompt_gradient_tensors": len(prompt_gradients),
        "prompt_gradient_nonzero_tensors": sum(
            row["gradient_nonzero"] > 0 for row in prompt_gradients
        ),
        "lora_gradient_tensors": len(lora_gradients),
        "lora_gradient_nonzero_tensors": sum(
            row["gradient_nonzero"] > 0 for row in lora_gradients
        ),
        "scale_gradient_tensors": len(scale_gradients),
        "scale_gradient_nonzero_tensors": sum(
            row["gradient_nonzero"] > 0 for row in scale_gradients
        ),
        "checkpoint_missing_keys": missing_keys,
        "checkpoint_unexpected_keys": unexpected_keys,
        "gradients": gradients,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        ),
        "peak_cuda_reserved_gib": (
            torch.cuda.max_memory_reserved() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("output={}".format(output))
    print(
        "loss={:.6f} trainable={} nonzero={}/{} prompt_nonzero={}/{} lora_nonzero={}/{}".format(
            result["loss"],
            result["trainable_parameters"],
            result["gradient_nonzero_tensors"],
            result["trainable_parameter_tensors"],
            result["prompt_gradient_nonzero_tensors"],
            result["prompt_gradient_tensors"],
            result["lora_gradient_nonzero_tensors"],
            result["lora_gradient_tensors"],
        )
    )


if __name__ == "__main__":
    main()
