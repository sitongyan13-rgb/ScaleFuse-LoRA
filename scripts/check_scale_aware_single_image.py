#!/usr/bin/env python
"""Run one real validation image through the scale-aware MM-GDINO model."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "third_party" / "mmdetection-3.3.0"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint
from torch.utils.data import DataLoader

from mmdet.registry import DATASETS, MODELS

import research.mmdet_plugins  # noqa: F401


def collate_one(batch):
    return {"inputs": [batch[0]["inputs"]], "data_samples": [batch[0]["data_samples"]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    init_default_scope("mmdet")
    cfg = Config.fromfile(str(ROOT / args.config))
    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint = _load_checkpoint(str(ROOT / cfg.load_from), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if "lora_down" not in key and "lora_up" not in key
        and "scale_aware_fusion" not in key
        and not key.startswith("bbox_head.cls_branches.")
        and key != "dn_query_generator.label_embedding.weight"
    ]
    if invalid_missing:
        raise RuntimeError("Unexpected missing keys: {}".format(invalid_missing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = DATASETS.build(dict(cfg.val_dataloader.dataset))
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_one)
    raw_batch = next(iter(loader))
    batch = model.data_preprocessor(raw_batch, training=False)
    with torch.no_grad():
        prediction = model.predict(batch["inputs"], batch["data_samples"], rescale=True)[0]
    instances = prediction.pred_instances.to("cpu")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "checkpoint": cfg.load_from,
        "split": "validation",
        "img_id": prediction.img_id,
        "img_path": prediction.img_path,
        "text_prompt": list(cfg.metainfo["classes"]),
        "prediction_count": len(instances),
        "top_scores": instances.scores[:10].tolist(),
        "top_labels": instances.labels[:10].tolist(),
        "top_boxes_xyxy": instances.bboxes[:10].tolist(),
        "finite": all(
            bool(torch.isfinite(value).all())
            for value in (instances.scores, instances.bboxes)
        ),
        "device": str(device),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
    }
    if not result["finite"]:
        raise RuntimeError("Non-finite single-image prediction")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
