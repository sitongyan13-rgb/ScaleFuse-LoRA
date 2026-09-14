#!/usr/bin/env python
"""Audit the object-aware crop on every Pothole training image once."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "third_party" / "mmdetection-3.3.0"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS

import research.mmdet_plugins  # noqa: F401,E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/proposed/odinw_pothole_object_aware_zoom_scale_lora_r16.py",
    )
    parser.add_argument(
        "--output",
        default="experiments/raw_metrics/object_aware_zoom_dataset_audit.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    init_default_scope("mmdet")
    config = Config.fromfile(str((ROOT / args.config).resolve()))
    dataset = DATASETS.build(dict(config.train_dataloader.dataset))
    applied = []
    retained_boxes = 0
    total_output_boxes = 0
    empty_samples = 0
    invalid_boxes = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        data_sample = sample["data_samples"]
        meta = data_sample.metainfo
        boxes = data_sample.gt_instances.bboxes
        total_output_boxes += len(boxes)
        if len(boxes) == 0:
            empty_samples += 1
        image_height, image_width = meta["img_shape"]
        if len(boxes):
            box_tensor = boxes.tensor if hasattr(boxes, "tensor") else boxes
            invalid_boxes += int((
                (box_tensor[:, 0] < 0) | (box_tensor[:, 1] < 0)
                | (box_tensor[:, 2] > image_width)
                | (box_tensor[:, 3] > image_height)
                | (box_tensor[:, 2] <= box_tensor[:, 0])
                | (box_tensor[:, 3] <= box_tensor[:, 1])
            ).sum())
        if meta.get("zoom_crop_applied", False):
            original_fraction = float(meta["zoom_crop_target_original_area_fraction"])
            cropped_fraction = float(meta["zoom_crop_target_cropped_area_fraction"])
            retained = int(meta["zoom_crop_retained_boxes"])
            retained_boxes += retained
            applied.append({
                "dataset_index": index,
                "image_id": int(meta["img_id"]),
                "crop_scale": float(meta["zoom_crop_scale"]),
                "retained_boxes": retained,
                "target_original_area": float(meta["zoom_crop_target_original_area"]),
                "target_original_area_fraction": original_fraction,
                "target_cropped_area_fraction": cropped_fraction,
                "target_relative_area_magnification": cropped_fraction / original_fraction,
            })

    if empty_samples or invalid_boxes:
        raise RuntimeError(
            "Invalid transformed data: empty={} invalid_boxes={}".format(
                empty_samples, invalid_boxes))
    if not applied or any(row["retained_boxes"] < 1 for row in applied):
        raise RuntimeError("Crop did not retain a target on every applied sample")
    magnifications = [row["target_relative_area_magnification"] for row in applied]
    crop_scales = [row["crop_scale"] for row in applied]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "seed": args.seed,
        "training_images": len(dataset),
        "crop_applied_images": len(applied),
        "crop_applied_fraction": len(applied) / len(dataset),
        "output_boxes": total_output_boxes,
        "empty_samples": empty_samples,
        "invalid_boxes": invalid_boxes,
        "selected_small_targets": sum(
            row["target_original_area"] < 32 ** 2 for row in applied),
        "mean_retained_boxes_when_cropped": retained_boxes / len(applied),
        "crop_scale_minimum": min(crop_scales),
        "crop_scale_mean": sum(crop_scales) / len(crop_scales),
        "crop_scale_maximum": max(crop_scales),
        "target_relative_area_magnification_minimum": min(magnifications),
        "target_relative_area_magnification_mean": sum(magnifications) / len(magnifications),
        "target_relative_area_magnification_maximum": max(magnifications),
        "held_out_test_used": False,
        "applied_samples": applied,
    }
    if payload["target_relative_area_magnification_minimum"] < 1.0:
        raise RuntimeError("A selected target was not magnified")
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "applied_samples"}, indent=2))


if __name__ == "__main__":
    main()
