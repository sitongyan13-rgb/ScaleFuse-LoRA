#!/usr/bin/env python
"""Evaluate an MMDetection DumpDetResults pickle with COCOeval.

The native MMDetection logger rounds AP to three decimals.  This utility
re-evaluates the immutable prediction dump and records the unrounded COCO
statistics together with hashes of both inputs.
"""

import argparse
import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mmengine.fileio import load
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_rows(path, category_id, top_k):
    samples = load(str(path))
    rows = []
    seen = set()
    for sample in samples:
        image_id = int(sample["img_id"])
        if image_id in seen:
            raise RuntimeError("Duplicate image ID {}".format(image_id))
        seen.add(image_id)
        pred = sample["pred_instances"]
        boxes = np.asarray(pred["bboxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(pred["scores"], dtype=np.float64).reshape(-1)
        labels = np.asarray(pred["labels"], dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(labels)):
            raise RuntimeError("Prediction lengths differ for image {}".format(image_id))
        valid = (
            (labels == 0)
            & np.isfinite(scores)
            & np.isfinite(boxes).all(axis=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        indices = np.flatnonzero(valid)
        indices = indices[np.argsort(-scores[indices], kind="stable")[:top_k]]
        for index in indices:
            x1, y1, x2, y2 = map(float, boxes[index])
            rows.append({
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[index]),
            })
    return samples, seen, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--predictions-pkl", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    annotation = args.annotation.resolve()
    predictions = args.predictions_pkl.resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation))
    category_ids = coco_gt.getCatIds()
    if len(category_ids) != 1:
        raise RuntimeError("Expected one category, found {}".format(len(category_ids)))
    samples, image_ids, rows = prediction_rows(
        predictions, category_ids[0], args.top_k)
    expected_ids = set(coco_gt.getImgIds())
    if image_ids != expected_ids:
        raise RuntimeError(
            "Prediction/annotation image IDs differ: missing={}, extra={}".format(
                sorted(expected_ids - image_ids), sorted(image_ids - expected_ids)))

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(rows)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": "pycocotools.COCOeval",
        "split": args.split,
        "annotation": str(annotation),
        "annotation_sha256": sha256(annotation),
        "predictions_pkl": str(predictions),
        "predictions_sha256": sha256(predictions),
        "image_count": len(samples),
        "detection_count": len(rows),
        "category_ids": category_ids,
        "top_k_per_image": args.top_k,
        "metrics": {
            name: float(value)
            for name, value in zip(METRIC_NAMES, evaluator.stats)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
