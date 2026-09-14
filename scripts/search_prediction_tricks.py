#!/usr/bin/env python
"""Validation-only NMS and prediction-ensemble search for Pothole.

Consumes immutable MMDetection ``DumpDetResults`` files, evaluates every
registered setting with pycocotools, and saves the complete search table plus
the selected COCO detections. It never reads or infers a test split.
"""

import argparse
import contextlib
import csv
import io
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from mmengine.fileio import load
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms as torchvision_nms


NMS_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
SCORE_FLOORS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20)
METRIC_NAMES = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
)


def pairwise_iou(box, boxes):
    """Return IoU between one xyxy box and an array of xyxy boxes."""
    box = np.asarray(box, dtype=np.float64).reshape(4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    intersection = wh[:, 0] * wh[:, 1]
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0.0, None)
    union = area + other - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def greedy_nms(boxes, scores, threshold):
    """Stable greedy NMS returning indices in descending-score order."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have matching lengths")
    if not 0 <= threshold <= 1:
        raise ValueError("NMS threshold must be in [0, 1]")
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[pairwise_iou(boxes[current], boxes[rest]) <= threshold]
    return np.asarray(keep, dtype=np.int64)


def fast_nms(boxes, scores, threshold):
    """C++ NMS used by the exhaustive search; equivalent for unique scores."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have matching lengths")
    if not 0 <= threshold <= 1:
        raise ValueError("NMS threshold must be in [0, 1]")
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    kept = torchvision_nms(
        torch.from_numpy(boxes), torch.from_numpy(scores), float(threshold))
    return kept.cpu().numpy().astype(np.int64, copy=False)


def load_predictions(path):
    samples = load(str(path))
    output = {}
    for sample in samples:
        image_id = int(sample["img_id"])
        if image_id in output:
            raise RuntimeError("Duplicate image ID {} in {}".format(image_id, path))
        pred = sample["pred_instances"]
        boxes = np.asarray(pred["bboxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(pred["scores"], dtype=np.float64).reshape(-1)
        labels = np.asarray(pred["labels"], dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(labels)):
            raise RuntimeError("Prediction tensor lengths differ in {}".format(path))
        valid = ((labels == 0) & np.isfinite(scores)
                 & np.isfinite(boxes).all(axis=1)
                 & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        output[image_id] = {"boxes": boxes[valid], "scores": scores[valid]}
    return output


def combine_predictions(sources, names, threshold, score_floor, top_k=100):
    image_ids = set(sources[names[0]])
    if any(set(sources[name]) != image_ids for name in names[1:]):
        raise RuntimeError("Constituent prediction image IDs differ")
    output = {}
    for image_id in sorted(image_ids):
        boxes = np.concatenate([sources[name][image_id]["boxes"] for name in names])
        scores = np.concatenate([sources[name][image_id]["scores"] for name in names])
        candidate = np.flatnonzero(scores >= score_floor)
        boxes, scores = boxes[candidate], scores[candidate]
        keep = fast_nms(boxes, scores, threshold)[:top_k]
        output[image_id] = {"boxes": boxes[keep], "scores": scores[keep]}
    return output


def coco_rows(predictions, category_id):
    rows = []
    for image_id, pred in predictions.items():
        order = np.argsort(-pred["scores"], kind="stable")[:100]
        for index in order:
            x1, y1, x2, y2 = map(float, pred["boxes"][index])
            rows.append({
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(pred["scores"][index]),
            })
    return rows


def evaluate(coco_gt, rows):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(rows)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="Named dump as NAME=PATH; repeat for every constituent model")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--selected-detections", type=Path, required=True)
    args = parser.parse_args()

    named_paths = {}
    for item in args.prediction:
        if "=" not in item:
            raise ValueError("--prediction must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or name in named_paths:
            raise ValueError("Prediction names must be non-empty and unique")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        named_paths[name] = path
    if len(named_paths) < 2:
        raise ValueError("At least two prediction dumps are required")
    annotation = args.annotation.resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation))
    category_ids = coco_gt.getCatIds()
    if len(category_ids) != 1:
        raise RuntimeError("This search expects exactly one Pothole category")
    expected_ids = set(coco_gt.getImgIds())
    sources = {name: load_predictions(path) for name, path in named_paths.items()}
    for name, predictions in sources.items():
        if set(predictions) != expected_ids:
            raise RuntimeError("{} image IDs do not match annotations".format(name))

    names = tuple(named_paths)
    subsets = [(name,) for name in names]
    for count in range(2, len(names) + 1):
        subsets.extend(itertools.combinations(names, count))
    records = []
    best = None
    best_predictions = None
    total = len(subsets) * len(NMS_THRESHOLDS) * len(SCORE_FLOORS)
    completed = 0
    for subset in subsets:
        for threshold in NMS_THRESHOLDS:
            for score_floor in SCORE_FLOORS:
                predictions = combine_predictions(
                    sources, subset, threshold, score_floor)
                metrics = evaluate(
                    coco_gt, coco_rows(predictions, category_ids[0]))
                row = {
                    "models": "+".join(subset),
                    "model_count": len(subset),
                    "nms_iou": threshold,
                    "score_floor": score_floor,
                    **metrics,
                }
                records.append(row)
                selection_key = (metrics["AP"], metrics["AP75"], metrics["APs"])
                if best is None or selection_key > best[0]:
                    best = (selection_key, row.copy())
                    best_predictions = predictions
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print("completed={}/{} current_best_AP={:.6f}".format(
                        completed, total, best[0][0]), flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    selected_rows = coco_rows(best_predictions, category_ids[0])
    args.selected_detections.parent.mkdir(parents=True, exist_ok=True)
    args.selected_detections.write_text(
        json.dumps(selected_rows, indent=2) + "\n", encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "annotation": str(annotation),
        "prediction_sources": {name: str(path) for name, path in named_paths.items()},
        "search_space": {
            "subsets": len(subsets),
            "nms_thresholds": list(NMS_THRESHOLDS),
            "score_floors": list(SCORE_FLOORS),
            "configurations": len(records),
        },
        "selection_rule": "maximum AP; tie-break AP75 then APs",
        "baseline_exact_AP": 0.540192,
        "selected": best[1],
        "strictly_improves_baseline": best[1]["AP"] > 0.540192,
        "held_out_test_used": False,
        "search_csv": str(args.output_csv.resolve()),
        "selected_detections": str(args.selected_detections.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
