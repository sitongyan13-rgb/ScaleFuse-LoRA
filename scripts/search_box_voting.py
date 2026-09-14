#!/usr/bin/env python
"""Validation-only coordinate box-voting search for two prediction dumps."""

import argparse
import contextlib
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

try:
    from scripts.search_prediction_tricks import (
        METRIC_NAMES,
        coco_rows,
        evaluate,
        fast_nms,
        load_predictions,
        pairwise_iou,
    )
except ModuleNotFoundError:  # Direct execution sets scripts/ as sys.path[0].
    from search_prediction_tricks import (
        METRIC_NAMES,
        coco_rows,
        evaluate,
        fast_nms,
        load_predictions,
        pairwise_iou,
    )


NMS_THRESHOLDS = (0.60, 0.70, 0.80)
VOTE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
SCORE_POWERS = (1.0, 2.0)


def box_vote_predictions(
        sources, names, nms_threshold, vote_threshold, score_power,
        top_k=100):
    """NMS seeds followed by score-weighted coordinate voting."""
    if not names:
        raise ValueError("At least one prediction source is required")
    if score_power <= 0:
        raise ValueError("score_power must be positive")
    image_ids = set(sources[names[0]])
    if any(set(sources[name]) != image_ids for name in names[1:]):
        raise RuntimeError("Constituent prediction image IDs differ")
    output = {}
    for image_id in sorted(image_ids):
        boxes = np.concatenate([sources[name][image_id]["boxes"] for name in names])
        scores = np.concatenate([sources[name][image_id]["scores"] for name in names])
        seeds = fast_nms(boxes, scores, nms_threshold)[:top_k]
        voted_boxes = []
        for seed in seeds:
            neighbors = np.flatnonzero(
                pairwise_iou(boxes[seed], boxes) >= vote_threshold)
            weights = np.maximum(scores[neighbors], 1e-12) ** score_power
            voted_boxes.append(np.average(boxes[neighbors], axis=0, weights=weights))
        output[image_id] = {
            "boxes": np.asarray(voted_boxes, dtype=np.float64).reshape(-1, 4),
            # Preserve NMS-seed confidence; voting changes localization only.
            "scores": scores[seeds].astype(np.float64, copy=True),
        }
    return output


def parse_predictions(items):
    named_paths = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--prediction must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or name in named_paths:
            raise ValueError("Prediction names must be non-empty and unique")
        if not path.is_file():
            raise FileNotFoundError(path)
        named_paths[name] = path
    if len(named_paths) != 2:
        raise ValueError("Box-voting search is pre-registered for exactly two sources")
    return named_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--selected-detections", type=Path, required=True)
    args = parser.parse_args()

    named_paths = parse_predictions(args.prediction)
    annotation = args.annotation.resolve()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation))
    category_ids = coco_gt.getCatIds()
    if len(category_ids) != 1:
        raise RuntimeError("This search expects exactly one category")
    sources = {name: load_predictions(path) for name, path in named_paths.items()}
    expected_ids = set(coco_gt.getImgIds())
    if any(set(predictions) != expected_ids for predictions in sources.values()):
        raise RuntimeError("Prediction image IDs do not match annotations")

    names = tuple(named_paths)
    records = []
    best = None
    best_predictions = None
    for nms_iou in NMS_THRESHOLDS:
        for vote_iou in VOTE_THRESHOLDS:
            for power in SCORE_POWERS:
                predictions = box_vote_predictions(
                    sources, names, nms_iou, vote_iou, power)
                metrics = evaluate(
                    coco_gt, coco_rows(predictions, category_ids[0]))
                row = {
                    "models": "+".join(names),
                    "nms_iou": nms_iou,
                    "vote_iou": vote_iou,
                    "score_power": power,
                    **metrics,
                }
                records.append(row)
                key = (metrics["AP"], metrics["AP75"], metrics["APs"])
                if best is None or key > best[0]:
                    best = (key, row.copy())
                    best_predictions = predictions

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
            "nms_thresholds": list(NMS_THRESHOLDS),
            "vote_thresholds": list(VOTE_THRESHOLDS),
            "score_powers": list(SCORE_POWERS),
            "configurations": len(records),
        },
        "selection_rule": "maximum AP; tie-break AP75 then APs",
        "baseline_exact_AP": 0.54019153629905,
        "selected": best[1],
        "strictly_improves_baseline": best[1]["AP"] > 0.54019153629905,
        "held_out_test_used": False,
        "search_csv": str(args.output_csv.resolve()),
        "selected_detections": str(args.selected_detections.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
