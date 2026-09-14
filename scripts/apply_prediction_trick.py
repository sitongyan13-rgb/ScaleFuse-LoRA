#!/usr/bin/env python
"""Apply one frozen prediction-level NMS ensemble without parameter search.

This script intentionally accepts exactly one NMS threshold and score floor. It
is suitable for applying a configuration selected on validation to a held-out
split without accidentally scanning the held-out annotations.
"""

import argparse
import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from pycocotools.coco import COCO

try:
    from scripts.search_prediction_tricks import (
        METRIC_NAMES,
        coco_rows,
        combine_predictions,
        evaluate,
        load_predictions,
    )
except ModuleNotFoundError:  # Direct execution sets scripts/ as sys.path[0].
    from search_prediction_tricks import (
        METRIC_NAMES,
        coco_rows,
        combine_predictions,
        evaluate,
        load_predictions,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="Named dump as NAME=PATH; repeat for each frozen constituent")
    parser.add_argument("--nms-iou", type=float, required=True)
    parser.add_argument("--score-floor", type=float, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-detections", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in [0, 1]")
    if not 0.0 <= args.score_floor <= 1.0:
        raise ValueError("--score-floor must be in [0, 1]")

    named_paths = {}
    for item in args.prediction:
        if "=" not in item:
            raise ValueError("--prediction must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or name in named_paths:
            raise ValueError("Prediction names must be non-empty and unique")
        if not path.is_file():
            raise FileNotFoundError(path)
        named_paths[name] = path
    if not named_paths:
        raise ValueError("At least one prediction dump is required")

    annotation = args.annotation.resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation))
    category_ids = coco_gt.getCatIds()
    if len(category_ids) != 1:
        raise RuntimeError("This utility currently expects one category")
    expected_ids = set(coco_gt.getImgIds())

    sources = {name: load_predictions(path) for name, path in named_paths.items()}
    for name, predictions in sources.items():
        if set(predictions) != expected_ids:
            raise RuntimeError("{} image IDs do not match annotations".format(name))
    names = tuple(named_paths)
    predictions = combine_predictions(
        sources, names, args.nms_iou, args.score_floor)
    rows = coco_rows(predictions, category_ids[0])
    metrics = evaluate(coco_gt, rows)

    args.output_detections.parent.mkdir(parents=True, exist_ok=True)
    args.output_detections.write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply_frozen_parameters_only",
        "split": args.split,
        "annotation": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "prediction_sources": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in named_paths.items()
        },
        "frozen_parameters": {
            "models": list(names),
            "nms_iou": args.nms_iou,
            "score_floor": args.score_floor,
            "max_detections_per_image": 100,
        },
        "metrics": {name: metrics[name] for name in METRIC_NAMES},
        "detections": str(args.output_detections.resolve()),
        "detections_sha256": sha256_file(args.output_detections.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
