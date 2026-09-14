#!/usr/bin/env python
"""Independently evaluate a COCO detection JSON and save all 12 metrics."""

import argparse
import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotation = args.annotation.resolve()
    detections = args.detections.resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not detections.is_file():
        raise FileNotFoundError(detections)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation))
        coco_dt = coco_gt.loadRes(str(detections))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": "pycocotools.COCOeval",
        "split": args.split,
        "annotation": str(annotation),
        "detections": str(detections),
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
