#!/usr/bin/env python
"""Render deterministic COCO ground-truth/prediction comparison images."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw_box(draw, bbox, color, width, text=None):
    x, y, w, h = bbox
    draw.rectangle((x, y, x + w, y + h), outline=color, width=width)
    if text:
        left, top = int(x), max(0, int(y) - 14)
        draw.rectangle((left, top, left + 92, top + 14), fill=color)
        draw.text((left + 2, top), text, fill="white", font=ImageFont.load_default())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in annotation["images"]}
    gt_by_image = {image_id: [] for image_id in images}
    pred_by_image = {image_id: [] for image_id in images}
    for item in annotation["annotations"]:
        if float(item["bbox"][2]) > 0 and float(item["bbox"][3]) > 0:
            gt_by_image[int(item["image_id"])].append(item)
    for item in detections:
        if float(item["score"]) >= args.score_threshold:
            pred_by_image[int(item["image_id"])].append(item)

    # Prefer informative examples, deterministically: most GT objects, then ID.
    selected = sorted(
        images, key=lambda image_id: (-len(gt_by_image[image_id]), image_id)
    )[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for image_id in selected:
        info = images[image_id]
        source = args.image_root / info["file_name"]
        with Image.open(source) as loaded:
            canvas = loaded.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for item in gt_by_image[image_id]:
            draw_box(draw, item["bbox"], "#00A651", 3, "GT pothole")
        predictions = sorted(
            pred_by_image[image_id], key=lambda item: -float(item["score"]))
        for item in predictions:
            draw_box(
                draw, item["bbox"], "#ED1C24", 3,
                "pred {:.2f}".format(float(item["score"])))
        destination = args.output_dir / "{:04d}_{}".format(image_id, source.name)
        canvas.save(destination, quality=95)
        records.append({
            "image_id": image_id,
            "source": str(source.resolve()),
            "output": str(destination.resolve()),
            "ground_truth_boxes": len(gt_by_image[image_id]),
            "predictions_at_threshold": len(predictions),
        })
    payload = {
        "annotation": str(args.annotation.resolve()),
        "detections": str(args.detections.resolve()),
        "score_threshold": args.score_threshold,
        "legend": {"green": "ground truth", "red": "prediction"},
        "selection": "descending valid ground-truth count, then image ID",
        "images": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
