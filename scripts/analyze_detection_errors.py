#!/usr/bin/env python
"""Validation-only, size-stratified detection error analysis.

The script consumes MMDetection ``DumpDetResults`` files and immutable COCO
annotations.  It never changes model selection or evaluates a held-out split.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mmengine.fileio import load
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[1]
SIZE_NAMES = ("small", "medium", "large")


def size_name(area):
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def pairwise_iou(boxes1, boxes2):
    boxes1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    left_top = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(right_bottom - left_top, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], 0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], 0, None)
    union = area1[:, None] + area2[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def xywh_to_xyxy(box):
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def load_ground_truth(annotation):
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    by_image = defaultdict(list)
    for ann in payload["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        box = xywh_to_xyxy(ann["bbox"])
        area = float(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
        if area <= 0 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        by_image[int(ann["image_id"])].append(
            {"id": int(ann["id"]), "box": box, "area": area,
             "size": size_name(area)})
    return images, by_image


def load_predictions(path):
    samples = load(str(path))
    by_image = {}
    for sample in samples:
        image_id = int(sample["img_id"])
        pred = sample["pred_instances"]
        boxes = np.asarray(pred["bboxes"], dtype=np.float64)
        scores = np.asarray(pred["scores"], dtype=np.float64)
        labels = np.asarray(pred["labels"], dtype=np.int64)
        order = np.argsort(-scores, kind="stable")
        by_image[image_id] = {"boxes": boxes[order], "scores": scores[order],
                              "labels": labels[order]}
    return by_image


def coco_detections(predictions, category_id):
    rows = []
    for image_id, pred in predictions.items():
        for box, score, label in zip(pred["boxes"][:100], pred["scores"][:100], pred["labels"][:100]):
            if int(label) != 0:
                continue
            x1, y1, x2, y2 = map(float, box)
            rows.append({"image_id": image_id, "category_id": category_id,
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": float(score)})
    return rows


def coco_stats(annotation, predictions):
    coco_gt = COCO(str(annotation))
    cat_ids = coco_gt.getCatIds()
    if len(cat_ids) != 1:
        raise RuntimeError("This analysis expects the one-class Pothole task")
    detections = coco_detections(predictions, cat_ids[0])
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    names = ("AP", "AP50", "AP75", "APs", "APm", "APl",
             "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


def gt_best_iou(predictions, ground_truth, top_k=100):
    rows = []
    for image_id, annotations in ground_truth.items():
        pred_boxes = predictions[image_id]["boxes"][:top_k]
        gt_boxes = [ann["box"] for ann in annotations]
        overlaps = pairwise_iou(gt_boxes, pred_boxes)
        best = overlaps.max(axis=1) if overlaps.shape[1] else np.zeros(len(gt_boxes))
        for ann, value in zip(annotations, best):
            rows.append({"image_id": image_id, "annotation_id": ann["id"],
                         "size": ann["size"], "area": ann["area"],
                         "best_iou": float(value)})
    return rows


def ranked_error_counts(predictions, ground_truth, score_threshold=0.25, iou_threshold=0.5):
    counts = Counter()
    per_image = {}
    for image_id, pred in predictions.items():
        keep = np.flatnonzero(pred["scores"] >= score_threshold)[:100]
        boxes = pred["boxes"][keep]
        annotations = ground_truth.get(image_id, [])
        gt_boxes = [ann["box"] for ann in annotations]
        overlaps = pairwise_iou(boxes, gt_boxes)
        matched = set()
        image_counts = Counter()
        for pred_index in range(len(boxes)):
            if not len(gt_boxes):
                image_counts["background"] += 1
                continue
            best_gt = int(np.argmax(overlaps[pred_index]))
            best_iou = float(overlaps[pred_index, best_gt])
            if best_iou >= iou_threshold and best_gt not in matched:
                matched.add(best_gt); image_counts["true_positive"] += 1
            elif best_iou >= iou_threshold:
                image_counts["duplicate"] += 1
            elif best_iou >= 0.1:
                image_counts["localization"] += 1
            else:
                image_counts["background"] += 1
        image_counts["false_negative"] = len(gt_boxes) - len(matched)
        image_counts["predictions"] = len(boxes)
        counts.update(image_counts)
        per_image[image_id] = dict(image_counts)
    return dict(counts), per_image


def _average_ranks(values):
    """Return one-based average ranks, including exact-value tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def score_iou_alignment(predictions, ground_truth, top_k=100):
    """Measure whether prediction confidence ranks boxes by localization quality."""
    scores, best_ious = [], []
    for image_id, pred in predictions.items():
        keep = np.flatnonzero(pred["labels"] == 0)[:top_k]
        boxes = pred["boxes"][keep]
        image_scores = pred["scores"][keep]
        gt_boxes = [ann["box"] for ann in ground_truth.get(image_id, [])]
        overlaps = pairwise_iou(boxes, gt_boxes)
        image_ious = (overlaps.max(axis=1) if overlaps.shape[1]
                      else np.zeros(len(boxes), dtype=np.float64))
        scores.extend(map(float, image_scores))
        best_ious.extend(map(float, image_ious))
    scores = np.asarray(scores, dtype=np.float64)
    best_ious = np.asarray(best_ious, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    top_count = min(100, len(order))
    return {
        "predictions": int(len(scores)),
        "pearson_score_vs_best_iou": _pearson(scores, best_ious),
        "spearman_score_vs_best_iou": _pearson(
            _average_ranks(scores), _average_ranks(best_ious)),
        "mean_best_iou_all": float(best_ious.mean()) if len(best_ious) else None,
        "mean_best_iou_top_100_scores": (
            float(best_ious[order[:top_count]].mean()) if top_count else None),
        "fraction_iou_0.50_top_100_scores": (
            float(np.mean(best_ious[order[:top_count]] >= 0.5)) if top_count else None),
    }


def summarize_gt(rows):
    output = {}
    for group in ("all",) + SIZE_NAMES:
        values = np.asarray([row["best_iou"] for row in rows
                             if group == "all" or row["size"] == group])
        output[group] = {
            "ground_truth": int(len(values)),
            "mean_best_iou": float(values.mean()) if len(values) else None,
            "recall_iou_0.50": float(np.mean(values >= 0.5)) if len(values) else None,
            "recall_iou_0.75": float(np.mean(values >= 0.75)) if len(values) else None,
            "miss_iou_below_0.10": int(np.sum(values < 0.1)),
            "localization_iou_0.10_to_0.50": int(np.sum((values >= 0.1) & (values < 0.5))),
        }
    return output


def paired_summary(baseline_rows, proposed_rows):
    baseline = {(row["image_id"], row["annotation_id"]): row for row in baseline_rows}
    proposed = {(row["image_id"], row["annotation_id"]): row for row in proposed_rows}
    if baseline.keys() != proposed.keys():
        raise RuntimeError("Ground-truth keys differ between prediction dumps")
    rows, output = [], {}
    for key in sorted(baseline):
        before, after = baseline[key], proposed[key]
        delta = after["best_iou"] - before["best_iou"]
        rows.append({**key_to_dict(key), "size": before["size"],
                     "baseline_best_iou": before["best_iou"],
                     "proposed_best_iou": after["best_iou"], "delta": delta})
    for group in ("all",) + SIZE_NAMES:
        values = np.asarray([row["delta"] for row in rows
                             if group == "all" or row["size"] == group])
        output[group] = {"count": int(len(values)), "mean_delta": float(values.mean()),
                         "median_delta": float(np.median(values)),
                         "improved_gt": int(np.sum(values > 1e-6)),
                         "worsened_gt": int(np.sum(values < -1e-6)),
                         "unchanged_gt": int(np.sum(np.abs(values) <= 1e-6))}
    return rows, output


def key_to_dict(key):
    return {"image_id": key[0], "annotation_id": key[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--proposed-label", default="Proposed")
    args = parser.parse_args()
    annotation, baseline_path, proposed_path = (
        args.annotation.resolve(), args.baseline.resolve(), args.proposed.resolve())
    for path in (annotation, baseline_path, proposed_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    images, ground_truth = load_ground_truth(annotation)
    baseline, proposed = load_predictions(baseline_path), load_predictions(proposed_path)
    expected_ids = set(images)
    if set(baseline) != expected_ids or set(proposed) != expected_ids:
        raise RuntimeError("Prediction image IDs do not exactly match annotations")
    baseline_rows = gt_best_iou(baseline, ground_truth)
    proposed_rows = gt_best_iou(proposed, ground_truth)
    paired_rows, paired = paired_summary(baseline_rows, proposed_rows)
    baseline_errors, baseline_per_image = ranked_error_counts(
        baseline, ground_truth, args.score_threshold)
    proposed_errors, proposed_per_image = ranked_error_counts(
        proposed, ground_truth, args.score_threshold)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"dataset": "ODinW Pothole", "split": "validation",
                     "images": len(images), "score_threshold": args.score_threshold,
                     "max_detections": 100, "held_out_test_evaluated": False},
        "baseline": {"predictions": str(baseline_path),
                     "coco": coco_stats(annotation, baseline),
                     "gt_best_iou": summarize_gt(baseline_rows),
                     "score_iou_alignment": score_iou_alignment(
                         baseline, ground_truth),
                     "fixed_threshold_errors": baseline_errors},
        "proposed": {"predictions": str(proposed_path),
                     "coco": coco_stats(annotation, proposed),
                     "gt_best_iou": summarize_gt(proposed_rows),
                     "score_iou_alignment": score_iou_alignment(
                         proposed, ground_truth),
                     "fixed_threshold_errors": proposed_errors},
        "paired_gt_best_iou": paired,
        "per_image_fixed_threshold": {
            "baseline": baseline_per_image, "proposed": proposed_per_image},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader(); writer.writerows(paired_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.25))
    for label, rows, color in ((args.baseline_label, baseline_rows, "#009E73"),
                               (args.proposed_label, proposed_rows, "#0072B2")):
        values = np.sort([row["best_iou"] for row in rows])
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values),
                     label=label, color=color, linewidth=1.8)
    axes[0].set(xlabel="Best IoU per ground-truth object (top 100)",
                ylabel="Empirical cumulative fraction", xlim=(0, 1))
    axes[0].grid(alpha=0.22); axes[0].legend(fontsize=8)
    categories = ("true_positive", "duplicate", "localization", "background", "false_negative")
    x = np.arange(len(categories)); width = 0.36
    axes[1].bar(x - width / 2, [baseline_errors.get(key, 0) for key in categories],
                width, label=args.baseline_label, color="#009E73")
    axes[1].bar(x + width / 2, [proposed_errors.get(key, 0) for key in categories],
                width, label=args.proposed_label, color="#0072B2")
    axes[1].set_xticks(x, ["TP", "Duplicate", "Localization", "Background", "FN"], rotation=20)
    axes[1].set_ylabel("Count at score >= {:.2f}".format(args.score_threshold))
    axes[1].grid(axis="y", alpha=0.22); axes[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(args.figure_prefix) + "." + suffix,
                    dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"baseline_coco": payload["baseline"]["coco"],
                      "proposed_coco": payload["proposed"]["coco"],
                      "paired": paired,
                      "baseline_errors": baseline_errors,
                      "proposed_errors": proposed_errors}, indent=2))


if __name__ == "__main__":
    main()
