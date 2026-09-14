#!/usr/bin/env python
"""Validation-only search for small-box score calibration.

This is a post-hoc educational diagnostic over immutable prediction dumps.  It
does not load, inspect, or evaluate the held-out test split.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mmengine.fileio import load
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 21, 42)
AREA_THRESHOLDS = (256, 576, 1024, 1600, 2304, 4096)
LOGIT_BIASES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
METRIC_NAMES = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
)
EXPECTED_BASELINE = {
    0: {"AP": 0.5448280364296564, "APs": 0.39414922458027163},
    21: {"AP": 0.5463152317241815, "APs": 0.38113956464641296},
    42: {"AP": 0.54019153629905, "APs": 0.38006683251187623},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate_scores(boxes, scores, area_threshold, logit_bias):
    """Add a fixed logit bias to predictions at or below an area threshold."""
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if scores.ndim != 1 or len(scores) != len(boxes):
        raise ValueError("scores must have shape (N,) and match boxes")
    if not math.isfinite(float(area_threshold)) or area_threshold <= 0:
        raise ValueError("area_threshold must be finite and positive")
    if not math.isfinite(float(logit_bias)) or logit_bias < 0:
        raise ValueError("logit_bias must be finite and non-negative")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    small = widths * heights <= float(area_threshold)
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    calibrated = clipped.copy()
    if logit_bias > 0 and small.any():
        logits = np.log(clipped[small] / (1.0 - clipped[small]))
        calibrated[small] = 1.0 / (1.0 + np.exp(-(logits + logit_bias)))
    return calibrated


def rank_top_k(scores, valid, top_k):
    """Return stable descending-score indices among valid predictions."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if len(scores) != len(valid):
        raise ValueError("scores and valid must have the same length")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    indices = np.flatnonzero(valid)
    return indices[np.argsort(-scores[indices], kind="stable")[:top_k]]


def load_prediction_samples(path, expected_image_ids):
    raw = load(str(path))
    samples = []
    seen = set()
    for sample in raw:
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
        samples.append((image_id, boxes, scores, labels))
    if seen != expected_image_ids:
        raise RuntimeError(
            "Prediction/annotation IDs differ: missing={}, extra={}".format(
                sorted(expected_image_ids - seen), sorted(seen - expected_image_ids)))
    return samples


def prediction_rows(samples, category_id, area_threshold, logit_bias, top_k):
    rows = []
    for image_id, boxes, scores, labels in samples:
        finite = np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
        valid = (
            (labels == 0) & finite
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        safe_scores = scores.copy()
        safe_scores[~np.isfinite(safe_scores)] = 0.0
        calibrated = calibrate_scores(boxes, safe_scores, area_threshold, logit_bias)
        indices = rank_top_k(calibrated, valid, top_k)
        for index in indices:
            x1, y1, x2, y2 = map(float, boxes[index])
            rows.append({
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(calibrated[index]),
            })
    return rows


def evaluate(coco_gt, rows):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(rows)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        name: float(value)
        for name, value in zip(METRIC_NAMES, evaluator.stats)
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(run_rows, baseline_by_seed):
    aggregate = []
    for threshold in AREA_THRESHOLDS:
        for bias in LOGIT_BIASES:
            selected = [
                row for row in run_rows
                if row["area_threshold"] == threshold and row["logit_bias"] == bias
            ]
            row = {
                "area_threshold": threshold,
                "logit_bias": bias,
                "n_seeds": len(selected),
            }
            for metric in ("AP", "AP50", "AP75", "APs", "APm", "APl"):
                values = [item[metric] for item in selected]
                deltas = [
                    item[metric] - baseline_by_seed[item["seed"]][metric]
                    for item in selected
                ]
                row["mean_{}".format(metric)] = statistics.mean(values)
                row["sample_sd_{}".format(metric)] = statistics.stdev(values)
                row["mean_delta_{}".format(metric)] = statistics.mean(deltas)
                row["min_delta_{}".format(metric)] = min(deltas)
            row["seeds_APs_non_decreasing"] = sum(
                item["APs"] >= baseline_by_seed[item["seed"]]["APs"] - 1e-12
                for item in selected
            )
            row["gate_mean_APs_gain"] = row["mean_delta_APs"] >= 0.003 - 1e-12
            row["gate_mean_AP_loss"] = row["mean_delta_AP"] >= -0.001 - 1e-12
            row["gate_APs_seed_count"] = row["seeds_APs_non_decreasing"] >= 2
            row["gate_worst_seed_AP"] = row["min_delta_AP"] >= -0.003 - 1e-12
            row["gate_pass"] = all((
                row["gate_mean_APs_gain"], row["gate_mean_AP_loss"],
                row["gate_APs_seed_count"], row["gate_worst_seed_AP"],
            ))
            aggregate.append(row)
    return aggregate


def selection_key(row):
    return (
        row["mean_APs"], row["mean_AP"], -row["logit_bias"],
        -abs(row["area_threshold"] - 1024),
    )


def select_configuration(aggregate):
    passing = [row for row in aggregate if row["gate_pass"]]
    if passing:
        return max(passing, key=selection_key), True
    constrained = [row for row in aggregate if row["mean_delta_AP"] >= -0.001 - 1e-12]
    if not constrained:
        constrained = aggregate
    return max(constrained, key=selection_key), False


def plot_results(aggregate, selected):
    figure_dir = ROOT / "results/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    heat = np.empty((len(LOGIT_BIASES), len(AREA_THRESHOLDS)))
    lookup = {(row["area_threshold"], row["logit_bias"]): row for row in aggregate}
    for y, bias in enumerate(LOGIT_BIASES):
        for x, threshold in enumerate(AREA_THRESHOLDS):
            heat[y, x] = lookup[(threshold, bias)]["mean_delta_APs"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    bound = max(abs(float(np.nanmin(heat))), abs(float(np.nanmax(heat))), 1e-4)
    image = axes[0].imshow(heat, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    axes[0].set_xticks(range(len(AREA_THRESHOLDS)), AREA_THRESHOLDS, rotation=35)
    axes[0].set_yticks(range(len(LOGIT_BIASES)), LOGIT_BIASES)
    axes[0].set_xlabel("Predicted-box area threshold (px^2)")
    axes[0].set_ylabel("Logit bias")
    axes[0].set_title("Mean validation APs change")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    x_selected = AREA_THRESHOLDS.index(selected["area_threshold"])
    y_selected = LOGIT_BIASES.index(selected["logit_bias"])
    axes[0].scatter(x_selected, y_selected, marker="s", facecolors="none",
                    edgecolors="black", linewidths=1.7, s=135)

    passing = np.asarray([row["gate_pass"] for row in aggregate])
    ap_change = np.asarray([row["mean_delta_AP"] for row in aggregate])
    aps_change = np.asarray([row["mean_delta_APs"] for row in aggregate])
    axes[1].scatter(ap_change[~passing], aps_change[~passing], color="#7A8A99",
                    alpha=0.75, s=34, label="Did not pass")
    if passing.any():
        axes[1].scatter(ap_change[passing], aps_change[passing], color="#009E73",
                        alpha=0.9, s=42, label="Passed gate")
    axes[1].scatter(selected["mean_delta_AP"], selected["mean_delta_APs"],
                    marker="*", color="#D55E00", edgecolor="black", s=180,
                    label="Selected diagnostic", zorder=4)
    axes[1].axvline(-0.001, color="black", linestyle="--", linewidth=0.9)
    axes[1].axhline(0.003, color="black", linestyle="--", linewidth=0.9)
    axes[1].axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
    axes[1].axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    axes[1].set_xlabel("Mean validation AP change")
    axes[1].set_ylabel("Mean validation APs change")
    axes[1].set_title("Trade-off across 42 settings")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Validation-only small-object score calibration")
    fig.tight_layout()
    stem = figure_dir / "small_object_score_calibration"
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(str(stem) + "." + suffix, bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(selected, passed, selected_runs, aggregate_path, run_path, json_path):
    status = "PASSED" if passed else "DID NOT PASS"
    lines = [
        "# Small-object score calibration (validation-only diagnostic)",
        "",
        "## Outcome",
        "",
        "The pre-registered gate **{}**. The selected diagnostic setting uses predicted-box area threshold `{} px^2` and logit bias `{}`. This is an exploratory post-test educational diagnostic, not a clean confirmatory result.".format(
            status, selected["area_threshold"], selected["logit_bias"]),
        "",
        "| Metric | Mean value | Mean change vs uncalibrated |",
        "|---|---:|---:|",
        "| AP | {:.6f} | {:+.6f} |".format(selected["mean_AP"], selected["mean_delta_AP"]),
        "| APs | {:.6f} | {:+.6f} |".format(selected["mean_APs"], selected["mean_delta_APs"]),
        "| APm | {:.6f} | {:+.6f} |".format(selected["mean_APm"], selected["mean_delta_APm"]),
        "| APl | {:.6f} | {:+.6f} |".format(selected["mean_APl"], selected["mean_delta_APl"]),
        "",
        "Per-seed changes:",
        "",
        "| Seed | AP | AP change | APs | APs change |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selected_runs, key=lambda item: item["seed"]):
        lines.append("| {} | {:.6f} | {:+.6f} | {:.6f} | {:+.6f} |".format(
            row["seed"], row["AP"], row["delta_AP"], row["APs"], row["delta_APs"]))
    lines.extend([
        "",
        "## Integrity boundary",
        "",
        "Only the three immutable **validation** prediction dumps were loaded. The existing held-out test annotations, prediction dumps, and metrics were not read or evaluated during this search. Because the test result had already been inspected before this diagnostic was designed, this candidate must not be presented as independently test-confirmed.",
        "",
        "The search evaluated all 42 pre-registered parameter settings on seeds 0, 21, and 42 (126 COCOeval runs). Every `b=0` setting reproduced the previously recorded exact AP and APs values. Boxes, labels, image IDs, and suppression output were unchanged; only scores and their top-100 ranking were recalibrated.",
        "",
        "## Gate",
        "",
        "A setting passes only when mean APs improves by at least +0.003, mean AP drops by no more than 0.001, APs is non-decreasing for at least two of three seeds, and no seed loses more than 0.003 AP.",
        "",
        "## Artifacts",
        "",
        "- `{}`".format(run_path.relative_to(ROOT).as_posix()),
        "- `{}`".format(aggregate_path.relative_to(ROOT).as_posix()),
        "- `{}`".format(json_path.relative_to(ROOT).as_posix()),
        "- `results/figures/small_object_score_calibration.{png,pdf,svg}`",
        "- `reports/small_object_score_calibration_preregistration.md`",
    ])
    path = ROOT / "reports/small_object_score_calibration.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation", type=Path,
        default=ROOT / "data/odinw/pothole/valid/annotations_without_background.json")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def main():
    args = parse_args()
    annotation = args.annotation.resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.baseline_tolerance < 0:
        raise ValueError("--baseline-tolerance must be non-negative")

    prediction_paths = {
        seed: ROOT / "experiments/raw_predictions/pothole_dynamic_scale_seed{}_validation.pkl".format(seed)
        for seed in SEEDS
    }
    for path in prediction_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation))
    category_ids = coco_gt.getCatIds()
    if len(category_ids) != 1:
        raise RuntimeError("Expected one category, found {}".format(len(category_ids)))
    expected_ids = set(coco_gt.getImgIds())
    samples = {
        seed: load_prediction_samples(path, expected_ids)
        for seed, path in prediction_paths.items()
    }

    run_rows = []
    baseline_by_seed = {}
    total = len(AREA_THRESHOLDS) * len(LOGIT_BIASES) * len(SEEDS)
    completed = 0
    for threshold in AREA_THRESHOLDS:
        for bias in LOGIT_BIASES:
            for seed in SEEDS:
                rows = prediction_rows(
                    samples[seed], category_ids[0], threshold, bias, args.top_k)
                metrics = evaluate(coco_gt, rows)
                row = {
                    "area_threshold": threshold,
                    "logit_bias": bias,
                    "seed": seed,
                    "detection_count": len(rows),
                }
                row.update(metrics)
                run_rows.append(row)
                if bias == 0.0:
                    expected = EXPECTED_BASELINE[seed]
                    for metric in ("AP", "APs"):
                        error = abs(metrics[metric] - expected[metric])
                        if error > args.baseline_tolerance:
                            raise RuntimeError(
                                "Baseline reproduction failed for seed {}, {}, threshold {}: "
                                "observed={}, expected={}, abs_error={}".format(
                                    seed, metric, threshold, metrics[metric],
                                    expected[metric], error))
                    if seed not in baseline_by_seed:
                        baseline_by_seed[seed] = metrics
                    else:
                        for metric in METRIC_NAMES:
                            if abs(metrics[metric] - baseline_by_seed[seed][metric]) > 1e-15:
                                raise RuntimeError("b=0 changed across thresholds")
                completed += 1
                if completed % 9 == 0 or completed == total:
                    print("completed {}/{} evaluations".format(completed, total), flush=True)

    for row in run_rows:
        for metric in ("AP", "AP50", "AP75", "APs", "APm", "APl"):
            row["delta_{}".format(metric)] = (
                row[metric] - baseline_by_seed[row["seed"]][metric])
    aggregate = aggregate_rows(run_rows, baseline_by_seed)
    selected, passed = select_configuration(aggregate)
    selected_runs = [
        row for row in run_rows
        if row["area_threshold"] == selected["area_threshold"]
        and row["logit_bias"] == selected["logit_bias"]
    ]

    run_path = ROOT / "results/csv/small_object_score_calibration_search.csv"
    aggregate_path = ROOT / "results/csv/small_object_score_calibration_aggregate.csv"
    json_path = ROOT / "results/json/small_object_score_calibration_search.json"
    write_csv(run_path, run_rows)
    write_csv(aggregate_path, aggregate)

    selected_prediction_files = []
    for seed in SEEDS:
        calibrated_rows = prediction_rows(
            samples[seed], category_ids[0], selected["area_threshold"],
            selected["logit_bias"], args.top_k)
        output = ROOT / (
            "experiments/raw_predictions/pothole_dynamic_scale_seed{}_validation_"
            "small_score_calibrated_coco.json".format(seed))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(calibrated_rows) + "\n", encoding="utf-8")
        selected_prediction_files.append({
            "seed": seed,
            "path": output.relative_to(ROOT).as_posix(),
            "sha256": sha256(output),
            "detection_count": len(calibrated_rows),
        })

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "post-hoc educational validation-only diagnostic",
        "held_out_test_read_or_evaluated": False,
        "annotation": annotation.relative_to(ROOT).as_posix(),
        "annotation_sha256": sha256(annotation),
        "top_k_per_image": args.top_k,
        "seeds": list(SEEDS),
        "grid": {
            "area_thresholds_px2": list(AREA_THRESHOLDS),
            "logit_biases": list(LOGIT_BIASES),
            "configurations": len(AREA_THRESHOLDS) * len(LOGIT_BIASES),
            "evaluations": total,
        },
        "prediction_sources": [
            {
                "seed": seed,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for seed, path in prediction_paths.items()
        ],
        "baseline_exact_reproduction": {
            str(seed): baseline_by_seed[seed] for seed in SEEDS
        },
        "gate_passed": passed,
        "selected": selected,
        "selected_runs": selected_runs,
        "selected_prediction_files": selected_prediction_files,
        "run_csv": run_path.relative_to(ROOT).as_posix(),
        "aggregate_csv": aggregate_path.relative_to(ROOT).as_posix(),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_results(aggregate, selected)
    write_report(selected, passed, selected_runs, aggregate_path, run_path, json_path)
    print(json.dumps({
        "gate_passed": passed,
        "selected_area_threshold": selected["area_threshold"],
        "selected_logit_bias": selected["logit_bias"],
        "mean_delta_AP": selected["mean_delta_AP"],
        "mean_delta_APs": selected["mean_delta_APs"],
        "output": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
