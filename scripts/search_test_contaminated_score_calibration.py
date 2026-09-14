#!/usr/bin/env python
"""Explicitly test-contaminated score calibration for educational diagnosis.

This script intentionally selects post-processing parameters on the Pothole
test split. Its output is not an independent held-out result and is kept in a
separate namespace from the frozen main experiment.
"""

import contextlib
import csv
import hashlib
import io
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO

from search_small_object_score_calibration import (
    AREA_THRESHOLDS,
    LOGIT_BIASES,
    METRIC_NAMES,
    evaluate,
    load_prediction_samples,
    prediction_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 21, 42)
SUMMARY_METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(run_rows, baseline_by_seed):
    output = []
    for threshold in AREA_THRESHOLDS:
        for bias in LOGIT_BIASES:
            selected = [
                row for row in run_rows
                if row["area_threshold"] == threshold and row["logit_bias"] == bias
            ]
            aggregate = {
                "area_threshold": threshold,
                "logit_bias": bias,
                "n_seeds": len(selected),
            }
            for metric in SUMMARY_METRICS:
                values = [row[metric] for row in selected]
                deltas = [
                    row[metric] - baseline_by_seed[row["seed"]][metric]
                    for row in selected
                ]
                aggregate["mean_{}".format(metric)] = statistics.mean(values)
                aggregate["sample_sd_{}".format(metric)] = statistics.stdev(values)
                aggregate["mean_delta_{}".format(metric)] = statistics.mean(deltas)
                aggregate["min_delta_{}".format(metric)] = min(deltas)
                aggregate["max_delta_{}".format(metric)] = max(deltas)
            aggregate["seeds_AP_improved"] = sum(
                row["AP"] > baseline_by_seed[row["seed"]]["AP"] + 1e-12
                for row in selected
            )
            aggregate["seeds_APs_improved"] = sum(
                row["APs"] > baseline_by_seed[row["seed"]]["APs"] + 1e-12
                for row in selected
            )
            output.append(aggregate)
    return output


def selection_key(row):
    return (
        row["mean_AP"],
        row["mean_APs"],
        -row["logit_bias"],
        -abs(row["area_threshold"] - 1024),
    )


def plot_results(aggregate, selected):
    lookup = {(row["area_threshold"], row["logit_bias"]): row for row in aggregate}
    heat_ap = np.empty((len(LOGIT_BIASES), len(AREA_THRESHOLDS)))
    heat_aps = np.empty_like(heat_ap)
    for y, bias in enumerate(LOGIT_BIASES):
        for x, threshold in enumerate(AREA_THRESHOLDS):
            row = lookup[(threshold, bias)]
            heat_ap[y, x] = row["mean_delta_AP"]
            heat_aps[y, x] = row["mean_delta_APs"]

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2))
    for axis, values, title in zip(
            axes, (heat_ap, heat_aps), ("Mean test AP change", "Mean test APs change")):
        bound = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1e-4)
        image = axis.imshow(
            values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_xticks(range(len(AREA_THRESHOLDS)), AREA_THRESHOLDS, rotation=35)
        axis.set_yticks(range(len(LOGIT_BIASES)), LOGIT_BIASES)
        axis.set_xlabel("Predicted-box area threshold (px^2)")
        axis.set_ylabel("Logit bias")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.scatter(
            AREA_THRESHOLDS.index(selected["area_threshold"]),
            LOGIT_BIASES.index(selected["logit_bias"]),
            marker="s", facecolors="none", edgecolors="black",
            linewidths=1.8, s=145)
    fig.suptitle("TEST-CONTAMINATED parameter search (diagnostic only)", color="#B2182B")
    fig.tight_layout()
    stem = ROOT / "results/figures/test_contaminated_score_calibration"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(str(stem) + "." + suffix, bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(selected, selected_runs, baseline_by_seed):
    improved = selected["logit_bias"] != 0.0 and selected["mean_delta_AP"] > 0.0
    lines = [
        "# Test-contaminated score calibration",
        "",
        "> **Invalid as independent test evidence.** Parameters in this report were selected directly on the test annotations at the user's explicit request for an educational diagnostic.",
        "",
        "## Test-optimized result",
        "",
        "The 42-setting search selected area threshold `{} px^2` and logit bias `{}`. The tuning result is **{}** relative to the uncalibrated predictions.".format(
            selected["area_threshold"], selected["logit_bias"],
            "positive" if improved else "negative"),
        "",
        "| Metric | Uncalibrated mean | Test-optimized mean | Change |",
        "|---|---:|---:|---:|",
    ]
    for metric in SUMMARY_METRICS:
        original = statistics.mean(baseline_by_seed[seed][metric] for seed in SEEDS)
        lines.append("| {} | {:.6f} | {:.6f} | {:+.6f} |".format(
            metric, original, selected["mean_{}".format(metric)],
            selected["mean_delta_{}".format(metric)]))
    lines.extend([
        "",
        "| Seed | Original AP | Tuned AP | AP change | Original APs | Tuned APs | APs change |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(selected_runs, key=lambda item: item["seed"]):
        base = baseline_by_seed[row["seed"]]
        lines.append("| {} | {:.6f} | {:.6f} | {:+.6f} | {:.6f} | {:.6f} | {:+.6f} |".format(
            row["seed"], base["AP"], row["AP"], row["delta_AP"],
            base["APs"], row["APs"], row["delta_APs"]))
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The unchanged, independently evaluated test means remain the valid main results. The optimized values quantify how much this narrow score-calibration family can overfit this particular test set; they are not evidence of performance on unseen data. A new untouched dataset or nested cross-validation would be required for a clean assessment of the selected parameters.",
        "",
        "## Artifacts",
        "",
        "- `reports/test_contaminated_calibration_protocol.md`",
        "- `results/csv/test_contaminated_score_calibration_search.csv`",
        "- `results/csv/test_contaminated_score_calibration_aggregate.csv`",
        "- `results/json/test_contaminated_score_calibration_search.json`",
        "- `results/figures/test_contaminated_score_calibration.{png,pdf,svg}`",
    ])
    path = ROOT / "reports/test_contaminated_score_calibration.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    annotation = ROOT / "data/odinw/pothole/test/annotations_without_background.json"
    prediction_paths = {
        seed: ROOT / "experiments/raw_predictions/pothole_dynamic_scale_seed{}_test.pkl".format(seed)
        for seed in SEEDS
    }
    exact_paths = {
        seed: ROOT / "experiments/raw_metrics/pothole_dynamic_scale_seed{}_test_exact.json".format(seed)
        for seed in SEEDS
    }
    for path in [annotation] + list(prediction_paths.values()) + list(exact_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    recorded = {
        seed: json.loads(path.read_text(encoding="utf-8"))["metrics"]
        for seed, path in exact_paths.items()
    }
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
                detections = prediction_rows(
                    samples[seed], category_ids[0], threshold, bias, 100)
                metrics = evaluate(coco_gt, detections)
                row = {
                    "area_threshold": threshold,
                    "logit_bias": bias,
                    "seed": seed,
                    "detection_count": len(detections),
                }
                row.update(metrics)
                run_rows.append(row)
                if bias == 0.0:
                    for metric in METRIC_NAMES:
                        error = abs(metrics[metric] - float(recorded[seed][metric]))
                        if error > 1e-12:
                            raise RuntimeError(
                                "Uncalibrated reproduction failed: seed={}, metric={}, "
                                "observed={}, recorded={}, error={}".format(
                                    seed, metric, metrics[metric], recorded[seed][metric], error))
                    if seed not in baseline_by_seed:
                        baseline_by_seed[seed] = metrics
                    else:
                        for metric in METRIC_NAMES:
                            if metrics[metric] != baseline_by_seed[seed][metric]:
                                raise RuntimeError("bias 0 changed across thresholds")
                completed += 1
                if completed % 9 == 0 or completed == total:
                    print("completed {}/{} test evaluations".format(completed, total), flush=True)

    for row in run_rows:
        for metric in SUMMARY_METRICS:
            row["delta_{}".format(metric)] = (
                row[metric] - baseline_by_seed[row["seed"]][metric])
    aggregate = aggregate_rows(run_rows, baseline_by_seed)
    selected = max(aggregate, key=selection_key)
    selected_runs = [
        row for row in run_rows
        if row["area_threshold"] == selected["area_threshold"]
        and row["logit_bias"] == selected["logit_bias"]
    ]

    run_path = ROOT / "results/csv/test_contaminated_score_calibration_search.csv"
    aggregate_path = ROOT / "results/csv/test_contaminated_score_calibration_aggregate.csv"
    json_path = ROOT / "results/json/test_contaminated_score_calibration_search.json"
    write_csv(run_path, run_rows)
    write_csv(aggregate_path, aggregate)

    output_predictions = []
    for seed in SEEDS:
        detections = prediction_rows(
            samples[seed], category_ids[0], selected["area_threshold"],
            selected["logit_bias"], 100)
        path = ROOT / (
            "experiments/raw_predictions/pothole_dynamic_scale_seed{}_test_"
            "test_contaminated_score_calibrated_coco.json".format(seed))
        path.write_text(json.dumps(detections) + "\n", encoding="utf-8")
        output_predictions.append({
            "seed": seed,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "detection_count": len(detections),
        })

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "TEST-CONTAMINATED: parameters selected on test; not independent evidence",
        "user_explicitly_requested_test_tuning": True,
        "selection_objective": "highest three-seed mean test AP",
        "annotation": annotation.relative_to(ROOT).as_posix(),
        "annotation_sha256": sha256(annotation),
        "seeds": list(SEEDS),
        "grid": {
            "area_thresholds_px2": list(AREA_THRESHOLDS),
            "logit_biases": list(LOGIT_BIASES),
            "configurations": len(aggregate),
            "evaluations": len(run_rows),
        },
        "prediction_sources": [
            {"seed": seed, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for seed, path in prediction_paths.items()
        ],
        "recorded_metric_sources": [
            {"seed": seed, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for seed, path in exact_paths.items()
        ],
        "uncalibrated_exact_reproduction": {
            str(seed): baseline_by_seed[seed] for seed in SEEDS
        },
        "selected": selected,
        "selected_runs": selected_runs,
        "selected_prediction_files": output_predictions,
        "run_csv": run_path.relative_to(ROOT).as_posix(),
        "aggregate_csv": aggregate_path.relative_to(ROOT).as_posix(),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_results(aggregate, selected)
    write_report(selected, selected_runs, baseline_by_seed)
    print(json.dumps({
        "warning": payload["warning"],
        "selected_area_threshold": selected["area_threshold"],
        "selected_logit_bias": selected["logit_bias"],
        "mean_AP": selected["mean_AP"],
        "mean_delta_AP": selected["mean_delta_AP"],
        "mean_APs": selected["mean_APs"],
        "mean_delta_APs": selected["mean_delta_APs"],
        "output": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
