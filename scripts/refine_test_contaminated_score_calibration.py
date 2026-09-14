#!/usr/bin/env python
"""Adaptive refinement of the explicitly test-contaminated calibration."""

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
    METRIC_NAMES,
    evaluate,
    load_prediction_samples,
    prediction_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 21, 42)
AREA_THRESHOLDS = (64, 100, 144, 196, 256, 324, 400, 484)
LOGIT_BIASES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75)
SUMMARY_METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl")
STEM = "test_contaminated_score_calibration_refine"


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
        row["mean_AP"], row["mean_APs"], -row["logit_bias"],
        -abs(row["area_threshold"] - 256),
    )


def plot_results(aggregate, selected):
    lookup = {(row["area_threshold"], row["logit_bias"]): row for row in aggregate}
    values = np.empty((len(LOGIT_BIASES), len(AREA_THRESHOLDS)))
    for y, bias in enumerate(LOGIT_BIASES):
        for x, threshold in enumerate(AREA_THRESHOLDS):
            values[y, x] = lookup[(threshold, bias)]["mean_delta_AP"]
    bound = max(abs(float(values.min())), abs(float(values.max())), 1e-5)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    image = axes[0].imshow(
        values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    axes[0].set_xticks(range(len(AREA_THRESHOLDS)), AREA_THRESHOLDS, rotation=35)
    axes[0].set_yticks(range(len(LOGIT_BIASES)), LOGIT_BIASES)
    axes[0].set_xlabel("Predicted-box area threshold (px^2)")
    axes[0].set_ylabel("Logit bias")
    axes[0].set_title("Mean test AP change")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].scatter(
        AREA_THRESHOLDS.index(selected["area_threshold"]),
        LOGIT_BIASES.index(selected["logit_bias"]), marker="s",
        facecolors="none", edgecolors="black", linewidths=1.8, s=145)

    x = np.asarray([row["mean_delta_AP"] for row in aggregate])
    y = np.asarray([row["mean_delta_APs"] for row in aggregate])
    axes[1].scatter(x, y, color="#6C8EA3", alpha=0.72, s=34)
    axes[1].scatter(
        selected["mean_delta_AP"], selected["mean_delta_APs"], marker="*",
        color="#D55E00", edgecolor="black", s=190, label="Test-selected", zorder=4)
    axes[1].axvline(0.0, color="black", linewidth=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Mean test AP change")
    axes[1].set_ylabel("Mean test APs change")
    axes[1].set_title("Adaptive test-search trade-off")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.suptitle("ADAPTIVE TEST OVERFIT: boundary refinement", color="#B2182B")
    fig.tight_layout()
    stem = ROOT / "results/figures" / STEM
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(str(stem) + "." + suffix, bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(selected, selected_runs, baseline_by_seed):
    lines = [
        "# Adaptive test-contaminated calibration refinement",
        "",
        "> **This is deliberate test-set overfitting, not a held-out result.** The refinement grid was chosen after observing the coarse test optimum at its threshold boundary.",
        "",
        "## Final test-selected diagnostic",
        "",
        "Area threshold: `{} px^2`; logit bias: `{}`.".format(
            selected["area_threshold"], selected["logit_bias"]),
        "",
        "| Metric | Original mean | Test-selected mean | Change |",
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
        "## Required interpretation",
        "",
        "The original uncalibrated three-seed test mean remains the only independently evaluated main result. This test-selected score rule must be evaluated on a new untouched dataset before it can support a generalization statement.",
        "",
        "## Artifacts",
        "",
        "- `results/csv/{}_search.csv`".format(STEM),
        "- `results/csv/{}_aggregate.csv`".format(STEM),
        "- `results/json/{}_search.json`".format(STEM),
        "- `results/figures/{}.{{png,pdf,svg}}`".format(STEM),
    ])
    (ROOT / "reports/test_contaminated_score_calibration_refinement.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


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
    recorded = {
        seed: json.loads(path.read_text(encoding="utf-8"))["metrics"]
        for seed, path in exact_paths.items()
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation))
    category_ids = coco_gt.getCatIds()
    if len(category_ids) != 1:
        raise RuntimeError("Expected exactly one category")
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
                        if abs(metrics[metric] - float(recorded[seed][metric])) > 1e-12:
                            raise RuntimeError(
                                "Uncalibrated reproduction failure: seed={}, metric={}".format(
                                    seed, metric))
                    if seed not in baseline_by_seed:
                        baseline_by_seed[seed] = metrics
                completed += 1
                if completed % 12 == 0 or completed == total:
                    print("completed {}/{} refinement evaluations".format(
                        completed, total), flush=True)

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

    run_path = ROOT / "results/csv/{}_search.csv".format(STEM)
    aggregate_path = ROOT / "results/csv/{}_aggregate.csv".format(STEM)
    json_path = ROOT / "results/json/{}_search.json".format(STEM)
    write_csv(run_path, run_rows)
    write_csv(aggregate_path, aggregate)

    outputs = []
    for seed in SEEDS:
        detections = prediction_rows(
            samples[seed], category_ids[0], selected["area_threshold"],
            selected["logit_bias"], 100)
        path = ROOT / (
            "experiments/raw_predictions/pothole_dynamic_scale_seed{}_test_"
            "adaptive_test_overfit_score_calibrated_coco.json".format(seed))
        path.write_text(json.dumps(detections) + "\n", encoding="utf-8")
        outputs.append({
            "seed": seed, "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path), "detection_count": len(detections),
        })

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "ADAPTIVE TEST OVERFIT: invalid as independent evidence",
        "coarse_test_results_were_observed_before_grid_design": True,
        "selection_objective": "highest three-seed mean test AP",
        "annotation": annotation.relative_to(ROOT).as_posix(),
        "annotation_sha256": sha256(annotation),
        "seeds": list(SEEDS),
        "grid": {
            "area_thresholds_px2": list(AREA_THRESHOLDS),
            "logit_biases": list(LOGIT_BIASES),
            "configurations": len(aggregate), "evaluations": len(run_rows),
        },
        "prediction_sources": [
            {"seed": seed, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for seed, path in prediction_paths.items()
        ],
        "uncalibrated_exact_reproduction": {
            str(seed): baseline_by_seed[seed] for seed in SEEDS
        },
        "selected": selected,
        "selected_runs": selected_runs,
        "selected_prediction_files": outputs,
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
