#!/usr/bin/env python
"""Summarize the frozen three-seed Pothole validation comparison."""

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 21, 42)
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl")
T_975_DF2 = 4.302652729911275
RUNS = {
    "Fusion-LoRA-r16": {
        0: ("experiments/raw_metrics/pothole_fusion_lora_r16_seed0_validation_exact.json", 12,
            "experiments/odinw_pothole_fusion_lora_r16_seed0/best_coco_bbox_mAP_epoch_12.pth"),
        21: ("experiments/raw_metrics/pothole_fusion_lora_r16_seed21_validation_exact.json", 12,
             "experiments/odinw_pothole_fusion_lora_r16_seed21/best_coco_bbox_mAP_epoch_12.pth"),
        42: ("experiments/raw_metrics/pothole_fusion_lora_r16_seed42_validation_exact_recomputed.json", 9,
             "experiments/odinw_pothole_fusion_lora_r16_seed42/best_coco_bbox_mAP_epoch_9.pth"),
    },
    "Dynamic Scale-Aware Fusion LoRA": {
        0: ("experiments/raw_metrics/pothole_dynamic_scale_seed0_validation_exact.json", 10,
            "experiments/odinw_pothole_scale_aware_lora_r16_seed0/best_coco_bbox_mAP_epoch_10.pth"),
        21: ("experiments/raw_metrics/pothole_dynamic_scale_seed21_validation_exact.json", 12,
             "experiments/odinw_pothole_scale_aware_lora_r16_seed21/best_coco_bbox_mAP_epoch_12.pth"),
        42: ("experiments/raw_metrics/pothole_dynamic_scale_seed42_validation_exact_recomputed.json", 11,
             "experiments/odinw_pothole_scale_aware_lora_r16_seed42_retry/best_coco_bbox_mAP_epoch_11.pth"),
    },
}


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


def load_runs():
    rows = []
    sources = []
    for method, by_seed in RUNS.items():
        for seed in SEEDS:
            metric_rel, epoch, checkpoint_rel = by_seed[seed]
            metric_path = ROOT / metric_rel
            checkpoint_path = ROOT / checkpoint_rel
            if not metric_path.is_file():
                raise FileNotFoundError(metric_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            row = {
                "method": method,
                "seed": seed,
                "best_epoch": epoch,
                "checkpoint": checkpoint_rel,
                "metric_json": metric_rel,
            }
            row.update({metric: float(metrics[metric]) for metric in METRICS})
            rows.append(row)
            sources.append({
                "path": metric_rel,
                "sha256": sha256(metric_path),
                "prediction_sha256": payload["predictions_sha256"],
            })
    return rows, sources


def aggregate(rows):
    output = []
    for method in RUNS:
        selected = [row for row in rows if row["method"] == method]
        for metric in METRICS:
            values = [row[metric] for row in selected]
            output.append({
                "method": method,
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            })
    return output


def paired(rows):
    lookup = {(row["method"], row["seed"]): row for row in rows}
    baseline = "Fusion-LoRA-r16"
    proposed = "Dynamic Scale-Aware Fusion LoRA"
    run_rows = []
    summary = {}
    for seed in SEEDS:
        row = {"seed": seed}
        for metric in METRICS:
            row["delta_{}".format(metric)] = (
                lookup[(proposed, seed)][metric] - lookup[(baseline, seed)][metric])
        run_rows.append(row)
    for metric in METRICS:
        values = [row["delta_{}".format(metric)] for row in run_rows]
        mean = statistics.mean(values)
        sample_sd = statistics.stdev(values)
        half_width = T_975_DF2 * sample_sd / math.sqrt(len(values))
        summary[metric] = {
            "by_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
            "n": len(values),
            "mean": mean,
            "sample_sd": sample_sd,
            "t_critical_df2": T_975_DF2,
            "ci95_t": [mean - half_width, mean + half_width],
        }
    return run_rows, summary


def plot(rows, paired_summary):
    figure_dir = ROOT / "results/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    colors = ("#607D8B", "#E76F51")
    for index, method in enumerate(RUNS):
        values = [row["AP"] for row in rows if row["method"] == method]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        axes[0].bar(index, mean, yerr=sd, capsize=5, color=colors[index], alpha=0.82)
        axes[0].scatter([index - 0.08, index, index + 0.08], values,
                        color="black", s=24, zorder=3)
    axes[0].set_xticks((0, 1), ("Fusion-LoRA\nr16", "Dynamic scale-aware\nfusion LoRA"))
    axes[0].set_ylabel("Validation AP")
    axes[0].set_ylim(0.515, 0.555)
    axes[0].grid(axis="y", alpha=0.25)
    deltas = [paired_summary["AP"]["by_seed"][str(seed)] for seed in SEEDS]
    axes[1].bar([str(seed) for seed in SEEDS], deltas, color="#2A9D8F", alpha=0.86)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Paired AP change")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Pothole validation: frozen three-seed comparison")
    fig.tight_layout()
    stem = figure_dir / "scale_aware_multiseed_validation"
    for suffix, kwargs in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        fig.savefig(str(stem) + "." + suffix, bbox_inches="tight", **kwargs)
    plt.close(fig)


def report(rows, aggregates, paired_summary):
    by_method = {}
    for method in RUNS:
        selected = [row for row in aggregates if row["method"] == method]
        by_method[method] = {row["metric"]: row for row in selected}
    pair = paired_summary["AP"]
    lines = [
        "# Dynamic Scale-Aware Fusion LoRA: three-seed validation",
        "",
        "## Frozen protocol",
        "",
        "Seeds 0, 21, and 42 use the same Pothole train/validation split, 12-epoch budget, input pipeline, prompt, evaluator, and validation-only checkpoint selection. Exact metrics are independently recomputed from immutable per-image prediction dumps with pycocotools. Held-out test data is not read in this aggregation.",
        "",
        "## Exact results",
        "",
        "| Method | Seed 0 AP | Seed 21 AP | Seed 42 AP | Mean +/- sample SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in RUNS:
        method_rows = {row["seed"]: row for row in rows if row["method"] == method}
        ap = by_method[method]["AP"]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} +/- {:.6f} |".format(
            method, method_rows[0]["AP"], method_rows[21]["AP"], method_rows[42]["AP"],
            ap["mean"], ap["sample_sd"]))
    lines.extend([
        "",
        "The paired AP changes for seeds 0/21/42 are {:+.6f}, {:+.6f}, and {:+.6f}. The mean paired change is {:+.6f} with sample SD {:.6f}; the descriptive 95% t interval is [{:+.6f}, {:+.6f}]. With only three seeds, this interval is reported for workflow completeness and is not treated as a strong significance or publication claim.".format(
            pair["by_seed"]["0"], pair["by_seed"]["21"], pair["by_seed"]["42"],
            pair["mean"], pair["sample_sd"], *pair["ci95_t"]),
        "",
        "All three seeds improve over their matched Fusion-LoRA baseline on validation AP. Test evaluation remains a separate, one-time post-selection step.",
        "",
        "## Artifacts",
        "",
        "- `results/csv/scale_aware_multiseed_validation_runs.csv`",
        "- `results/csv/scale_aware_multiseed_validation_aggregate.csv`",
        "- `results/csv/scale_aware_multiseed_validation_paired.csv`",
        "- `results/json/scale_aware_multiseed_validation_summary.json`",
        "- `results/figures/scale_aware_multiseed_validation.{png,pdf,svg}`",
    ])
    path = ROOT / "reports/scale_aware_multiseed_validation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows, sources = load_runs()
    aggregates = aggregate(rows)
    paired_rows, paired_summary = paired(rows)
    write_csv(ROOT / "results/csv/scale_aware_multiseed_validation_runs.csv", rows)
    write_csv(ROOT / "results/csv/scale_aware_multiseed_validation_aggregate.csv", aggregates)
    write_csv(ROOT / "results/csv/scale_aware_multiseed_validation_paired.csv", paired_rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "pothole_validation",
        "seeds": list(SEEDS),
        "metric_source": "pycocotools.COCOeval over immutable prediction PKLs",
        "runs": rows,
        "aggregate": aggregates,
        "paired_proposed_minus_baseline": paired_summary,
        "sources": sources,
        "held_out_test_read": False,
    }
    output = ROOT / "results/json/scale_aware_multiseed_validation_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(rows, paired_summary)
    report(rows, aggregates, paired_summary)
    print(json.dumps({
        "output": str(output),
        "dynamic_AP_mean": next(row["mean"] for row in aggregates
                                if row["method"].startswith("Dynamic") and row["metric"] == "AP"),
        "baseline_AP_mean": next(row["mean"] for row in aggregates
                                 if row["method"].startswith("Fusion") and row["metric"] == "AP"),
        "paired_AP": paired_summary["AP"],
    }, indent=2))


if __name__ == "__main__":
    main()
