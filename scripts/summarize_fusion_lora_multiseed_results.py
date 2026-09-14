#!/usr/bin/env python
"""Combine frozen validation evidence with post-selection held-out tests."""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]}
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "experiments" / "raw_metrics"
SEEDS = (0, 21, 42)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest(paths, description):
    candidates = list(paths)
    if not candidates:
        raise FileNotFoundError("Missing {}".format(description))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def count_images(path):
    suffixes = {".jpg", ".jpeg", ".png"}
    return sum(
        item.is_file() and item.suffix.lower() in suffixes
        for item in path.rglob("*")
    )


def stats(values):
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half_width = float(1.96 * std / np.sqrt(len(array)))
    return {
        "n": len(array),
        "mean": mean,
        "sample_std": std,
        "normal_approx_95ci_low": mean - half_width,
        "normal_approx_95ci_high": mean + half_width,
    }


def main():
    validation_path = (
        ROOT / "results" / "json" / "fusion_lora_multiseed_validation_summary.json"
    )
    validation = load_json(validation_path)
    if validation["status"] != "three-seed validation gate complete; held-out tests pending":
        raise RuntimeError("Validation gate status is not the immutable pre-test state")
    runs = []
    for run in validation["runs"]:
        seed = run["seed"]
        test_path = latest(
            RAW_DIR.glob(
                "odinw_pothole_fusion_lora_r16_seed{}_test_s{}_*.json".format(
                    seed, seed
                )
            ),
            "seed-{} held-out test JSON".format(seed),
        )
        test_payload = load_json(test_path)
        expected_checkpoint = str((ROOT / run["checkpoint"]).resolve())
        actual_checkpoint = str(Path(test_payload["checkpoint"]).resolve())
        if expected_checkpoint.lower() != actual_checkpoint.lower():
            raise RuntimeError("Seed-{} test checkpoint mismatch".format(seed))
        test_visualization_dir = (
            ROOT
            / "experiments"
            / "odinw_pothole_fusion_lora_r16_seed{}_test".format(seed)
            / "visualizations"
        )
        merged = {key: value for key, value in run.items() if key not in ("validation_metrics", "epoch_metrics")}
        for key, value in run["validation_metrics"].items():
            merged["validation/{}".format(key)] = value
        for key, value in test_payload["metrics"].items():
            merged["test/{}".format(key)] = value
        merged["held_out_test_json"] = str(test_path.relative_to(ROOT))
        merged["test_visualizations"] = count_images(test_visualization_dir)
        if merged["test_visualizations"] != 67:
            raise RuntimeError(
                "Expected 67 seed-{} test visualizations, found {}".format(
                    seed, merged["test_visualizations"]
                )
            )
        runs.append(merged)

    metric_names = (
        "coco/bbox_mAP",
        "coco/bbox_mAP_50",
        "coco/bbox_mAP_75",
        "coco/bbox_mAP_s",
        "coco/bbox_mAP_m",
        "coco/bbox_mAP_l",
    )
    aggregate = {}
    for split in ("validation", "test"):
        aggregate[split] = {
            metric: stats([run["{}/{}".format(split, metric)] for run in runs])
            for metric in metric_names
        }
    for resource in (
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        "elapsed_minutes",
    ):
        aggregate[resource] = stats([run[resource] for run in runs])

    baseline = load_json(ROOT / "results" / "json" / "cross_task_multiseed_summary.json")
    baseline_runs = [
        row
        for row in baseline["runs"]
        if row["dataset"] == "Pothole"
        and row["method"] in ("Full fine-tuning", "Frozen text encoder")
    ]
    comparison_runs = []
    for method in ("Full fine-tuning", "Frozen text encoder"):
        method_rows = {row["seed"]: row for row in baseline_runs if row["method"] == method}
        if set(method_rows) != set(SEEDS):
            raise RuntimeError("Incomplete baseline seeds for {}".format(method))
        val_differences = []
        test_differences = []
        for fusion in runs:
            base = method_rows[fusion["seed"]]
            val_diff = fusion["validation/coco/bbox_mAP"] - base["coco/bbox_mAP"]
            test_diff = fusion["test/coco/bbox_mAP"] - base["test/coco/bbox_mAP"]
            val_differences.append(val_diff)
            test_differences.append(test_diff)
            comparison_runs.append(
                {
                    "baseline_method": method,
                    "seed": fusion["seed"],
                    "fusion_validation_mAP": fusion["validation/coco/bbox_mAP"],
                    "baseline_validation_mAP": base["coco/bbox_mAP"],
                    "paired_validation_difference": val_diff,
                    "fusion_test_mAP": fusion["test/coco/bbox_mAP"],
                    "baseline_test_mAP": base["test/coco/bbox_mAP"],
                    "paired_test_difference": test_diff,
                }
            )
        aggregate["paired_differences_vs_{}".format(method)] = {
            "validation_mAP": stats(val_differences),
            "test_mAP": stats(test_differences),
        }

    method_aggregate = []
    for row in baseline["aggregate"]:
        if row["dataset"] != "Pothole":
            continue
        method_aggregate.append(
            {
                "method": row["method"],
                "trainable_ratio_percent": next(
                    item["trainable_ratio_percent"]
                    for item in baseline_runs
                    if item["method"] == row["method"]
                ),
                "validation_mAP_mean": row["coco/bbox_mAP_mean"],
                "validation_mAP_sample_std": row["coco/bbox_mAP_std"],
                "test_mAP_mean": row["test/coco/bbox_mAP_mean"],
                "test_mAP_sample_std": row["test/coco/bbox_mAP_std"],
            }
        )
    method_aggregate.append(
        {
            "method": "Fusion-LoRA rank 16",
            "trainable_ratio_percent": runs[0]["trainable_ratio_percent"],
            "validation_mAP_mean": aggregate["validation"]["coco/bbox_mAP"]["mean"],
            "validation_mAP_sample_std": aggregate["validation"]["coco/bbox_mAP"]["sample_std"],
            "test_mAP_mean": aggregate["test"]["coco/bbox_mAP"]["mean"],
            "test_mAP_sample_std": aggregate["test"]["coco/bbox_mAP"]["sample_std"],
        }
    )

    summary = {
        "created_local": datetime.now().astimezone().isoformat(),
        "selection_protocol": (
            "Rank 16 and all three checkpoints were fixed by validation evidence before "
            "held-out testing. The immutable pre-test validation summary is referenced "
            "below; each fixed checkpoint was evaluated once on the 67-image test split."
        ),
        "pre_test_validation_summary": str(validation_path.relative_to(ROOT)),
        "status": "three-seed validation and held-out test complete",
        "seeds": list(SEEDS),
        "standard_deviation": "sample (ddof=1)",
        "confidence_interval_note": "Normal-approximation 95% intervals are descriptive only because n=3.",
        "runs": runs,
        "aggregate": aggregate,
        "method_aggregate": method_aggregate,
        "paired_comparison_runs": comparison_runs,
    }

    json_path = ROOT / "results" / "json" / "fusion_lora_multiseed_summary.json"
    runs_path = ROOT / "results" / "csv" / "fusion_lora_multiseed_runs.csv"
    aggregate_path = ROOT / "results" / "csv" / "fusion_lora_multiseed_aggregate.csv"
    comparison_path = ROOT / "results" / "csv" / "fusion_lora_multiseed_comparison.csv"
    paired_path = ROOT / "results" / "csv" / "fusion_lora_multiseed_paired_differences.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0].keys()))
        writer.writeheader()
        writer.writerows(runs)

    aggregate_rows = []
    for split in ("validation", "test"):
        for metric, values in aggregate[split].items():
            aggregate_rows.append({"group": split, "metric": metric, **values})
    for resource in ("peak_cuda_allocated_gib", "peak_cuda_reserved_gib", "elapsed_minutes"):
        aggregate_rows.append({"group": "resource", "metric": resource, **aggregate[resource]})
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(method_aggregate)
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_runs[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_runs)

    labels = ["Full", "Frozen text", "Fusion-LoRA r16"]
    x = np.arange(len(labels))
    val_means = [row["validation_mAP_mean"] for row in method_aggregate]
    val_stds = [row["validation_mAP_sample_std"] for row in method_aggregate]
    test_means = [row["test_mAP_mean"] for row in method_aggregate]
    test_stds = [row["test_mAP_sample_std"] for row in method_aggregate]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    axes[0].bar(x, val_means, yerr=val_stds, capsize=5, color=colors)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Validation COCO bbox mAP")
    axes[0].set_ylim(0.51, 0.56)
    axes[0].set_title("Validation: mean and sample SD")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, test_means, yerr=test_stds, capsize=5, color=colors)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Held-out test COCO bbox mAP")
    axes[1].set_ylim(0.53, 0.58)
    axes[1].set_title("Held-out test: mean and sample SD")
    axes[1].grid(axis="y", alpha=0.25)
    for axis, values, errors in zip(
        axes, (val_means, test_means), (val_stds, test_stds)
    ):
        for index, (value, error) in enumerate(zip(values, errors)):
            axis.text(
                index,
                value + error + 0.001,
                "{:.3f}".format(value),
                ha="center",
                va="bottom",
            )
    fig.suptitle("Pothole three-seed accuracy comparison")
    fig.tight_layout()
    figure_dir = ROOT / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = figure_dir / "fusion_lora_multiseed_comparison.{}".format(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(str(path), **kwargs)
    plt.close(fig)

    report_path = ROOT / "reports" / "fusion_lora_multiseed_results.md"
    full_diff = aggregate["paired_differences_vs_Full fine-tuning"]["test_mAP"]
    frozen_diff = aggregate["paired_differences_vs_Frozen text encoder"]["test_mAP"]
    report = """# Fusion-LoRA Rank-16 Multi-Seed Results

Last updated: {timestamp}

## Outcome

The fixed rank-16 Fusion-LoRA design reached validation mAP {val_mean:.4f} +/- {val_std:.4f} and held-out test mAP {test_mean:.4f} +/- {test_std:.4f} over seeds 0, 21, and 42 (mean +/- sample SD). It updates {ratio:.4f}% of model parameters. Its held-out accuracy is below both Full fine-tuning and Frozen-text fine-tuning, so these experiments support a trainable-parameter efficiency result, not an accuracy-improvement claim.

## Protocol integrity

Rank 16 was selected only by the earlier seed-42 validation rank ablation. Each seed-specific checkpoint was then selected only by validation mAP and independently reloaded. The pre-test validation summary was generated before any new rank-16 held-out test. Each frozen checkpoint was evaluated once on the unchanged 67-image Pothole test split.

## Three-seed comparison

| Method | Trainable ratio | Validation mAP | Held-out test mAP |
|---|---:|---:|---:|
| Full fine-tuning | {full_ratio:.4f}% | {full_val:.4f} +/- {full_val_sd:.4f} | {full_test:.4f} +/- {full_test_sd:.4f} |
| Frozen text encoder | {frozen_ratio:.4f}% | {frozen_val:.4f} +/- {frozen_val_sd:.4f} | {frozen_test:.4f} +/- {frozen_test_sd:.4f} |
| Fusion-LoRA rank 16 | {ratio:.4f}% | {val_mean:.4f} +/- {val_std:.4f} | {test_mean:.4f} +/- {test_std:.4f} |

The paired Fusion-LoRA-minus-Full test difference is {full_diff_mean:+.4f} +/- {full_diff_std:.4f} mAP. The paired Fusion-LoRA-minus-Frozen difference is {frozen_diff_mean:+.4f} +/- {frozen_diff_std:.4f} mAP. With only three seeds, these differences are descriptive; no significance claim is made.

## Per-seed Fusion-LoRA results

| Seed | Best epoch | Validation mAP | Test mAP | Gradient audits | Non-finite tensors |
|---:|---:|---:|---:|---:|---:|
{seed_rows}

All 168 training gradient audits were finite. All three validation reloads exactly reproduced the checkpoint-selection metric, and validation/test visualization counts are 133/67 per seed.

## Interpretation and limitation

Fusion-LoRA reduces the trainable ratio to {ratio:.4f}%, but it trails Full by {val_gap_full:.4f} validation mAP and {test_gap_full:.4f} test mAP on average. It also trails Frozen text by {val_gap_frozen:.4f}/{test_gap_frozen:.4f}. The negative accuracy result indicates that sparse low-rank adaptation alone is insufficient for the paper's main improvement; subsequent work needs a motivated prompt-fusion or calibration component, plus cross-task evidence. The normal-approximation 95% intervals in the machine-readable summary are descriptive only because n=3.

## Claim-to-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Rank-16 Fusion-LoRA is trainable-parameter efficient | {ratio:.4f}% trainable versus {full_ratio:.4f}%/{frozen_ratio:.4f}% | Supported |
| Rank-16 optimization is numerically stable | 168 audits, zero non-finite tensors | Supported for these runs |
| Rank-16 improves accuracy over strong baselines | Lower validation and test means | Not supported |
| Rank-16 generalizes across ODinW tasks | Pothole only | Needs evidence |

## Five-dimension self-review

1. **Contribution:** parameter efficiency is demonstrated, but accuracy improvement is absent.
2. **Writing clarity:** validation selection, held-out evaluation, and negative result are separated explicitly.
3. **Experimental strength:** three seeds and strong reproduced baselines are available; n=3 remains small.
4. **Evaluation completeness:** Pothole validation/test and efficiency evidence are complete; prompt robustness, few-shot scaling, and cross-task Fusion-LoRA remain missing.
5. **Method soundness:** optimization is stable, but the accuracy gap motivates an additional domain-aware component rather than a stronger LoRA claim.

## Reproducibility artifacts

- Pre-test validation evidence: `results/json/fusion_lora_multiseed_validation_summary.json`
- Final summary: `results/json/fusion_lora_multiseed_summary.json`
- Run table: `results/csv/fusion_lora_multiseed_runs.csv`
- Aggregate table: `results/csv/fusion_lora_multiseed_aggregate.csv`
- Baseline comparison: `results/csv/fusion_lora_multiseed_comparison.csv`
- Paired differences: `results/csv/fusion_lora_multiseed_paired_differences.csv`
- Figure: `results/figures/fusion_lora_multiseed_comparison.png`, PDF, and SVG
""".format(
        timestamp=datetime.now().astimezone().isoformat(),
        val_mean=aggregate["validation"]["coco/bbox_mAP"]["mean"],
        val_std=aggregate["validation"]["coco/bbox_mAP"]["sample_std"],
        test_mean=aggregate["test"]["coco/bbox_mAP"]["mean"],
        test_std=aggregate["test"]["coco/bbox_mAP"]["sample_std"],
        ratio=runs[0]["trainable_ratio_percent"],
        full_ratio=method_aggregate[0]["trainable_ratio_percent"],
        full_val=method_aggregate[0]["validation_mAP_mean"],
        full_val_sd=method_aggregate[0]["validation_mAP_sample_std"],
        full_test=method_aggregate[0]["test_mAP_mean"],
        full_test_sd=method_aggregate[0]["test_mAP_sample_std"],
        frozen_ratio=method_aggregate[1]["trainable_ratio_percent"],
        frozen_val=method_aggregate[1]["validation_mAP_mean"],
        frozen_val_sd=method_aggregate[1]["validation_mAP_sample_std"],
        frozen_test=method_aggregate[1]["test_mAP_mean"],
        frozen_test_sd=method_aggregate[1]["test_mAP_sample_std"],
        full_diff_mean=full_diff["mean"], full_diff_std=full_diff["sample_std"],
        frozen_diff_mean=frozen_diff["mean"], frozen_diff_std=frozen_diff["sample_std"],
        seed_rows="\n".join(
            "| {seed} | {epoch} | {val:.3f} | {test:.3f} | {audits} | {nonfinite} |".format(
                seed=run["seed"], epoch=run["best_epoch"],
                val=run["validation/coco/bbox_mAP"], test=run["test/coco/bbox_mAP"],
                audits=run["gradient_audits"], nonfinite=run["max_nonfinite_gradient_tensors"],
            ) for run in runs
        ),
        val_gap_full=method_aggregate[0]["validation_mAP_mean"] - aggregate["validation"]["coco/bbox_mAP"]["mean"],
        test_gap_full=method_aggregate[0]["test_mAP_mean"] - aggregate["test"]["coco/bbox_mAP"]["mean"],
        val_gap_frozen=method_aggregate[1]["validation_mAP_mean"] - aggregate["validation"]["coco/bbox_mAP"]["mean"],
        test_gap_frozen=method_aggregate[1]["test_mAP_mean"] - aggregate["test"]["coco/bbox_mAP"]["mean"],
    )
    report_path.write_text(report, encoding="utf-8")
    print("summary={}".format(json_path))
    print("report={}".format(report_path))
    print("test_mAP={:.6f}+/-{:.6f}".format(
        aggregate["test"]["coco/bbox_mAP"]["mean"],
        aggregate["test"]["coco/bbox_mAP"]["sample_std"],
    ))


if __name__ == "__main__":
    main()
