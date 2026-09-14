#!/usr/bin/env python
"""Aggregate fixed-rank Fusion-LoRA Pothole seeds from raw run artifacts."""

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]}
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "experiments" / "raw_metrics"
MANIFEST_DIR = ROOT / "experiments" / "manifests"
SEEDS = (0, 21, 42)


def latest(paths, description):
    candidates = list(paths)
    if not candidates:
        raise FileNotFoundError("Missing {}".format(description))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def count_images(path):
    suffixes = {".jpg", ".jpeg", ".png"}
    return sum(
        item.is_file() and item.suffix.lower() in suffixes
        for item in path.rglob("*")
    )


def completed_manifest(seed):
    pattern = "odinw_pothole_fusion_lora_r16_seed{}_s{}_*.json".format(
        seed, seed
    )
    candidates = []
    for path in MANIFEST_DIR.glob(pattern):
        payload = load_json(path)
        if payload.get("returncode") == 0 and payload.get("completed_utc"):
            candidates.append((path, payload))
    if not candidates:
        raise FileNotFoundError("Missing completed seed-{} manifest".format(seed))
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def parse_seed(seed):
    run_name = "odinw_pothole_fusion_lora_r16_seed{}".format(seed)
    work_dir = ROOT / "experiments" / run_name
    log_path = latest(work_dir.rglob("*.log"), "seed-{} log".format(seed))
    text = log_path.read_text(encoding="utf-8", errors="replace")
    val_pattern = re.compile(
        r"Epoch\(val\) \[(\d+)\]\[133/133\].*?"
        r"coco/bbox_mAP: ([0-9.]+).*?"
        r"coco/bbox_mAP_50: ([0-9.]+).*?"
        r"coco/bbox_mAP_75: ([0-9.]+).*?"
        r"coco/bbox_mAP_s: ([0-9.]+).*?"
        r"coco/bbox_mAP_m: ([0-9.]+).*?"
        r"coco/bbox_mAP_l: ([0-9.]+)"
    )
    epochs = []
    for match in val_pattern.finditer(text):
        values = [float(value) for value in match.groups()[1:]]
        epochs.append(
            {
                "seed": seed,
                "epoch": int(match.group(1)),
                "coco/bbox_mAP": values[0],
                "coco/bbox_mAP_50": values[1],
                "coco/bbox_mAP_75": values[2],
                "coco/bbox_mAP_s": values[3],
                "coco/bbox_mAP_m": values[4],
                "coco/bbox_mAP_l": values[5],
            }
        )
    if len(epochs) != 12:
        raise RuntimeError("Expected 12 epochs for seed {}, found {}".format(seed, len(epochs)))
    best = max(epochs, key=lambda row: row["coco/bbox_mAP"])

    param_match = re.search(
        r"SetTrainableModulesHook: trainable=(\d+) total=(\d+) ratio=([0-9.]+)",
        text,
    )
    if not param_match:
        raise RuntimeError("Missing seed-{} parameter audit".format(seed))
    trainable = int(param_match.group(1))
    total = int(param_match.group(2))

    audit_pattern = re.compile(
        r"gradient_audit iter=(\d+) tensors=(\d+) nonzero=(\d+) "
        r"nonfinite=(\d+) global_norm=([0-9.eE+-]+) "
        r"cuda_peak_allocated_bytes=(\d+) cuda_peak_reserved_bytes=(\d+)"
    )
    audits = [match.groups() for match in audit_pattern.finditer(text)]
    if len(audits) != 56:
        raise RuntimeError("Expected 56 audits for seed {}, found {}".format(seed, len(audits)))
    if max(int(row[3]) for row in audits) != 0:
        raise RuntimeError("Non-finite gradient found for seed {}".format(seed))

    checkpoint = work_dir / "best_coco_bbox_mAP_epoch_{}.pth".format(best["epoch"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    validation_path = latest(
        RAW_DIR.glob("{}_best_eval_s{}_*.json".format(run_name, seed)),
        "seed-{} independent validation JSON".format(seed),
    )
    validation = load_json(validation_path)["metrics"]
    if validation["coco/bbox_mAP"] != best["coco/bbox_mAP"]:
        raise RuntimeError("Seed-{} validation reload mismatch".format(seed))
    manifest_path, manifest = completed_manifest(seed)
    started = datetime.fromisoformat(manifest["created_utc"])
    completed = datetime.fromisoformat(manifest["completed_utc"])
    elapsed_minutes = (completed - started).total_seconds() / 60.0

    run = {
        "seed": seed,
        "rank": 16,
        "best_epoch": best["epoch"],
        "validation_mAP": validation["coco/bbox_mAP"],
        "final_validation_mAP": epochs[-1]["coco/bbox_mAP"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio_percent": 100.0 * trainable / total,
        "gradient_audits": len(audits),
        "max_nonfinite_gradient_tensors": max(int(row[3]) for row in audits),
        "max_gradient_global_norm": max(float(row[4]) for row in audits),
        "peak_cuda_allocated_gib": max(int(row[5]) for row in audits) / (1024 ** 3),
        "peak_cuda_reserved_gib": max(int(row[6]) for row in audits) / (1024 ** 3),
        "elapsed_minutes": elapsed_minutes,
        "validation_visualizations": count_images(
            ROOT / "experiments" / "{}_best_eval".format(run_name) / "visualizations"
        ),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest": str(manifest_path.relative_to(ROOT)),
        "training_log": str(log_path.relative_to(ROOT)),
        "independent_validation_json": str(validation_path.relative_to(ROOT)),
        "validation_metrics": validation,
        "epoch_metrics": epochs,
    }
    return run


def metric_stats(runs, key):
    values = np.array([row["validation_metrics"][key] for row in runs], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    half_width = float(1.96 * std / np.sqrt(len(values)))
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": std,
        "normal_approx_95ci_low": mean - half_width,
        "normal_approx_95ci_high": mean + half_width,
    }


def main():
    runs = [parse_seed(seed) for seed in SEEDS]
    metrics = [
        "coco/bbox_mAP",
        "coco/bbox_mAP_50",
        "coco/bbox_mAP_75",
        "coco/bbox_mAP_s",
        "coco/bbox_mAP_m",
        "coco/bbox_mAP_l",
    ]
    aggregate = {metric: metric_stats(runs, metric) for metric in metrics}

    baseline_summary = load_json(ROOT / "results" / "json" / "cross_task_multiseed_summary.json")
    comparisons = []
    fusion_mean = aggregate["coco/bbox_mAP"]["mean"]
    for row in baseline_summary["aggregate"]:
        if row["dataset"] != "Pothole":
            continue
        comparisons.append(
            {
                "method": row["method"],
                "validation_mAP_mean": row["coco/bbox_mAP_mean"],
                "validation_mAP_sample_std": row["coco/bbox_mAP_std"],
                "fusion_lora_minus_method_mean_mAP": (
                    fusion_mean - row["coco/bbox_mAP_mean"]
                ),
            }
        )
    comparisons.append(
        {
            "method": "Fusion-LoRA rank 16",
            "validation_mAP_mean": fusion_mean,
            "validation_mAP_sample_std": aggregate["coco/bbox_mAP"]["sample_std"],
            "fusion_lora_minus_method_mean_mAP": 0.0,
        }
    )

    summary = {
        "created_local": datetime.now().astimezone().isoformat(),
        "selection_protocol": (
            "Rank 16 was fixed by the earlier seed-42 validation-only rank ablation. "
            "For each seed, its checkpoint was fixed by validation mAP and independently "
            "reloaded. This summary was generated before rank-16 seeds 0/21 held-out tests."
        ),
        "status": "three-seed validation gate complete; held-out tests pending",
        "seeds": list(SEEDS),
        "standard_deviation": "sample (ddof=1)",
        "confidence_interval_note": (
            "Normal-approximation 95% intervals are descriptive only because n=3."
        ),
        "runs": runs,
        "aggregate": aggregate,
        "validation_comparison": comparisons,
    }

    json_path = ROOT / "results" / "json" / "fusion_lora_multiseed_validation_summary.json"
    runs_path = ROOT / "results" / "csv" / "fusion_lora_multiseed_validation_runs.csv"
    aggregate_path = ROOT / "results" / "csv" / "fusion_lora_multiseed_validation_aggregate.csv"
    epochs_path = ROOT / "results" / "csv" / "fusion_lora_multiseed_validation_epochs.csv"
    figure_dir = ROOT / "results" / "figures"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    run_fields = [key for key in runs[0] if key not in ("validation_metrics", "epoch_metrics")]
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in run_fields} for row in runs)

    aggregate_rows = []
    for metric, values in aggregate.items():
        aggregate_rows.append({"metric": metric, **values})
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    epoch_rows = [epoch for run in runs for epoch in run["epoch_metrics"]]
    with epochs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    colors = {0: "#4c78a8", 21: "#f58518", 42: "#54a24b"}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    for run in runs:
        axes[0].plot(
            [row["epoch"] for row in run["epoch_metrics"]],
            [row["coco/bbox_mAP"] for row in run["epoch_metrics"]],
            marker="o", markersize=3.5, linewidth=2,
            color=colors[run["seed"]], label="seed {}".format(run["seed"]),
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation COCO bbox mAP")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_ylim(0.38, 0.55)
    axes[0].set_title("Rank-16 validation trajectories")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = ["Full", "Frozen text", "Fusion-LoRA r16"]
    comparison_map = {row["method"]: row for row in comparisons}
    ordered = [
        comparison_map["Full fine-tuning"],
        comparison_map["Frozen text encoder"],
        comparison_map["Fusion-LoRA rank 16"],
    ]
    values = [row["validation_mAP_mean"] for row in ordered]
    errors = [row["validation_mAP_sample_std"] for row in ordered]
    x = np.arange(len(labels))
    bars = axes[1].bar(x, values, yerr=errors, capsize=5, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Validation COCO bbox mAP")
    axes[1].set_ylim(0.51, 0.56)
    axes[1].set_title("Three-seed mean and sample SD")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value, error in zip(bars, values, errors):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + error + 0.001,
            "{:.3f}".format(value),
            ha="center",
            va="bottom",
        )

    fig.suptitle("Pothole multi-seed validation comparison")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        path = figure_dir / "fusion_lora_multiseed_validation.{}".format(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(str(path), **kwargs)
    plt.close(fig)
    print("summary={}".format(json_path))
    print("validation_mAP={:.6f}+/-{:.6f}".format(
        aggregate["coco/bbox_mAP"]["mean"],
        aggregate["coco/bbox_mAP"]["sample_std"],
    ))


if __name__ == "__main__":
    main()
