#!/usr/bin/env python
"""Aggregate the validation-only Pothole Fusion-LoRA rank ablation."""

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
RANKS = (4, 8, 16)


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


def count_visualizations(path):
    suffixes = {".jpg", ".jpeg", ".png"}
    return sum(
        item.is_file() and item.suffix.lower() in suffixes
        for item in path.rglob("*")
    )


def real_manifest(rank):
    candidates = []
    pattern = "odinw_pothole_fusion_lora_r{}_seed42_s42_*.json".format(rank)
    for path in MANIFEST_DIR.glob(pattern):
        payload = load_json(path)
        if payload.get("returncode") == 0 and payload.get("completed_utc"):
            candidates.append((path, payload))
    if not candidates:
        raise FileNotFoundError("Missing completed rank-{} manifest".format(rank))
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def parse_rank(rank):
    run_name = "odinw_pothole_fusion_lora_r{}_seed42".format(rank)
    work_dir = ROOT / "experiments" / run_name
    train_log = latest(work_dir.rglob("*.log"), "rank-{} training log".format(rank))
    text = train_log.read_text(encoding="utf-8", errors="replace")

    val_pattern = re.compile(
        r"Epoch\(val\) \[(\d+)\].*?"
        r"coco/bbox_mAP: ([0-9.]+).*?"
        r"coco/bbox_mAP_50: ([0-9.]+).*?"
        r"coco/bbox_mAP_75: ([0-9.]+).*?"
        r"coco/bbox_mAP_s: ([0-9.]+).*?"
        r"coco/bbox_mAP_m: ([0-9.]+).*?"
        r"coco/bbox_mAP_l: ([0-9.]+)"
    )
    epochs = []
    for match in val_pattern.finditer(text):
        metrics = [float(value) for value in match.groups()[1:]]
        epochs.append(
            {
                "rank": rank,
                "epoch": int(match.group(1)),
                "coco/bbox_mAP": metrics[0],
                "coco/bbox_mAP_50": metrics[1],
                "coco/bbox_mAP_75": metrics[2],
                "coco/bbox_mAP_s": metrics[3],
                "coco/bbox_mAP_m": metrics[4],
                "coco/bbox_mAP_l": metrics[5],
            }
        )
    if len(epochs) != 12:
        raise RuntimeError(
            "Expected 12 validation epochs for rank {}, found {}".format(
                rank, len(epochs)
            )
        )
    best = max(epochs, key=lambda row: row["coco/bbox_mAP"])

    trainable_match = re.search(
        r"SetTrainableModulesHook: trainable=(\d+) total=(\d+) ratio=([0-9.]+)",
        text,
    )
    if not trainable_match:
        raise RuntimeError("Missing rank-{} parameter audit".format(rank))
    trainable = int(trainable_match.group(1))
    total = int(trainable_match.group(2))

    audit_pattern = re.compile(
        r"gradient_audit iter=(\d+) tensors=(\d+) nonzero=(\d+) "
        r"nonfinite=(\d+) global_norm=([0-9.eE+-]+) "
        r"cuda_peak_allocated_bytes=(\d+) cuda_peak_reserved_bytes=(\d+)"
    )
    audits = [
        {
            "iteration": int(match.group(1)),
            "tensors": int(match.group(2)),
            "nonzero": int(match.group(3)),
            "nonfinite": int(match.group(4)),
            "global_norm": float(match.group(5)),
            "cuda_peak_allocated_bytes": int(match.group(6)),
            "cuda_peak_reserved_bytes": int(match.group(7)),
        }
        for match in audit_pattern.finditer(text)
    ]
    if len(audits) != 56:
        raise RuntimeError(
            "Expected 56 gradient audits for rank {}, found {}".format(
                rank, len(audits)
            )
        )
    if max(row["nonfinite"] for row in audits) != 0:
        raise RuntimeError("Non-finite rank-{} gradient detected".format(rank))

    checkpoint = work_dir / "best_coco_bbox_mAP_epoch_{}.pth".format(
        best["epoch"]
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    validation_path = latest(
        RAW_DIR.glob("{}_best_eval_s42_*.json".format(run_name)),
        "rank-{} independent validation JSON".format(rank),
    )
    validation = load_json(validation_path)["metrics"]
    if validation["coco/bbox_mAP"] != best["coco/bbox_mAP"]:
        raise RuntimeError(
            "Rank-{} independent validation does not reproduce best mAP".format(
                rank
            )
        )

    manifest_path, manifest = real_manifest(rank)
    started = datetime.fromisoformat(manifest["created_utc"])
    completed = datetime.fromisoformat(manifest["completed_utc"])
    elapsed_minutes = (completed - started).total_seconds() / 60.0
    visualization_dir = (
        ROOT / "experiments" / "{}_best_eval".format(run_name) / "visualizations"
    )

    return {
        "rank": rank,
        "seed": 42,
        "best_epoch": best["epoch"],
        "validation_mAP": validation["coco/bbox_mAP"],
        "final_validation_mAP": epochs[-1]["coco/bbox_mAP"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio_percent": 100.0 * trainable / total,
        "gradient_audits": len(audits),
        "max_nonfinite_gradient_tensors": max(row["nonfinite"] for row in audits),
        "max_gradient_global_norm": max(row["global_norm"] for row in audits),
        "peak_cuda_allocated_gib": max(
            row["cuda_peak_allocated_bytes"] for row in audits
        ) / (1024 ** 3),
        "peak_cuda_reserved_gib": max(
            row["cuda_peak_reserved_bytes"] for row in audits
        ) / (1024 ** 3),
        "elapsed_minutes": elapsed_minutes,
        "validation_visualizations": count_visualizations(visualization_dir),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest": str(manifest_path.relative_to(ROOT)),
        "training_log": str(train_log.relative_to(ROOT)),
        "independent_validation_json": str(validation_path.relative_to(ROOT)),
        "validation_metrics": validation,
        "epoch_metrics": epochs,
    }


def main():
    runs = [parse_rank(rank) for rank in RANKS]
    selected = max(runs, key=lambda row: row["validation_mAP"])
    summary = {
        "created_local": datetime.now().astimezone().isoformat(),
        "selection_protocol": (
            "Select rank only by Pothole validation COCO bbox mAP at seed 42. "
            "Held-out test metrics are excluded from every rank comparison and "
            "were not run for ranks 4 or 16."
        ),
        "status": "single-seed validation-only rank ablation",
        "selected_rank": selected["rank"],
        "selected_validation_mAP": selected["validation_mAP"],
        "runs": runs,
    }

    json_path = ROOT / "results" / "json" / "fusion_lora_rank_ablation_summary.json"
    runs_path = ROOT / "results" / "csv" / "fusion_lora_rank_ablation_runs.csv"
    epochs_path = ROOT / "results" / "csv" / "fusion_lora_rank_ablation_epochs.csv"
    figure_dir = ROOT / "results" / "figures"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    run_fields = [
        "rank",
        "seed",
        "best_epoch",
        "validation_mAP",
        "final_validation_mAP",
        "trainable_parameters",
        "total_parameters",
        "trainable_ratio_percent",
        "gradient_audits",
        "max_nonfinite_gradient_tensors",
        "max_gradient_global_norm",
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        "elapsed_minutes",
        "validation_visualizations",
        "checkpoint",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "training_manifest",
        "training_log",
        "independent_validation_json",
    ]
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in run_fields} for row in runs)

    epoch_rows = [epoch for row in runs for epoch in row["epoch_metrics"]]
    with epochs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    colors = {4: "#4c78a8", 8: "#f58518", 16: "#54a24b"}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    for row in runs:
        axes[0].plot(
            [epoch["epoch"] for epoch in row["epoch_metrics"]],
            [epoch["coco/bbox_mAP"] for epoch in row["epoch_metrics"]],
            marker="o",
            markersize=3.5,
            linewidth=2,
            color=colors[row["rank"]],
            label="rank {}".format(row["rank"]),
        )
        axes[0].scatter(
            row["best_epoch"],
            row["validation_mAP"],
            s=65,
            color=colors[row["rank"]],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation COCO bbox mAP")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_ylim(0.28, 0.55)
    axes[0].set_title("Validation trajectories (seed 42)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    ranks = [row["rank"] for row in runs]
    validation = [row["validation_mAP"] for row in runs]
    ratios = [row["trainable_ratio_percent"] for row in runs]
    x = np.arange(len(ranks))
    bars = axes[1].bar(x, validation, color=[colors[rank] for rank in ranks])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["rank {}".format(rank) for rank in ranks])
    axes[1].set_ylabel("Best validation COCO bbox mAP")
    axes[1].set_ylim(0.49, 0.54)
    axes[1].set_title("Accuracy and trainable-parameter ratio")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, validation):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            "{:.3f}".format(value),
            ha="center",
            va="bottom",
        )
    ratio_axis = axes[1].twinx()
    ratio_axis.plot(x, ratios, "D--", color="#b279a2", linewidth=1.8)
    ratio_axis.set_ylabel("Trainable parameters (%)", color="#8f5c88")
    ratio_axis.tick_params(axis="y", labelcolor="#8f5c88")
    ratio_axis.set_ylim(0, max(ratios) * 1.35)

    fig.suptitle("Pothole Fusion-LoRA rank ablation: validation-only selection")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        path = figure_dir / "fusion_lora_rank_ablation.{}".format(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(str(path), **kwargs)
    plt.close(fig)

    print("summary={}".format(json_path))
    print("runs={}".format(runs_path))
    print("epochs={}".format(epochs_path))
    print(
        "selected_rank={} validation_mAP={:.3f}".format(
            selected["rank"], selected["validation_mAP"]
        )
    )


if __name__ == "__main__":
    main()
