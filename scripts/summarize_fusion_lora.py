#!/usr/bin/env python
"""Aggregate the Pothole Fusion-LoRA screening run without manual metrics."""

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "experiments" / "odinw_pothole_fusion_lora_r8_seed42"
RAW_DIR = ROOT / "experiments" / "raw_metrics"
MANIFEST_DIR = ROOT / "experiments" / "manifests"


def latest(paths, description):
    candidates = list(paths)
    if not candidates:
        raise FileNotFoundError("Missing {}".format(description))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_visualizations(path):
    suffixes = {".jpg", ".jpeg", ".png"}
    return sum(
        1 for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in suffixes
    )


def main():
    train_log = latest(WORK_DIR.rglob("*.log"), "Fusion-LoRA training log")
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
        values = [float(value) for value in match.groups()[1:]]
        epochs.append(
            {
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
        raise RuntimeError("Expected 12 validation epochs, found {}".format(len(epochs)))
    best = max(epochs, key=lambda row: row["coco/bbox_mAP"])

    trainable_match = re.search(
        r"SetTrainableModulesHook: trainable=(\d+) total=(\d+) ratio=([0-9.]+)",
        text,
    )
    if not trainable_match:
        raise RuntimeError("Missing trainable-parameter audit")
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
        raise RuntimeError("Expected 56 gradient audits, found {}".format(len(audits)))
    if max(row["nonfinite"] for row in audits) != 0:
        raise RuntimeError("Non-finite gradient detected")

    checkpoint = WORK_DIR / "best_coco_bbox_mAP_epoch_{}.pth".format(
        best["epoch"]
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    validation_path = latest(
        RAW_DIR.glob("odinw_pothole_fusion_lora_r8_seed42_best_eval*.json"),
        "independent validation JSON",
    )
    test_path = latest(
        RAW_DIR.glob("odinw_pothole_fusion_lora_r8_seed42_test*.json"),
        "held-out test JSON",
    )
    validation = load_json(validation_path)["metrics"]
    test = load_json(test_path)["metrics"]
    if validation["coco/bbox_mAP"] != best["coco/bbox_mAP"]:
        raise RuntimeError("Independent validation does not reproduce best mAP")

    manifest_path = latest(
        MANIFEST_DIR.glob("odinw_pothole_fusion_lora_r8_seed42_s42_*.json"),
        "training manifest",
    )
    manifest = load_json(manifest_path)
    if manifest.get("returncode") != 0:
        raise RuntimeError("Training manifest did not complete successfully")
    started = datetime.fromisoformat(manifest["created_utc"])
    completed = datetime.fromisoformat(manifest["completed_utc"])
    elapsed_minutes = (completed - started).total_seconds() / 60.0

    lora_run = {
        "method": "Fusion-LoRA r=8",
        "seed": 42,
        "best_epoch": best["epoch"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio_percent": 100.0 * trainable / total,
        "gradient_audits": len(audits),
        "max_nonfinite_gradient_tensors": max(
            row["nonfinite"] for row in audits
        ),
        "max_gradient_global_norm": max(row["global_norm"] for row in audits),
        "peak_cuda_allocated_gib": max(
            row["cuda_peak_allocated_bytes"] for row in audits
        ) / (1024 ** 3),
        "peak_cuda_reserved_gib": max(
            row["cuda_peak_reserved_bytes"] for row in audits
        ) / (1024 ** 3),
        "elapsed_minutes": elapsed_minutes,
        "validation_visualizations": count_visualizations(
            ROOT
            / "experiments"
            / "odinw_pothole_fusion_lora_r8_seed42_best_eval"
            / "visualizations"
        ),
        "test_visualizations": count_visualizations(
            ROOT
            / "experiments"
            / "odinw_pothole_fusion_lora_r8_seed42_test"
            / "visualizations"
        ),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest": str(manifest_path.relative_to(ROOT)),
        "training_log": str(train_log.relative_to(ROOT)),
        "independent_validation_json": str(validation_path.relative_to(ROOT)),
        "held_out_test_json": str(test_path.relative_to(ROOT)),
        "validation_metrics": validation,
        "test_metrics": test,
        "epoch_metrics": epochs,
    }

    cross_task = load_json(
        ROOT / "results" / "json" / "cross_task_multiseed_summary.json"
    )
    baseline_runs = []
    for row in cross_task["runs"]:
        if row["dataset"] != "Pothole" or row["seed"] != 42:
            continue
        if row["method"] not in ("Full fine-tuning", "Frozen text encoder"):
            continue
        baseline_runs.append(row)
    if len(baseline_runs) != 2:
        raise RuntimeError("Expected Full/Frozen seed-42 Pothole baselines")

    comparison = []
    for method in ("Full fine-tuning", "Frozen text encoder"):
        row = next(item for item in baseline_runs if item["method"] == method)
        comparison.append(
            {
                "method": row["method"],
                "trainable_parameters": row["trainable_parameters"],
                "total_parameters": row["total_parameters"],
                "trainable_ratio_percent": row["trainable_ratio_percent"],
                "validation_mAP": row["coco/bbox_mAP"],
                "held_out_test_mAP": row["test/coco/bbox_mAP"],
                "peak_cuda_allocated_gib": row["peak_cuda_allocated_gib"],
                "peak_cuda_reserved_gib": row["peak_cuda_reserved_gib"],
                "elapsed_minutes": row["elapsed_minutes"],
            }
        )
    comparison.append(
        {
            "method": lora_run["method"],
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_ratio_percent": lora_run["trainable_ratio_percent"],
            "validation_mAP": validation["coco/bbox_mAP"],
            "held_out_test_mAP": test["coco/bbox_mAP"],
            "peak_cuda_allocated_gib": lora_run["peak_cuda_allocated_gib"],
            "peak_cuda_reserved_gib": lora_run["peak_cuda_reserved_gib"],
            "elapsed_minutes": elapsed_minutes,
        }
    )

    summary = {
        "created_utc": datetime.now().astimezone().isoformat(),
        "selection_protocol": (
            "Rank-8 checkpoint selected only by Pothole validation mAP; "
            "held-out test evaluated after selection."
        ),
        "screening_status": "single-seed descriptive result",
        "lora_run": lora_run,
        "comparison": comparison,
    }

    json_path = ROOT / "results" / "json" / "fusion_lora_screening_summary.json"
    csv_path = ROOT / "results" / "csv" / "fusion_lora_comparison.csv"
    figure_dir = ROOT / "results" / "figures"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

    labels = ["Full", "Frozen text", "Fusion-LoRA"]
    x = np.arange(len(labels))
    validation_values = [row["validation_mAP"] for row in comparison]
    test_values = [row["held_out_test_mAP"] for row in comparison]
    trainable_values = [row["trainable_ratio_percent"] for row in comparison]
    memory_values = [row["peak_cuda_allocated_gib"] for row in comparison]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.36
    axes[0].bar(x - width / 2, validation_values, width, label="Validation")
    axes[0].bar(x + width / 2, test_values, width, label="Held-out test")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("COCO mAP")
    axes[0].set_ylim(0.48, 0.58)
    axes[0].set_title("Pothole seed-42 accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    bars = axes[1].bar(x, trainable_values, color="#4c78a8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Trainable parameters (%)", color="#4c78a8")
    axes[1].tick_params(axis="y", labelcolor="#4c78a8")
    axes[1].set_title("Parameter and memory efficiency")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, trainable_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            "{:.3g}%".format(value),
            ha="center",
            va="bottom",
        )
    memory_axis = axes[1].twinx()
    memory_axis.plot(x, memory_values, "o-", color="#f58518", linewidth=2)
    memory_axis.set_ylabel("Peak CUDA allocated (GiB)", color="#f58518")
    memory_axis.tick_params(axis="y", labelcolor="#f58518")
    memory_axis.set_ylim(0, max(memory_values) * 1.25)

    fig.tight_layout()
    png_path = figure_dir / "fusion_lora_accuracy_efficiency.png"
    pdf_path = figure_dir / "fusion_lora_accuracy_efficiency.pdf"
    fig.savefig(str(png_path), dpi=220, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

    print("summary={}".format(json_path))
    print(
        "Fusion-LoRA r8 | val={:.3f} | test={:.3f} | trainable={:.4f}%".format(
            validation["coco/bbox_mAP"],
            test["coco/bbox_mAP"],
            lora_run["trainable_ratio_percent"],
        )
    )


if __name__ == "__main__":
    main()
