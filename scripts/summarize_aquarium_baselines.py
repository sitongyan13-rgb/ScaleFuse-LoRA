#!/usr/bin/env python
"""Summarize the completed Aquarium baseline runs without hand transcription."""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    {
        "method": "Full fine-tuning",
        "slug": "full_finetune",
        "work_dir": "experiments/odinw_aquarium_full_finetune",
        "manifest_prefix": "odinw_aquarium_full_finetune_s42_",
        "eval_prefix": "odinw_aquarium_full_finetune_best_eval_s42_",
        "test_prefix": "odinw_aquarium_full_finetune_test_s42_",
    },
    {
        "method": "Frozen text encoder",
        "slug": "frozen_text",
        "work_dir": "experiments/odinw_aquarium_frozen_text",
        "manifest_prefix": "odinw_aquarium_frozen_text_s42_",
        "eval_prefix": "odinw_aquarium_frozen_text_best_eval_s42_",
        "test_prefix": "odinw_aquarium_frozen_text_test_s42_",
    },
    {
        "method": "Residual adapters",
        "slug": "adapter",
        "work_dir": "experiments/odinw_aquarium_adapter",
        "manifest_prefix": "odinw_aquarium_adapter_s42_",
        "eval_prefix": "odinw_aquarium_adapter_best_eval_s42_",
        "test_prefix": "odinw_aquarium_adapter_test_s42_",
    },
)
METRICS = (
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError("{} in {}".format(pattern, directory))
    return matches[-1]


def discover_run_files(
    run: Dict[str, str]
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    work_dir = ROOT / run["work_dir"]
    run_dirs = sorted(
        (
            item
            for item in work_dir.iterdir()
            if item.is_dir() and (item / "vis_data" / "scalars.json").is_file()
        ),
        key=lambda item: item.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError("No completed run directory in {}".format(work_dir))
    run_dir = run_dirs[-1]
    log_path = latest_file(run_dir, "*.log")
    scalars_path = run_dir / "vis_data" / "scalars.json"
    manifest_path = latest_file(
        ROOT / "experiments" / "manifests",
        run["manifest_prefix"] + "*.json",
    )
    eval_path = latest_file(
        ROOT / "experiments" / "raw_metrics",
        run["eval_prefix"] + "*.json",
    )
    test_path = latest_file(
        ROOT / "experiments" / "raw_metrics",
        run["test_prefix"] + "*.json",
    )
    checkpoint_path = latest_file(work_dir, "best_coco_bbox_mAP_epoch_*.pth")
    return (
        log_path,
        scalars_path,
        manifest_path,
        eval_path,
        test_path,
        checkpoint_path,
    )


def parse_training_log(path: Path) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    validation_rows: List[Dict[str, float]] = []
    audit: Dict[str, float] = {
        "trainable_parameters": 0,
        "total_parameters": 0,
        "trainable_ratio": 0.0,
        "gradient_audits": 0,
        "max_nonfinite_gradient_tensors": 0,
        "max_gradient_global_norm": 0.0,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
    }
    trainable_pattern = re.compile(
        r"SetTrainableModulesHook: trainable=(\d+) total=(\d+) ratio=([0-9.]+)"
    )
    audit_pattern = re.compile(
        r"gradient_audit iter=(\d+) tensors=(\d+) nonzero=(\d+) "
        r"nonfinite=(\d+) global_norm=([0-9.eE+-]+) "
        r"cuda_peak_allocated_bytes=(\d+) cuda_peak_reserved_bytes=(\d+)"
    )
    val_pattern = re.compile(r"Epoch\(val\) \[(\d+)\]\[127/127\](.*)")
    metric_pattern = re.compile(r"(coco/bbox_mAP(?:_50|_75|_s|_m|_l)?): ([0-9.-]+)")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            trainable_match = trainable_pattern.search(line)
            if trainable_match:
                audit["trainable_parameters"] = int(trainable_match.group(1))
                audit["total_parameters"] = int(trainable_match.group(2))
                audit["trainable_ratio"] = float(trainable_match.group(3))
            audit_match = audit_pattern.search(line)
            if audit_match:
                audit["gradient_audits"] += 1
                audit["max_nonfinite_gradient_tensors"] = max(
                    audit["max_nonfinite_gradient_tensors"],
                    int(audit_match.group(4)),
                )
                audit["max_gradient_global_norm"] = max(
                    audit["max_gradient_global_norm"],
                    float(audit_match.group(5)),
                )
                audit["peak_cuda_allocated_bytes"] = max(
                    audit["peak_cuda_allocated_bytes"],
                    int(audit_match.group(6)),
                )
                audit["peak_cuda_reserved_bytes"] = max(
                    audit["peak_cuda_reserved_bytes"],
                    int(audit_match.group(7)),
                )
            val_match = val_pattern.search(line)
            if val_match:
                row: Dict[str, float] = {"epoch": int(val_match.group(1))}
                for name, value in metric_pattern.findall(val_match.group(2)):
                    row[name] = float(value)
                if all(name in row for name in METRICS):
                    validation_rows.append(row)
    if len(validation_rows) != 12:
        raise RuntimeError(
            "Expected 12 validation epochs in {}, found {}".format(
                path, len(validation_rows)
            )
        )
    if not audit["trainable_parameters"]:
        raise RuntimeError("Missing trainable parameter audit in {}".format(path))
    return validation_rows, audit


def load_losses(path: Path) -> List[Dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if "iter" in item and "loss" in item and "epoch" in item:
                rows.append(
                    {
                        "iteration": int(item["iter"]),
                        "epoch": int(item["epoch"]),
                        "loss": float(item["loss"]),
                    }
                )
    if not rows:
        raise RuntimeError("No training losses in {}".format(path))
    return rows


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def duration_seconds(manifest: Dict) -> Optional[float]:
    if "completed_utc" not in manifest:
        return None
    start = datetime.fromisoformat(manifest["created_utc"])
    end = datetime.fromisoformat(manifest["completed_utc"])
    return (end - start).total_seconds()


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_figures(
    epoch_rows: List[Dict],
    losses_by_method: Dict[str, List[Dict]],
    summary: List[Dict],
    zero_validation: Dict[str, float],
    zero_test: Dict[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Full fine-tuning": "#1f77b4",
        "Frozen text encoder": "#2ca02c",
        "Residual adapters": "#d62728",
    }

    fig, axis = plt.subplots(figsize=(7.4, 4.7))
    for method, color in colors.items():
        rows = [row for row in epoch_rows if row["method"] == method]
        axis.plot(
            [row["epoch"] for row in rows],
            [row["coco/bbox_mAP"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=method,
            color=color,
        )
    axis.axhline(
        zero_validation["coco/bbox_mAP"],
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label="Zero-shot",
    )
    axis.set(xlabel="Epoch", ylabel="COCO mAP", title="Aquarium validation mAP")
    axis.set_xticks(range(1, 13))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "aquarium_validation_map.png", dpi=600)
    fig.savefig(figure_dir / "aquarium_validation_map.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 4.7))
    for method, rows in losses_by_method.items():
        axis.plot(
            [row["iteration"] for row in rows],
            [row["loss"] for row in rows],
            linewidth=1.2,
            label=method,
            color=colors[method],
            alpha=0.9,
        )
    axis.set(
        xlabel="Iteration",
        ylabel="Logged total loss",
        title="Aquarium training loss",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "aquarium_training_loss.png", dpi=600)
    fig.savefig(figure_dir / "aquarium_training_loss.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 4.7))
    label_offsets = {
        "Full fine-tuning": (-6, 10),
        "Frozen text encoder": (-6, -20),
        "Residual adapters": (6, 5),
    }
    for row in summary:
        method = row["method"]
        axis.scatter(
            row["trainable_ratio_percent"],
            row["coco/bbox_mAP"],
            s=65,
            color=colors[method],
        )
        axis.annotate(
            method,
            (row["trainable_ratio_percent"], row["coco/bbox_mAP"]),
            xytext=label_offsets[method],
            textcoords="offset points",
            fontsize=8,
            horizontalalignment=(
                "right" if method != "Residual adapters" else "left"
            ),
        )
    axis.set_xscale("log")
    axis.set(
        xlabel="Trainable parameters (%) — log scale",
        ylabel="Best validation mAP",
        title="Accuracy–parameter trade-off",
    )
    axis.grid(alpha=0.25)
    axis.set_ylim(0.295, 0.56)
    fig.tight_layout()
    fig.savefig(figure_dir / "aquarium_accuracy_parameter_tradeoff.png", dpi=600)
    fig.savefig(figure_dir / "aquarium_accuracy_parameter_tradeoff.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.6, 4.7))
    methods = ["Zero-shot"] + [row["method"] for row in summary]
    metric_names = ("coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75")
    metric_labels = ("mAP", "AP50", "AP75")
    x_positions = list(range(len(methods)))
    width = 0.23
    for metric_index, (metric_name, metric_label) in enumerate(
        zip(metric_names, metric_labels)
    ):
        values = [zero_test[metric_name]] + [
            row["test/" + metric_name] for row in summary
        ]
        offsets = [
            position + (metric_index - 1) * width for position in x_positions
        ]
        axis.bar(offsets, values, width=width, label=metric_label)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(methods, rotation=12, horizontalalignment="right")
    axis.set(
        ylabel="COCO metric",
        title="Aquarium held-out test results",
        ylim=(0.0, 0.9),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "aquarium_test_metrics.png", dpi=600)
    fig.savefig(figure_dir / "aquarium_test_metrics.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/json/aquarium_baseline_summary.json"),
    )
    args = parser.parse_args()

    epoch_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    losses_by_method: Dict[str, List[Dict]] = {}
    provenance: Dict[str, Dict] = {}
    for run in RUNS:
        (
            log_path,
            scalars_path,
            manifest_path,
            eval_path,
            test_path,
            checkpoint_path,
        ) = discover_run_files(run)
        validation, audit = parse_training_log(log_path)
        losses = load_losses(scalars_path)
        manifest = load_json(manifest_path)
        eval_payload = load_json(eval_path)
        test_payload = load_json(test_path)
        metrics = eval_payload["metrics"]
        test_metrics = test_payload["metrics"]
        best_row = max(validation, key=lambda row: row["coco/bbox_mAP"])
        for row in validation:
            epoch_rows.append({"method": run["method"], **row})
        losses_by_method[run["method"]] = losses
        summary = {
            "method": run["method"],
            "best_epoch": int(best_row["epoch"]),
            "trainable_parameters": int(audit["trainable_parameters"]),
            "total_parameters": int(audit["total_parameters"]),
            "trainable_ratio_percent": 100.0 * float(audit["trainable_ratio"]),
            "gradient_audits": int(audit["gradient_audits"]),
            "max_nonfinite_gradient_tensors": int(
                audit["max_nonfinite_gradient_tensors"]
            ),
            "max_gradient_global_norm": float(audit["max_gradient_global_norm"]),
            "peak_cuda_allocated_gib": float(audit["peak_cuda_allocated_bytes"])
            / (1024 ** 3),
            "peak_cuda_reserved_gib": float(audit["peak_cuda_reserved_bytes"])
            / (1024 ** 3),
            "elapsed_minutes": float(duration_seconds(manifest) or 0.0) / 60.0,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "independent_eval_json": str(eval_path.relative_to(ROOT)),
            **{name: float(metrics[name]) for name in METRICS},
            **{
                "test/" + name: float(test_metrics[name])
                for name in METRICS
            },
        }
        if abs(summary["coco/bbox_mAP"] - best_row["coco/bbox_mAP"]) > 1e-12:
            raise RuntimeError(
                "{} independent mAP does not match logged best".format(run["method"])
            )
        summary_rows.append(summary)
        provenance[run["slug"]] = {
            "log": str(log_path.relative_to(ROOT)),
            "scalars": str(scalars_path.relative_to(ROOT)),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "independent_eval": str(eval_path.relative_to(ROOT)),
            "held_out_test": str(test_path.relative_to(ROOT)),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        }

    zero_payload = load_json(
        ROOT
        / "experiments"
        / "raw_metrics"
        / "odinw13_zero_shot_s42_20260730T010615Z.json"
    )
    zero_metrics = {
        name: float(zero_payload["metrics"]["Aquarium/" + name])
        for name in METRICS
    }
    zero_test_path = latest_file(
        ROOT / "experiments" / "raw_metrics",
        "odinw_aquarium_zero_shot_test_s42_*.json",
    )
    zero_test_payload = load_json(zero_test_path)
    zero_test_metrics = {
        name: float(zero_test_payload["metrics"][name]) for name in METRICS
    }
    output = {
        "dataset": "ODinW-13 Aquarium validation and held-out test",
        "seed": 42,
        "zero_shot_validation": zero_metrics,
        "zero_shot_test": zero_test_metrics,
        "baselines": summary_rows,
        "epochs": epoch_rows,
        "provenance": provenance,
    }
    output_path = args.output_json
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    write_csv(ROOT / "results" / "csv" / "aquarium_baseline_summary.csv", summary_rows)
    write_csv(ROOT / "results" / "csv" / "aquarium_epoch_metrics.csv", epoch_rows)
    render_figures(
        epoch_rows,
        losses_by_method,
        summary_rows,
        zero_metrics,
        zero_test_metrics,
    )
    print("summary={}".format(output_path))
    print("methods={} epochs={}".format(len(summary_rows), len(epoch_rows)))
    for row in summary_rows:
        print(
            "{}: best_epoch={} val_mAP={:.3f} test_mAP={:.3f} "
            "trainable={:.6f}% elapsed={:.2f}min "
            "peak_reserved={:.3f}GiB".format(
                row["method"],
                row["best_epoch"],
                row["coco/bbox_mAP"],
                row["test/coco/bbox_mAP"],
                row["trainable_ratio_percent"],
                row["elapsed_minutes"],
                row["peak_cuda_reserved_gib"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
