#!/usr/bin/env python
"""Aggregate the completed Aquarium Full/Frozen runs over seeds 0, 21, 42."""

import csv
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
)
RUNS = (
    {
        "method": "Full fine-tuning",
        "slug": "full_finetune",
        "seed": 0,
        "work_dir": "experiments/odinw_aquarium_full_finetune_seed0",
        "train_prefix": "odinw_aquarium_full_finetune_seed0_s0_",
        "eval_prefix": "odinw_aquarium_full_finetune_seed0_best_eval_s0_",
        "test_prefix": "odinw_aquarium_full_finetune_seed0_test_s0_",
    },
    {
        "method": "Full fine-tuning",
        "slug": "full_finetune",
        "seed": 21,
        "work_dir": "experiments/odinw_aquarium_full_finetune_seed21",
        "train_prefix": "odinw_aquarium_full_finetune_seed21_s21_",
        "eval_prefix": "odinw_aquarium_full_finetune_seed21_best_eval_s21_",
        "test_prefix": "odinw_aquarium_full_finetune_seed21_test_s21_",
    },
    {
        "method": "Full fine-tuning",
        "slug": "full_finetune",
        "seed": 42,
        "work_dir": "experiments/odinw_aquarium_full_finetune",
        "train_prefix": "odinw_aquarium_full_finetune_s42_",
        "eval_prefix": "odinw_aquarium_full_finetune_best_eval_s42_",
        "test_prefix": "odinw_aquarium_full_finetune_test_s42_",
    },
    {
        "method": "Frozen text encoder",
        "slug": "frozen_text",
        "seed": 0,
        "work_dir": "experiments/odinw_aquarium_frozen_text_seed0",
        "train_prefix": "odinw_aquarium_frozen_text_seed0_s0_",
        "eval_prefix": "odinw_aquarium_frozen_text_seed0_best_eval_s0_",
        "test_prefix": "odinw_aquarium_frozen_text_seed0_test_s0_",
    },
    {
        "method": "Frozen text encoder",
        "slug": "frozen_text",
        "seed": 21,
        "work_dir": "experiments/odinw_aquarium_frozen_text_seed21",
        "train_prefix": "odinw_aquarium_frozen_text_seed21_s21_",
        "eval_prefix": "odinw_aquarium_frozen_text_seed21_best_eval_s21_",
        "test_prefix": "odinw_aquarium_frozen_text_seed21_test_s21_",
    },
    {
        "method": "Frozen text encoder",
        "slug": "frozen_text",
        "seed": 42,
        "work_dir": "experiments/odinw_aquarium_frozen_text",
        "train_prefix": "odinw_aquarium_frozen_text_s42_",
        "eval_prefix": "odinw_aquarium_frozen_text_best_eval_s42_",
        "test_prefix": "odinw_aquarium_frozen_text_test_s42_",
    },
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_matching(directory: Path, pattern: str, completed=False) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime)
    if completed:
        candidates = [
            path for path in candidates if load_json(path).get("status") == "completed"
        ]
    if not candidates:
        raise FileNotFoundError("{} in {}".format(pattern, directory))
    return candidates[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> Dict:
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
    result = {
        "trainable_parameters": 0,
        "total_parameters": 0,
        "trainable_ratio": 0.0,
        "gradient_audits": 0,
        "max_nonfinite_gradient_tensors": 0,
        "max_gradient_global_norm": 0.0,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
        "epochs": [],
    }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = trainable_pattern.search(line)
            if match:
                result["trainable_parameters"] = int(match.group(1))
                result["total_parameters"] = int(match.group(2))
                result["trainable_ratio"] = float(match.group(3))
            match = audit_pattern.search(line)
            if match:
                result["gradient_audits"] += 1
                result["max_nonfinite_gradient_tensors"] = max(
                    result["max_nonfinite_gradient_tensors"], int(match.group(4))
                )
                result["max_gradient_global_norm"] = max(
                    result["max_gradient_global_norm"], float(match.group(5))
                )
                result["peak_cuda_allocated_bytes"] = max(
                    result["peak_cuda_allocated_bytes"], int(match.group(6))
                )
                result["peak_cuda_reserved_bytes"] = max(
                    result["peak_cuda_reserved_bytes"], int(match.group(7))
                )
            match = val_pattern.search(line)
            if match:
                row = {"epoch": int(match.group(1))}
                row.update(
                    {
                        key: float(value)
                        for key, value in metric_pattern.findall(match.group(2))
                    }
                )
                if all(metric in row for metric in METRICS):
                    result["epochs"].append(row)
    if len(result["epochs"]) != 12:
        raise RuntimeError(
            "Expected 12 validation epochs in {}, found {}".format(
                path, len(result["epochs"])
            )
        )
    if not result["trainable_parameters"]:
        raise RuntimeError("Missing trainable parameter audit in {}".format(path))
    return result


def elapsed_minutes(manifest: Dict) -> float:
    start = datetime.fromisoformat(manifest["created_utc"])
    end = datetime.fromisoformat(manifest["completed_utc"])
    return (end - start).total_seconds() / 60.0


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_run(spec: Dict) -> Dict:
    work_dir = ROOT / spec["work_dir"]
    run_dirs = sorted(
        (
            path
            for path in work_dir.iterdir()
            if path.is_dir() and list(path.glob("*.log"))
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError("No training run in {}".format(work_dir))
    log_path = sorted(run_dirs[-1].glob("*.log"))[-1]
    parsed = parse_log(log_path)
    best_logged = max(parsed["epochs"], key=lambda row: row["coco/bbox_mAP"])
    manifest_path = latest_matching(
        ROOT / "experiments/manifests", spec["train_prefix"] + "*.json", completed=True
    )
    eval_path = latest_matching(
        ROOT / "experiments/raw_metrics", spec["eval_prefix"] + "*.json"
    )
    test_path = latest_matching(
        ROOT / "experiments/raw_metrics", spec["test_prefix"] + "*.json"
    )
    checkpoint = latest_matching(work_dir, "best_coco_bbox_mAP_epoch_*.pth")
    manifest = load_json(manifest_path)
    val_metrics = load_json(eval_path)["metrics"]
    test_metrics = load_json(test_path)["metrics"]
    if abs(float(val_metrics["coco/bbox_mAP"]) - best_logged["coco/bbox_mAP"]) > 1e-12:
        raise RuntimeError(
            "{} seed {} best reload differs from training log".format(
                spec["method"], spec["seed"]
            )
        )
    return {
        "method": spec["method"],
        "seed": spec["seed"],
        "best_epoch": best_logged["epoch"],
        "trainable_parameters": parsed["trainable_parameters"],
        "total_parameters": parsed["total_parameters"],
        "trainable_ratio_percent": 100.0 * parsed["trainable_ratio"],
        "gradient_audits": parsed["gradient_audits"],
        "max_nonfinite_gradient_tensors": parsed[
            "max_nonfinite_gradient_tensors"
        ],
        "max_gradient_global_norm": parsed["max_gradient_global_norm"],
        "peak_cuda_allocated_gib": parsed["peak_cuda_allocated_bytes"] / (1024 ** 3),
        "peak_cuda_reserved_gib": parsed["peak_cuda_reserved_bytes"] / (1024 ** 3),
        "elapsed_minutes": elapsed_minutes(manifest),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest": str(manifest_path.relative_to(ROOT)),
        "training_log": str(log_path.relative_to(ROOT)),
        "independent_validation_json": str(eval_path.relative_to(ROOT)),
        "held_out_test_json": str(test_path.relative_to(ROOT)),
        **{metric: float(val_metrics[metric]) for metric in METRICS},
        **{
            "test/" + metric: float(test_metrics[metric])
            for metric in METRICS
        },
    }


def aggregate_runs(rows: List[Dict]) -> List[Dict]:
    aggregate = []
    numeric_fields = (
        *METRICS,
        *("test/" + metric for metric in METRICS),
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        "elapsed_minutes",
    )
    for method in ("Full fine-tuning", "Frozen text encoder"):
        method_rows = [row for row in rows if row["method"] == method]
        if [row["seed"] for row in method_rows] != [0, 21, 42]:
            raise RuntimeError("Unexpected seed set for {}".format(method))
        item = {"method": method, "n": len(method_rows), "seeds": "0;21;42"}
        for field in numeric_fields:
            values = [float(row[field]) for row in method_rows]
            item[field + "_mean"] = statistics.mean(values)
            item[field + "_std"] = statistics.stdev(values)
        aggregate.append(item)
    return aggregate


def render_plot(aggregate: List[Dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "results/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["method"] for row in aggregate]
    x = range(len(labels))
    val_mean = [row["coco/bbox_mAP_mean"] for row in aggregate]
    val_std = [row["coco/bbox_mAP_std"] for row in aggregate]
    test_mean = [row["test/coco/bbox_mAP_mean"] for row in aggregate]
    test_std = [row["test/coco/bbox_mAP_std"] for row in aggregate]
    fig, axis = plt.subplots(figsize=(7.0, 4.7))
    offsets = [-0.13, 0.13]
    axis.errorbar(
        [value + offsets[0] for value in x],
        val_mean,
        yerr=val_std,
        marker="o",
        capsize=5,
        linestyle="none",
        label="Validation",
    )
    axis.errorbar(
        [value + offsets[1] for value in x],
        test_mean,
        yerr=test_std,
        marker="s",
        capsize=5,
        linestyle="none",
        label="Held-out test",
    )
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels)
    axis.set(
        ylabel="COCO mAP (mean ± sample SD)",
        title="Aquarium baseline stability (seeds 0, 21, 42)",
        ylim=(0.52, 0.56),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "aquarium_multiseed_map.png", dpi=600)
    fig.savefig(figure_dir / "aquarium_multiseed_map.pdf")
    plt.close(fig)


def main() -> int:
    rows = [collect_run(spec) for spec in RUNS]
    aggregate = aggregate_runs(rows)
    full = {row["seed"]: row for row in rows if row["method"] == "Full fine-tuning"}
    frozen = {
        row["seed"]: row for row in rows if row["method"] == "Frozen text encoder"
    }
    paired_test_differences = {
        str(seed): frozen[seed]["test/coco/bbox_mAP"]
        - full[seed]["test/coco/bbox_mAP"]
        for seed in (0, 21, 42)
    }
    paired_values = list(paired_test_differences.values())
    payload = {
        "dataset": "ODinW-13 Aquarium",
        "seeds": [0, 21, 42],
        "standard_deviation": "sample (ddof=1)",
        "runs": rows,
        "aggregate": aggregate,
        "paired_test_mAP_frozen_minus_full": {
            "by_seed": paired_test_differences,
            "mean": statistics.mean(paired_values),
            "std": statistics.stdev(paired_values),
        },
    }
    output = ROOT / "results/json/aquarium_multiseed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(ROOT / "results/csv/aquarium_multiseed_runs.csv", rows)
    write_csv(ROOT / "results/csv/aquarium_multiseed_aggregate.csv", aggregate)
    render_plot(aggregate)
    print("summary={}".format(output))
    for row in aggregate:
        print(
            "{}: val={:.4f}+/-{:.4f}, test={:.4f}+/-{:.4f}".format(
                row["method"],
                row["coco/bbox_mAP_mean"],
                row["coco/bbox_mAP_std"],
                row["test/coco/bbox_mAP_mean"],
                row["test/coco/bbox_mAP_std"],
            )
        )
    print(
        "paired frozen-full test mAP: {:.4f}+/-{:.4f}".format(
            payload["paired_test_mAP_frozen_minus_full"]["mean"],
            payload["paired_test_mAP_frozen_minus_full"]["std"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
