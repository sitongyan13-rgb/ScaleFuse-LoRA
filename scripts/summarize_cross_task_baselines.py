#!/usr/bin/env python
"""Aggregate auditable Pothole and thermalDogsAndPeople baseline runs."""

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
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
ZERO_SHOT_VAL = (
    ROOT / "experiments/raw_metrics/odinw13_zero_shot_s42_20260730T010615Z.json"
)
DATASETS = (
    {
        "dataset": "Pothole",
        "metric_prefix": "pothole",
        "val_images": 133,
        "test_images": 67,
        "zero_test_prefix": "odinw_pothole_zero_shot_test_s42_",
    },
    {
        "dataset": "thermalDogsAndPeople",
        "metric_prefix": "thermalDogsAndPeople",
        "val_images": 41,
        "test_images": 20,
        "zero_test_prefix": "odinw_thermal_dogs_and_people_zero_shot_test_s42_",
    },
)
RUNS = (
    {
        "dataset": "Pothole",
        "method": "Full fine-tuning",
        "seed": 42,
        "work_dir": "experiments/odinw_pothole_full_finetune",
        "train_prefix": "odinw_pothole_full_finetune_s42_",
        "eval_prefix": "odinw_pothole_full_finetune_best_eval_s42_",
        "test_prefix": "odinw_pothole_full_finetune_test_s42_",
        "val_images": 133,
        "test_images": 67,
    },
    {
        "dataset": "Pothole",
        "method": "Frozen text encoder",
        "seed": 42,
        "work_dir": "experiments/odinw_pothole_frozen_text",
        "train_prefix": "odinw_pothole_frozen_text_s42_",
        "eval_prefix": "odinw_pothole_frozen_text_best_eval_s42_",
        "test_prefix": "odinw_pothole_frozen_text_test_s42_",
        "val_images": 133,
        "test_images": 67,
    },
    {
        "dataset": "thermalDogsAndPeople",
        "method": "Full fine-tuning",
        "seed": 42,
        "work_dir": "experiments/odinw_thermal_dogs_and_people_full_finetune",
        "train_prefix": "odinw_thermal_dogs_and_people_full_finetune_s42_",
        "eval_prefix": (
            "odinw_thermal_dogs_and_people_full_finetune_best_eval_s42_"
        ),
        "test_prefix": "odinw_thermal_dogs_and_people_full_finetune_test_s42_",
        "val_images": 41,
        "test_images": 20,
    },
    {
        "dataset": "thermalDogsAndPeople",
        "method": "Frozen text encoder",
        "seed": 42,
        "work_dir": "experiments/odinw_thermal_dogs_and_people_frozen_text",
        "train_prefix": "odinw_thermal_dogs_and_people_frozen_text_s42_",
        "eval_prefix": (
            "odinw_thermal_dogs_and_people_frozen_text_best_eval_s42_"
        ),
        "test_prefix": "odinw_thermal_dogs_and_people_frozen_text_test_s42_",
        "val_images": 41,
        "test_images": 20,
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


def parse_log(path: Path, expected_val_images: int) -> Dict:
    trainable_pattern = re.compile(
        r"SetTrainableModulesHook: trainable=(\d+) total=(\d+) ratio=([0-9.]+)"
    )
    audit_pattern = re.compile(
        r"gradient_audit iter=(\d+) tensors=(\d+) nonzero=(\d+) "
        r"nonfinite=(\d+) global_norm=([0-9.eE+-]+) "
        r"cuda_peak_allocated_bytes=(\d+) cuda_peak_reserved_bytes=(\d+)"
    )
    val_pattern = re.compile(
        r"Epoch\(val\) \[(\d+)\]\[{0}/{0}\](.*)".format(expected_val_images)
    )
    metric_pattern = re.compile(
        r"(coco/bbox_mAP(?:_50|_75|_s|_m|_l)?): ([0-9.-]+)"
    )
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


def count_visualizations(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


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


def find_training_log(work_dir: Path) -> Path:
    candidates = []
    for path in work_dir.iterdir():
        if path.is_dir():
            candidates.extend(path.glob("*.log"))
    if not candidates:
        raise FileNotFoundError("No training log in {}".format(work_dir))
    return max(candidates, key=lambda item: item.stat().st_mtime)


def collect_run(spec: Dict) -> Dict:
    work_dir = ROOT / spec["work_dir"]
    log_path = find_training_log(work_dir)
    parsed = parse_log(log_path, spec["val_images"])
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
            "{} {} best reload differs from training log".format(
                spec["dataset"], spec["method"]
            )
        )
    seed = int(spec.get("seed", 42))
    seed_marker = "_s{}_".format(seed)
    val_visualizations = count_visualizations(
        ROOT
        / "experiments"
        / (
            spec["eval_prefix"]
            .replace(seed_marker, "")
            .rstrip("_")
        )
        / "visualizations"
    )
    test_visualizations = count_visualizations(
        ROOT
        / "experiments"
        / (
            spec["test_prefix"]
            .replace(seed_marker, "")
            .rstrip("_")
        )
        / "visualizations"
    )
    if val_visualizations != spec["val_images"]:
        raise RuntimeError(
            "{} {} validation visualization count {} != {}".format(
                spec["dataset"],
                spec["method"],
                val_visualizations,
                spec["val_images"],
            )
        )
    if test_visualizations != spec["test_images"]:
        raise RuntimeError(
            "{} {} test visualization count {} != {}".format(
                spec["dataset"],
                spec["method"],
                test_visualizations,
                spec["test_images"],
            )
        )
    row = {
        "dataset": spec["dataset"],
        "method": spec["method"],
        "seed": seed,
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
        "validation_visualizations": val_visualizations,
        "test_visualizations": test_visualizations,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest": str(manifest_path.relative_to(ROOT)),
        "training_log": str(log_path.relative_to(ROOT)),
        "independent_validation_json": str(eval_path.relative_to(ROOT)),
        "held_out_test_json": str(test_path.relative_to(ROOT)),
    }
    row.update({metric: float(val_metrics[metric]) for metric in METRICS})
    row.update(
        {"test/" + metric: float(test_metrics[metric]) for metric in METRICS}
    )
    return row


def collect_zero_shot() -> List[Dict]:
    payload = load_json(ZERO_SHOT_VAL)
    rows = []
    for spec in DATASETS:
        test_path = latest_matching(
            ROOT / "experiments/raw_metrics", spec["zero_test_prefix"] + "*.json"
        )
        test_metrics = load_json(test_path)["metrics"]
        row = {
            "dataset": spec["dataset"],
            "method": "Official zero-shot",
            "seed": 42,
            "best_epoch": "",
            "trainable_parameters": 0,
            "total_parameters": 172912413,
            "trainable_ratio_percent": 0.0,
            "gradient_audits": 0,
            "max_nonfinite_gradient_tensors": 0,
            "max_gradient_global_norm": 0.0,
            "peak_cuda_allocated_gib": "",
            "peak_cuda_reserved_gib": "",
            "elapsed_minutes": "",
            "validation_visualizations": "",
            "test_visualizations": count_visualizations(
                ROOT
                / "experiments"
                / spec["zero_test_prefix"].replace("_s42_", "").rstrip("_")
                / "visualizations"
            ),
            "checkpoint": "weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth",
            "checkpoint_bytes": (
                ROOT / "weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
            ).stat().st_size,
            "checkpoint_sha256": (
                "822D7E9DB9CE6FF2119B72DC6E78606A1B0A2C307234798ADF0CAB50F1B424E3"
            ),
            "training_manifest": "",
            "training_log": "",
            "independent_validation_json": str(ZERO_SHOT_VAL.relative_to(ROOT)),
            "held_out_test_json": str(test_path.relative_to(ROOT)),
        }
        for metric in METRICS:
            row[metric] = float(
                payload["metrics"]["{}/{}".format(spec["metric_prefix"], metric)]
            )
            row["test/" + metric] = float(test_metrics[metric])
        if row["test_visualizations"] != spec["test_images"]:
            raise RuntimeError(
                "{} zero-shot test visualization count {} != {}".format(
                    spec["dataset"], row["test_visualizations"], spec["test_images"]
                )
            )
        rows.append(row)
    return rows


def epoch_rows(run_rows: List[Dict]) -> List[Dict]:
    rows = []
    for spec, run in zip(RUNS, run_rows):
        parsed = parse_log(ROOT / run["training_log"], spec["val_images"])
        for item in parsed["epochs"]:
            rows.append(
                {
                    "dataset": spec["dataset"],
                    "method": spec["method"],
                    **item,
                }
            )
    return rows


def render_plot(rows: List[Dict], epochs: List[Dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "results/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    colors = {
        "Official zero-shot": "#777777",
        "Full fine-tuning": "#1565c0",
        "Frozen text encoder": "#ef6c00",
    }
    for axis, dataset in zip(axes, ("Pothole", "thermalDogsAndPeople")):
        for method in ("Full fine-tuning", "Frozen text encoder"):
            subset = [
                row
                for row in epochs
                if row["dataset"] == dataset and row["method"] == method
            ]
            axis.plot(
                [row["epoch"] for row in subset],
                [row["coco/bbox_mAP"] for row in subset],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=colors[method],
                label=method,
            )
        zero = next(
            row
            for row in rows
            if row["dataset"] == dataset and row["method"] == "Official zero-shot"
        )
        axis.axhline(
            zero["coco/bbox_mAP"],
            linestyle="--",
            linewidth=1.3,
            color=colors["Official zero-shot"],
            label="Official zero-shot",
        )
        axis.set_title(dataset)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation COCO mAP")
        axis.set_xticks(range(1, 13))
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for extension in ("png", "pdf"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(
            figure_dir / ("cross_task_validation_curves." + extension), **kwargs
        )
    plt.close(fig)


def main() -> int:
    trained_rows = [collect_run(spec) for spec in RUNS]
    zero_rows = collect_zero_shot()
    rows = []
    for dataset in ("Pothole", "thermalDogsAndPeople"):
        rows.extend(row for row in zero_rows if row["dataset"] == dataset)
        rows.extend(row for row in trained_rows if row["dataset"] == dataset)
    epochs = epoch_rows(trained_rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "selection_protocol": (
            "Best fine-tuned checkpoint selected by validation mAP only; "
            "held-out test used once after selection."
        ),
        "rows": rows,
    }
    output = ROOT / "results/json/cross_task_baseline_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(ROOT / "results/csv/cross_task_baseline_runs.csv", rows)
    write_csv(ROOT / "results/csv/cross_task_epoch_metrics.csv", epochs)
    render_plot(rows, epochs)
    print("summary={}".format(output))
    for row in rows:
        print(
            "{} | {} | val mAP={:.3f} | test mAP={:.3f}".format(
                row["dataset"],
                row["method"],
                row["coco/bbox_mAP"],
                row["test/coco/bbox_mAP"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
