#!/usr/bin/env python
"""Aggregate Pothole and Thermal Full/Frozen runs over seeds 0, 21, 42."""

import statistics
from datetime import datetime, timezone

import summarize_cross_task_baselines as base


SEEDS = (0, 21, 42)
METHODS = (
    ("Full fine-tuning", "full_finetune"),
    ("Frozen text encoder", "frozen_text"),
)
DATASETS = (
    ("Pothole", "pothole", 133, 67),
    ("thermalDogsAndPeople", "thermal_dogs_and_people", 41, 20),
)


def make_specs():
    specs = []
    for dataset, task_slug, val_images, test_images in DATASETS:
        for method, method_slug in METHODS:
            for seed in SEEDS:
                seed_suffix = "" if seed == 42 else "_seed{}".format(seed)
                stem = "odinw_{}_{}{}".format(task_slug, method_slug, seed_suffix)
                specs.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "work_dir": "experiments/{}".format(stem),
                        "train_prefix": "{}_s{}_".format(stem, seed),
                        "eval_prefix": "{}_best_eval_s{}_".format(stem, seed),
                        "test_prefix": "{}_test_s{}_".format(stem, seed),
                        "val_images": val_images,
                        "test_images": test_images,
                    }
                )
    return specs


def aggregate_runs(rows):
    aggregate = []
    numeric_fields = (
        *base.METRICS,
        *("test/" + metric for metric in base.METRICS),
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        "elapsed_minutes",
    )
    for dataset, _, _, _ in DATASETS:
        for method, _ in METHODS:
            subset = sorted(
                (
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: row["seed"],
            )
            if [row["seed"] for row in subset] != list(SEEDS):
                raise RuntimeError(
                    "Unexpected seed set for {} {}".format(dataset, method)
                )
            item = {
                "dataset": dataset,
                "method": method,
                "n": len(subset),
                "seeds": ";".join(str(seed) for seed in SEEDS),
            }
            for field in numeric_fields:
                values = [float(row[field]) for row in subset]
                item[field + "_mean"] = statistics.mean(values)
                item[field + "_std"] = statistics.stdev(values)
            aggregate.append(item)
    return aggregate


def paired_differences(rows):
    output = {}
    for dataset, _, _, _ in DATASETS:
        full = {
            row["seed"]: row
            for row in rows
            if row["dataset"] == dataset and row["method"] == "Full fine-tuning"
        }
        frozen = {
            row["seed"]: row
            for row in rows
            if row["dataset"] == dataset
            and row["method"] == "Frozen text encoder"
        }
        val = {
            str(seed): frozen[seed]["coco/bbox_mAP"]
            - full[seed]["coco/bbox_mAP"]
            for seed in SEEDS
        }
        test = {
            str(seed): frozen[seed]["test/coco/bbox_mAP"]
            - full[seed]["test/coco/bbox_mAP"]
            for seed in SEEDS
        }
        output[dataset] = {
            "frozen_minus_full_validation_mAP": {
                "by_seed": val,
                "mean": statistics.mean(val.values()),
                "std": statistics.stdev(val.values()),
            },
            "frozen_minus_full_test_mAP": {
                "by_seed": test,
                "mean": statistics.mean(test.values()),
                "std": statistics.stdev(test.values()),
            },
        }
    return output


def render_plot(aggregate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = base.ROOT / "results/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5))
    for axis, (dataset, _, _, _) in zip(axes, DATASETS):
        subset = [row for row in aggregate if row["dataset"] == dataset]
        positions = range(len(subset))
        axis.errorbar(
            [position - 0.11 for position in positions],
            [row["coco/bbox_mAP_mean"] for row in subset],
            yerr=[row["coco/bbox_mAP_std"] for row in subset],
            marker="o",
            capsize=5,
            linestyle="none",
            label="Validation",
        )
        axis.errorbar(
            [position + 0.11 for position in positions],
            [row["test/coco/bbox_mAP_mean"] for row in subset],
            yerr=[row["test/coco/bbox_mAP_std"] for row in subset],
            marker="s",
            capsize=5,
            linestyle="none",
            label="Held-out test",
        )
        axis.set_xticks(list(positions))
        axis.set_xticklabels([row["method"] for row in subset], rotation=8)
        axis.set_title(dataset)
        axis.set_ylabel("COCO mAP (mean +/- sample SD)")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for extension in ("png", "pdf"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(
            figure_dir / ("cross_task_multiseed_map." + extension), **kwargs
        )
    plt.close(fig)


def main():
    specs = make_specs()
    rows = [base.collect_run(spec) for spec in specs]
    aggregate = aggregate_runs(rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "standard_deviation": "sample (ddof=1)",
        "selection_protocol": (
            "Best checkpoint selected independently by validation mAP for "
            "each task, method, and seed; held-out test used after selection."
        ),
        "runs": rows,
        "aggregate": aggregate,
        "paired_differences": paired_differences(rows),
    }
    output = base.ROOT / "results/json/cross_task_multiseed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        base.json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    base.write_csv(base.ROOT / "results/csv/cross_task_multiseed_runs.csv", rows)
    base.write_csv(
        base.ROOT / "results/csv/cross_task_multiseed_aggregate.csv", aggregate
    )
    render_plot(aggregate)
    print("summary={}".format(output))
    for row in aggregate:
        print(
            "{} | {} | val={:.4f}+/-{:.4f} | test={:.4f}+/-{:.4f}".format(
                row["dataset"],
                row["method"],
                row["coco/bbox_mAP_mean"],
                row["coco/bbox_mAP_std"],
                row["test/coco/bbox_mAP_mean"],
                row["test/coco/bbox_mAP_std"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
