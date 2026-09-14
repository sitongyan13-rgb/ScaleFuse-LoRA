#!/usr/bin/env python
"""Summarize the one-time frozen three-seed Pothole test evaluation."""

import json
import statistics
from datetime import datetime, timezone

import matplotlib.pyplot as plt

import summarize_scale_aware_multiseed as base


TEST_RUNS = {
    "Fusion-LoRA-r16": {
        0: "experiments/raw_metrics/pothole_fusion_lora_r16_seed0_test_exact.json",
        21: "experiments/raw_metrics/pothole_fusion_lora_r16_seed21_test_exact.json",
        42: "experiments/raw_metrics/pothole_fusion_lora_r16_seed42_test_exact.json",
    },
    "Dynamic Scale-Aware Fusion LoRA": {
        0: "experiments/raw_metrics/pothole_dynamic_scale_seed0_test_exact.json",
        21: "experiments/raw_metrics/pothole_dynamic_scale_seed21_test_exact.json",
        42: "experiments/raw_metrics/pothole_dynamic_scale_seed42_test_exact.json",
    },
}


def load_test_runs():
    rows = []
    sources = []
    for method, by_seed in TEST_RUNS.items():
        for seed in base.SEEDS:
            relative = by_seed[seed]
            path = base.ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = {"method": method, "seed": seed, "metric_json": relative}
            row.update({metric: float(payload["metrics"][metric])
                        for metric in base.METRICS})
            rows.append(row)
            sources.append({"path": relative, "sha256": base.sha256(path)})
    return rows, sources


def plot(rows, paired_summary):
    figure_dir = base.ROOT / "results/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    colors = ("#607D8B", "#E76F51")
    for index, method in enumerate(TEST_RUNS):
        values = [row["AP"] for row in rows if row["method"] == method]
        axes[0].bar(index, statistics.mean(values), yerr=statistics.stdev(values),
                    capsize=5, color=colors[index], alpha=0.82)
        axes[0].scatter([index - 0.08, index, index + 0.08], values,
                        color="black", s=22, zorder=3)
    axes[0].set_xticks((0, 1), ("Fusion-LoRA\nr16", "Dynamic scale-aware\nfusion LoRA"))
    axes[0].set_ylabel("Held-out test AP")
    axes[0].set_ylim(0.54, 0.585)
    axes[0].grid(axis="y", alpha=0.25)

    ap_deltas = [paired_summary["AP"]["by_seed"][str(seed)] for seed in base.SEEDS]
    axes[1].bar([str(seed) for seed in base.SEEDS], ap_deltas,
                color="#2A9D8F", alpha=0.86)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Paired test AP change")
    axes[1].grid(axis="y", alpha=0.25)

    aps_deltas = [paired_summary["APs"]["by_seed"][str(seed)] for seed in base.SEEDS]
    axes[2].bar([str(seed) for seed in base.SEEDS], aps_deltas,
                color="#D1495B", alpha=0.86)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Seed")
    axes[2].set_ylabel("Paired test APs change")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Pothole held-out test: frozen three-seed evaluation")
    fig.tight_layout()
    stem = figure_dir / "scale_aware_multiseed_test"
    for suffix, kwargs in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        fig.savefig(str(stem) + "." + suffix, bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(rows, aggregates, paired_summary):
    by_method = {}
    for method in TEST_RUNS:
        by_method[method] = {
            row["metric"]: row for row in aggregates if row["method"] == method}
    lines = [
        "# Dynamic Scale-Aware Fusion LoRA: final three-seed results",
        "",
        "## Integrity statement",
        "",
        "All validation metrics and checkpoint paths were frozen in `reports/scale_aware_multiseed_validation.md` before the seed 0/21 held-out test was read. The test results below are a one-time evaluation of those fixed checkpoints and did not alter method, prompt, post-processing, epoch budget, or checkpoint selection.",
        "",
        "## Exact held-out results",
        "",
        "| Method | Seed 0 AP | Seed 21 AP | Seed 42 AP | Mean +/- sample SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in TEST_RUNS:
        method_rows = {row["seed"]: row for row in rows if row["method"] == method}
        ap = by_method[method]["AP"]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} +/- {:.6f} |".format(
            method, method_rows[0]["AP"], method_rows[21]["AP"], method_rows[42]["AP"],
            ap["mean"], ap["sample_sd"]))
    ap_pair = paired_summary["AP"]
    aps_pair = paired_summary["APs"]
    lines.extend([
        "",
        "Paired overall-AP changes for seeds 0/21/42 are {:+.6f}, {:+.6f}, and {:+.6f}. Their mean is {:+.6f} (sample SD {:.6f}; descriptive 95% t interval [{:+.6f}, {:+.6f}]). All three overall-AP changes are positive.".format(
            ap_pair["by_seed"]["0"], ap_pair["by_seed"]["21"],
            ap_pair["by_seed"]["42"], ap_pair["mean"], ap_pair["sample_sd"],
            *ap_pair["ci95_t"]),
        "",
        "The improvement is not uniform across object sizes. Test APs changes are {:+.6f}, {:+.6f}, and {:+.6f} (mean {:+.6f}); all three are negative. The overall gain is therefore driven by medium/large-object or localization behavior, not better small-object AP. This limitation must accompany any summary of the main result.".format(
            aps_pair["by_seed"]["0"], aps_pair["by_seed"]["21"],
            aps_pair["by_seed"]["42"], aps_pair["mean"]),
        "",
        "With only three seeds and 67 held-out images, these statistics are descriptive and do not support SOTA or strong significance claims.",
        "",
        "## Artifacts",
        "",
        "- `results/csv/scale_aware_multiseed_test_runs.csv`",
        "- `results/csv/scale_aware_multiseed_test_aggregate.csv`",
        "- `results/csv/scale_aware_multiseed_test_paired.csv`",
        "- `results/json/scale_aware_multiseed_results.json`",
        "- `results/figures/scale_aware_multiseed_test.{png,pdf,svg}`",
    ])
    (base.ROOT / "reports/scale_aware_multiseed_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows, sources = load_test_runs()
    aggregates = base.aggregate(rows)
    paired_rows, paired_summary = base.paired(rows)
    base.write_csv(base.ROOT / "results/csv/scale_aware_multiseed_test_runs.csv", rows)
    base.write_csv(base.ROOT / "results/csv/scale_aware_multiseed_test_aggregate.csv", aggregates)
    base.write_csv(base.ROOT / "results/csv/scale_aware_multiseed_test_paired.csv", paired_rows)
    validation_path = base.ROOT / "results/json/scale_aware_multiseed_validation_summary.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": "results/json/scale_aware_multiseed_validation_summary.json",
        "selection_source_sha256": base.sha256(validation_path),
        "test_split": "pothole_test",
        "test_image_count": 67,
        "seeds": list(base.SEEDS),
        "runs": rows,
        "aggregate": aggregates,
        "paired_proposed_minus_baseline": paired_summary,
        "sources": sources,
        "model_selection_changed_after_test": False,
    }
    output = base.ROOT / "results/json/scale_aware_multiseed_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(rows, paired_summary)
    write_report(rows, aggregates, paired_summary)
    dynamic_ap = next(row for row in aggregates
                      if row["method"].startswith("Dynamic") and row["metric"] == "AP")
    baseline_ap = next(row for row in aggregates
                       if row["method"].startswith("Fusion") and row["metric"] == "AP")
    print(json.dumps({
        "output": str(output),
        "dynamic_test_AP_mean": dynamic_ap["mean"],
        "baseline_test_AP_mean": baseline_ap["mean"],
        "paired_test_AP": paired_summary["AP"],
        "paired_test_APs": paired_summary["APs"],
    }, indent=2))


if __name__ == "__main__":
    main()
