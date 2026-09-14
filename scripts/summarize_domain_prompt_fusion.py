#!/usr/bin/env python
"""Summarize validation-only domain-prompt experiments and the matched baseline."""

import csv
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 21, 42)
METRICS = (
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
)
VAL_PATTERN = re.compile(
    r"Epoch\(val\) \[(\d+)\]\[133/133\].*?"
    r"coco/bbox_mAP: ([0-9.]+).*?"
    r"coco/bbox_mAP_50: ([0-9.]+).*?"
    r"coco/bbox_mAP_75: ([0-9.]+).*?"
    r"coco/bbox_mAP_s: ([0-9.]+).*?"
    r"coco/bbox_mAP_m: ([0-9.]+).*?"
    r"coco/bbox_mAP_l: ([0-9.]+)"
)
AUDIT_PATTERN = re.compile(
    r"gradient_audit iter=(\d+) tensors=(\d+) nonzero=(\d+) "
    r"nonfinite=(\d+) global_norm=([0-9.eE+-]+) "
    r"cuda_peak_allocated_bytes=(\d+) cuda_peak_reserved_bytes=(\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def completed_manifest(seed: int):
    pattern = "odinw_pothole_domain_prompt_fusion_lora_r16_seed{}_s{}_*.json".format(
        seed, seed
    )
    candidates = []
    for path in (ROOT / "experiments" / "manifests").glob(pattern):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed" and payload.get("returncode") == 0:
            candidates.append((path, payload))
    if len(candidates) != 1:
        raise RuntimeError("Expected one completed manifest for seed {}: {}".format(seed, candidates))
    return candidates[0]


def completed_log(seed: int) -> Path:
    pattern = "odinw_pothole_domain_prompt_fusion_lora_r16_seed{}_s{}_*.log".format(
        seed, seed
    )
    candidates = sorted((ROOT / "experiments" / "logs").glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError("Expected one completed log for seed {}: {}".format(seed, candidates))
    return candidates[0]


def load_baselines():
    path = ROOT / "results" / "csv" / "fusion_lora_multiseed_runs.csv"
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            if seed in SEEDS:
                rows[seed] = {
                    metric: float(row["validation/{}".format(metric)])
                    for metric in METRICS
                }
    if set(rows) != set(SEEDS):
        raise RuntimeError("Missing matched baseline seed")
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_figure(runs, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    x = list(range(len(SEEDS)))
    baseline = [runs[seed]["baseline/coco/bbox_mAP"] for seed in SEEDS]
    proposed = [runs[seed]["validation/coco/bbox_mAP"] for seed in SEEDS]
    axes[0].plot(x, baseline, marker="o", linewidth=1.8, label="Fusion-LoRA r16")
    axes[0].plot(x, proposed, marker="o", linewidth=1.8, label="+ domain prompts")
    axes[0].set_xticks(x, [str(seed) for seed in SEEDS])
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel("Validation mAP")
    axes[0].set_title("Paired validation results")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    means = [summary["baseline_mean"], summary["proposed_mean"]]
    errors = [summary["baseline_sample_sd"], summary["proposed_sample_sd"]]
    bars = axes[1].bar(
        [0, 1], means, yerr=errors, capsize=5, color=["#6c8ebf", "#d79b00"]
    )
    axes[1].set_xticks([0, 1], ["Fusion-LoRA\nr16", "+ domain\nprompts"])
    axes[1].set_ylabel("Validation mAP")
    axes[1].set_title("Mean +/- sample SD (3 seeds)")
    axes[1].grid(axis="y", alpha=0.25)
    lower = min(means) - max(errors) - 0.01
    upper = max(means) + max(errors) + 0.012
    axes[1].set_ylim(lower, upper)
    for bar, value in zip(bars, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            "{:.4f}".format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle("Pothole validation: image-conditioned domain prompt fusion")
    fig.tight_layout()
    prefix = ROOT / "results" / "figures" / "domain_prompt_fusion_validation"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix.with_suffix(".png")), dpi=600)
    fig.savefig(str(prefix.with_suffix(".pdf")))
    fig.savefig(str(prefix.with_suffix(".svg")))
    plt.close(fig)


def main() -> int:
    baselines = load_baselines()
    runs = {}
    epoch_rows = []
    for seed in SEEDS:
        log_path = completed_log(seed)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        epochs = []
        for match in VAL_PATTERN.finditer(log_text):
            values = [float(value) for value in match.groups()[1:]]
            row = {"seed": seed, "epoch": int(match.group(1))}
            row.update(dict(zip(METRICS, values)))
            epochs.append(row)
            epoch_rows.append(row)
        if len(epochs) != 12:
            raise RuntimeError("Expected 12 validation epochs for seed {}".format(seed))

        metric_path = (
            ROOT
            / "experiments"
            / "raw_metrics"
            / "odinw_pothole_domain_prompt_fusion_lora_r16_seed{}_best_eval.json".format(seed)
        )
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        best_row = max(epochs, key=lambda item: item["coco/bbox_mAP"])
        if any(abs(metrics[key] - best_row[key]) > 1e-12 for key in METRICS):
            raise RuntimeError("Independent metrics disagree for seed {}".format(seed))

        audits = [
            {
                "iteration": int(match.group(1)),
                "tensors": int(match.group(2)),
                "nonzero": int(match.group(3)),
                "nonfinite": int(match.group(4)),
                "global_norm": float(match.group(5)),
                "allocated": int(match.group(6)),
                "reserved": int(match.group(7)),
            }
            for match in AUDIT_PATTERN.finditer(log_text)
        ]
        if len(audits) != 56 or max(item["nonfinite"] for item in audits) != 0:
            raise RuntimeError("Gradient audit failed for seed {}".format(seed))

        checkpoint = Path(payload["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        manifest_path, manifest = completed_manifest(seed)
        elapsed_minutes = (
            datetime.fromisoformat(manifest["completed_utc"])
            - datetime.fromisoformat(manifest["created_utc"])
        ).total_seconds() / 60.0
        visuals = list(
            (
                ROOT
                / "experiments"
                / "odinw_pothole_domain_prompt_fusion_lora_r16_seed{}_eval".format(seed)
                / "visualizations"
            ).glob("*.*")
        )
        valid_visuals = [path for path in visuals if path.suffix.lower() in (".jpg", ".jpeg", ".png")]
        run = {
            "seed": seed,
            "best_epoch": best_row["epoch"],
            "final_validation_mAP": epochs[-1]["coco/bbox_mAP"],
            "trainable_parameters": 2527304,
            "total_parameters": 175439717,
            "trainable_ratio_percent": 100.0 * 2527304 / 175439717,
            "gradient_audits": len(audits),
            "max_nonfinite_gradient_tensors": max(item["nonfinite"] for item in audits),
            "max_gradient_global_norm": max(item["global_norm"] for item in audits),
            "peak_cuda_allocated_gib": max(item["allocated"] for item in audits) / (1024 ** 3),
            "peak_cuda_reserved_gib": max(item["reserved"] for item in audits) / (1024 ** 3),
            "validation_visualizations": len(valid_visuals),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            "training_manifest": str(manifest_path.relative_to(ROOT)),
            "training_log": str(log_path.relative_to(ROOT)),
            "independent_validation_json": str(metric_path.relative_to(ROOT)),
            "manifest_returncode": manifest["returncode"],
            "elapsed_minutes": elapsed_minutes,
        }
        for key in METRICS:
            run["validation/{}".format(key)] = float(metrics[key])
            run["baseline/{}".format(key)] = baselines[seed][key]
            run["paired_delta/{}".format(key)] = float(metrics[key]) - baselines[seed][key]
        runs[seed] = run

    proposed_values = [runs[seed]["validation/coco/bbox_mAP"] for seed in SEEDS]
    baseline_values = [runs[seed]["baseline/coco/bbox_mAP"] for seed in SEEDS]
    deltas = [runs[seed]["paired_delta/coco/bbox_mAP"] for seed in SEEDS]
    summary = {
        "seeds": list(SEEDS),
        "proposed_mean": statistics.mean(proposed_values),
        "proposed_sample_sd": statistics.stdev(proposed_values),
        "baseline_mean": statistics.mean(baseline_values),
        "baseline_sample_sd": statistics.stdev(baseline_values),
        "paired_delta_mean": statistics.mean(deltas),
        "paired_delta_sample_sd": statistics.stdev(deltas),
        "paired_deltas": deltas,
        "positive_delta_seeds": sum(delta > 0 for delta in deltas),
        "nonnegative_delta_seeds": sum(delta >= 0 for delta in deltas),
        "stability_gate": "failed",
        "stability_gate_reason": (
            "Paired mAP changes have mixed signs; the mean +0.0010 is smaller "
            "than the 0.0066 sample SD of paired changes."
        ),
        "held_out_test_evaluated": False,
    }
    output = {
        "protocol": {
            "dataset": "ODinW Pothole",
            "selection_split": "validation",
            "seeds": list(SEEDS),
            "checkpoint_selection": "maximum validation coco/bbox_mAP per seed",
            "held_out_test_policy": "not evaluated because validation stability gate failed",
        },
        "summary": summary,
        "runs": [runs[seed] for seed in SEEDS],
    }
    json_path = ROOT / "results" / "json" / "domain_prompt_fusion_validation_summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(
        ROOT / "results" / "csv" / "domain_prompt_fusion_validation_runs.csv",
        [runs[seed] for seed in SEEDS],
    )
    write_csv(
        ROOT / "results" / "csv" / "domain_prompt_fusion_validation_epochs.csv",
        epoch_rows,
    )
    render_figure(runs, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
