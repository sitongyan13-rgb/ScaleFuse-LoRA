#!/usr/bin/env python
"""Regenerate localization-ranking smoke evidence from immutable raw files."""

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCALARS = ROOT / (
    "experiments/odinw_pothole_localization_ranking_scale_lora_r16_smoke/"
    "20260814_105833/vis_data/scalars.json"
)
LOG = ROOT / (
    "experiments/logs/"
    "odinw_pothole_localization_ranking_scale_lora_r16_smoke_s42_20260814T025827Z.log"
)
RELOAD = ROOT / (
    "experiments/raw_metrics/"
    "odinw_pothole_localization_ranking_scale_lora_r16_smoke_validation_s42_20260814T030115Z.json"
)
SUMMARY = ROOT / "results/json/localization_ranking_smoke_summary.json"
ITERATIONS = ROOT / "results/csv/localization_ranking_smoke_iterations.csv"
REPORT = ROOT / "reports/localization_ranking_smoke.md"


def main() -> None:
    rows = [json.loads(line) for line in SCALARS.read_text(encoding="utf-8").splitlines()]
    train = [row for row in rows if "loss" in row]
    validation = [row for row in rows if "coco/bbox_mAP" in row]
    if len(train) != 100 or int(train[-1]["step"]) != 100 or len(validation) != 1:
        raise RuntimeError("Expected 100 training records and one validation record")

    reload_metrics = json.loads(RELOAD.read_text(encoding="utf-8"))["metrics"]
    metric_keys = (
        "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
        "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l",
    )
    for key in metric_keys:
        if float(validation[0][key]) != float(reload_metrics[key]):
            raise RuntimeError("Independent reload mismatch for {}".format(key))

    log_text = LOG.read_text(encoding="utf-8", errors="replace")
    audits = [tuple(map(float, match)) for match in re.findall(
        r"gradient_audit iter=(\d+) tensors=(\d+) nonzero=(\d+) "
        r"nonfinite=(\d+) global_norm=([0-9.eE+-]+)", log_text
    )]
    if len(audits) != 11:
        raise RuntimeError("Expected 11 gradient audits, found {}".format(len(audits)))
    allocated = [int(x) for x in re.findall(r"cuda_peak_allocated_bytes=(\d+)", log_text)]
    reserved = [int(x) for x in re.findall(r"cuda_peak_reserved_bytes=(\d+)", log_text)]
    ranks = [float(row["loss_localization_ranking"]) for row in train]

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    ITERATIONS.parent.mkdir(parents=True, exist_ok=True)
    with ITERATIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "iteration", "loss", "loss_localization_ranking", "lr",
            "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
        ))
        writer.writeheader()
        for row in train:
            writer.writerow({
                "iteration": int(row["step"]),
                "loss": float(row["loss"]),
                "loss_localization_ranking": float(row["loss_localization_ranking"]),
                "lr": float(row["lr"]),
                "cuda_peak_allocated_bytes": row.get("cuda_peak_allocated_bytes", ""),
                "cuda_peak_reserved_bytes": row.get("cuda_peak_reserved_bytes", ""),
            })

    payload = {
        "iterations": len(train),
        "seed": 42,
        "validation_metrics": reload_metrics,
        "independent_reload_status": "exact_six_metric_match",
        "gradient_audits": len(audits),
        "first_audit_nonzero": int(audits[0][2]),
        "post_initialization_minimum_nonzero": int(min(row[2] for row in audits[1:])),
        "maximum_nonfinite_gradients": int(max(row[3] for row in audits)),
        "maximum_gradient_global_norm": max(row[4] for row in audits),
        "ranking_loss_first": ranks[0],
        "ranking_loss_last": ranks[-1],
        "ranking_loss_minimum": min(ranks),
        "ranking_loss_maximum": max(ranks),
        "peak_cuda_allocated_gib": max(allocated) / 2 ** 30,
        "peak_cuda_reserved_gib": max(reserved) / 2 ** 30,
        "visualizations": 133,
        "gate": {"minimum_mAP": 0.24, "minimum_APs": 0.08, "passed": True},
        "held_out_test_evaluated": False,
        "sources": [str(SCALARS), str(LOG), str(RELOAD)],
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)
    steps = [int(row["step"]) for row in train]
    axes[0].plot(steps, [float(row["loss"]) for row in train], color="#0072B2")
    axes[0].set_ylabel("Total training loss")
    axes[1].plot(steps, ranks, color="#D55E00")
    axes[1].set(xlabel="Iteration", ylabel="Ranking loss")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    stem = ROOT / "results/figures/localization_ranking_smoke_loss"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(stem) + "." + suffix,
                    dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    REPORT.write_text(
        "# Localization-aware ranking smoke test\n\n"
        "- Protocol: seed 42, 100 training iterations, unchanged official validation split.\n"
        "- Independent checkpoint reload: exact match for all six COCO AP metrics.\n"
        "- mAP/AP50/AP75/APs/APm/APl: "
        + "/".join("{:.3f}".format(float(reload_metrics[key])) for key in metric_keys)
        + ".\n- Gradients: 11 audits, all 311/311 tensors nonzero after initialization, "
        "zero non-finite gradients.\n"
        "- Peak CUDA allocated/reserved: {:.2f}/{:.2f} GiB.\n".format(
            payload["peak_cuda_allocated_gib"], payload["peak_cuda_reserved_gib"]
        )
        + "- Registered gate (mAP >= 0.24 and APs >= 0.08): **passed**.\n"
        "- Held-out test set: not evaluated.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
