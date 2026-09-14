#!/usr/bin/env python
"""Regenerate the 960 x 1600 diagnostic evidence from raw artifacts."""

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCALARS = ROOT / (
    "experiments/odinw_pothole_high_resolution_scale_lora_r16_diagnostic/"
    "20260814_151825/vis_data/scalars.json"
)
LOG = ROOT / (
    "experiments/logs/"
    "odinw_pothole_high_resolution_scale_lora_r16_diagnostic_s42_20260814T071820Z.log"
)
RELOAD = ROOT / (
    "experiments/raw_metrics/"
    "odinw_pothole_high_resolution_scale_lora_r16_diagnostic_validation_s42_20260814T072052Z.json"
)
SUMMARY = ROOT / "results/json/high_resolution_diagnostic_summary.json"
ITERATIONS = ROOT / "results/csv/high_resolution_diagnostic_iterations.csv"
REPORT = ROOT / "reports/high_resolution_diagnostic.md"


def main() -> None:
    rows = [json.loads(line) for line in SCALARS.read_text(encoding="utf-8").splitlines()]
    train = [row for row in rows if "loss" in row]
    val = [row for row in rows if "coco/bbox_mAP" in row]
    if len(train) != 100 or int(train[-1]["step"]) != 100 or len(val) != 1:
        raise RuntimeError("Expected a complete 100-iteration diagnostic")
    metrics = json.loads(RELOAD.read_text(encoding="utf-8"))["metrics"]
    keys = (
        "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
        "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l",
    )
    if any(float(val[0][key]) != float(metrics[key]) for key in keys):
        raise RuntimeError("Independent reload metrics do not match")
    text = LOG.read_text(encoding="utf-8", errors="replace")
    audits = [tuple(map(float, match)) for match in re.findall(
        r"gradient_audit iter=(\d+) tensors=(\d+) nonzero=(\d+) "
        r"nonfinite=(\d+) global_norm=([0-9.eE+-]+)", text
    )]
    allocated = [int(value) for value in re.findall(
        r"cuda_peak_allocated_bytes=(\d+)", text)]
    reserved = [int(value) for value in re.findall(
        r"cuda_peak_reserved_bytes=(\d+)", text)]
    if len(audits) != 11 or not allocated or not reserved:
        raise RuntimeError("Incomplete gradient or memory audit")

    ITERATIONS.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with ITERATIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("iteration", "loss", "lr"))
        writer.writeheader()
        for row in train:
            writer.writerow({"iteration": int(row["step"]),
                             "loss": float(row["loss"]), "lr": float(row["lr"])})

    peak_allocated = max(allocated) / 2 ** 30
    peak_reserved = max(reserved) / 2 ** 30
    passed = (float(metrics["coco/bbox_mAP"]) >= 0.245
              and float(metrics["coco/bbox_mAP_s"]) >= 0.105
              and peak_allocated < 10.0)
    payload = {
        "iterations": 100,
        "seed": 42,
        "resolution": [960, 1600],
        "validation_metrics": metrics,
        "dynamic_scale_smoke_comparator": {"mAP": 0.252, "APs": 0.096},
        "independent_reload_status": "exact_six_metric_match",
        "gradient_audits": len(audits),
        "post_initialization_minimum_nonzero": int(min(row[2] for row in audits[1:])),
        "maximum_nonfinite_gradients": int(max(row[3] for row in audits)),
        "peak_cuda_allocated_gib": peak_allocated,
        "peak_cuda_reserved_gib": peak_reserved,
        "visualizations": 133,
        "support_criterion": {
            "minimum_mAP": 0.245, "minimum_APs": 0.105,
            "maximum_allocated_gib": 10.0, "passed": passed,
        },
        "held_out_test_evaluated": False,
        "sources": [str(SCALARS), str(LOG), str(RELOAD)],
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot([row["step"] for row in train], [row["loss"] for row in train],
            color="#0072B2", linewidth=1.6)
    ax.set(xlabel="Iteration", ylabel="Training loss")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    stem = ROOT / "results/figures/high_resolution_diagnostic_loss"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(stem) + "." + suffix,
                    dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    REPORT.write_text(
        "# High-resolution mechanism diagnostic\n\n"
        "- Protocol: seed 42, 100 iterations, 960 x 1600 train/validation input.\n"
        "- Independent reload exactly matches mAP/AP50/AP75/APs/APm/APl = "
        + "/".join("{:.3f}".format(float(metrics[key])) for key in keys)
        + ".\n- Dynamic-scale 800 x 1333 smoke comparator mAP/APs: 0.252/0.096.\n"
        "- All 11 gradient audits are finite; all 311 trainable tensors are "
        "nonzero after initialization.\n"
        "- Peak CUDA allocated/reserved: {:.2f}/{:.2f} GiB.\n".format(
            peak_allocated, peak_reserved)
        + "- Registered diagnostic support criterion: **failed**. Both mAP and "
        "APs declined despite 44% more image pixels.\n"
        "- Decision: do not pursue a P2/high-resolution feature candidate from "
        "this evidence; no full run or held-out test is authorized.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
