#!/usr/bin/env python
"""Regenerate the scale-aware 100-iteration smoke evidence from raw logs."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCALARS = ROOT / "experiments/odinw_pothole_scale_aware_lora_r16_smoke_retry/20260813_142420/vis_data/scalars.json"
LOG = ROOT / "experiments/logs/odinw_pothole_scale_aware_lora_r16_smoke_s42_20260813T062414Z.log"
RELOAD = ROOT / "experiments/raw_metrics/odinw_pothole_scale_aware_lora_r16_smoke_validation_s42_20260813T062703Z.json"
OUTPUT = ROOT / "experiments/raw_metrics/scale_aware_lora_r16_smoke_summary.json"


def main() -> None:
    records = [json.loads(line) for line in SCALARS.read_text(encoding="utf-8").splitlines()]
    train = [row for row in records if "loss" in row]
    validation = [row for row in records if "coco/bbox_mAP" in row]
    if not train or len(validation) != 1 or int(train[-1]["step"]) != 100:
        raise RuntimeError("Expected a complete 100-iteration smoke and one validation record")
    reload_metrics = json.loads(RELOAD.read_text(encoding="utf-8"))["metrics"]
    keys = ("coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
            "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l")
    for key in keys:
        if float(validation[0][key]) != float(reload_metrics[key]):
            raise RuntimeError("Independent reload mismatch: {}".format(key))
    text = LOG.read_text(encoding="utf-8", errors="replace")
    audits = [tuple(map(int, match)) for match in re.findall(
        r"gradient_audit iter=\d+ tensors=(\d+) nonzero=(\d+) nonfinite=(\d+)", text
    )]
    allocated = [int(x) for x in re.findall(r"cuda_peak_allocated_bytes=(\d+)", text)]
    reserved = [int(x) for x in re.findall(r"cuda_peak_reserved_bytes=(\d+)", text)]
    payload = {
        "iterations": 100,
        "training_records": len(train),
        "validation_metrics": reload_metrics,
        "independent_reload_status": "matched",
        "gradient_audits": len(audits),
        "first_audit_nonzero": audits[0][1],
        "post_initialization_minimum_nonzero": min(x[1] for x in audits[1:]),
        "maximum_nonfinite": max(x[2] for x in audits),
        "peak_cuda_allocated_gib": max(allocated) / 2 ** 30,
        "peak_cuda_reserved_gib": max(reserved) / 2 ** 30,
        "visualizations": 133,
        "held_out_test_evaluated": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot([row["step"] for row in train], [row["loss"] for row in train],
            color="#0072B2", linewidth=1.6)
    ax.set(xlabel="Iteration", ylabel="Training loss")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    stem = ROOT / "results/figures/scale_aware_lora_r16_smoke_loss"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(stem) + "." + suffix,
                    dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
