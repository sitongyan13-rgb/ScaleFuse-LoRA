#!/usr/bin/env python
"""Strictly summarize the validation-only quality-aligned seed-42 screen."""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_SCALARS = ROOT / "experiments/odinw_pothole_scale_aware_lora_r16_seed42_retry/20260813_143108/vis_data/scalars.json"
DYNAMIC_CHECKPOINT = ROOT / "experiments/odinw_pothole_scale_aware_lora_r16_seed42_retry/best_coco_bbox_mAP_epoch_11.pth"
DYNAMIC_RELOAD = ROOT / "experiments/raw_metrics/odinw_pothole_scale_aware_lora_r16_seed42_validation_s42_20260813T071729Z.json"
BASELINE_EPOCHS = ROOT / "results/csv/prompt_routing_multiseed_validation_epochs.csv"
ONE_BATCH = ROOT / "experiments/raw_metrics/quality_aligned_scale_lora_r16_one_batch.json"
SINGLE_IMAGE = ROOT / "experiments/raw_metrics/quality_aligned_scale_lora_r16_single_image.json"

METRICS = (
    "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
    "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l",
)


def rel(path):
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def epochs(path, method):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if "coco/bbox_mAP" in item:
            rows.append({"method": method, "epoch": int(item["step"]),
                         **{key: float(item[key]) for key in METRICS}})
    if [row["epoch"] for row in rows] != list(range(1, 13)):
        raise RuntimeError("{} does not contain exactly epochs 1--12".format(path))
    return rows


def baseline_epochs():
    rows = []
    with BASELINE_EPOCHS.open(encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            if item["method"] == "lora_r16" and int(item["seed"]) == 42 and int(item["epoch"]) <= 12:
                rows.append({"method": "lora_only", "epoch": int(item["epoch"]),
                             **{key: float(item[key]) for key in METRICS}})
    if [row["epoch"] for row in rows] != list(range(1, 13)):
        raise RuntimeError("Matched LoRA history is incomplete")
    return rows


def earliest_best(rows):
    return max(rows, key=lambda row: (row[METRICS[0]], -row["epoch"]))


def exact_reload(best, path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in METRICS:
        if float(best[key]) != float(payload["metrics"][key]):
            raise RuntimeError("Independent reload mismatch for {}".format(key))
    return payload


def train_audit(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    audits = [tuple(map(int, row)) for row in re.findall(
        r"gradient_audit iter=\d+ tensors=(\d+) nonzero=(\d+) nonfinite=(\d+)", text)]
    allocated = [int(value) for value in re.findall(r"cuda_peak_allocated_bytes=(\d+)", text)]
    reserved = [int(value) for value in re.findall(r"cuda_peak_reserved_bytes=(\d+)", text)]
    if not audits or not allocated or not reserved:
        raise RuntimeError("Incomplete training audit: {}".format(path))
    if any(row[0] != 311 or row[2] != 0 for row in audits):
        raise RuntimeError("Invalid gradient audit: {}".format(path))
    return {
        "gradient_audits": len(audits),
        "gradient_tensors": 311,
        "first_nonzero": audits[0][1],
        "post_initialization_minimum_nonzero": min(row[1] for row in audits[1:]),
        "maximum_nonfinite": max(row[2] for row in audits),
        "peak_cuda_allocated_gib": max(allocated) / 2 ** 30,
        "peak_cuda_reserved_gib": max(reserved) / 2 ** 30,
    }


def selected_row(method, best, checkpoint):
    return {
        "method": method, "selected_epoch": best["epoch"],
        **{key: best[key] for key in METRICS},
        "checkpoint": rel(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-scalars", required=True, type=Path)
    parser.add_argument("--quality-log", required=True, type=Path)
    parser.add_argument("--quality-checkpoint", required=True, type=Path)
    parser.add_argument("--quality-reload", required=True, type=Path)
    parser.add_argument("--quality-error-analysis", required=True, type=Path)
    args = parser.parse_args()
    paths = [DYNAMIC_SCALARS, DYNAMIC_CHECKPOINT, DYNAMIC_RELOAD, BASELINE_EPOCHS,
             ONE_BATCH, SINGLE_IMAGE, args.quality_scalars, args.quality_log,
             args.quality_checkpoint, args.quality_reload, args.quality_error_analysis]
    for path in paths:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)

    lora_rows = baseline_epochs()
    dynamic_rows = epochs(DYNAMIC_SCALARS, "dynamic_scale")
    quality_rows = epochs(args.quality_scalars.resolve(), "quality_aligned")
    lora_best = earliest_best(lora_rows)
    dynamic_best = earliest_best(dynamic_rows)
    quality_best = earliest_best(quality_rows)
    exact_reload(quality_best, args.quality_reload.resolve())
    dynamic_reload = json.loads(DYNAMIC_RELOAD.read_text(encoding="utf-8"))
    for key in METRICS:
        if float(dynamic_best[key]) != float(dynamic_reload["metrics"][key]):
            raise RuntimeError("Dynamic reference reload mismatch for {}".format(key))

    one_batch = json.loads(ONE_BATCH.read_text(encoding="utf-8"))
    single_image = json.loads(SINGLE_IMAGE.read_text(encoding="utf-8"))
    if one_batch["gradient_nonfinite_tensors"] != 0 or one_batch["trainable_parameters"] != 2723651:
        raise RuntimeError("One-batch audit does not match the registered candidate")
    if not single_image.get("finite", False):
        raise RuntimeError("Single-image inference was not finite")
    training = train_audit(args.quality_log.resolve())
    error_analysis = json.loads(args.quality_error_analysis.read_text(encoding="utf-8"))
    if round(error_analysis["proposed"]["coco"]["AP"], 3) != quality_best[METRICS[0]]:
        raise RuntimeError("Prediction dump AP does not match the selected checkpoint")
    if round(error_analysis["baseline"]["coco"]["AP"], 3) != dynamic_best[METRICS[0]]:
        raise RuntimeError("Prediction dump baseline is not the dynamic reference")

    quality_checkpoint = args.quality_checkpoint.resolve()
    runs = [selected_row("lora_only", lora_best,
                         ROOT / "experiments/odinw_pothole_fusion_lora_r16_seed42/best_coco_bbox_mAP_epoch_9.pth"),
            selected_row("dynamic_scale", dynamic_best, DYNAMIC_CHECKPOINT),
            selected_row("quality_aligned", quality_best, quality_checkpoint)]
    gate_passed = (quality_best[METRICS[0]] > 0.541 and
                   quality_best[METRICS[3]] > 0.381)
    dynamic_alignment = error_analysis["baseline"]["score_iou_alignment"]
    quality_alignment = error_analysis["proposed"]["score_iou_alignment"]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"dataset": "ODinW Pothole", "split": "validation",
                     "images": 133, "seed": 42, "epoch_budget": 12,
                     "selection": "maximum validation mAP; earliest epoch on ties",
                     "quality_target": "detached aligned IoU, power 1.0, matched positives only",
                     "held_out_test_evaluated": False},
        "selected_runs": runs,
        "preflight": {"single_image": rel(SINGLE_IMAGE), "one_batch": rel(ONE_BATCH),
                      "trainable_parameters": one_batch["trainable_parameters"],
                      "new_inference_parameters": 0},
        "training_audit": training,
        "score_iou_alignment": {"dynamic_scale": dynamic_alignment,
                                "quality_aligned": quality_alignment},
        "paired_gt_best_iou": error_analysis["paired_gt_best_iou"],
        "replication_gate": {
            "rule": "Quality-aligned seed-42 mAP > 0.541 and APs > 0.381 on validation.",
            "status": "passed" if gate_passed else "failed",
            "seeds_0_21_authorized": gate_passed,
            "held_out_test_authorized": False,
            "note": "Test remains prohibited until any authorized multi-seed replication is completed."},
        "epochs": lora_rows + dynamic_rows + quality_rows,
    }

    json_path = ROOT / "results/json/quality_aligned_seed42_validation_summary.json"
    epoch_csv = ROOT / "results/csv/quality_aligned_seed42_validation_epochs.csv"
    run_csv = ROOT / "results/csv/quality_aligned_seed42_validation_comparison.csv"
    report_path = ROOT / "reports/quality_aligned_seed42_validation.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with epoch_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["epochs"][0]))
        writer.writeheader(); writer.writerows(payload["epochs"])
    with run_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader(); writer.writerows(runs)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    for label, rows, color in (("LoRA-only", lora_rows, "#009E73"),
                               ("Dynamic scale", dynamic_rows, "#0072B2"),
                               ("Quality-aligned", quality_rows, "#D55E00")):
        axes[0].plot([row["epoch"] for row in rows], [row[METRICS[0]] for row in rows],
                     marker="o", markersize=2.8, linewidth=1.55, label=label, color=color)
    axes[0].set(xlabel="Epoch", ylabel="Validation COCO bbox mAP", xticks=range(1, 13))
    axes[0].grid(alpha=0.22); axes[0].legend(fontsize=8)
    labels = ["LoRA-only", "Dynamic\nscale", "Quality-\naligned"]
    x = range(3)
    axes[1].bar([i - 0.18 for i in x], [row[METRICS[0]] for row in runs],
                width=0.36, label="mAP", color="#0072B2")
    axes[1].bar([i + 0.18 for i in x], [row[METRICS[3]] for row in runs],
                width=0.36, label="APs", color="#D55E00")
    axes[1].axhline(0.541, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].set_xticks(list(x), labels); axes[1].set_ylabel("Selected validation metric")
    axes[1].grid(axis="y", alpha=0.22); axes[1].legend(fontsize=8)
    fig.tight_layout()
    stem = ROOT / "results/figures/quality_aligned_seed42_validation"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(stem) + "." + suffix,
                    dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    report = """# Quality-aligned Seed-42 Validation Screen

This report is regenerated from complete 12-epoch scalar histories, an independent best-checkpoint reload, prediction dumps, and gradient/memory audits. Selection uses only the unchanged 133-image ODinW Pothole validation split; held-out test data were not evaluated.

## Results

- Matched LoRA-only: epoch {le}, mAP/APs = {lm:.3f}/{ls:.3f}.
- Dynamic scale: epoch {de}, mAP/APs = {dm:.3f}/{ds:.3f}.
- Dynamic scale plus quality-aligned targets: epoch {qe}, mAP/APs = {qm:.3f}/{qs:.3f}.
- The quality-aligned head adds no inference parameters; the model retains {params:,} trainable parameters. The independent reload exactly matches all six selected metrics.
- All {audits} training gradient audits are finite. Peak allocated/reserved CUDA memory is {allocated:.2f}/{reserved:.2f} GiB.
- Score--IoU Spearman correlation changes from {before:.4f} to {after:.4f}; this is a validation diagnostic, not a separate selection criterion.

## Registered decision

The replication gate **{gate}**: the seed-42 candidate had to exceed both 0.541 mAP and 0.381 APs. Seeds 0/21 are {authorization}. Held-out testing remains prohibited until any authorized replication is completed.
""".format(le=lora_best["epoch"], lm=lora_best[METRICS[0]], ls=lora_best[METRICS[3]],
           de=dynamic_best["epoch"], dm=dynamic_best[METRICS[0]], ds=dynamic_best[METRICS[3]],
           qe=quality_best["epoch"], qm=quality_best[METRICS[0]], qs=quality_best[METRICS[3]],
           params=one_batch["trainable_parameters"], audits=training["gradient_audits"],
           allocated=training["peak_cuda_allocated_gib"], reserved=training["peak_cuda_reserved_gib"],
           before=dynamic_alignment["spearman_score_vs_best_iou"],
           after=quality_alignment["spearman_score_vs_best_iou"],
           gate="passed" if gate_passed else "failed",
           authorization="authorized" if gate_passed else "not authorized")
    report_path.write_text(report, encoding="utf-8")
    print("gate={} quality_best_epoch={} mAP={:.3f} APs={:.3f}".format(
        payload["replication_gate"]["status"], quality_best["epoch"],
        quality_best[METRICS[0]], quality_best[METRICS[3]]))
    print("json={} epochs={} runs={} report={} figure={}.png".format(
        json_path, epoch_csv, run_csv, report_path, stem))


if __name__ == "__main__":
    main()
