#!/usr/bin/env python
"""Summarize the validation-only localization-ranking seed-42 screen."""

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "results/json/quality_aligned_seed42_validation_summary.json"
SCALARS = ROOT / (
    "experiments/odinw_pothole_localization_ranking_scale_lora_r16_seed42/"
    "20260814_110337/vis_data/scalars.json"
)
LOG = ROOT / (
    "experiments/logs/"
    "odinw_pothole_localization_ranking_scale_lora_r16_seed42_s42_20260814T030331Z.log"
)
CHECKPOINT = ROOT / (
    "experiments/odinw_pothole_localization_ranking_scale_lora_r16_seed42/"
    "best_coco_bbox_mAP_epoch_11.pth"
)
RELOAD = ROOT / (
    "experiments/raw_metrics/"
    "odinw_pothole_localization_ranking_scale_lora_r16_seed42_validation_s42_20260814T035205Z.json"
)
ERRORS = ROOT / "experiments/raw_metrics/pothole_localization_ranking_error_analysis.json"
ONE_BATCH = ROOT / "experiments/raw_metrics/localization_ranking_scale_lora_r16_one_batch.json"
SINGLE_IMAGE = ROOT / "experiments/raw_metrics/localization_ranking_scale_lora_r16_single_image.json"
METRICS = (
    "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
    "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l",
)


def relative(path):
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ranking_epochs():
    rows = []
    for line in SCALARS.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if "coco/bbox_mAP" in item:
            rows.append({"method": "localization_ranking", "epoch": int(item["step"]),
                         **{key: float(item[key]) for key in METRICS}})
    if [row["epoch"] for row in rows] != list(range(1, 13)):
        raise RuntimeError("Localization-ranking scalar history is incomplete")
    return rows


def train_audit():
    text = LOG.read_text(encoding="utf-8", errors="replace")
    audits = [tuple(map(int, row)) for row in re.findall(
        r"gradient_audit iter=\d+ tensors=(\d+) nonzero=(\d+) nonfinite=(\d+)", text
    )]
    allocated = [int(x) for x in re.findall(r"cuda_peak_allocated_bytes=(\d+)", text)]
    reserved = [int(x) for x in re.findall(r"cuda_peak_reserved_bytes=(\d+)", text)]
    if len(audits) != 56 or any(row[0] != 311 or row[2] != 0 for row in audits):
        raise RuntimeError("Unexpected gradient audit")
    return {
        "gradient_audits": len(audits),
        "gradient_tensors": 311,
        "first_nonzero": audits[0][1],
        "post_initialization_minimum_nonzero": min(row[1] for row in audits[1:]),
        "maximum_nonfinite": max(row[2] for row in audits),
        "peak_cuda_allocated_gib": max(allocated) / 2 ** 30,
        "peak_cuda_reserved_gib": max(reserved) / 2 ** 30,
    }


def main():
    for path in (PREVIOUS, SCALARS, LOG, CHECKPOINT, RELOAD, ERRORS,
                 ONE_BATCH, SINGLE_IMAGE):
        if not path.is_file():
            raise FileNotFoundError(path)
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    old_epochs = previous["epochs"]
    rank_epochs = ranking_epochs()
    rank_best = max(rank_epochs, key=lambda row: (row[METRICS[0]], -row["epoch"]))
    reload_metrics = json.loads(RELOAD.read_text(encoding="utf-8"))["metrics"]
    for key in METRICS:
        if rank_best[key] != float(reload_metrics[key]):
            raise RuntimeError("Independent reload mismatch for {}".format(key))
    if rank_best["epoch"] != 11:
        raise RuntimeError("Expected epoch 11 selection")

    one_batch = json.loads(ONE_BATCH.read_text(encoding="utf-8"))
    single_image = json.loads(SINGLE_IMAGE.read_text(encoding="utf-8"))
    if one_batch["gradient_nonfinite_tensors"] != 0 or not single_image["finite"]:
        raise RuntimeError("Preflight evidence is not finite")
    errors = json.loads(ERRORS.read_text(encoding="utf-8"))
    if round(errors["proposed"]["coco"]["AP"], 3) != rank_best[METRICS[0]]:
        raise RuntimeError("Prediction dump and selected checkpoint disagree")

    ranking_selected = {
        "method": "localization_ranking",
        "selected_epoch": rank_best["epoch"],
        **{key: rank_best[key] for key in METRICS},
        "checkpoint": relative(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": sha256(CHECKPOINT),
    }
    selected = previous["selected_runs"] + [ranking_selected]
    dynamic = errors["baseline"]
    ranking = errors["proposed"]
    gate = rank_best[METRICS[0]] > 0.541 and rank_best[METRICS[3]] > 0.381
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "ODinW Pothole", "split": "validation", "images": 133,
            "seed": 42, "epoch_budget": 12,
            "selection": "maximum validation mAP; earliest epoch on ties",
            "held_out_test_evaluated": False,
        },
        "method": {
            "auxiliary": "final matching-layer localization-aware logit ranking",
            "weight": 0.1, "temperature": 0.5, "iou_gap": 0.05,
            "hard_negative_topk": 20, "hard_negative_weight": 0.25,
            "new_inference_parameters": 0,
        },
        "preflight": {
            "single_image": relative(SINGLE_IMAGE), "one_batch": relative(ONE_BATCH),
            "trainable_parameters": one_batch["trainable_parameters"],
        },
        "training_audit": train_audit(),
        "selected_runs": selected,
        "exact_offline_comparison": {
            "dynamic_scale": dynamic["coco"],
            "localization_ranking": ranking["coco"],
            "delta_AP": ranking["coco"]["AP"] - dynamic["coco"]["AP"],
            "delta_APs": ranking["coco"]["APs"] - dynamic["coco"]["APs"],
            "delta_spearman_score_iou": (
                ranking["score_iou_alignment"]["spearman_score_vs_best_iou"]
                - dynamic["score_iou_alignment"]["spearman_score_vs_best_iou"]
            ),
            "paired_gt_best_iou": errors["paired_gt_best_iou"],
            "fixed_threshold_errors": {
                "dynamic_scale": dynamic["fixed_threshold_errors"],
                "localization_ranking": ranking["fixed_threshold_errors"],
            },
        },
        "replication_gate": {
            "rule": "Seed-42 validation mAP > 0.541 and APs > 0.381.",
            "status": "passed" if gate else "failed",
            "seeds_0_21_authorized": gate,
            "held_out_test_authorized": False,
        },
        "epochs": old_epochs + rank_epochs,
    }

    json_path = ROOT / "results/json/localization_ranking_seed42_validation_summary.json"
    epoch_csv = ROOT / "results/csv/localization_ranking_seed42_validation_epochs.csv"
    selected_csv = ROOT / "results/csv/localization_ranking_seed42_validation_comparison.csv"
    report_path = ROOT / "reports/localization_ranking_seed42_validation.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with epoch_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["epochs"][0]))
        writer.writeheader(); writer.writerows(payload["epochs"])
    with selected_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)

    histories = {}
    for row in payload["epochs"]:
        histories.setdefault(row["method"], []).append(row)
    labels = {
        "lora_only": "LoRA-only", "dynamic_scale": "Dynamic scale",
        "quality_aligned": "Quality-aligned", "localization_ranking": "Localization ranking",
    }
    colors = {
        "lora_only": "#009E73", "dynamic_scale": "#0072B2",
        "quality_aligned": "#CC79A7", "localization_ranking": "#D55E00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.35))
    for method in ("lora_only", "dynamic_scale", "quality_aligned", "localization_ranking"):
        rows = histories[method]
        axes[0].plot([row["epoch"] for row in rows], [row[METRICS[0]] for row in rows],
                     marker="o", markersize=2.7, linewidth=1.5,
                     color=colors[method], label=labels[method])
    axes[0].set(xlabel="Epoch", ylabel="Validation COCO bbox mAP", xticks=range(1, 13))
    axes[0].grid(alpha=0.22); axes[0].legend(fontsize=7.8)
    x = list(range(4))
    axes[1].bar([i - 0.18 for i in x], [row[METRICS[0]] for row in selected],
                width=0.36, color="#0072B2", label="mAP")
    axes[1].bar([i + 0.18 for i in x], [row[METRICS[3]] for row in selected],
                width=0.36, color="#D55E00", label="APs")
    axes[1].axhline(0.541, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].set_xticks(x, ["LoRA", "Dynamic\nscale", "Quality\naligned", "Localization\nranking"])
    axes[1].set_ylabel("Selected validation metric")
    axes[1].grid(axis="y", alpha=0.22); axes[1].legend(fontsize=8)
    fig.tight_layout()
    stem = ROOT / "results/figures/localization_ranking_seed42_validation"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(stem) + "." + suffix,
                    dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    audit = payload["training_audit"]
    report_path.write_text(
        "# Localization-aware ranking seed-42 validation screen\n\n"
        "Selection uses only the unchanged 133-image ODinW Pothole validation split; "
        "the held-out test set was not evaluated.\n\n"
        "## Outcome\n\n"
        "- Selected epoch: 11 of 12 (maximum validation mAP).\n"
        "- Independent mAP/AP50/AP75/APs/APm/APl: "
        + "/".join("{:.3f}".format(rank_best[key]) for key in METRICS) + ".\n"
        "- Exact offline AP/APs: {:.6f}/{:.6f}; dynamic-scale reference: {:.6f}/{:.6f}.\n".format(
            ranking["coco"]["AP"], ranking["coco"]["APs"],
            dynamic["coco"]["AP"], dynamic["coco"]["APs"])
        + "- Score--IoU Spearman changes from {:.4f} to {:.4f}; paired mean best-IoU "
          "delta is {:.4f} overall and {:.4f} for small objects.\n".format(
              dynamic["score_iou_alignment"]["spearman_score_vs_best_iou"],
              ranking["score_iou_alignment"]["spearman_score_vs_best_iou"],
              errors["paired_gt_best_iou"]["all"]["mean_delta"],
              errors["paired_gt_best_iou"]["small"]["mean_delta"])
        + "- {} gradient audits are finite; peak allocated/reserved CUDA memory is "
          "{:.2f}/{:.2f} GiB.\n".format(
              audit["gradient_audits"], audit["peak_cuda_allocated_gib"],
              audit["peak_cuda_reserved_gib"])
        + "\n## Registered decision\n\n"
        "The gate **failed**: the selected checkpoint did not strictly exceed both "
        "0.541 mAP and 0.381 APs. Seeds 0/21 and held-out testing are not authorized.\n",
        encoding="utf-8",
    )
    print("gate={} epoch={} mAP={:.3f} APs={:.3f} exact_delta_AP={:+.8f}".format(
        payload["replication_gate"]["status"], rank_best["epoch"],
        rank_best[METRICS[0]], rank_best[METRICS[3]],
        payload["exact_offline_comparison"]["delta_AP"]))


if __name__ == "__main__":
    main()
