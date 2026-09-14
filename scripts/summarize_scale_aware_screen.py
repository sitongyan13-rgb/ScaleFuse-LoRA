#!/usr/bin/env python
"""Strictly summarize the dynamic/static scale-aware seed-42 screen."""

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
DYNAMIC_LOG = ROOT / "experiments/logs/odinw_pothole_scale_aware_lora_r16_seed42_s42_20260813T063102Z.log"
DYNAMIC_CHECKPOINT = ROOT / "experiments/odinw_pothole_scale_aware_lora_r16_seed42_retry/best_coco_bbox_mAP_epoch_11.pth"
DYNAMIC_RELOAD = ROOT / "experiments/raw_metrics/odinw_pothole_scale_aware_lora_r16_seed42_validation_s42_20260813T071729Z.json"
DYNAMIC_AUDIT = ROOT / "experiments/raw_metrics/odinw_pothole_scale_aware_lora_r16_seed42_fusion_audit.json"
INTERVENTION = ROOT / "experiments/raw_metrics/odinw_pothole_static_scale_aware_lora_r16_checkpoint_intervention_s42_20260813T072205Z.json"
BASELINE_EPOCHS = ROOT / "results/csv/prompt_routing_multiseed_validation_epochs.csv"

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


def earliest_best(rows):
    return max(rows, key=lambda row: (row["coco/bbox_mAP"], -row["epoch"]))


def exact_metrics(expected, payload, label):
    actual = payload["metrics"]
    for key in METRICS:
        if float(expected[key]) != float(actual[key]):
            raise RuntimeError("{} mismatch for {}".format(label, key))


def train_audit(path, expected_tensors):
    text = path.read_text(encoding="utf-8", errors="replace")
    audits = [tuple(map(int, match)) for match in re.findall(
        r"gradient_audit iter=\d+ tensors=(\d+) nonzero=(\d+) nonfinite=(\d+)", text)]
    allocated = [int(x) for x in re.findall(r"cuda_peak_allocated_bytes=(\d+)", text)]
    reserved = [int(x) for x in re.findall(r"cuda_peak_reserved_bytes=(\d+)", text)]
    if not audits or not allocated or not reserved:
        raise RuntimeError("Incomplete gradient/memory audit: {}".format(path))
    if max(row[2] for row in audits) != 0 or max(row[0] for row in audits) != expected_tensors:
        raise RuntimeError("Invalid gradient audit: {}".format(path))
    return {
        "gradient_audits": len(audits),
        "gradient_tensors": expected_tensors,
        "first_nonzero": audits[0][1],
        "post_initialization_minimum_nonzero": min(row[1] for row in audits[1:]),
        "maximum_nonfinite": max(row[2] for row in audits),
        "peak_cuda_allocated_gib": max(allocated) / 2 ** 30,
        "peak_cuda_reserved_gib": max(reserved) / 2 ** 30,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-scalars", required=True, type=Path)
    parser.add_argument("--static-log", required=True, type=Path)
    parser.add_argument("--static-checkpoint", required=True, type=Path)
    parser.add_argument("--static-reload", required=True, type=Path)
    parser.add_argument("--static-audit", required=True, type=Path)
    args = parser.parse_args()
    paths = [DYNAMIC_SCALARS, DYNAMIC_LOG, DYNAMIC_CHECKPOINT, DYNAMIC_RELOAD,
             DYNAMIC_AUDIT, INTERVENTION, BASELINE_EPOCHS, args.static_scalars,
             args.static_log, args.static_checkpoint, args.static_reload,
             args.static_audit]
    for path in paths:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)

    all_epochs = epochs(DYNAMIC_SCALARS, "dynamic_gate") + epochs(
        args.static_scalars.resolve(), "static_residual")
    dynamic_rows = [row for row in all_epochs if row["method"] == "dynamic_gate"]
    static_rows = [row for row in all_epochs if row["method"] == "static_residual"]
    dynamic_best, static_best = earliest_best(dynamic_rows), earliest_best(static_rows)
    dynamic_reload = json.loads(DYNAMIC_RELOAD.read_text(encoding="utf-8"))
    static_reload = json.loads(args.static_reload.read_text(encoding="utf-8"))
    exact_metrics(dynamic_best, dynamic_reload, "dynamic independent reload")
    exact_metrics(static_best, static_reload, "static independent reload")

    dynamic_mechanism = json.loads(DYNAMIC_AUDIT.read_text(encoding="utf-8"))
    static_mechanism = json.loads(args.static_audit.read_text(encoding="utf-8"))
    for key in METRICS:
        if float(dynamic_mechanism["validation_metrics"][key]) != dynamic_best[key]:
            raise RuntimeError("Dynamic mechanism-audit mismatch for {}".format(key))
        if float(static_mechanism["validation_metrics"][key]) != static_best[key]:
            raise RuntimeError("Static mechanism-audit mismatch for {}".format(key))
    if any(abs(float(value) - 1.0) > 1e-12 for value in static_mechanism["gate_mean"]):
        raise RuntimeError("Static constant-one gate audit is not exactly one")

    intervention = json.loads(INTERVENTION.read_text(encoding="utf-8"))["metrics"]
    dynamic_train = train_audit(DYNAMIC_LOG, 311)
    static_train = train_audit(args.static_log.resolve(), 293)
    gate_passed = (static_best[METRICS[0]] > 0.541 and
                   static_best[METRICS[3]] > 0.381)
    summaries = []
    for label, best, checkpoint in (
            ("dynamic_gate", dynamic_best, DYNAMIC_CHECKPOINT),
            ("static_residual", static_best, args.static_checkpoint.resolve())):
        summaries.append({
            "method": label, "selected_epoch": best["epoch"],
            **{key: best[key] for key in METRICS},
            "checkpoint": rel(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
        })
    summaries.append({"method": "static_checkpoint_intervention",
                      "selected_epoch": dynamic_best["epoch"],
                      **{key: float(intervention[key]) for key in METRICS},
                      "checkpoint": rel(DYNAMIC_CHECKPOINT),
                      "checkpoint_bytes": DYNAMIC_CHECKPOINT.stat().st_size,
                      "checkpoint_sha256": sha256(DYNAMIC_CHECKPOINT)})

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"dataset": "ODinW Pothole", "split": "validation",
                     "images": 133, "seed": 42, "epoch_budget": 12,
                     "selection": "maximum validation mAP; earliest epoch on ties",
                     "held_out_test_evaluated": False},
        "selected_runs": summaries,
        "training_audits": {"dynamic_gate": dynamic_train,
                            "static_residual": static_train},
        "mechanism": {
            "dynamic_gate_mean": dynamic_mechanism["gate_mean"],
            "dynamic_relative_residual_rms": dynamic_mechanism["relative_residual_rms_mean"],
            "static_gate_mean": static_mechanism["gate_mean"],
            "static_relative_residual_rms": static_mechanism["relative_residual_rms_mean"],
        },
        "replication_gate": {
            "rule": "Fresh static seed-42 mAP > 0.541 and APs > 0.381 on validation.",
            "status": "passed" if gate_passed else "failed",
            "seeds_0_21_authorized": gate_passed,
            "held_out_test_authorized": gate_passed,
        },
        "epochs": all_epochs,
    }

    json_path = ROOT / "results/json/scale_aware_seed42_validation_summary.json"
    epoch_csv = ROOT / "results/csv/scale_aware_seed42_validation_epochs.csv"
    run_csv = ROOT / "results/csv/scale_aware_seed42_validation_comparison.csv"
    report_path = ROOT / "reports/scale_aware_seed42_validation.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with epoch_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_epochs[0]))
        writer.writeheader(); writer.writerows(all_epochs)
    with run_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)

    curves = {"Dynamic gate": [(row["epoch"], row[METRICS[0]]) for row in dynamic_rows],
              "Static residual": [(row["epoch"], row[METRICS[0]]) for row in static_rows]}
    with BASELINE_EPOCHS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) == 42 and int(row["epoch"]) <= 12 and row["method"] in ("lora_r16", "uniform_gate"):
                label = {"lora_r16": "LoRA-only", "uniform_gate": "Uniform prompt"}[row["method"]]
                curves.setdefault(label, []).append((int(row["epoch"]), float(row[METRICS[0]])))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    for label, points in curves.items():
        points.sort()
        axes[0].plot([p[0] for p in points], [p[1] for p in points], marker="o",
                     markersize=2.7, linewidth=1.5, label=label)
    axes[0].set(xlabel="Epoch", ylabel="Validation COCO bbox mAP", xticks=range(1, 13))
    axes[0].grid(alpha=0.22); axes[0].legend(fontsize=8)
    labels = ["Dynamic\ngate", "Static\nresidual", "Static\nintervention"]
    x = range(3)
    axes[1].bar([i - 0.18 for i in x], [row[METRICS[0]] for row in summaries],
                width=0.36, label="mAP", color="#0072B2")
    axes[1].bar([i + 0.18 for i in x], [row[METRICS[3]] for row in summaries],
                width=0.36, label="APs", color="#D55E00")
    axes[1].set_xticks(list(x), labels); axes[1].set_ylabel("Validation metric")
    axes[1].grid(axis="y", alpha=0.22); axes[1].legend(fontsize=8)
    fig.tight_layout()
    stem = ROOT / "results/figures/scale_aware_seed42_validation"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(stem) + "." + suffix,
                    dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    report = """# Scale-aware Seed-42 Validation Screen

This report is regenerated from complete 12-epoch scalar histories, independent checkpoint reloads, gradient/memory audits, and mechanism audits. Selection uses only the unchanged 133-image ODinW Pothole validation split; held-out test data were not evaluated.

## Results

- Dynamic gate: earliest-best epoch {de}, mAP/APs = {dm:.3f}/{ds:.3f}.
- Fresh static residual: earliest-best epoch {se}, mAP/APs = {sm:.3f}/{ss:.3f}.
- Constant-one intervention on the dynamic checkpoint: mAP/APs = {im:.3f}/{is_:.3f}.
- Independent reloads exactly match all six selected metrics. All gradient audits are finite. Peak reserved CUDA memory is {dr:.2f} GiB (dynamic) and {sr:.2f} GiB (static).

## Registered decision

The static replication gate **{gate}**: a fresh run had to exceed both 0.541 mAP and 0.381 APs. Seeds 0/21 and held-out test are therefore {authorization} for this candidate. The checkpoint intervention is causal evidence about inference-time gating, but it is not a substitute for independent-seed replication.
""".format(de=dynamic_best["epoch"], dm=dynamic_best[METRICS[0]], ds=dynamic_best[METRICS[3]],
           se=static_best["epoch"], sm=static_best[METRICS[0]], ss=static_best[METRICS[3]],
           im=intervention[METRICS[0]], is_=intervention[METRICS[3]],
           dr=dynamic_train["peak_cuda_reserved_gib"], sr=static_train["peak_cuda_reserved_gib"],
           gate="passed" if gate_passed else "failed",
           authorization="authorized" if gate_passed else "not authorized")
    report_path.write_text(report, encoding="utf-8")
    print("gate={} static_best_epoch={} mAP={:.3f} APs={:.3f}".format(
        payload["replication_gate"]["status"], static_best["epoch"],
        static_best[METRICS[0]], static_best[METRICS[3]]))
    print("json={} epochs={} runs={} report={} figure={}.png".format(
        json_path, epoch_csv, run_csv, report_path, stem))


if __name__ == "__main__":
    main()
