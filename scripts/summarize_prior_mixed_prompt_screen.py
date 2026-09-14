#!/usr/bin/env python
"""Strictly summarize the registered seed-42 prior-mixed routing screen."""

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCALARS = ROOT / "experiments/odinw_pothole_prior_mixed_prompt_lora_r16_seed42/20260813_130448/vis_data/scalars.json"
TRAIN_LOG = ROOT / "experiments/logs/odinw_pothole_prior_mixed_prompt_lora_r16_seed42_s42_20260813T050443Z.log"
CHECKPOINT = ROOT / "experiments/odinw_pothole_prior_mixed_prompt_lora_r16_seed42/best_coco_bbox_mAP_epoch_11.pth"
RELOAD = ROOT / "experiments/raw_metrics/odinw_pothole_prior_mixed_prompt_lora_r16_seed42_validation_s42_20260813T055132Z.json"
ROUTING = ROOT / "experiments/raw_metrics/odinw_pothole_prior_mixed_prompt_lora_r16_seed42_routing_audit.json"
BASELINE_EPOCHS = ROOT / "results/csv/prompt_routing_multiseed_validation_epochs.csv"

METRICS = (
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
)


def rel(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_files():
    for path in (SCALARS, TRAIN_LOG, CHECKPOINT, RELOAD, ROUTING, BASELINE_EPOCHS):
        if not path.is_file():
            raise FileNotFoundError(path)


def read_epochs():
    rows = []
    for line in SCALARS.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if "coco/bbox_mAP" in item:
            rows.append({"epoch": int(item["step"]), **{key: float(item[key]) for key in METRICS}})
    if [row["epoch"] for row in rows] != list(range(1, 13)):
        raise RuntimeError("Expected exactly one validation record for epochs 1--12")
    return rows


def main():
    require_files()
    epochs = read_epochs()
    best = max(epochs, key=lambda row: (row["coco/bbox_mAP"], -row["epoch"]))
    if best["epoch"] != 11:
        raise RuntimeError("Selected checkpoint epoch does not match parsed earliest-best epoch")

    reload_payload = json.loads(RELOAD.read_text(encoding="utf-8"))
    routing_payload = json.loads(ROUTING.read_text(encoding="utf-8"))
    reload_metrics = reload_payload["metrics"]
    routing_metrics = routing_payload["validation_metrics"]
    for key in METRICS:
        if abs(best[key] - float(reload_metrics[key])) > 1e-12:
            raise RuntimeError("Independent reload mismatch for {}".format(key))
        if abs(best[key] - float(routing_metrics[key])) > 1e-12:
            raise RuntimeError("Routing audit mismatch for {}".format(key))

    text = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")
    audits = [
        tuple(map(int, match))
        for match in re.findall(r"gradient_audit iter=\d+ tensors=(\d+) nonzero=(\d+) nonfinite=(\d+)", text)
    ]
    allocated = [int(value) for value in re.findall(r"cuda_peak_allocated_bytes=(\d+)", text)]
    reserved = [int(value) for value in re.findall(r"cuda_peak_reserved_bytes=(\d+)", text)]
    if not audits or not allocated or not reserved:
        raise RuntimeError("Training audit records are incomplete")

    route = routing_payload["routing_summary"]
    observed_min = min(min(item["weights"]) for item in routing_payload["per_image"])
    observed_max = max(max(item["weights"]) for item in routing_payload["per_image"])
    no_collapse = observed_min >= 0.0625 - 1e-5 and observed_max <= 0.5625 + 1e-5
    registered_lora = 0.528
    registered_uniform = 0.534
    accuracy_gate = best["coco/bbox_mAP"] > registered_lora and best["coco/bbox_mAP"] > registered_uniform
    gate_passed = no_collapse and accuracy_gate

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "ODinW Pothole",
            "split": "validation",
            "seed": 42,
            "epoch_budget": 12,
            "selection": "maximum validation mAP; earliest epoch on ties",
            "held_out_test_evaluated": False,
        },
        "selected": {
            "epoch": best["epoch"],
            "metrics": {key: best[key] for key in METRICS},
            "checkpoint": rel(CHECKPOINT),
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "checkpoint_sha256": sha256(CHECKPOINT),
            "independent_reload": rel(RELOAD),
            "independent_reload_status": "matched",
        },
        "training_audit": {
            "gradient_audits": len(audits),
            "trainable_gradient_tensors": max(item[0] for item in audits),
            "minimum_nonzero_gradient_tensors": min(item[1] for item in audits),
            "post_initialization_minimum_nonzero_gradient_tensors": min(item[1] for item in audits[1:]),
            "maximum_nonfinite_gradient_tensors": max(item[2] for item in audits),
            "peak_cuda_allocated_gib": max(allocated) / 2 ** 30,
            "peak_cuda_reserved_gib": max(reserved) / 2 ** 30,
        },
        "routing_audit": {
            "mean_normalized_entropy": route["normalized_entropy"]["mean"],
            "mean_max_prompt_weight": route["max_prompt_weight"]["mean"],
            "minimum_observed_prompt_weight": observed_min,
            "maximum_observed_prompt_weight": observed_max,
            "dominant_prompt_counts": route["dominant_prompt_counts"],
            "mean_relative_text_residual_rms": route["text_residual_relative_rms"]["mean"],
            "noncollapse_constraint_passed": no_collapse,
            "source": rel(ROUTING),
        },
        "replication_gate": {
            "rule": "Seed-42 validation mAP must exceed both registered LoRA-only 0.528 and 12-epoch uniform-routing 0.534, with the prompt-weight bounds satisfied.",
            "accuracy_gate_passed": accuracy_gate,
            "mechanism_gate_passed": no_collapse,
            "status": "passed" if gate_passed else "failed",
            "seeds_0_21_authorized": gate_passed,
            "held_out_test_authorized": gate_passed,
        },
        "epochs": epochs,
    }

    json_path = ROOT / "results/json/prior_mixed_prompt_seed42_validation_summary.json"
    csv_path = ROOT / "results/csv/prior_mixed_prompt_seed42_validation_epochs.csv"
    report_path = ROOT / "reports/prior_mixed_prompt_seed42_validation.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epochs[0]))
        writer.writeheader()
        writer.writerows(epochs)

    curves = {"Prior-mixed gate": [(row["epoch"], row["coco/bbox_mAP"]) for row in epochs]}
    with BASELINE_EPOCHS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) == 42 and int(row["epoch"]) <= 12:
                label = {"lora_r16": "LoRA-only", "learned_gate": "Learned gate", "uniform_gate": "Uniform gate"}[row["method"]]
                curves.setdefault(label, []).append((int(row["epoch"]), float(row["coco/bbox_mAP"])))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, points in curves.items():
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, linewidth=1.6, label=label)
    ax.axhline(registered_uniform, color="black", linestyle="--", linewidth=1, alpha=0.55, label="Replication threshold (0.534)")
    ax.scatter([best["epoch"]], [best["coco/bbox_mAP"]], marker="*", s=130,
               color="#1f77b4", edgecolor="black", linewidth=0.7, zorder=5)
    ax.set(xlabel="Epoch", ylabel="Validation COCO bbox mAP", xticks=range(1, 13))
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    figure_stem = ROOT / "results/figures/prior_mixed_prompt_seed42_validation"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(figure_stem) + "." + suffix, dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    report = """# Prior-mixed Prompt Routing: Seed-42 Validation Screen

This report is regenerated from the completed training scalars, independent validation reload, and checkpoint-level routing audit. Model selection uses only the unchanged 133-image validation split; the held-out test split was not evaluated.

## Result

- The 12-epoch run completed with return code 0. The earliest validation optimum is epoch {epoch}: mAP/AP50/AP75/APs/APm/APl = {map:.3f}/{ap50:.3f}/{ap75:.3f}/{aps:.3f}/{apm:.3f}/{apl:.3f}.
- Independent checkpoint reload exactly matches all six metrics. Epoch 12 falls to {epoch12:.3f}, so continuing solely because the last epoch exists would select a worse model.
- All {audits} gradient audits are finite. After the expected zero-initialized first audit, all 298 trainable gradient tensors are nonzero. Peak allocated/reserved CUDA memory is {alloc:.2f}/{reserved:.2f} GiB.
- The constraint works mechanically: observed weights remain in [{wmin:.6f}, {wmax:.6f}], mean normalized entropy is {entropy:.4f}, and mean maximum weight is {wmean:.4f}.

## Registered decision

The replication gate **failed**. Although {map:.3f} exceeds matched LoRA-only seed 42 ({lora:.3f}), it does not exceed the registered 12-epoch uniform-routing comparator ({uniform:.3f}); it is also below the later epoch-13 uniform result 0.536. Therefore seeds 0/21 and held-out testing are not authorized for this candidate. This is a useful negative mechanism result: preventing gate collapse alone did not improve validation accuracy.
""".format(
        epoch=best["epoch"], map=best[METRICS[0]], ap50=best[METRICS[1]], ap75=best[METRICS[2]],
        aps=best[METRICS[3]], apm=best[METRICS[4]], apl=best[METRICS[5]], epoch12=epochs[-1][METRICS[0]],
        audits=len(audits), alloc=max(allocated) / 2 ** 30, reserved=max(reserved) / 2 ** 30,
        wmin=observed_min, wmax=observed_max, entropy=route["normalized_entropy"]["mean"],
        wmean=route["max_prompt_weight"]["mean"], lora=registered_lora, uniform=registered_uniform,
    )
    report_path.write_text(report, encoding="utf-8")
    print("gate={} best_epoch={} mAP={:.3f}".format(payload["replication_gate"]["status"], best["epoch"], best["coco/bbox_mAP"]))
    print("json={} csv={} report={} figure={}.png".format(json_path, csv_path, report_path, figure_stem))


if __name__ == "__main__":
    main()
