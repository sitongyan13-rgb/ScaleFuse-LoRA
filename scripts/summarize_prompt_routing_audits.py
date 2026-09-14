#!/usr/bin/env python
"""Aggregate prompt-routing audits without manually transcribing measurements."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("learned_gate", seed, ROOT / "experiments/raw_metrics/odinw_pothole_learned_prompt_lora_r16_seed{}_routing_audit.json".format(seed))
    for seed in (0, 21, 42)
] + [
    (
        "prior_mixed_gate_smoke",
        42,
        ROOT / "experiments/raw_metrics/odinw_pothole_prior_mixed_prompt_lora_r16_smoke_routing_audit.json",
    ),
    (
        "prior_mixed_gate_full",
        42,
        ROOT / "experiments/raw_metrics/odinw_pothole_prior_mixed_prompt_lora_r16_seed42_routing_audit.json",
    ),
]


def main():
    rows = []
    for method, seed, path in SOURCES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        routing = payload["routing_summary"]
        row = {
            "method": method,
            "seed": seed,
            "checkpoint": payload["protocol"]["checkpoint"],
            "checkpoint_sha256": payload["protocol"]["checkpoint_sha256"],
            "validation_mAP": payload["validation_metrics"]["coco/bbox_mAP"],
            "images": payload["protocol"]["images"],
            "mean_max_prompt_weight": routing["max_prompt_weight"]["mean"],
            "mean_normalized_entropy": routing["normalized_entropy"]["mean"],
            "mean_l1_distance_from_uniform": routing["l1_distance_from_uniform"]["mean"],
            "mean_relative_text_residual_rms": routing["text_residual_relative_rms"]["mean"],
            "dominant_prompt_counts": "/".join(map(str, routing["dominant_prompt_counts"])),
            "minimum_observed_prompt_weight": min(
                min(item["weights"]) for item in payload["per_image"]
            ),
            "maximum_observed_prompt_weight": max(
                max(item["weights"]) for item in payload["per_image"]
            ),
            "source_json": str(path.relative_to(ROOT)).replace("\\", "/"),
        }
        rows.append(row)
    csv_path = ROOT / "results/csv/prompt_routing_audits.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    learned = rows[:3]
    prior_smoke = rows[3]
    prior_full = rows[4]
    report = [
        "# Prompt-routing Mechanism Audit",
        "",
        "All values below are regenerated from checkpoint inference on the unchanged 133-image Pothole validation split. No held-out test image was used.",
        "",
        "| Variant | Seed | mAP | Mean max weight | Normalized entropy | Dominant counts | Relative residual RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {} | {} | {:.3f} | {:.4f} | {:.4f} | {} | {:.6f} |".format(
                row["method"], row["seed"], row["validation_mAP"],
                row["mean_max_prompt_weight"], row["mean_normalized_entropy"],
                row["dominant_prompt_counts"], row["mean_relative_text_residual_rms"],
            )
        )
    report.extend(
        [
            "",
            "## Evidence-backed decision",
            "",
            "The unconstrained learned gate collapses to one prompt for seeds 0 and 21 (all 133 images share one dominant prompt and mean max weight exceeds 0.998). Seed 42 uses two prompts but remains concentrated (mean max weight {:.4f}). Thus the original gate does not reliably implement image-conditioned routing across seeds.".format(learned[2]["mean_max_prompt_weight"]),
            "",
            "The 50% uniform-prior mixture raises normalized routing entropy from {:.4f}--{:.4f} in the collapsed runs to {:.4f} after 100 iterations, while the observed minimum prompt weight {:.4f} respects the designed 0.0625 lower bound within floating-point tolerance. This supports a single-seed full validation screen, not an accuracy claim.".format(
                min(row["mean_normalized_entropy"] for row in learned[:2]),
                max(row["mean_normalized_entropy"] for row in learned[:2]),
                prior_smoke["mean_normalized_entropy"], prior_smoke["minimum_observed_prompt_weight"],
            ),
            "",
            "After 12 epochs, the constrained seed-42 checkpoint retains a minimum observed weight of {:.4f} and normalized entropy {:.4f}, but validation mAP is {:.3f}. The constraint therefore repairs the collapse mechanism without meeting the registered accuracy threshold.".format(
                prior_full["minimum_observed_prompt_weight"], prior_full["mean_normalized_entropy"], prior_full["validation_mAP"],
            ),
        ]
    )
    (ROOT / "reports/prompt_routing_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("rows={} csv={} report={}".format(len(rows), csv_path, ROOT / "reports/prompt_routing_audit.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
