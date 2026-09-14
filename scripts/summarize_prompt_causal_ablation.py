#!/usr/bin/env python
"""Summarize the fixed seed-42 causal ablation after independent evaluation."""

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
)
VARIANTS = (
    {
        "variant": "LoRA-only r16",
        "visual_conditioned_gate": False,
        "trainable_prompt": False,
        "trainable_lora": True,
        "trainable_parameters": 2425280,
        "metric": "experiments/raw_metrics/odinw_pothole_fusion_lora_r16_seed42_best_eval_s42_20260731T180116Z.json",
        "visuals": "experiments/odinw_pothole_fusion_lora_r16_seed42_best_eval/visualizations",
    },
    {
        "variant": "learned gate + prompt + LoRA r16",
        "visual_conditioned_gate": True,
        "trainable_prompt": True,
        "trainable_lora": True,
        "trainable_parameters": 2527304,
        "metric": "experiments/raw_metrics/odinw_pothole_domain_prompt_fusion_lora_r16_seed42_best_eval.json",
        "visuals": "experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed42_eval/visualizations",
    },
    {
        "variant": "prompt-only",
        "visual_conditioned_gate": True,
        "trainable_prompt": True,
        "trainable_lora": False,
        "trainable_parameters": 102024,
        "metric": "experiments/raw_metrics/odinw_pothole_prompt_only_seed42_best_eval.json",
        "visuals": "experiments/odinw_pothole_prompt_only_seed42_eval/visualizations",
    },
    {
        "variant": "uniform gate + prompt + LoRA r16",
        "visual_conditioned_gate": False,
        "trainable_prompt": True,
        "trainable_lora": True,
        "trainable_parameters": 2492864,
        "metric": "experiments/raw_metrics/odinw_pothole_uniform_prompt_lora_r16_seed42_best_eval.json",
        "visuals": "experiments/odinw_pothole_uniform_prompt_lora_r16_seed42_eval/visualizations",
    },
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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def find_training_log(checkpoint):
    work_dir = checkpoint.parent
    candidates = sorted(work_dir.glob("*/**/*.log"))
    candidates = [path for path in candidates if "reload_eval" not in str(path)]
    if not candidates:
        raise FileNotFoundError("No training log below {}".format(work_dir))
    return candidates[-1]


def validation_rows(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in VAL_PATTERN.finditer(text):
        row = {"epoch": int(match.group(1))}
        row.update(dict(zip(METRICS, map(float, match.groups()[1:]))))
        rows.append(row)
    return rows, text


def load_variant(spec):
    metric_path = ROOT / spec["metric"]
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    checkpoint = Path(payload["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metrics = payload["metrics"]
    log_path = find_training_log(checkpoint)
    epochs, log_text = validation_rows(log_path)
    if len(epochs) != 12:
        raise RuntimeError("{} has {} validation epochs".format(spec["variant"], len(epochs)))
    best = max(epochs, key=lambda row: row["coco/bbox_mAP"])
    for metric in METRICS:
        if abs(float(metrics[metric]) - best[metric]) > 1e-12:
            raise RuntimeError("Independent reload mismatch: {} {}".format(spec["variant"], metric))
    audits = re.findall(r"gradient_audit .*?nonfinite=(\d+)", log_text)
    if not audits or any(int(value) for value in audits):
        raise RuntimeError("Gradient audit failed: {}".format(spec["variant"]))
    visuals = list((ROOT / spec["visuals"]).glob("*.*"))
    row = dict(spec)
    row.pop("metric")
    row.pop("visuals")
    row.update(
        {
            "seed": 42,
            "best_epoch": best["epoch"],
            "final_validation_mAP": epochs[-1]["coco/bbox_mAP"],
            "gradient_audits": len(audits),
            "max_nonfinite_gradient_tensors": max(map(int, audits)),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            "training_log": str(log_path.relative_to(ROOT)),
            "independent_validation_json": spec["metric"],
            "validation_visualizations": len(
                [p for p in visuals if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            ),
        }
    )
    row.update({"validation/{}".format(key): float(metrics[key]) for key in METRICS})
    return row, epochs


def render(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["LoRA\nonly", "Learned gate\n+ prompt + LoRA", "Prompt\nonly", "Uniform gate\n+ prompt + LoRA"]
    values = [row["validation/coco/bbox_mAP"] for row in rows]
    colors = ["#6c8ebf", "#d79b00", "#82b366", "#9673a6"]
    fig, axis = plt.subplots(figsize=(8.2, 4.7))
    bars = axis.bar(range(4), values, color=colors, width=0.68)
    axis.set_xticks(range(4), labels)
    axis.set_ylabel("Validation mAP")
    axis.set_title("Pothole seed-42 causal prompt ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(max(0.0, min(values) - 0.04), max(values) + 0.025)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.002, "{:.3f}".format(value), ha="center")
    fig.tight_layout()
    prefix = ROOT / "results" / "figures" / "prompt_causal_ablation_seed42"
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"dpi": 600} if suffix == ".png" else {}
        fig.savefig(str(prefix.with_suffix(suffix)), **kwargs)
    plt.close(fig)


def main():
    rows = []
    epoch_rows = []
    for spec in VARIANTS:
        row, epochs = load_variant(spec)
        rows.append(row)
        for epoch in epochs:
            epoch_row = {"variant": row["variant"], "seed": 42}
            epoch_row.update(epoch)
            epoch_rows.append(epoch_row)
    baseline = rows[0]["validation/coco/bbox_mAP"]
    learned = rows[1]["validation/coco/bbox_mAP"]
    for row in rows:
        row["delta_vs_lora_only_mAP"] = row["validation/coco/bbox_mAP"] - baseline
        row["delta_vs_learned_gate_mAP"] = row["validation/coco/bbox_mAP"] - learned
    csv_path = ROOT / "results" / "csv" / "prompt_causal_ablation_seed42.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    epoch_csv_path = ROOT / "results" / "csv" / "prompt_causal_ablation_seed42_epochs.csv"
    with epoch_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)
    output = {
        "protocol": {
            "dataset": "ODinW Pothole",
            "selection_split": "validation",
            "seed": 42,
            "checkpoint_selection": "maximum validation coco/bbox_mAP",
            "held_out_test_evaluated": False,
        },
        "runs": rows,
    }
    json_path = ROOT / "results" / "json" / "prompt_causal_ablation_seed42.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    render(rows)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
