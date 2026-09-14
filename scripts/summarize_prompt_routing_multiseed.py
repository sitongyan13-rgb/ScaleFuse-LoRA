#!/usr/bin/env python
"""Audit and summarize the registered 13-epoch prompt-routing comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
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
METHODS = {
    "lora_r16": {
        "label": "LoRA-only r16",
        "trainable_parameters": 2425280,
        "original_dir": "experiments/odinw_pothole_fusion_lora_r16_seed{seed}",
        "extension_dir": "experiments/odinw_pothole_lora_r16_seed{seed}_extend13",
        "original_log": {
            0: "experiments/odinw_pothole_fusion_lora_r16_seed0/20260812_101006/20260812_101006.log",
            21: "experiments/odinw_pothole_fusion_lora_r16_seed21/20260812_110115/20260812_110115.log",
            42: "experiments/odinw_pothole_fusion_lora_r16_seed42/20260801_003115/20260801_003115.log",
        },
        "extension_glob": {
            0: "experiments/logs/odinw_pothole_lora_r16_seed0_extend13_s0_*.log",
            21: "experiments/logs/odinw_pothole_lora_r16_seed21_extend13_s21_*.log",
            42: "experiments/logs/odinw_pothole_lora_r16_seed42_extend20_s42_*.log",
        },
        "reload": {
            0: "experiments/raw_metrics/odinw_pothole_fusion_lora_r16_seed0_best_eval_s0_20260812T025944Z.json",
            21: "experiments/raw_metrics/odinw_pothole_fusion_lora_r16_seed21_best_eval_s21_20260812T034620Z.json",
            42: "experiments/raw_metrics/odinw_pothole_fusion_lora_r16_seed42_best_eval_s42_20260731T180116Z.json",
        },
    },
    "learned_gate": {
        "label": "Learned gate + prompt + LoRA",
        "trainable_parameters": 2527304,
        "original_dir": "experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed{seed}",
        "extension_dir": "experiments/odinw_pothole_learned_prompt_lora_r16_seed{seed}_extend13",
        "original_log": {
            0: "experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed0/20260812_174424/20260812_174424.log",
            21: "experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed21/20260812_183045/20260812_183045.log",
            42: "experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed42/20260812_165432/20260812_165432.log",
        },
        "extension_glob": {
            0: "experiments/logs/odinw_pothole_learned_prompt_lora_r16_seed0_extend13_s0_*.log",
            21: "experiments/logs/odinw_pothole_learned_prompt_lora_r16_seed21_extend13_s21_*.log",
            42: "experiments/logs/odinw_pothole_learned_prompt_lora_r16_seed42_extend20_s42_*.log",
        },
        "reload": {
            0: "experiments/raw_metrics/odinw_pothole_domain_prompt_fusion_lora_r16_seed0_best_eval.json",
            21: "experiments/raw_metrics/odinw_pothole_learned_prompt_lora_r16_seed21_epoch13selected_validation_s21_*.json",
            42: "experiments/raw_metrics/odinw_pothole_domain_prompt_fusion_lora_r16_seed42_best_eval.json",
        },
    },
    "uniform_gate": {
        "label": "Uniform gate + prompt + LoRA",
        "trainable_parameters": 2492864,
        "original_dir": "experiments/odinw_pothole_uniform_prompt_lora_r16_seed{seed}",
        "extension_dir": "experiments/odinw_pothole_uniform_prompt_lora_r16_seed{seed}_extend13",
        "extension_dir_seed42": "experiments/odinw_pothole_uniform_prompt_lora_r16_seed42_extend20",
        "original_log": {
            0: "experiments/logs/odinw_pothole_uniform_prompt_lora_r16_seed0_s0_*.log",
            21: "experiments/logs/odinw_pothole_uniform_prompt_lora_r16_seed21_s21_*.log",
            42: "experiments/odinw_pothole_uniform_prompt_lora_r16_seed42/20260812_222851/20260812_222851.log",
        },
        "extension_glob": {
            0: "experiments/logs/odinw_pothole_uniform_prompt_lora_r16_seed0_extend13_s0_*.log",
            21: "experiments/logs/odinw_pothole_uniform_prompt_lora_r16_seed21_extend13_s21_*.log",
            42: "experiments/logs/odinw_pothole_uniform_prompt_lora_r16_seed42_extend20_s42_*.log",
        },
        "reload": {
            0: "experiments/raw_metrics/odinw_pothole_uniform_prompt_lora_r16_seed0_epoch13selected_validation_s0_*.json",
            21: "experiments/raw_metrics/odinw_pothole_uniform_prompt_lora_r16_seed21_epoch13selected_validation_s21_*.json",
            42: "experiments/raw_metrics/odinw_pothole_uniform_prompt_lora_r16_seed42_extend20_best_eval.json",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-reload", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_unique(pattern: str) -> Path:
    path = ROOT / pattern
    if "*" not in pattern:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    matches = sorted(ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError("Expected one match for {!r}, found {}".format(pattern, matches))
    return matches[0]


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    epochs = []
    for match in VAL_PATTERN.finditer(text):
        row = {"epoch": int(match.group(1)), "source_log": rel(path)}
        row.update(dict(zip(METRICS, map(float, match.groups()[1:]))))
        epochs.append(row)
    audits = []
    for match in AUDIT_PATTERN.finditer(text):
        audits.append(
            {
                "iteration": int(match.group(1)),
                "tensors": int(match.group(2)),
                "nonzero": int(match.group(3)),
                "nonfinite": int(match.group(4)),
                "global_norm": float(match.group(5)),
                "allocated": int(match.group(6)),
                "reserved": int(match.group(7)),
            }
        )
    return epochs, audits


def merge_curve(method: str, seed: int, meta: dict):
    original_spec = meta["original_log"][seed]
    original = resolve_unique(original_spec)
    extensions = sorted(ROOT.glob(meta["extension_glob"][seed]))
    if not extensions:
        raise RuntimeError("Missing extension log for {} seed {}".format(method, seed))
    epoch_map = {}
    audits = []
    sources = [original] + extensions
    for path in sources:
        rows, log_audits = parse_log(path)
        audits.extend(log_audits)
        for row in rows:
            epoch = row["epoch"]
            if epoch > 13:
                continue
            if epoch in epoch_map:
                if any(epoch_map[epoch][key] != row[key] for key in METRICS):
                    raise RuntimeError(
                        "Conflicting metrics for {} seed {} epoch {}".format(
                            method, seed, epoch
                        )
                    )
                continue
            epoch_map[epoch] = row
    missing = [epoch for epoch in range(1, 14) if epoch not in epoch_map]
    if missing:
        raise RuntimeError("Missing {} seed {} epochs {}".format(method, seed, missing))
    if not audits or max(item["nonfinite"] for item in audits) != 0:
        raise RuntimeError("Gradient audit failed for {} seed {}".format(method, seed))
    if min(item["nonzero"] for item in audits) <= 0:
        raise RuntimeError("Zero-gradient audit for {} seed {}".format(method, seed))
    return [epoch_map[epoch] for epoch in range(1, 14)], audits, list(map(rel, sources))


def checkpoint_for(meta: dict, seed: int, epoch: int) -> Path:
    directory = ROOT / meta["original_dir"].format(seed=seed)
    if epoch == 13:
        extension_key = "extension_dir_seed42" if seed == 42 and "extension_dir_seed42" in meta else "extension_dir"
        directory = ROOT / meta[extension_key].format(seed=seed)
    if epoch <= 12:
        online = directory / "best_coco_bbox_mAP_epoch_{}.pth".format(epoch)
        if online.is_file():
            return online
    checkpoint = directory / "epoch_{}.pth".format(epoch)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def load_reload(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    metrics = payload.get("metrics", payload)
    return payload, {key: float(metrics[key]) for key in METRICS}


def selected_run(method: str, seed: int, meta: dict, curve, audits, sources, require_reload):
    # Chronological max deliberately chooses the earliest epoch when mAP ties.
    best = max(curve, key=lambda row: row["coco/bbox_mAP"])
    checkpoint = checkpoint_for(meta, seed, best["epoch"])
    reload_pattern = meta["reload"][seed]
    reload_path = None
    reload_status = "missing"
    reload_metrics = None
    try:
        reload_path = resolve_unique(reload_pattern)
    except (FileNotFoundError, RuntimeError):
        if require_reload:
            raise
    if reload_path is not None:
        payload, reload_metrics = load_reload(reload_path)
        mismatch = {
            key: (best[key], reload_metrics[key])
            for key in METRICS
            if best[key] != reload_metrics[key]
        }
        if mismatch:
            raise RuntimeError(
                "Reload mismatch for {} seed {}: {}".format(method, seed, mismatch)
            )
        payload_checkpoint = Path(payload["checkpoint"]).resolve()
        if payload_checkpoint != checkpoint.resolve():
            raise RuntimeError(
                "Reload checkpoint mismatch for {} seed {}: {} != {}".format(
                    method, seed, payload_checkpoint, checkpoint
                )
            )
        reload_status = "matched"
    return {
        "method": method,
        "label": meta["label"],
        "seed": seed,
        "best_epoch": best["epoch"],
        "epoch12_mAP": curve[11]["coco/bbox_mAP"],
        "epoch13_mAP": curve[12]["coco/bbox_mAP"],
        "epoch13_delta_vs_epoch12": (
            curve[12]["coco/bbox_mAP"] - curve[11]["coco/bbox_mAP"]
        ),
        "trainable_parameters": meta["trainable_parameters"],
        "gradient_audits": len(audits),
        "min_nonzero_gradient_tensors": min(item["nonzero"] for item in audits),
        "max_nonfinite_gradient_tensors": max(item["nonfinite"] for item in audits),
        "peak_cuda_allocated_gib": max(item["allocated"] for item in audits) / (1024 ** 3),
        "peak_cuda_reserved_gib": max(item["reserved"] for item in audits) / (1024 ** 3),
        "checkpoint": rel(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "independent_validation_json": rel(reload_path) if reload_path else None,
        "independent_reload_status": reload_status,
        "source_logs": "|".join(sources),
        **{"validation/{}".format(key): best[key] for key in METRICS},
    }


def mean_sd_ci(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    # Student-t 0.975 quantile for df=2; n is fixed to the registered 3 seeds.
    margin = 4.302652729 * sd / math.sqrt(3)
    return mean, sd, [mean - margin, mean + margin]


def summarize(runs):
    by_method = {}
    for method in METHODS:
        values = [
            row["validation/coco/bbox_mAP"]
            for row in runs
            if row["method"] == method
        ]
        mean, sd, ci = mean_sd_ci(values)
        by_method[method] = {
            "label": METHODS[method]["label"],
            "values": values,
            "mean": mean,
            "sample_sd": sd,
            "descriptive_95pct_t_interval": ci,
        }
    pairs = {}
    for candidate, reference in (
        ("learned_gate", "lora_r16"),
        ("uniform_gate", "lora_r16"),
        ("learned_gate", "uniform_gate"),
    ):
        deltas = []
        for seed in SEEDS:
            candidate_row = next(
                row for row in runs if row["method"] == candidate and row["seed"] == seed
            )
            reference_row = next(
                row for row in runs if row["method"] == reference and row["seed"] == seed
            )
            deltas.append(
                candidate_row["validation/coco/bbox_mAP"]
                - reference_row["validation/coco/bbox_mAP"]
            )
        mean, sd, ci = mean_sd_ci(deltas)
        pairs["{}_minus_{}".format(candidate, reference)] = {
            "paired_deltas": deltas,
            "mean": mean,
            "sample_sd": sd,
            "descriptive_95pct_t_interval": ci,
            "positive_seeds": sum(delta > 0 for delta in deltas),
            "nonnegative_seeds": sum(delta >= 0 for delta in deltas),
        }
    gate_pair = pairs["uniform_gate_minus_lora_r16"]
    gate_passed = (
        gate_pair["positive_seeds"] == len(SEEDS)
        and gate_pair["mean"] > gate_pair["sample_sd"]
    )
    return {
        "methods": by_method,
        "paired_comparisons": pairs,
        "uniform_validation_stability_gate": {
            "status": "passed" if gate_passed else "failed",
            "rule": (
                "Pass only if uniform-minus-LoRA mAP is positive for all three "
                "registered seeds and its paired mean exceeds its paired sample SD."
            ),
            "held_out_test_authorized": gate_passed,
        },
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for row in rows:
        normalized.append(
            {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def render_figure(runs, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"lora_r16": "#0072B2", "learned_gate": "#D55E00", "uniform_gate": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    x = list(range(len(SEEDS)))
    for method, meta in METHODS.items():
        values = [
            next(
                row["validation/coco/bbox_mAP"]
                for row in runs
                if row["method"] == method and row["seed"] == seed
            )
            for seed in SEEDS
        ]
        axes[0].plot(x, values, marker="o", linewidth=1.8, color=colors[method], label=meta["label"])
    axes[0].set_xticks(x, list(map(str, SEEDS)))
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel("Validation mAP")
    axes[0].set_title("Validation-selected epoch (1–13)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    methods = list(METHODS)
    means = [summary["methods"][method]["mean"] for method in methods]
    errors = [summary["methods"][method]["sample_sd"] for method in methods]
    bars = axes[1].bar(
        range(3), means, yerr=errors, capsize=5, color=[colors[method] for method in methods]
    )
    axes[1].set_xticks(range(3), ["LoRA-only", "Learned\ngate", "Uniform\ngate"])
    axes[1].set_ylabel("Validation mAP")
    axes[1].set_title("Mean ± sample SD (3 seeds)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(min(means) - max(errors) - 0.01, max(means) + max(errors) + 0.012)
    for bar, value in zip(bars, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            "{:.4f}".format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle("ODinW Pothole prompt-routing comparison (validation only)")
    fig.tight_layout()
    prefix = ROOT / "results" / "figures" / "prompt_routing_multiseed_validation"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix.with_suffix(".png")), dpi=600)
    fig.savefig(str(prefix.with_suffix(".pdf")))
    fig.savefig(str(prefix.with_suffix(".svg")))
    plt.close(fig)


def render_report(summary, runs):
    gate = summary["uniform_validation_stability_gate"]
    lines = [
        "# Prompt-routing Multi-seed Validation",
        "",
        "Generated from raw logs and independent validation JSON files; no held-out test metric is used.",
        "",
        "## Validation result",
        "",
        "| Method | Seed mAP (0/21/42) | Mean ± sample SD | Best epochs |",
        "|---|---:|---:|---:|",
    ]
    for method, meta in METHODS.items():
        selected = [row for row in runs if row["method"] == method]
        item = summary["methods"][method]
        lines.append(
            "| {} | {} | {:.4f} ± {:.4f} | {} |".format(
                meta["label"],
                "/".join("{:.3f}".format(row["validation/coco/bbox_mAP"]) for row in selected),
                item["mean"],
                item["sample_sd"],
                "/".join(str(row["best_epoch"]) for row in selected),
            )
        )
    pair = summary["paired_comparisons"]["uniform_gate_minus_lora_r16"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Uniform-minus-LoRA paired changes are {} (mean {:+.4f}, sample SD {:.4f}).".format(
                "/".join("{:+.3f}".format(value) for value in pair["paired_deltas"]),
                pair["mean"],
                pair["sample_sd"],
            ),
            "The pre-registered stability gate **{}**. Held-out-test evaluation is {}authorized.".format(
                gate["status"], "" if gate["held_out_test_authorized"] else "not "
            ),
            "",
            "The three-seed t intervals are descriptive because n=3; they are not used to claim statistical significance.",
        ]
    )
    path = ROOT / "reports" / "prompt_routing_multiseed_validation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_latex(summary, runs):
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Seed 0 & Seed 21 & Seed 42 & Mean $\pm$ SD & Best epochs \\",
        r"\midrule",
    ]
    for method, meta in METHODS.items():
        selected = [row for row in runs if row["method"] == method]
        item = summary["methods"][method]
        label = meta["label"].replace("+", r"$+$")
        lines.append(
            "{} & {} & {} & {} & {:.4f} $\\pm$ {:.4f} & {} \\\\".format(
                label,
                *("{:.3f}".format(row["validation/coco/bbox_mAP"]) for row in selected),
                item["mean"],
                item["sample_sd"],
                "/".join(str(row["best_epoch"]) for row in selected),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path = ROOT / "results" / "latex" / "prompt_routing_multiseed_validation.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    runs = []
    epoch_rows = []
    for method, meta in METHODS.items():
        for seed in SEEDS:
            curve, audits, sources = merge_curve(method, seed, meta)
            for row in curve:
                epoch_rows.append({"method": method, "seed": seed, **row})
            runs.append(
                selected_run(method, seed, meta, curve, audits, sources, args.require_reload)
            )
    summary = summarize(runs)
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "ODinW Pothole",
            "selection_split": "validation",
            "seeds": list(SEEDS),
            "epoch_budget": 13,
            "checkpoint_selection": "maximum validation mAP; earliest epoch on ties",
            "held_out_test_evaluated": False,
        },
        "summary": summary,
        "runs": runs,
    }
    json_path = ROOT / "results" / "json" / "prompt_routing_multiseed_validation_summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(ROOT / "results" / "csv" / "prompt_routing_multiseed_validation_runs.csv", runs)
    write_csv(ROOT / "results" / "csv" / "prompt_routing_multiseed_validation_epochs.csv", epoch_rows)
    render_figure(runs, summary)
    render_report(summary, runs)
    render_latex(summary, runs)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
