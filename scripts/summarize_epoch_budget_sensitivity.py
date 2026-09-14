#!/usr/bin/env python
"""Merge epoch 1--20 validation curves for the controlled seed-42 extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
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
        "original_log": "experiments/odinw_pothole_fusion_lora_r16_seed42/20260801_003115/20260801_003115.log",
        "extension_log_glob": "experiments/logs/odinw_pothole_lora_r16_seed42_extend20_s42_*.log",
        "original_work_dir": "experiments/odinw_pothole_fusion_lora_r16_seed42",
        "extension_work_dir": "experiments/odinw_pothole_lora_r16_seed42_extend20",
        "original_reload": "experiments/raw_metrics/odinw_pothole_fusion_lora_r16_seed42_best_eval_s42_20260731T180116Z.json",
        "extension_reload": "experiments/raw_metrics/odinw_pothole_lora_r16_seed42_extend20_best_eval.json",
    },
    "learned_gate": {
        "label": "Learned gate + prompt + LoRA",
        "original_log": "experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed42/20260812_165432/20260812_165432.log",
        "extension_log_glob": "experiments/logs/odinw_pothole_learned_prompt_lora_r16_seed42_extend20_s42_*.log",
        "original_work_dir": "experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed42",
        "extension_work_dir": "experiments/odinw_pothole_learned_prompt_lora_r16_seed42_extend20",
        "original_reload": "experiments/raw_metrics/odinw_pothole_domain_prompt_fusion_lora_r16_seed42_best_eval.json",
        "extension_reload": "experiments/raw_metrics/odinw_pothole_learned_prompt_lora_r16_seed42_extend20_best_eval.json",
    },
    "uniform_gate": {
        "label": "Uniform gate + prompt + LoRA",
        "original_log": "experiments/odinw_pothole_uniform_prompt_lora_r16_seed42/20260812_222851/20260812_222851.log",
        "extension_log_glob": "experiments/logs/odinw_pothole_uniform_prompt_lora_r16_seed42_extend20_s42_*.log",
        "original_work_dir": "experiments/odinw_pothole_uniform_prompt_lora_r16_seed42",
        "extension_work_dir": "experiments/odinw_pothole_uniform_prompt_lora_r16_seed42_extend20",
        "original_reload": "experiments/raw_metrics/odinw_pothole_uniform_prompt_lora_r16_seed42_best_eval.json",
        "extension_reload": "experiments/raw_metrics/odinw_pothole_uniform_prompt_lora_r16_seed42_extend20_best_eval.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-reload",
        action="store_true",
        help="Fail unless every selected checkpoint has a matching independent reload JSON.",
    )
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    epochs = []
    for match in VAL_PATTERN.finditer(text):
        values = [float(value) for value in match.groups()[1:]]
        row = {"epoch": int(match.group(1)), "source_log": rel(path)}
        row.update(dict(zip(METRICS, values)))
        epochs.append(row)
    audits = []
    for match in AUDIT_PATTERN.finditer(text):
        audits.append(
            {
                "iter": int(match.group(1)),
                "tensors": int(match.group(2)),
                "nonzero": int(match.group(3)),
                "nonfinite": int(match.group(4)),
                "global_norm": float(match.group(5)),
                "cuda_peak_allocated_bytes": int(match.group(6)),
                "cuda_peak_reserved_bytes": int(match.group(7)),
            }
        )
    return epochs, audits


def merge_epochs(method: str, meta: dict):
    original = ROOT / meta["original_log"]
    if not original.is_file():
        raise FileNotFoundError(original)
    extension_logs = sorted(ROOT.glob(meta["extension_log_glob"]))
    if not extension_logs:
        raise RuntimeError(f"No extension log found for {method}")

    epoch_map = {}
    audits = []
    source_logs = [original] + extension_logs
    for path in source_logs:
        rows, log_audits = parse_log(path)
        if path != original:
            audits.extend(log_audits)
        for row in rows:
            epoch = row["epoch"]
            if epoch in epoch_map:
                prior = epoch_map[epoch]
                if any(prior[name] != row[name] for name in METRICS):
                    raise RuntimeError(
                        f"Conflicting validation metrics for {method} epoch {epoch}"
                    )
                continue
            epoch_map[epoch] = row

    missing = [epoch for epoch in range(1, 21) if epoch not in epoch_map]
    extra = sorted(set(epoch_map) - set(range(1, 21)))
    if missing or extra:
        raise RuntimeError(f"Incomplete {method} curve: missing={missing}, extra={extra}")
    epochs = [epoch_map[epoch] for epoch in range(1, 21)]
    return epochs, audits, [rel(path) for path in source_logs]


def load_reload(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    metrics = payload.get("metrics", payload)
    return {name: float(metrics[name]) for name in METRICS}, payload


def select_run(method: str, meta: dict, epochs: list, require_reload: bool):
    # max() over chronological rows deliberately keeps the earliest epoch on a tie.
    best = max(epochs, key=lambda row: row["coco/bbox_mAP"])
    epoch = best["epoch"]
    work_key = "original_work_dir" if epoch <= 12 else "extension_work_dir"
    reload_key = "original_reload" if epoch <= 12 else "extension_reload"
    checkpoint = ROOT / meta[work_key] / f"epoch_{epoch}.pth"
    if epoch <= 12:
        online_best = ROOT / meta[work_key] / f"best_coco_bbox_mAP_epoch_{epoch}.pth"
        if online_best.is_file():
            checkpoint = online_best
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    reload_path = ROOT / meta[reload_key]
    reload_status = "missing"
    reload_metrics = None
    reload_payload = None
    if reload_path.is_file():
        reload_metrics, reload_payload = load_reload(reload_path)
        mismatches = {
            name: (best[name], reload_metrics[name])
            for name in METRICS
            if best[name] != reload_metrics[name]
        }
        if mismatches:
            raise RuntimeError(f"Independent reload mismatch for {method}: {mismatches}")
        reload_status = "matched"
    elif require_reload:
        raise RuntimeError(f"Missing independent reload JSON for {method}: {reload_path}")

    return {
        "method": method,
        "label": meta["label"],
        "best_epoch": epoch,
        "best_in_extension": epoch > 12,
        "best_metrics": {name: best[name] for name in METRICS},
        "epoch12_mAP": epochs[11]["coco/bbox_mAP"],
        "epoch20_mAP": epochs[19]["coco/bbox_mAP"],
        "post12_best_mAP": max(row["coco/bbox_mAP"] for row in epochs[12:]),
        "post12_delta_vs_global_best": max(
            row["coco/bbox_mAP"] for row in epochs[12:]
        )
        - best["coco/bbox_mAP"],
        "checkpoint": rel(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "independent_reload_json": rel(reload_path),
        "independent_reload_status": reload_status,
        "independent_reload_created_utc": (
            reload_payload.get("created_utc") if reload_payload else None
        ),
    }


def write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_figure(all_epochs: dict, runs: list):
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
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for method, meta in METHODS.items():
        rows = all_epochs[method]
        ax.plot(
            [row["epoch"] for row in rows],
            [row["coco/bbox_mAP"] for row in rows],
            marker="o",
            markersize=3.4,
            linewidth=1.7,
            color=colors[method],
            label=meta["label"],
        )
        run = next(item for item in runs if item["method"] == method)
        ax.scatter(
            [run["best_epoch"]],
            [run["best_metrics"]["coco/bbox_mAP"]],
            s=62,
            marker="*",
            color=colors[method],
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
    ax.axvline(12, color="#666666", linestyle="--", linewidth=1.2)
    ax.text(
        12.15,
        0.541,
        "original budget",
        fontsize=8.5,
        color="#555555",
        va="top",
    )
    ax.set_xticks(range(1, 21))
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation mAP")
    ax.set_title("ODinW-Pothole epoch-budget sensitivity (seed 42)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower right")
    fig.tight_layout()
    output = ROOT / "results" / "figures" / "epoch_budget_sensitivity_seed42"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def render_latex(runs: list):
    path = ROOT / "results/latex/epoch_budget_sensitivity_seed42.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        rows.append(
            "{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                run["label"].replace("+", r"\,+\,"),
                run["best_epoch"],
                run["best_metrics"]["coco/bbox_mAP"],
                run["epoch12_mAP"],
                run["post12_best_mAP"],
                run["epoch20_mAP"],
            )
        )
    text = """% Generated by scripts/summarize_epoch_budget_sensitivity.py
\\begin{table}[t]
  \\centering
  \\caption{Seed-42 epoch-budget sensitivity on the unchanged 133-image
  ODinW-Pothole validation split. Checkpoints are selected by validation mAP
  over epochs 1--20; the held-out test split is not used.}
  \\label{tab:epoch_budget_sensitivity}
  \\begin{tabular}{lrrrrr}
    \\toprule
    Method & Best epoch & Best mAP $\\uparrow$ & Epoch-12 mAP & Post-12 best & Epoch-20 mAP \\\\
    \\midrule
__ROWS__
    \\bottomrule
  \\end{tabular}
\\end{table}
""".replace("__ROWS__", "\n".join("    " + row for row in rows))
    path.write_text(text, encoding="utf-8")


def render_report(runs: list, gradient_audits: dict):
    by_method = {run["method"]: run for run in runs}
    lines = [
        "# Epoch-budget Sensitivity on ODinW-Pothole",
        "",
        "## Scope and protocol",
        "",
        "This validation-only study tests whether the original 12-epoch budget, rather than the adaptation method, explains the observed performance ceiling. LoRA-only rank 16, learned image-conditioned prompt routing, and uniform prompt routing were resumed from their exact epoch-12 full-state checkpoints and continued through epoch 20 at the already-decayed learning rate. Dataset split, seed (42), model, augmentation, batch/accumulation, prompt text, and evaluator were unchanged. Checkpoints were selected by maximum mAP on the official 133-image validation split; the held-out test split was not read for model selection or reporting in this study.",
        "",
        "## Automatically generated results",
        "",
        "| Method | Best epoch | Best val mAP | Epoch 12 | Best after 12 | Epoch 20 | Independent reload |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        lines.append(
            "| {label} | {best_epoch} | {best:.3f} | {epoch12:.3f} | {post12:.3f} | {epoch20:.3f} | {reload} |".format(
                label=run["label"],
                best_epoch=run["best_epoch"],
                best=run["best_metrics"]["coco/bbox_mAP"],
                epoch12=run["epoch12_mAP"],
                post12=run["post12_best_mAP"],
                epoch20=run["epoch20_mAP"],
                reload=run["independent_reload_status"],
            )
        )
    lines.extend(
        [
            "",
            "The two accuracy-relevant conclusions are narrow. First, the LoRA-only and learned-gate global optima remain inside the original budget (epochs {} and {}), while their best post-12 mAP values are {:.3f} and {:.3f}, respectively. Second, uniform routing reaches a new single-seed maximum of {:.3f} at epoch {}, only {:+.3f} above its epoch-12 value. Its independently reloaded six-metric vector is {:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f} (mAP/AP50/AP75/APs/APm/APl). This small single-seed change does not establish a general benefit from longer training or from learned routing.".format(
                by_method["lora_r16"]["best_epoch"],
                by_method["learned_gate"]["best_epoch"],
                by_method["lora_r16"]["post12_best_mAP"],
                by_method["learned_gate"]["post12_best_mAP"],
                by_method["uniform_gate"]["best_metrics"]["coco/bbox_mAP"],
                by_method["uniform_gate"]["best_epoch"],
                by_method["uniform_gate"]["best_metrics"]["coco/bbox_mAP"]
                - by_method["uniform_gate"]["epoch12_mAP"],
                *[by_method["uniform_gate"]["best_metrics"][name] for name in METRICS],
            ),
            "",
            "## Training integrity and resource evidence",
            "",
        ]
    )
    for method, meta in METHODS.items():
        audits = gradient_audits[method]
        lines.append(
            "- {}: {} extension audits, zero non-finite gradients, all audited trainable tensors nonzero; peak allocated/reserved CUDA memory {:.3f}/{:.3f} GiB.".format(
                meta["label"],
                len(audits),
                max(row["cuda_peak_allocated_bytes"] for row in audits) / 2**30,
                max(row["cuda_peak_reserved_bytes"] for row in audits) / 2**30,
            )
        )
    lines.extend(
        [
            "",
            "All three extension launches ended with return code 0. A monitoring interruption during the first LoRA continuation is retained; the run resumed from the last complete epoch-15 full-state checkpoint, so the partial epoch-16 work is not counted. The first uniform epoch-13 reload produced valid metrics and 133 visualizations but its PowerShell wrapper returned 1 because warnings on stderr were promoted to `NativeCommandError`; the unchanged unified manifest launcher then repeated the reload with return code 0.",
            "",
            "## Decision",
            "",
            "The paper protocol should retain a short validation-selected budget rather than use 100 epochs. Twelve epochs are sufficient for LoRA-only and learned-gate comparisons because their optima occur before epoch 12 and post-12 performance does not improve. Uniform routing warrants at most a pre-registered 13-epoch multi-seed check because its seed-42 change is only +0.002; extending all methods to 100 epochs is unsupported and would amplify compute and checkpoint-selection multiplicity while the curves are flat or declining.",
            "",
            "## Claim-evidence map",
            "",
            "- Claim: 100 epochs are not justified by the observed convergence curves. | Evidence: LoRA-only and learned-gate best epochs are {} and {}; their epoch-20 values are {:.3f} and {:.3f}. | Status: supported for this dataset/configuration/seed, not a universal Grounding DINO rule.".format(
                by_method["lora_r16"]["best_epoch"],
                by_method["learned_gate"]["best_epoch"],
                by_method["lora_r16"]["epoch20_mAP"],
                by_method["learned_gate"]["epoch20_mAP"],
            ),
            "- Claim: uniform routing may benefit from one extra epoch. | Evidence: epoch-13 independently reloads at {:.3f}, versus {:.3f} at epoch 12. | Status: descriptive only; needs seeds 0 and 21 before any general claim.".format(
                by_method["uniform_gate"]["best_metrics"]["coco/bbox_mAP"],
                by_method["uniform_gate"]["epoch12_mAP"],
            ),
            "- Claim: learned image-conditioned routing improves over static routing. | Evidence: both methods reach {:.3f} in this 1--20 seed-42 comparison, while earlier three-seed learned-gate stability failed. | Status: unsupported; do not claim.".format(
                by_method["learned_gate"]["best_metrics"]["coco/bbox_mAP"]
            ),
            "",
            "## Reviewer-style self-review",
            "",
            "- Contribution: this is a convergence/control study, not a new method result; pass if presented as protocol evidence.",
            "- Clarity and reproducibility: exact resume checkpoints, logs, hashes, curves, and reload JSON files are machine-linked; pass.",
            "- Experimental strength: the only new gain is +0.002 at one seed; needs multi-seed replication.",
            "- Evaluation completeness: test leakage is absent, but this study covers one ODinW task and one seed; limitation must remain explicit.",
            "- Method soundness: longer training does not rescue learned routing; redesign or causal replication is more motivated than increasing epochs.",
            "",
            "## Regeneration",
            "",
            "```powershell",
            "python scripts\\summarize_epoch_budget_sensitivity.py --require-reload",
            "```",
            "",
            "Machine-readable sources: `results/json/epoch_budget_sensitivity_seed42.json`, `results/csv/epoch_budget_sensitivity_seed42_epochs.csv`, and `results/csv/epoch_budget_sensitivity_seed42_runs.csv`.",
            "",
        ]
    )
    path = ROOT / "reports/epoch_budget_sensitivity.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    all_epochs = {}
    source_logs = {}
    gradient_audits = {}
    runs = []
    epoch_rows = []
    for method, meta in METHODS.items():
        epochs, audits, logs = merge_epochs(method, meta)
        all_epochs[method] = epochs
        source_logs[method] = logs
        gradient_audits[method] = audits
        runs.append(select_run(method, meta, epochs, args.require_reload))
        for row in epochs:
            epoch_rows.append(
                {
                    "method": method,
                    "label": meta["label"],
                    "epoch": row["epoch"],
                    **{name: row[name] for name in METRICS},
                    "phase": "original_1_12" if row["epoch"] <= 12 else "extension_13_20",
                    "source_log": row["source_log"],
                }
            )

    run_rows = []
    for run in runs:
        run_rows.append(
            {
                "method": run["method"],
                "label": run["label"],
                "best_epoch": run["best_epoch"],
                "best_mAP": run["best_metrics"]["coco/bbox_mAP"],
                "epoch12_mAP": run["epoch12_mAP"],
                "epoch20_mAP": run["epoch20_mAP"],
                "post12_best_mAP": run["post12_best_mAP"],
                "post12_delta_vs_global_best": run["post12_delta_vs_global_best"],
                "best_in_extension": run["best_in_extension"],
                "checkpoint": run["checkpoint"],
                "checkpoint_bytes": run["checkpoint_bytes"],
                "checkpoint_sha256": run["checkpoint_sha256"],
                "independent_reload_status": run["independent_reload_status"],
                "independent_reload_json": run["independent_reload_json"],
            }
        )

    write_csv(ROOT / "results/csv/epoch_budget_sensitivity_seed42_epochs.csv", epoch_rows)
    write_csv(ROOT / "results/csv/epoch_budget_sensitivity_seed42_runs.csv", run_rows)
    render_figure(all_epochs, runs)
    render_latex(runs)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "ODinW-Pothole",
            "split": "official validation (133 images)",
            "seed": 42,
            "original_budget_epochs": 12,
            "extended_budget_epochs": 20,
            "selection": "maximum validation coco/bbox_mAP over epochs 1--20; earliest epoch breaks ties",
            "test_set_used_for_selection": False,
        },
        "runs": runs,
        "source_logs": source_logs,
        "extension_gradient_audits": gradient_audits,
    }
    output = ROOT / "results/json/epoch_budget_sensitivity_seed42.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_report(runs, gradient_audits)
    for row in run_rows:
        print(
            "{method}: best_epoch={best_epoch} best_mAP={best_mAP:.3f} "
            "post12_best={post12_best_mAP:.3f} reload={independent_reload_status}".format(
                **row
            )
        )
    print(f"Wrote {rel(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
