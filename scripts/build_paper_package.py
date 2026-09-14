#!/usr/bin/env python
"""Build a paper-ready evidence package from frozen experiment artifacts."""

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
SEEDS = (0, 21, 42)
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(relative):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(relative):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(relative, rows):
    path = PAPER / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_text(relative, text):
    path = PAPER / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def mean_sd(values):
    values = [float(value) for value in values]
    return statistics.mean(values), statistics.stdev(values)


def aggregate_lookup(payload):
    return {
        (row["method"], row["metric"]): row
        for row in payload["aggregate"]
    }


def pct(value):
    return 100.0 * float(value)


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    sources = {
        "dataset_statistics": "results/json/dataset_statistics.json",
        "cross_task_runs": "results/csv/cross_task_multiseed_runs.csv",
        "single_seed_baselines": "results/csv/cross_task_baseline_runs.csv",
        "lora_rank": "results/json/fusion_lora_rank_ablation_summary.json",
        "validation_exact": "results/json/scale_aware_multiseed_validation_summary.json",
        "test_exact": "results/json/scale_aware_multiseed_results.json",
        "scale_mechanism": "results/json/scale_aware_seed42_validation_summary.json",
        "error_analysis": "experiments/raw_metrics/pothole_scale_aware_error_analysis.json",
        "epoch_budget": "results/json/epoch_budget_sensitivity_seed42.json",
        "main_method_tricks": "results/csv/pothole_main_method_comparison.csv",
        "test_contaminated": "results/json/test_contaminated_score_calibration_refine_search.json",
        "one_batch_audit": "experiments/raw_metrics/scale_aware_lora_r16_one_batch.json",
    }
    for relative in sources.values():
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)

    cross_rows = read_csv(sources["cross_task_runs"])
    single_rows = read_csv(sources["single_seed_baselines"])
    validation = read_json(sources["validation_exact"])
    test = read_json(sources["test_exact"])
    scale = read_json(sources["scale_mechanism"])
    error = read_json(sources["error_analysis"])
    epoch = read_json(sources["epoch_budget"])
    contaminated = read_json(sources["test_contaminated"])
    one_batch = read_json(sources["one_batch_audit"])
    val_agg = aggregate_lookup(validation)
    test_agg = aggregate_lookup(test)

    pothole_cross = [row for row in cross_rows if row["dataset"] == "Pothole"]
    full = [row for row in pothole_cross if row["method"] == "Full fine-tuning"]
    frozen = [row for row in pothole_cross if row["method"] == "Frozen text encoder"]
    if len(full) != 3 or len(frozen) != 3:
        raise RuntimeError("Expected three Pothole Full and Frozen-text runs")
    zero = next(
        row for row in single_rows
        if row["dataset"] == "Pothole" and row["method"] == "Official zero-shot"
    )

    proposed_trainable = int(one_batch["trainable_parameters"])
    lora_trainable = 2425280
    proposed_total = 175636064
    proposed_ratio = 100.0 * proposed_trainable / proposed_total
    if proposed_trainable != 2723651:
        raise RuntimeError("Unexpected proposed trainable parameter count")

    main_rows = []
    main_rows.append({
        "method": "Official zero-shot (seed 42)",
        "trainable_parameters": 0,
        "trainable_ratio_percent": 0.0,
        "validation_AP_mean": float(zero["coco/bbox_mAP"]),
        "validation_AP_sample_sd": "",
        "test_AP_mean": float(zero["test/coco/bbox_mAP"]),
        "test_AP_sample_sd": "",
        "test_AP50_mean": float(zero["test/coco/bbox_mAP_50"]),
        "test_AP75_mean": float(zero["test/coco/bbox_mAP_75"]),
        "test_APs_mean": float(zero["test/coco/bbox_mAP_s"]),
        "test_APm_mean": float(zero["test/coco/bbox_mAP_m"]),
        "test_APl_mean": float(zero["test/coco/bbox_mAP_l"]),
        "evidence_scope": "single seed; not directly variance-matched",
    })
    for label, rows in (("Full fine-tuning", full), ("Frozen text encoder", frozen)):
        val_mean, val_sd = mean_sd(row["coco/bbox_mAP"] for row in rows)
        test_mean, test_sd = mean_sd(row["test/coco/bbox_mAP"] for row in rows)
        main_rows.append({
            "method": label,
            "trainable_parameters": int(rows[0]["trainable_parameters"]),
            "trainable_ratio_percent": float(rows[0]["trainable_ratio_percent"]),
            "validation_AP_mean": val_mean,
            "validation_AP_sample_sd": val_sd,
            "test_AP_mean": test_mean,
            "test_AP_sample_sd": test_sd,
            "test_AP50_mean": statistics.mean(float(row["test/coco/bbox_mAP_50"]) for row in rows),
            "test_AP75_mean": statistics.mean(float(row["test/coco/bbox_mAP_75"]) for row in rows),
            "test_APs_mean": statistics.mean(float(row["test/coco/bbox_mAP_s"]) for row in rows),
            "test_APm_mean": statistics.mean(float(row["test/coco/bbox_mAP_m"]) for row in rows),
            "test_APl_mean": statistics.mean(float(row["test/coco/bbox_mAP_l"]) for row in rows),
            "evidence_scope": "three seeds; native metrics rounded to 0.001",
        })
    for source_label, paper_label, trainable, ratio in (
        ("Fusion-LoRA-r16", "Fusion-LoRA-r16", lora_trainable,
         100.0 * lora_trainable / 175337693),
        ("Dynamic Scale-Aware Fusion LoRA", "ScaleFuse-LoRA (ours)",
         proposed_trainable, proposed_ratio),
    ):
        main_rows.append({
            "method": paper_label,
            "trainable_parameters": trainable,
            "trainable_ratio_percent": ratio,
            "validation_AP_mean": val_agg[(source_label, "AP")]["mean"],
            "validation_AP_sample_sd": val_agg[(source_label, "AP")]["sample_sd"],
            "test_AP_mean": test_agg[(source_label, "AP")]["mean"],
            "test_AP_sample_sd": test_agg[(source_label, "AP")]["sample_sd"],
            "test_AP50_mean": test_agg[(source_label, "AP50")]["mean"],
            "test_AP75_mean": test_agg[(source_label, "AP75")]["mean"],
            "test_APs_mean": test_agg[(source_label, "APs")]["mean"],
            "test_APm_mean": test_agg[(source_label, "APm")]["mean"],
            "test_APl_mean": test_agg[(source_label, "APl")]["mean"],
            "evidence_scope": "three seeds; exact pycocotools recomputation",
        })
    main_path = write_csv("tables/main_results.csv", main_rows)

    per_seed = []
    for row in validation["runs"]:
        match = next(
            item for item in test["runs"]
            if item["method"] == row["method"] and item["seed"] == row["seed"]
        )
        per_seed.append({
            "method": (
                "ScaleFuse-LoRA (ours)"
                if row["method"] == "Dynamic Scale-Aware Fusion LoRA"
                else row["method"]
            ),
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "validation_AP": row["AP"],
            "validation_APs": row["APs"],
            "test_AP": match["AP"],
            "test_AP50": match["AP50"],
            "test_AP75": match["AP75"],
            "test_APs": match["APs"],
            "test_APm": match["APm"],
            "test_APl": match["APl"],
        })
    seed_path = write_csv("tables/per_seed_results.csv", per_seed)

    lora_seed42 = next(
        row for row in validation["runs"]
        if row["method"] == "Fusion-LoRA-r16" and row["seed"] == 42
    )
    mechanism_rows = [{
        "variant": "Fusion-LoRA-r16",
        "training_or_intervention": "trained baseline",
        "seed": 42,
        "validation_AP": lora_seed42["AP"],
        "validation_APs": lora_seed42["APs"],
        "test_used_for_selection": False,
        "interpretation": "matched parameter-efficient baseline",
    }]
    labels = {
        "dynamic_gate": "ScaleFuse-LoRA (learned gate)",
        "static_residual": "ScaleFuse-LoRA (constant-one, fresh training)",
        "static_checkpoint_intervention": "Learned checkpoint with gate forced to one",
    }
    for row in scale["selected_runs"]:
        mechanism_rows.append({
            "variant": labels[row["method"]],
            "training_or_intervention": (
                "checkpoint intervention"
                if row["method"] == "static_checkpoint_intervention"
                else "trained variant"
            ),
            "seed": 42,
            "validation_AP": row["coco/bbox_mAP"],
            "validation_APs": row["coco/bbox_mAP_s"],
            "test_used_for_selection": False,
            "interpretation": (
                "gate saturation audit; not an independent trained model"
                if row["method"] == "static_checkpoint_intervention"
                else "cross-scale mechanism ablation"
            ),
        })
    mechanism_path = write_csv("tables/mechanism_ablation.csv", mechanism_rows)

    full42 = next(row for row in full if int(row["seed"]) == 42)
    frozen42 = next(row for row in frozen if int(row["seed"]) == 42)
    resource_rows = [
        {
            "method": "Full fine-tuning", "seed": 42,
            "trainable_parameters": int(full42["trainable_parameters"]),
            "trainable_ratio_percent": float(full42["trainable_ratio_percent"]),
            "peak_cuda_allocated_gib": float(full42["peak_cuda_allocated_gib"]),
            "peak_cuda_reserved_gib": float(full42["peak_cuda_reserved_gib"]),
            "wall_time_minutes": float(full42["elapsed_minutes"]),
            "comparability_note": "same task/split/schedule; rounded evaluator",
        },
        {
            "method": "Frozen text encoder", "seed": 42,
            "trainable_parameters": int(frozen42["trainable_parameters"]),
            "trainable_ratio_percent": float(frozen42["trainable_ratio_percent"]),
            "peak_cuda_allocated_gib": float(frozen42["peak_cuda_allocated_gib"]),
            "peak_cuda_reserved_gib": float(frozen42["peak_cuda_reserved_gib"]),
            "wall_time_minutes": float(frozen42["elapsed_minutes"]),
            "comparability_note": "same task/split/schedule; rounded evaluator",
        },
        {
            "method": "Fusion-LoRA-r16", "seed": 42,
            "trainable_parameters": lora_trainable,
            "trainable_ratio_percent": 100.0 * lora_trainable / 175337693,
            "peak_cuda_allocated_gib": 1.9445209503173828,
            "peak_cuda_reserved_gib": 5.935546875,
            "wall_time_minutes": 89.68140961666667,
            "comparability_note": "external GPU load inflated wall time; do not claim speedup",
        },
        {
            "method": "ScaleFuse-LoRA (ours)", "seed": 42,
            "trainable_parameters": proposed_trainable,
            "trainable_ratio_percent": proposed_ratio,
            "peak_cuda_allocated_gib": scale["training_audits"]["dynamic_gate"]["peak_cuda_allocated_gib"],
            "peak_cuda_reserved_gib": scale["training_audits"]["dynamic_gate"]["peak_cuda_reserved_gib"],
            "wall_time_minutes": 45.8167,
            "comparability_note": "derived from 14:31:13--15:16:57 seed-42 log interval",
        },
    ]
    resource_path = write_csv("tables/resource_summary.csv", resource_rows)

    paper_table_lines = [
        "# Paper-ready result tables",
        "",
        "All AP values below are shown on the conventional 0--100 scale. Mean +/- sample SD uses three seeds unless explicitly marked otherwise.",
        "",
        "## Main Pothole results",
        "",
        "| Method | Trainable | Ratio | Val AP | Test AP | AP50 | AP75 | APs | APm | APl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        val = "{:.3f}".format(pct(row["validation_AP_mean"]))
        test_value = "{:.3f}".format(pct(row["test_AP_mean"]))
        if row["validation_AP_sample_sd"] != "":
            val += " +/- {:.3f}".format(pct(row["validation_AP_sample_sd"]))
        if row["test_AP_sample_sd"] != "":
            test_value += " +/- {:.3f}".format(pct(row["test_AP_sample_sd"]))
        paper_table_lines.append(
            "| {method} | {params:,} | {ratio:.3f}% | {val} | {test} | {ap50:.3f} | {ap75:.3f} | {aps:.3f} | {apm:.3f} | {apl:.3f} |".format(
                method=row["method"], params=int(row["trainable_parameters"]),
                ratio=float(row["trainable_ratio_percent"]), val=val, test=test_value,
                ap50=pct(row["test_AP50_mean"]), ap75=pct(row["test_AP75_mean"]),
                aps=pct(row["test_APs_mean"]), apm=pct(row["test_APm_mean"]),
                apl=pct(row["test_APl_mean"])))
    paper_table_lines.extend([
        "",
        "Zero-shot is a single seed and must not be read as a variance-matched comparison. Full/Frozen metrics come from the native evaluator rounded to 0.001; Fusion-LoRA and ScaleFuse-LoRA are exact pycocotools recomputations from immutable prediction dumps.",
        "",
        "## Matched three-seed comparison",
        "",
        "| Method | Seed | Best epoch | Val AP | Test AP | Test APs | Test APm | Test APl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in per_seed:
        paper_table_lines.append(
            "| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                row["method"], row["seed"], row["best_epoch"],
                pct(row["validation_AP"]), pct(row["test_AP"]),
                pct(row["test_APs"]), pct(row["test_APm"]), pct(row["test_APl"])))
    paper_table_lines.extend([
        "",
        "Paired ScaleFuse-LoRA minus Fusion-LoRA changes are +1.332 AP on validation and +1.983 AP on test. The test change is positive for all three seeds. Test APs changes by -1.516 on average, while APm/APl improve by +3.295/+2.117; the small-object limitation must remain explicit.",
        "",
        "## Scale-path mechanism ablation (seed 42 validation)",
        "",
        "| Variant | AP | APs | Evidence type |",
        "|---|---:|---:|---|",
    ])
    for row in mechanism_rows:
        paper_table_lines.append("| {} | {:.3f} | {:.3f} | {} |".format(
            row["variant"], pct(row["validation_AP"]), pct(row["validation_APs"]),
            row["training_or_intervention"]))
    paper_table_lines.extend([
        "",
        "The learned gates average above 0.9996. Forcing them to one slightly improves the same checkpoint, whereas a freshly trained constant-one variant underperforms. Therefore the evidence supports the cross-scale residual pathway but does not isolate a benefit from image-conditioned gating.",
        "",
        "## Resource audit (seed 42)",
        "",
        "| Method | Trainable | Ratio | Peak allocated | Peak reserved | Wall time |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in resource_rows:
        paper_table_lines.append(
            "| {method} | {params:,} | {ratio:.3f}% | {alloc:.3f} GiB | {reserved:.3f} GiB | {time:.1f} min |".format(
                method=row["method"], params=int(row["trainable_parameters"]),
                ratio=float(row["trainable_ratio_percent"]),
                alloc=float(row["peak_cuda_allocated_gib"]),
                reserved=float(row["peak_cuda_reserved_gib"]),
                time=float(row["wall_time_minutes"])))
    paper_table_lines.extend([
        "",
        "Wall time is reported for transparency, not as a speed claim: external display/compute load differed, especially for Fusion-LoRA. Checkpoints are full-state (~700 MB), so trainable-parameter efficiency does not imply storage efficiency.",
    ])
    results_md = write_text("results_tables.md", "\n".join(paper_table_lines))

    inventory = """# Experiment inventory for paper writing

## Paper identity

- Working name: **ScaleFuse-LoRA**.
- Repository implementation: `Dynamic Scale-Aware Fusion LoRA` / `ScaleAwareFusionLoRAGroundingDINO`.
- Scope: parameter-efficient Grounding DINO adaptation for the ODinW Pothole task.
- Backbone: Grounding DINO Swin-T.
- Dataset: 465 train images / 1,256 boxes; 133 validation images / 330 boxes; 67 test images / 154 boxes.
- Prompt: one category, `pothole`.
- Schedule: 12 epochs, seeds 0/21/42, validation-only checkpoint selection.

## Evidence tiers

### Tier A: main-paper evidence

1. Official zero-shot reference reproduced locally (single seed).
2. Full and frozen-text fine-tuning across three seeds.
3. Fusion-LoRA rank-16 matched baseline across three seeds.
4. ScaleFuse-LoRA across three seeds with validation frozen before one-time test evaluation.
5. Seed-42 cross-scale pathway/gate intervention and validation error analysis.
6. Resource, gradient, dataset-integrity, and epoch-budget audits.

### Tier B: appendix or negative-result evidence

- LoRA rank 4/8/16 screen.
- Learned/uniform prompt-routing studies and gate-collapse audit.
- Quality-aligned targets and localization-ranking loss.
- Object-aware zoom, scale-adaptive regression, and high-resolution smoke diagnostics.
- Horizontal-flip TTA, box voting, checkpoint soup, and validation-selected NMS ensemble.
- Epochs 13--20 sensitivity study showing no support for a 100-epoch schedule.

### Tier C: excluded from claims and main tables

- Direct test-set score-calibration search and its adaptive refinement.
- Final contaminated diagnostic: AP 57.543 and APs 47.093 after selecting parameters on test.
- Reason for exclusion: parameters were selected using the test annotations after the clean result had already been read.

## Non-negotiable limitations

1. The proposed method is evaluated on one ODinW task, not the full ODinW-13 benchmark.
2. The test split contains only 67 images; n=3 intervals are descriptive.
3. Test APs is lower than Fusion-LoRA for every seed despite higher overall AP.
4. Learned gates saturate; dynamic routing is not established as the cause of improvement.
5. Full-state checkpoints do not demonstrate storage efficiency.
6. No SOTA, universal generalization, or statistical-significance claim is supported.
"""
    inventory_md = write_text("experiment_inventory.md", inventory)

    claims = """# Claim-evidence matrix

| Candidate claim | Evidence | Status | Permitted wording |
|---|---|---|---|
| ScaleFuse-LoRA improves matched Fusion-LoRA | Three-seed exact validation +1.332 AP and test +1.983 AP; every paired AP change is positive | Supported | "improves the matched Fusion-LoRA baseline on this Pothole split" |
| The method is parameter-efficient | 2,723,651 / 175,636,064 trainable parameters (1.551%); seed-42 peak allocated 1.989 GiB | Supported | "updates 1.55% of parameters" |
| Cross-scale residual fusion is useful | Dynamic residual model exceeds LoRA; forcing saturated gates to one preserves/slightly improves the checkpoint | Supported with scope | "the evidence attributes the gain primarily to the residual cross-scale path" |
| Image-conditioned gating causes the gain | Gates saturate above 0.9996; constant-one intervention is slightly better; fresh static run is lower | Unsupported | Do not claim a dynamic-routing advantage |
| The method improves small-object AP | Validation APs mean is nearly unchanged; test APs falls by 1.516 points | Contradicted | State the small-object limitation explicitly |
| The method improves small-object proposal recall | Seed-42 validation AR100-small rises by 5.059 points and small-GT best IoU by 0.0305 | Supported diagnostically | "improves small-object proposal recall on the validation diagnostic" |
| The method is better than full fine-tuning | Test mean is +1.238 AP but validation mean is -0.189 AP; one small dataset | Descriptive only | "achieves a higher test mean in this experiment"; do not claim general superiority |
| Twelve epochs are adequate | LoRA and learned-prompt optima occur at epochs 9/10; epoch-20 is lower; uniform routing gains only 0.2 at epoch 13 | Supported for this setup | "the observed curves do not justify 100 epochs" |
| The method generalizes across ODinW-13 | Proposed method was not trained/evaluated across all 13 tasks | Unsupported | Do not claim ODinW-13 generalization |
| Test-calibrated AP/APs are final results | Parameters were directly selected on test | Invalid evidence | Exclude from abstract, main tables, and conclusions |
| The work establishes SOTA | No comprehensive current-method comparison; one task and 67 test images | Unsupported | Never use "state of the art" |
"""
    claims_md = write_text("claim_evidence_matrix.md", claims)

    figures = """# Figure and table index

## Main-paper candidates

1. **Method overview:** `paper/figures/scalefuse_lora_overview.{pdf,svg,png}`. It shows Grounding DINO Swin-T with distributed LoRA, the zero-initialized coarse-to-fine residual path, frozen/trainable modules, and the clean three-seed headline result.
2. **Three-seed main result:** `results/figures/scale_aware_multiseed_test.{pdf,svg,png}`.
3. **Validation selection:** `results/figures/scale_aware_multiseed_validation.{pdf,svg,png}`.
4. **Scale/error diagnostic:** `results/figures/pothole_scale_aware_error_analysis.{pdf,svg,png}`.
5. **Parameter-efficiency context:** `results/figures/fusion_lora_accuracy_efficiency.{pdf,png}`.

## Appendix candidates

- LoRA rank: `results/figures/fusion_lora_rank_ablation.{pdf,svg,png}`.
- Epoch budget: `results/figures/epoch_budget_sensitivity_seed42.{pdf,svg,png}`.
- Prompt routing: `results/figures/prompt_routing_multiseed_validation.{pdf,svg,png}`.
- Quality/ranking failures: `results/figures/pothole_quality_aligned_error_analysis.*` and `pothole_localization_ranking_error_analysis.*`.
- Smoke diagnostics: object-aware zoom, scale-adaptive regression, and high-resolution figures.

## Excluded figures

`test_contaminated_score_calibration*` must not appear as held-out evidence. If retained for an educational appendix, the caption must say that parameters were selected directly on test.

## Visual QA notes

- Prefer PDF/SVG for paper insertion and PNG only for preview.
- Main result plots currently use 0--1 AP; final manuscript tables use 0--100 AP. Keep that convention explicit.
- A qualitative panel should be rebuilt from existing validation/test visualizations with matched images for Fusion-LoRA and ScaleFuse-LoRA; do not cherry-pick test examples by metric.
"""
    figures_md = write_text("figure_index.md", figures)

    reproducibility = """# Reproducibility entry point

## Frozen main method

- Config: `configs/proposed/odinw_pothole_scale_aware_lora_r16.py`
- Test config: `configs/proposed/odinw_pothole_scale_aware_lora_r16_test.py`
- Implementation: `research/mmdet_plugins.py` (`ScaleAwareCrossLevelFusion`, `ScaleAwareFusionLoRAGroundingDINO`)
- Training seeds: 0, 21, 42
- Epoch budget: 12
- Best epochs: 10, 12, 11
- Exact validation summary: `results/json/scale_aware_multiseed_validation_summary.json`
- Exact test summary: `results/json/scale_aware_multiseed_results.json`
- Complete executed command ledger: `reports/execution_commands.md`

## Regeneration commands

```powershell
python scripts\\summarize_scale_aware_multiseed.py
python scripts\\summarize_scale_aware_multiseed_results.py
python scripts\\build_paper_figures.py
python scripts\\build_paper_package.py
```

Do not run the test-contaminated calibration scripts when regenerating the main paper results.
"""
    reproducibility_md = write_text("reproducibility.md", reproducibility)

    readme = """# ScaleFuse-LoRA paper workspace

This directory is the clean entry point for manuscript writing. It separates frozen scientific evidence from exploratory and test-contaminated diagnostics.

## Recommended reading order

1. `experiment_inventory.md` -- what was run and what may enter the paper.
2. `claim_evidence_matrix.md` -- exactly which claims are supported.
3. `results_tables.md` -- paper-ready main, seed, mechanism, and resource tables.
4. `figure_index.md` -- main/appendix visual assets and missing visuals.
5. `reproducibility.md` -- configs, implementation, summaries, and regeneration.
6. `outline.md` and `draft.md` -- manuscript structure, title, abstract, and introduction.
7. `references.bib` -- verified primary-source metadata for the initial citations.

## Ready-to-use assets

- Method overview: `figures/scalefuse_lora_overview.pdf` (with SVG and PNG variants).
- Main tables: `tables/main_results.csv`, `tables/per_seed_results.csv`,
  `tables/mechanism_ablation.csv`, and `tables/resource_summary.csv`.
- Evidence ledger: `evidence_manifest.json`, including source and figure hashes.
- Latest automated integrity check: `package_validation.json`.
- Writing status and remaining evidence gaps: `writing_status.md`.

## Frozen story

ScaleFuse-LoRA adds a zero-initialized coarse-to-fine residual feature pathway to a distributed rank-16 LoRA adaptation of Grounding DINO. On the ODinW Pothole task, it updates 1.551% of parameters and improves the matched Fusion-LoRA baseline by 1.332 validation AP and 1.983 held-out test AP across three seeds. The improvement is concentrated in medium/large objects; small-object AP declines on test. Mechanism audits support the cross-scale residual path but not a benefit from image-conditioned gating.

## Integrity rule

The direct test-tuning result is deliberately excluded from the manuscript's main evidence. The valid proposed test mean is 57.405 AP, not the test-selected 57.543 diagnostic.
"""
    readme_md = write_text("README.md", readme)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_method_name": "ScaleFuse-LoRA",
        "repository_method_name": "Dynamic Scale-Aware Fusion LoRA",
        "main_test_AP_mean": test_agg[("Dynamic Scale-Aware Fusion LoRA", "AP")]["mean"],
        "matched_baseline_test_AP_mean": test_agg[("Fusion-LoRA-r16", "AP")]["mean"],
        "main_test_APs_mean": test_agg[("Dynamic Scale-Aware Fusion LoRA", "APs")]["mean"],
        "test_contaminated_result_excluded": True,
        "test_contaminated_AP_mean": contaminated["selected"]["mean_AP"],
        "trainable_parameters": proposed_trainable,
        "total_parameters": proposed_total,
        "trainable_ratio_percent": proposed_ratio,
        "sources": [
            {"name": name, "path": relative, "sha256": sha256(ROOT / relative)}
            for name, relative in sources.items()
        ],
        "generated": [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in (main_path, seed_path, mechanism_path, resource_path,
                         results_md, inventory_md, claims_md, figures_md,
                         reproducibility_md, readme_md)
        ],
        "paper_figures": [
            {
                "path": "paper/figures/scalefuse_lora_overview.{}".format(extension),
                "sha256": sha256(PAPER / "figures/scalefuse_lora_overview.{}".format(extension)),
            }
            for extension in ("png", "pdf", "svg")
        ],
        "diagnostic_facts": {
            "small_AR100_gain_seed42_validation": 0.05059,
            "small_best_iou_gain_seed42_validation": 0.03047,
            "test_APs_paired_mean_change": test["paired_proposed_minus_baseline"]["APs"]["mean"],
            "gates_saturated": all(
                value > 0.9996 for value in scale["mechanism"]["dynamic_gate_mean"]),
            "error_analysis_source_loaded": bool(error),
            "epoch_budget_source_loaded": bool(epoch),
        },
    }
    manifest_path = PAPER / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "paper_directory": str(PAPER),
        "generated_files": len(manifest["generated"]) + 1,
        "main_test_AP": manifest["main_test_AP_mean"],
        "baseline_test_AP": manifest["matched_baseline_test_AP_mean"],
        "trainable_ratio_percent": manifest["trainable_ratio_percent"],
        "contaminated_result_excluded": True,
    }, indent=2))


if __name__ == "__main__":
    main()
