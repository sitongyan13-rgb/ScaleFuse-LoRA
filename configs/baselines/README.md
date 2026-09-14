# Baseline execution

All benchmark jobs use Grounding DINO Swin-T, seed 42, a fixed
`800 x 1333` input envelope, horizontal-flip probability 0.5, batch size 1,
gradient accumulation 4, AMP with the locally verified fixed loss scale 1.0,
12 epochs, class-name prompts, and COCO bbox AP.

The generated directory contains 39 parsed configs: 13 ODinW tasks times
full fine-tuning, frozen-text fine-tuning, and residual-adapter fine-tuning.
Their task classes and paths are generated from the pinned official GLIP
ODinW-13 YAMLs. `generated/catalog.json` records every source SHA-256.

## Data preparation

For the locally supplied archives now present in `datasets/`, the command
actually used was:

```powershell
python scripts\download_dataset.py --members all --archive-dir datasets --extract-dir data\odinw --manifest data\odinw13_download_manifest.json --offline-existing
```

The network/resume form remains:

```powershell
python scripts\download_dataset.py --members all --archive-dir data\archives\odinw13 --extract-dir data\odinw --manifest data\odinw13_download_manifest.json
python scripts\verify_dataset.py --root data\odinw --archives data\archives\odinw13 --output results\json\dataset_verification.json
python scripts\analyze_dataset.py --root data\odinw --report reports\dataset_statistics.md --json-output results\json\dataset_statistics.json --csv-output results\csv\dataset_split_statistics.csv
```

The current manifest has 13 completed members and
`dataset_verification.json` has status `pass`.

## Zero-shot ODinW-13 evaluation

```powershell
python scripts\run_experiment.py configs\baselines\zero_shot.yaml --checkpoint weights\groundingdino_swint_ogc_mmdet-822d7e9d.pth --work-dir experiments\odinw13_zero_shot --execute
```

## Full-shot baselines

Aquarium is the first bounded task:

```powershell
python scripts\run_experiment.py configs\baselines\generated_specs\aquarium_full_finetune.yaml --execute
python scripts\run_experiment.py configs\baselines\generated_specs\aquarium_frozen_text.yaml --execute
python scripts\run_experiment.py configs\baselines\generated_specs\aquarium_adapter.yaml --execute
```

Replace `aquarium` with any catalog task after checking its local paths.
Run one job at a time on the audited single GPU. Each invocation creates a
manifest before launching and records the full log and return code.

## Metrics

```powershell
python scripts\collect_metrics.py --raw-dir experiments\raw_metrics --output results\csv\metrics_summary.csv
```

The 100-iteration fixture command already executed in the startup phase was:

```powershell
python third_party\mmdetection-3.3.0\tools\train.py configs\baselines\smoke_cat_100iter.py --work-dir experiments\smoke_cat_100iter
```

## Fusion-LoRA screening and rank ablation

`odinw_pothole_fusion_lora_r8.py` inserts checkpoint-compatible LoRA into
145 multimodal encoder, decoder, detection-head, memory-transform, and text
feature-map linear projections. Only `lora_down` and `lora_up` parameters
are trainable. Rank 8 completed the 100-iteration smoke test and the bounded
seed-42 screening run.

Ranks 4 and 16 keep the same split, schedule, input policy, augmentation,
prompt generation, evaluator, optimizer, and alpha/rank scaling as rank 8.
They are prepared for validation-only selection:

```powershell
python scripts\run_experiment.py configs\baselines\generated_specs\pothole_fusion_lora_r4_seed42.yaml --execute
python scripts\run_experiment.py configs\baselines\generated_specs\pothole_fusion_lora_r16_seed42.yaml --execute
```

Do not evaluate rank 4 or rank 16 on held-out test before selecting the rank
from validation. Full commands and dynamic best-checkpoint resolution are in
`reports/execution_commands.md`.
