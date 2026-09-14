# ScaleFuse-LoRA

Official source code for **"ScaleFuse-LoRA: parameter-efficient cross-scale adaptation of Grounding DINO for pothole detection."**

Repository: https://github.com/sitongyan13-rgb/ScaleFuse-LoRA

ScaleFuse-LoRA adapts MMDetection's Grounding DINO with distributed rank-16 LoRA and a zero-initialized coarse-to-fine residual pathway over the four mapped visual feature levels. The repository contains the model extensions, ODinW Pothole configurations, experiment launchers, analysis scripts, and unit tests used in the paper.

## Repository contents

- `research/mmdet_plugins.py`: LoRA layers, ScaleFuse module, model registrations, training hooks, and diagnostic utilities.
- `configs/baselines/`: matched full-finetuning and Fusion-LoRA configurations.
- `configs/proposed/`: ScaleFuse-LoRA training and held-out test configurations.
- `configs/ablations/`: gate and auxiliary ablation configurations.
- `scripts/`: experiment launch, evaluation, metric aggregation, audit, and figure/table generation utilities.
- `tests/`: deterministic unit tests for ScaleFuse and related components.

Datasets, pretrained weights, trained checkpoints, prediction dumps, and local run directories are intentionally excluded.

## Environment

The experiments reported in the paper used:

- Python 3.8
- PyTorch 2.4.1 with CUDA 12.1
- MMDetection 3.3.0
- MMEngine 0.10.7
- MMCV 2.1.0

Follow the official installation instructions for [MMDetection 3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0) and its compatible MMCV build. Then place the MMDetection source tree at:

```text
third_party/mmdetection-3.3.0/
```

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

Download the official [Grounding DINO Swin-T checkpoint](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth) to:

```text
weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth
```

The configs use an offline BERT directory at `weights/bert-base-uncased/`. Populate it from the standard `bert-base-uncased` model before running offline experiments.

## Dataset

The paper uses the official ODinW Pothole split distributed through the [Microsoft GLIP repository](https://github.com/microsoft/GLIP). Arrange the extracted dataset as follows:

```text
data/odinw/pothole/
  train/
  valid/
  test/
```

Each split directory must contain its images and `annotations_without_background.json` file. Use the dataset only under its original terms.

## Training

Run the three final ScaleFuse-LoRA seeds with the supplied manifest-first launcher:

```bash
python scripts/run_experiment.py configs/ablations/generated_specs/pothole_scale_aware_lora_r16_seed0.yaml --execute
python scripts/run_experiment.py configs/ablations/generated_specs/pothole_scale_aware_lora_r16_seed21.yaml --execute
python scripts/run_experiment.py configs/ablations/generated_specs/pothole_scale_aware_lora_r16_seed42.yaml --execute
```

The matched Fusion-LoRA commands are:

```bash
python scripts/run_experiment.py configs/baselines/generated_specs/pothole_fusion_lora_r16_seed0.yaml --execute
python scripts/run_experiment.py configs/baselines/generated_specs/pothole_fusion_lora_r16_seed21.yaml --execute
python scripts/run_experiment.py configs/baselines/generated_specs/pothole_fusion_lora_r16_seed42.yaml --execute
```

All runs use the same data split, augmentation, optimizer, 12-epoch budget, prompt, and evaluator. Checkpoints must be selected on validation data before using the held-out test configurations.

## Testing

Run the lightweight unit-test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The release was checked with 46 passing tests.

## Reproducibility notes

- `SetTrainableModulesHook` freezes all parameters except the declared LoRA and ScaleFuse parameters.
- ScaleFuse projection layers are zero-initialized, so the branch is an exact identity at initialization.
- Training manifests record the resolved config, hashes, command, environment, seed, and output paths.
- The held-out test configs are evaluation-only and should not be used for checkpoint, prompt, threshold, or ensemble selection.

## License and attribution

The project code is released under the Apache License 2.0. ScaleFuse-LoRA builds on Grounding DINO and MMDetection; their licenses and citation requirements continue to apply. See `NOTICE.md` for upstream references.

## Citation

If this repository is useful in your research, please cite the accompanying paper. Bibliographic details will be updated after publication.
