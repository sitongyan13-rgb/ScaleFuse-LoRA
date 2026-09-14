#!/usr/bin/env python
"""Evaluate an MMDetection config and persist the runner's raw metric mapping."""

import argparse
from copy import deepcopy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "third_party" / "mmdetection-3.3.0"):
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

from mmengine import ConfigDict
from mmengine.config import Config
from mmengine.runner import Runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--show-dir", type=Path)
    parser.add_argument(
        "--predictions-pkl",
        type=Path,
        help="Optional MMDetection prediction dump for offline error analysis",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Apply the config's TTA model/pipeline (or MMDet flip default)",
    )
    args = parser.parse_args()

    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = Config.fromfile(str(config))
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.launcher = "none"
    if args.tta:
        if "tta_model" not in cfg:
            cfg.tta_model = dict(
                type="DetTTAModel",
                tta_cfg=dict(
                    nms=dict(type="nms", iou_threshold=0.5),
                    max_per_img=100,
                ),
            )
        if "tta_pipeline" not in cfg:
            test_data_cfg = cfg.test_dataloader.dataset
            while "dataset" in test_data_cfg:
                test_data_cfg = test_data_cfg.dataset
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            cfg.tta_pipeline[-1] = dict(
                type="TestTimeAug",
                transforms=[
                    [
                        dict(type="RandomFlip", prob=1.0),
                        dict(type="RandomFlip", prob=0.0),
                    ],
                    [
                        dict(
                            type="PackDetInputs",
                            meta_keys=(
                                "img_id", "img_path", "ori_shape",
                                "img_shape", "scale_factor", "flip",
                                "flip_direction", "text", "custom_entities",
                            ),
                        )
                    ],
                ],
            )
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    if args.predictions_pkl:
        predictions = args.predictions_pkl.resolve()
        if predictions.suffix.lower() not in (".pkl", ".pickle"):
            raise ValueError("--predictions-pkl must end with .pkl or .pickle")
        predictions.parent.mkdir(parents=True, exist_ok=True)
        base_evaluator = cfg.test_evaluator
        if isinstance(base_evaluator, (list, tuple)):
            cfg.test_evaluator = list(base_evaluator) + [
                dict(type="DumpDetResults", out_file_path=str(predictions))
            ]
        else:
            cfg.test_evaluator = [
                base_evaluator,
                dict(type="DumpDetResults", out_file_path=str(predictions)),
            ]
    if args.show_dir:
        show_dir = args.show_dir.resolve()
        show_dir.mkdir(parents=True, exist_ok=True)
        cfg.default_hooks.visualization.draw = True
        cfg.default_hooks.visualization.interval = 1
        cfg.default_hooks.visualization.score_thr = 0.25
        cfg.default_hooks.visualization.test_out_dir = str(show_dir)

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "tta": bool(args.tta),
    }
    if args.predictions_pkl:
        payload["predictions_pkl"] = str(args.predictions_pkl.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
