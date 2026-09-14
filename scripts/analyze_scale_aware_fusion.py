#!/usr/bin/env python
"""Audit scale gates and residual magnitudes on fixed Pothole validation data."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "third_party/mmdetection-3.3.0"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

import research.mmdet_plugins  # noqa: F401


if os.name == "nt" and not sys.flags.utf8_mode:
    # Python 3.8 fixes locale.getpreferredencoding() at process startup, while
    # MMEngine decodes MSVC output with that value. Re-enter once in UTF-8 mode
    # so it agrees with the UTF-8 console configured below.
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    raise SystemExit(
        subprocess.call([sys.executable] + sys.argv, env=child_env)
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    init_default_scope("mmdet")
    cfg = Config.fromfile(str(args.config.resolve()))
    checkpoint = args.checkpoint.resolve()
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(args.work_dir.resolve())
    old_input_cp = old_output_cp = None
    if os.name == "nt":
        # MMEngine invokes MSVC while collecting the environment. Keep the
        # subprocess output and Python decoder on the same UTF-8 code page.
        kernel32 = ctypes.windll.kernel32
        old_input_cp = kernel32.GetConsoleCP()
        old_output_cp = kernel32.GetConsoleOutputCP()
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    try:
        runner = Runner.from_cfg(cfg)
    finally:
        if os.name == "nt" and old_input_cp and old_output_cp:
            kernel32.SetConsoleCP(old_input_cp)
            kernel32.SetConsoleOutputCP(old_output_cp)
    module = runner.model.scale_aware_fusion
    samples = []

    def hook(_module, inputs, outputs):
        originals = inputs[0]
        calibrated, gates = outputs
        per_level = []
        for fine, changed in zip(originals[:-1], calibrated[:-1]):
            residual = changed - fine
            base_rms = fine.float().square().mean(dim=(1, 2, 3)).sqrt()
            residual_rms = residual.float().square().mean(dim=(1, 2, 3)).sqrt()
            per_level.append((residual_rms / base_rms.clamp_min(1e-12)).detach().cpu())
        relative = torch.stack(per_level, dim=-1)
        for row_gate, row_relative in zip(gates.detach().cpu(), relative):
            samples.append({"gates": row_gate.tolist(),
                            "relative_residual_rms": row_relative.tolist()})

    handle = module.register_forward_hook(hook)
    metrics = runner.test()
    handle.remove()
    if len(samples) != 133:
        raise RuntimeError("Expected 133 validation images, got {}".format(len(samples)))
    gates = torch.tensor([row["gates"] for row in samples])
    residuals = torch.tensor([row["relative_residual_rms"] for row in samples])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"dataset": "ODinW Pothole", "split": "validation",
                     "images": len(samples), "held_out_test_evaluated": False,
                     "checkpoint": str(checkpoint),
                     "checkpoint_sha256": sha256(checkpoint)},
        "validation_metrics": metrics,
        "gate_mean": gates.mean(0).tolist(),
        "gate_sample_sd": gates.std(0, unbiased=True).tolist(),
        "gate_min": gates.min(0).values.tolist(),
        "gate_max": gates.max(0).values.tolist(),
        "relative_residual_rms_mean": residuals.mean(0).tolist(),
        "relative_residual_rms_sample_sd": residuals.std(0, unbiased=True).tolist(),
        "projection_frobenius_norms": [
            float(layer.weight.detach().float().norm())
            for layer in module.residual_projections
        ],
        "per_image": samples,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "validation_metrics", "gate_mean", "gate_sample_sd",
        "relative_residual_rms_mean", "projection_frobenius_norms")}, indent=2))


if __name__ == "__main__":
    main()
