#!/usr/bin/env python
"""Manifest-first launcher for the project's auditable baseline experiments."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MMDET_TRAIN = ROOT / "third_party" / "mmdetection-3.3.0" / "tools" / "train.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_command(command: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(shlex.quote(part) for part in command)


def load_spec(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Baseline spec must be a YAML mapping: {}".format(path))
    for key in ("name", "action", "native_config", "seed"):
        if key not in data:
            raise ValueError("Missing required key {!r} in {}".format(key, path))
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path, help="Baseline YAML specification")
    parser.add_argument("--execute", action="store_true", help="Run after manifest creation")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint for test action")
    parser.add_argument(
        "--resume", type=Path, help="Full-state checkpoint for a training resume"
    )
    parser.add_argument("--work-dir", type=Path, help="Override MMEngine work directory")
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    native_config = (ROOT / spec["native_config"]).resolve()
    if not native_config.is_file():
        raise FileNotFoundError(native_config)
    if not MMDET_TRAIN.is_file():
        raise FileNotFoundError(MMDET_TRAIN)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = "{}_s{}_{}".format(spec["name"], spec["seed"], timestamp)
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir
        else (ROOT / "experiments" / spec["name"]).resolve()
    )

    if spec["action"] == "train":
        resume = args.resume
        if resume is not None:
            resume = (
                resume.resolve()
                if resume.is_absolute()
                else (ROOT / resume).resolve()
            )
            if not resume.is_file():
                raise FileNotFoundError(resume)
        command = [
            sys.executable,
            str(MMDET_TRAIN),
            str(native_config),
            "--work-dir",
            str(work_dir),
            "--cfg-options",
            "randomness.seed={}".format(int(spec["seed"])),
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
    elif spec["action"] == "test":
        if args.resume is not None:
            raise ValueError("--resume is only valid for a training action")
        checkpoint = args.checkpoint or Path(spec.get("checkpoint", ""))
        if not str(checkpoint):
            raise ValueError("Test action requires --checkpoint or checkpoint in YAML")
        checkpoint = (
            checkpoint.resolve()
            if checkpoint.is_absolute()
            else (ROOT / checkpoint).resolve()
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = ROOT / "experiments" / "raw_metrics" / "{}.json".format(run_id)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_experiment.py"),
            "--config",
            str(native_config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--work-dir",
            str(work_dir),
        ]
        if spec.get("visualize", True):
            command.extend(
                ["--show-dir", str(work_dir / "visualizations")]
            )
        if spec.get("predictions"):
            predictions = Path(spec["predictions"])
            predictions = (
                predictions.resolve()
                if predictions.is_absolute()
                else (ROOT / predictions).resolve()
            )
            command.extend(["--predictions-pkl", str(predictions)])
    else:
        raise ValueError("Unsupported action {!r}".format(spec["action"]))

    manifest_dir = ROOT / "experiments" / "manifests"
    log_dir = ROOT / "experiments" / "logs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "{}.log".format(run_id)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "config": str(native_config),
        "config_sha256": sha256_file(native_config),
        "action": spec["action"],
        "seed": int(spec["seed"]),
        "git": {
            "available": False,
            "reason": "Supplied workspace contains no .git metadata",
            "source_manifest_sha256": (
                "349c9ba4293a229c051ceaf2bab90e6dfc7e98240ab89c2232cc607892f0d185"
            ),
        },
        "command": command,
        "command_display": display_command(command),
        "work_dir": str(work_dir),
        "log": str(log_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if spec.get("predictions"):
        manifest["predictions"] = str(predictions)
    if spec["action"] == "train" and resume is not None:
        manifest["resume"] = {
            "checkpoint": str(resume),
            "checkpoint_bytes": resume.stat().st_size,
            "checkpoint_sha256": sha256_file(resume),
        }
    manifest_path = manifest_dir / "{}.json".format(run_id)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("manifest={}".format(manifest_path))
    print("command={}".format(manifest["command_display"]))
    if not args.execute:
        print("dry-run: add --execute to launch")
        return 0

    env = os.environ.copy()
    python_paths = [str(ROOT), str(ROOT / "third_party" / "mmdetection-3.3.0")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONUTF8": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    manifest["status"] = "running"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    old_input_cp = old_output_cp = None
    if os.name == "nt":
        # MMEngine queries MSVC during environment collection. MSVC follows
        # the console code page while Python UTF-8 mode decodes as UTF-8.
        kernel32 = ctypes.windll.kernel32
        old_input_cp = kernel32.GetConsoleCP()
        old_output_cp = kernel32.GetConsoleOutputCP()
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        if os.name == "nt" and old_input_cp and old_output_cp:
            kernel32.SetConsoleCP(old_input_cp)
            kernel32.SetConsoleOutputCP(old_output_cp)
    manifest["status"] = "completed" if process.returncode == 0 else "failed"
    manifest["returncode"] = process.returncode
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["log_sha256"] = sha256_file(log_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("returncode={}".format(process.returncode))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
