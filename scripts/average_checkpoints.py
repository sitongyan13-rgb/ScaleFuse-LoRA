#!/usr/bin/env python
"""Create a traceable two-checkpoint parameter soup."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.first_weight <= 1.0:
        raise ValueError("--first-weight must be in [0, 1]")
    first_path, second_path = args.first.resolve(), args.second.resolve()
    output = args.output.resolve()
    if output in (first_path, second_path):
        raise ValueError("Output must not overwrite either source checkpoint")
    if not first_path.is_file() or not second_path.is_file():
        raise FileNotFoundError("Both source checkpoints must exist")

    first = torch.load(str(first_path), map_location="cpu")
    second = torch.load(str(second_path), map_location="cpu")
    first_state = first["state_dict"]
    second_state = second["state_dict"]
    if set(first_state) != set(second_state):
        raise RuntimeError("Checkpoint state_dict keys differ")

    averaged = 0
    copied_non_float = 0
    for name, tensor in first_state.items():
        other = second_state[name]
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            raise RuntimeError("Tensor mismatch for {}".format(name))
        if tensor.is_floating_point() or tensor.is_complex():
            tensor.mul_(args.first_weight).add_(
                other, alpha=1.0 - args.first_weight)
            averaged += 1
        else:
            if not torch.equal(tensor, other):
                raise RuntimeError("Non-floating state differs for {}".format(name))
            copied_non_float += 1

    # Training optimizer/message state is intentionally omitted: this artifact
    # is an inference checkpoint, not a resumable training checkpoint.
    soup = {"state_dict": first_state, "meta": dict(first.get("meta", {}))}
    soup["meta"]["checkpoint_soup"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "first": str(first_path),
        "second": str(second_path),
        "first_weight": args.first_weight,
        "second_weight": 1.0 - args.first_weight,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, str(output))
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "equal_run_adjacent_checkpoint_soup",
        "sources": [
            {"path": str(first_path), "sha256": sha256_file(first_path)},
            {"path": str(second_path), "sha256": sha256_file(second_path)},
        ],
        "weights": [args.first_weight, 1.0 - args.first_weight],
        "averaged_tensors": averaged,
        "copied_non_float_tensors": copied_non_float,
        "output": str(output),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "resume_training_supported": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
