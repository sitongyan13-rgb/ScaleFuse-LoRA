#!/usr/bin/env python
"""Collect raw metric JSON files and optionally render a smoke-test loss curve."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def flatten_numeric(prefix: str, value, output: Dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            name = "{}.{}".format(prefix, key) if prefix else str(key)
            flatten_numeric(name, child, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def load_raw_metrics(raw_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("metrics", payload)
        flat = {}
        flatten_numeric("", metrics, flat)
        row = {
            "run": path.stem,
            "source": str(path.resolve()),
            "config": payload.get("config", ""),
            "checkpoint": payload.get("checkpoint", ""),
        }
        row.update(flat)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed = ["run", "source", "config", "checkpoint"]
    extra = sorted({key for row in rows for key in row if key not in fixed})
    fields = fixed + extra
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_scalars(path: Path) -> Tuple[List[int], List[float]]:
    iterations = []
    losses = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if "iter" in item and "loss" in item:
                iterations.append(int(item["iter"]))
                losses.append(float(item["loss"]))
    return iterations, losses


def save_loss_curve(scalars: Path, prefix: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations, losses = load_scalars(scalars)
    if not iterations:
        raise RuntimeError("No training loss records found in {}".format(scalars))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(iterations, losses, linewidth=1.6, color="#1f77b4", label="total loss")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    axis.set_title("Grounding DINO Swin-T — 100-iteration smoke test")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(str(prefix.with_suffix("." + extension)), **kwargs)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("experiments/raw_metrics")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/csv/metrics_summary.csv")
    )
    parser.add_argument("--scalars", type=Path)
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("results/figures/smoke_cat_100iter_loss"),
    )
    args = parser.parse_args()
    rows = load_raw_metrics(args.raw_dir)
    write_csv(rows, args.output)
    print("metric_files={} csv={}".format(len(rows), args.output.resolve()))
    if args.scalars:
        save_loss_curve(args.scalars, args.figure_prefix)
        print("loss_curve_prefix={}".format(args.figure_prefix.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
