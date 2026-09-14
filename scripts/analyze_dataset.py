#!/usr/bin/env python
"""Create reproducible statistics from real COCO-style annotation files."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List


def split_name(path: Path) -> str:
    text = "/".join(part.lower() for part in path.parts)
    for name in ("train", "valid", "validation", "val", "test"):
        if "/{}/".format(name) in "/{}/".format(text) or path.stem.lower() == name:
            return "val" if name in ("valid", "validation") else name
    return path.stem


def load_coco(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not {
        "images",
        "annotations",
        "categories",
    }.issubset(payload):
        return None
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/odinw")
    parser.add_argument(
        "--report", default="reports/dataset_statistics.md"
    )
    parser.add_argument(
        "--json-output", default="results/json/dataset_statistics.json"
    )
    parser.add_argument(
        "--csv-output", default="results/csv/dataset_split_statistics.csv"
    )
    parser.add_argument(
        "--catalog",
        help=(
            "Optional generated ODinW catalog; when set, analyze only the "
            "exact train/val/test files selected by the 13-task protocol"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root)
    report = Path(args.report)
    json_output = Path(args.json_output)
    csv_output = Path(args.csv_output)
    if not root.is_dir():
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Dataset Statistics\n\n"
            "Status: **not available**. The selected ODinW-13 dataset has not "
            "been downloaded, so no image, annotation, class, corruption, "
            "duplicate, or split-overlap count is claimed.\n",
            encoding="utf-8",
        )
        print("ERROR: dataset root does not exist: {}".format(root), file=sys.stderr)
        return 2

    rows: List[Dict[str, object]] = []
    categories_total = set()
    selected_paths = []
    if args.catalog:
        catalog_path = Path(args.catalog)
        with catalog_path.open("r", encoding="utf-8") as handle:
            catalog = json.load(handle)
        for task in catalog["tasks"]:
            for selected_split in ("train", "val", "test"):
                item = task[selected_split]
                relative = item["ann_file"].replace("\\", "/")
                if relative.startswith("odinw/"):
                    relative = relative[len("odinw/") :]
                selected_paths.append(
                    (root / relative, selected_split, task["task"])
                )
    else:
        selected_paths = [
            (path, split_name(path.relative_to(root)), "")
            for path in sorted(root.rglob("*.json"))
        ]

    for path, selected_split, task_name in selected_paths:
        try:
            payload = load_coco(path)
        except Exception as exc:
            print("WARNING: skip {}: {}".format(path, exc), file=sys.stderr)
            continue
        if payload is None:
            continue
        category_map = {
            item["id"]: item.get("name", str(item["id"]))
            for item in payload["categories"]
        }
        category_counts = Counter(
            category_map.get(item.get("category_id"), str(item.get("category_id")))
            for item in payload["annotations"]
        )
        categories_total.update(category_map.values())
        small = medium = large = invalid = 0
        for annotation in payload["annotations"]:
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                invalid += 1
                continue
            area = float(annotation.get("area", bbox[2] * bbox[3]))
            if area < 32 ** 2:
                small += 1
            elif area < 96 ** 2:
                medium += 1
            else:
                large += 1
        rows.append(
            {
                "annotation_file": str(path.relative_to(root)),
                "split": selected_split,
                "task": task_name,
                "images": len(payload["images"]),
                "annotations": len(payload["annotations"]),
                "categories": len(payload["categories"]),
                "small_boxes": small,
                "medium_boxes": medium,
                "large_boxes": large,
                "invalid_boxes": invalid,
                "category_counts": dict(category_counts),
            }
        )
    if not rows:
        print("ERROR: no COCO annotation JSON was found under {}".format(root))
        return 2

    totals = {
        "annotation_files": len(rows),
        "image_records": sum(int(row["images"]) for row in rows),
        "annotations": sum(int(row["annotations"]) for row in rows),
        "unique_category_names": len(categories_total),
        "note": (
            "Exact generated ODinW-13 train/val/test catalog."
            if args.catalog
            else "Image records may repeat across alternate annotation JSONs; "
            "use verify_dataset.py for unique files and split leakage."
        ),
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8") as handle:
        json.dump({"root": str(root.resolve()), "totals": totals, "splits": rows},
                  handle, ensure_ascii=False, indent=2)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "annotation_file",
        "split",
        "images",
        "annotations",
        "categories",
        "small_boxes",
        "medium_boxes",
        "large_boxes",
        "invalid_boxes",
    ]
    with csv_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    lines = [
        "# Dataset Statistics",
        "",
        "Source root: `{}`".format(root.resolve()),
        "",
        "Annotation files: **{}**; image records: **{}**; boxes: **{}**; "
        "unique category names: **{}**; invalid/non-positive boxes: **{}**."
        .format(
            totals["annotation_files"],
            totals["image_records"],
            totals["annotations"],
            totals["unique_category_names"],
            sum(int(row["invalid_boxes"]) for row in rows),
        ),
        "",
        "These values were computed from the JSON files present on disk. "
        "Repeated image records in multiple annotation variants are not "
        "silently deduplicated.",
        "",
        "| Task | Annotation file | Split | Images | Boxes | Classes | Small | Medium | Large | Invalid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {annotation_file} | {split} | {images} | {annotations} | "
            "{categories} | {small_boxes} | {medium_boxes} | {large_boxes} | "
            "{invalid_boxes} |".format(**row)
        )
    lines.extend(
        [
            "",
            (
                "Catalog mode uses the exact 13 GLIP-selected tasks and their "
                "train/val/test registrations. Pascal VOC's official GLIP "
                "registration aliases test to its valid split."
                if args.catalog
                else ""
            ),
            "",
            "Raw machine-readable output: `{}`.".format(json_output),
            "",
            "Integrity, duplicate, and split-overlap results are generated by "
            "`scripts/verify_dataset.py` and are not inferred here.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("report={} json={} csv={}".format(report, json_output, csv_output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
