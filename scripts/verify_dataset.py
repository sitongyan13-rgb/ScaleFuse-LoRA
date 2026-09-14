#!/usr/bin/env python
"""Verify COCO-style datasets, images, duplicates, and split leakage."""

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_name(path: Path) -> str:
    text = "/".join(part.lower() for part in path.parts)
    for name in ("train", "valid", "validation", "val", "test"):
        if "/{}/".format(name) in "/{}/".format(text) or path.stem.lower() == name:
            return "val" if name in ("valid", "validation") else name
    return path.stem


def resolve_image(root: Path, annotation_path: Path, file_name: str) -> Optional[Path]:
    rel = Path(file_name.replace("\\", "/"))
    candidates = [
        annotation_path.parent / rel,
        annotation_path.parent.parent / rel,
        root / rel,
    ]
    for parent_name in ("images", "train", "valid", "val", "test"):
        candidates.extend(
            [
                annotation_path.parent / parent_name / rel,
                annotation_path.parent.parent / parent_name / rel,
                root / parent_name / rel,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(root.rglob(rel.name))
    return matches[0].resolve() if len(matches) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/odinw")
    parser.add_argument(
        "--output", default="results/json/dataset_verification.json"
    )
    parser.add_argument(
        "--archives", default="data/archives/odinw13", help="ZIP directory"
    )
    parser.add_argument(
        "--skip-pixel-check",
        action="store_true",
        help="Only check image references and hashes, not Pillow decode",
    )
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    if not root.is_dir():
        print("ERROR: dataset root does not exist: {}".format(root), file=sys.stderr)
        return 2

    annotation_paths = sorted(root.rglob("*.json"))
    errors: List[str] = []
    split_paths: Dict[str, Set[str]] = defaultdict(set)
    referenced: Set[Path] = set()
    annotation_summaries = []
    for annotation_path in annotation_paths:
        try:
            with annotation_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            errors.append("{}: JSON decode failed: {}".format(annotation_path, exc))
            continue
        if not isinstance(payload, dict) or "images" not in payload:
            continue
        split = split_name(annotation_path.relative_to(root))
        missing = []
        ids = set()
        for image in payload.get("images", []):
            image_id = image.get("id")
            if image_id in ids:
                errors.append(
                    "{}: duplicate image id {}".format(annotation_path, image_id)
                )
            ids.add(image_id)
            file_name = image.get("file_name")
            if not file_name:
                missing.append("<empty file_name>")
                continue
            resolved = resolve_image(root, annotation_path, file_name)
            if resolved is None:
                missing.append(file_name)
            else:
                referenced.add(resolved)
                split_paths[split].add(str(resolved).lower())
        annotation_summaries.append(
            {
                "path": str(annotation_path),
                "split": split,
                "images": len(payload.get("images", [])),
                "annotations": len(payload.get("annotations", [])),
                "categories": len(payload.get("categories", [])),
                "missing_image_references": len(missing),
                "missing_examples": missing[:20],
            }
        )
        if missing:
            errors.append(
                "{}: {} missing image reference(s)".format(annotation_path, len(missing))
            )

    all_images = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    corrupt = []
    hashes: Dict[str, List[str]] = defaultdict(list)
    for index, path in enumerate(all_images, 1):
        if not args.skip_pixel_check:
            try:
                with Image.open(str(path)) as image:
                    image.verify()
            except Exception as exc:
                corrupt.append({"path": str(path), "error": repr(exc)})
                continue
        hashes[sha256_file(path)].append(str(path.resolve()))
        if index % 500 == 0:
            print("checked {}/{} images".format(index, len(all_images)), flush=True)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]

    # Aggregate files such as annotations_all.json are useful integrity inputs
    # but are not independent partitions. Compare only recognized dataset
    # partitions so a deliberate aggregate does not become a leakage false
    # positive.
    partition_splits = {
        "train",
        "trainval",
        "val",
        "test",
    }
    split_names = sorted(
        split for split in split_paths if split.lower() in partition_splits
    )
    path_intersections = {}
    hash_intersections = {}
    hashes_by_split: Dict[str, Set[str]] = defaultdict(set)
    path_to_hash = {}
    for digest, paths in hashes.items():
        for path in paths:
            path_to_hash[path.lower()] = digest
    for split, paths in split_paths.items():
        hashes_by_split[split] = {
            path_to_hash[path] for path in paths if path in path_to_hash
        }
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            key = "{}__{}".format(left, right)
            common_paths = split_paths[left].intersection(split_paths[right])
            common_hashes = hashes_by_split[left].intersection(hashes_by_split[right])
            if common_paths:
                path_intersections[key] = sorted(common_paths)
            if common_hashes:
                hash_intersections[key] = sorted(common_hashes)

    archive_results = []
    archive_root = Path(args.archives)
    if archive_root.is_dir():
        for archive in sorted(archive_root.glob("*.zip")):
            try:
                with zipfile.ZipFile(str(archive)) as zf:
                    bad = zf.testzip()
                archive_results.append(
                    {
                        "path": str(archive),
                        "size_bytes": archive.stat().st_size,
                        "sha256": sha256_file(archive),
                        "zip_test": "ok" if bad is None else "corrupt: {}".format(bad),
                    }
                )
                if bad is not None:
                    errors.append("{}: corrupt ZIP member {}".format(archive, bad))
            except Exception as exc:
                errors.append("{}: ZIP check failed: {}".format(archive, exc))

    if corrupt:
        errors.append("{} corrupt image(s)".format(len(corrupt)))
    if path_intersections:
        errors.append("same image path is referenced by multiple splits")
    if hash_intersections:
        errors.append("same image bytes occur in multiple splits")

    result = {
        "root": str(root.resolve()),
        "status": "pass" if not errors else "fail",
        "annotation_files": annotation_summaries,
        "image_files_found": len(all_images),
        "referenced_images_found": len(referenced),
        "corrupt_images": corrupt,
        "duplicate_groups": duplicate_groups,
        "same_path_cross_split": path_intersections,
        "same_hash_cross_split": hash_intersections,
        "archives": archive_results,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print("status={} output={}".format(result["status"], output.resolve()))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
