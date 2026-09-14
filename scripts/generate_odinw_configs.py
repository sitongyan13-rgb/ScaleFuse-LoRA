#!/usr/bin/env python
"""Generate thin MMDetection configs from the pinned official GLIP ODinW-13 YAMLs."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GLIP_CONFIG_ROOT = ROOT / "third_party" / "GLIP-main" / "configs" / "odinw_13"
TASKS = (
    ("aerial_maritime_drone", "AerialMaritimeDrone_large.yaml"),
    ("aquarium", "Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml"),
    ("cottontail_rabbits", "CottontailRabbits.yaml"),
    ("ego_hands", "EgoHands_generic.yaml"),
    (
        "north_america_mushrooms",
        "NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco.yaml",
    ),
    ("packages", "Packages_Raw.yaml"),
    ("pascal_voc", "PascalVOC.yaml"),
    ("pistols", "pistols_export.yaml"),
    ("pothole", "pothole.yaml"),
    ("raccoon", "Raccoon_Raccoon.v2-raw.coco.yaml"),
    ("shellfish_open_images", "ShellfishOpenImages_raw.yaml"),
    ("thermal_dogs_and_people", "thermalDogsAndPeople.yaml"),
    ("vehicles_open_images", "VehiclesOpenImages_416x416.yaml"),
)
VARIANTS = {
    "full_finetune": "../odinw_aquarium_full_finetune.py",
    "frozen_text": "../odinw_aquarium_frozen_text.py",
    "adapter": "../odinw_aquarium_adapter.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_python_tuple(values):
    if len(values) == 1:
        return "({!r},)".format(values[0])
    return repr(tuple(values))


def normalize_data_path(value: str) -> str:
    value = value.replace("\\", "/").lstrip("/")
    return value if value.startswith("odinw/") else "odinw/" + value


def render_config(task, variant, base_config, classes, train, val):
    train_ann = normalize_data_path(train["ann_file"])
    train_img = normalize_data_path(train["img_dir"]).rstrip("/") + "/"
    val_ann = normalize_data_path(val["ann_file"])
    val_img = normalize_data_path(val["img_dir"]).rstrip("/") + "/"
    return """# Generated mechanically by scripts/generate_odinw_configs.py.
# Dataset paths/classes come from the pinned official GLIP ODinW-13 YAML.
_base_ = {base!r}

task_name = {task!r}
class_name = {classes}
metainfo = dict(classes=class_name)
data_root = 'data/'

model = dict(bbox_head=dict(num_classes=len(class_name)))
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file={train_ann!r},
        data_prefix=dict(img={train_img!r})))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file={val_ann!r},
        data_prefix=dict(img={val_img!r})))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + {val_ann!r})
test_evaluator = val_evaluator
work_dir = 'experiments/odinw/{task}/{variant}'
""".format(
        base=base_config,
        task=task,
        classes=as_python_tuple(classes),
        train_ann=train_ann,
        train_img=train_img,
        val_ann=val_ann,
        val_img=val_img,
        variant=variant,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "configs" / "baselines" / "generated",
    )
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=ROOT / "configs" / "baselines" / "generated_specs",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    spec_dir = args.spec_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    catalog = {"source_root": str(GLIP_CONFIG_ROOT), "tasks": []}
    generated = 0
    for task, yaml_name in TASKS:
        source = GLIP_CONFIG_ROOT / yaml_name
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        datasets = payload["DATASETS"]
        categories = json.loads(datasets["OVERRIDE_CATEGORY"])
        classes = tuple(item["name"] for item in categories)
        register = datasets["REGISTER"]
        train = register["train"]
        val = register["val"]
        test = register["test"]
        task_entry = {
            "task": task,
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256_file(source),
            "classes": list(classes),
            "train": train,
            "val": val,
            "test": test,
            "configs": [],
        }
        for variant, base_config in VARIANTS.items():
            destination = output_dir / "{}_{}.py".format(task, variant)
            content = render_config(
                task, variant, base_config, classes, train, val
            )
            if destination.exists() and not args.force:
                existing = destination.read_text(encoding="utf-8")
                if existing != content:
                    raise FileExistsError(
                        "{} differs; pass --force to replace".format(destination)
                    )
            else:
                destination.write_text(content, encoding="utf-8")
            task_entry["configs"].append(str(destination.relative_to(ROOT)))
            spec_path = spec_dir / "{}_{}.yaml".format(task, variant)
            spec_content = (
                "name: odinw_{task}_{variant}\n"
                "action: train\n"
                "native_config: {config}\n"
                "seed: 42\n"
                "task: {task}\n"
                "trainable: {trainable}\n"
            ).format(
                task=task,
                variant=variant,
                config=destination.relative_to(ROOT).as_posix(),
                trainable={
                    "full_finetune": "all",
                    "frozen_text": "all_except_language_model",
                    "adapter": "vision_and_text_residual_adapters_only",
                }[variant],
            )
            if spec_path.exists() and not args.force:
                existing_spec = spec_path.read_text(encoding="utf-8")
                if existing_spec != spec_content:
                    raise FileExistsError(
                        "{} differs; pass --force to replace".format(spec_path)
                    )
            else:
                spec_path.write_text(spec_content, encoding="utf-8")
            task_entry.setdefault("specs", []).append(
                str(spec_path.relative_to(ROOT))
            )
            generated += 1
        catalog["tasks"].append(task_entry)

    catalog_path = output_dir / "catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("generated={} catalog={}".format(generated, catalog_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
