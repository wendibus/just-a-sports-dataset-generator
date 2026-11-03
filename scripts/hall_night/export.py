"""COCO handoff with explicit group/identity splits, variable view resolutions."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image
from pycocotools import mask as coco_mask

from just_a_sports_dataset_generator.annotations.scene import load_frame, load_scene
from just_a_sports_dataset_generator.core import labels_for_sport
from just_a_sports_dataset_generator.storage import new_directory

from .common import read, sha, validate, write


def export_and_verify(root: Path, config, plans, pilot=False):
    target = root / ("pilot-dataset" if pilot else "dataset")
    if target.exists():
        verify(target)
        return target
    schema = read(Path(__file__).resolve().parents[2] / "schemas/scene-v2.0.json")
    validator = Draft202012Validator(schema)
    labels = labels_for_sport("basketball")
    categories = [
        {"id": i + 1, "name": name, "supercategory": "basketball"} for i, name in enumerate(labels)
    ]
    data = {
        split: {
            "info": {
                "description": "Local photographic hall composites. Same-scene crops and asset identities grouped by split."
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        for split in ("train", "val", "test")
    }
    image_id, annotation_id = 0, 0
    assets_by_split = {s: set() for s in data}
    counts = Counter()
    gender_counts = Counter()
    ball_counts = Counter()
    shots = Counter()
    partial_people = partial_balls = 0
    image_hashes = {}
    expected = len(plans) * 2
    with new_directory(target) as stage:
        for split in data:
            (stage / "coco" / ("valid" if split == "val" else split)).mkdir(parents=True)
        (stage / "native").mkdir()
        for index, plan in enumerate(plans):
            folder = root / "renders" / plan["id"]
            done = read(folder / "done.json")
            for filename, digest in done["files"].items():
                if sha(folder / filename) != digest:
                    raise ValueError("Source changed since rendering")
            metadata = read(folder / "view-metadata.json")
            assets = {a["asset"] for a in metadata["layers"] if not a["asset"].startswith("ball-")}
            split = plan["split"]
            assets_by_split[split].update(assets)
            for i in range(2):
                path = folder / f"scene-{i:03d}.json"
                validator.validate(read(path))
                scene = load_scene(path)
                rgb, _ids, annotations = load_frame(scene, scene.frames[0])
                image_id += 1
                name = f"{plan['id']}-v{i}.png"
                split_name = "valid" if split == "val" else split
                shutil.copyfile(scene.frames[0].rgb, stage / "coco" / split_name / name)
                shutil.copyfile(scene.frames[0].instance_ids, stage / "native" / name)
                digest = sha(scene.frames[0].rgb)
                if digest in image_hashes:
                    raise ValueError(f"Duplicate training image: {name}, {image_hashes[digest]}")
                image_hashes[digest] = name
                data[split]["images"].append(
                    {
                        "id": image_id,
                        "file_name": name,
                        "width": rgb.width,
                        "height": rgb.height,
                        "scene_group": plan["id"],
                        "sha256": digest,
                    }
                )
                for annotation in annotations:
                    annotation_id += 1
                    counts[annotation.label] += 1
                    data[split]["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": annotation.class_id + 1,
                            "bbox": list(annotation.bbox_xywh),
                            "area": annotation.visible_area,
                            "iscrowd": 0,
                            "instance_id": annotation.instance_id,
                            "segmentation": {
                                "size": [rgb.height, rgb.width],
                                "counts": list(annotation.rle_counts),
                            },
                        }
                    )
                stats = done["statistics"][i]
                gender_counts[stats["gender_mode"]] += 1
                ball_counts[stats["ball_mode"]] += 1
                shots[stats["shot"]] += 1
                occ = stats["occlusion_in_view"]
                partial_people += int(
                    any(0.03 < float(v) < 0.95 for k, v in occ.items() if 10 <= int(k) < 40)
                )
                partial_balls += int(
                    any(0.03 < float(v) < 0.95 for k, v in occ.items() if 1 <= int(k) < 5)
                )
            if index % 25 == 0:
                print(f"COCO export: {image_id}/{expected}", flush=True)
        assert image_id == expected
        if not pilot:
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                if assets_by_split[a] & assets_by_split[b]:
                    raise ValueError("Person identity leakage between splits")
        for split, coco in data.items():
            write(
                stage / "coco" / ("valid" if split == "val" else split) / "_annotations.coco.json",
                coco,
            )
        write(
            stage / "manifest.json",
            {
                "schema_version": "1.0",
                "format": "hall-photo-coco",
                "images": image_id,
                "annotations": annotation_id,
                "labels": list(labels),
                "background_sha256": config["background_sha256"],
                "scene_groups": len(plans),
                "counts": dict(counts),
                "gender_modes": dict(gender_counts),
                "ball_modes": dict(ball_counts),
                "shots": dict(shots),
                "images_with_partially_occluded_people": partial_people,
                "images_with_partially_occluded_balls": partial_balls,
                "person_asset_ids_by_split": {s: sorted(v) for s, v in assets_by_split.items()},
                "pilot_preview_only": pilot,
                "limitations": [
                    "One hall only; real held-out footage required for quality evaluation",
                    "Photographic 2.5D composites, not physical simulation",
                    "Requested demographics and poses need visual asset review",
                    "Static hoop envelopes are manually traced; photographic net holes are approximate",
                    "Reco imports boxes and rebuilds its splits; retain this export for grouped evaluation",
                ],
                "rights": "Source hall and ball reference sheets supplied by the user; no redistribution rights asserted.",
            },
        )
        verify(stage)
    return target


def verify(dataset: Path):
    labels = list(labels_for_sport("basketball"))
    total_images = total_annotations = 0
    for split in ("train", "valid", "test"):
        folder = dataset / "coco" / split
        coco = read(folder / "_annotations.coco.json")
        assert coco["categories"] == [
            {"id": i + 1, "name": name, "supercategory": "basketball"}
            for i, name in enumerate(labels)
        ]
        grouped = {}
        for ann in coco["annotations"]:
            grouped.setdefault(ann["image_id"], []).append(ann)
        for image in coco["images"]:
            path = folder / image["file_name"]
            if sha(path) != image["sha256"]:
                raise ValueError("Exported RGB checksum mismatch")
            with Image.open(dataset / "native" / image["file_name"]) as raw:
                ids = np.asarray(raw)
            assert ids.shape == (image["height"], image["width"])
            expected_ids = {int(v) for v in np.unique(ids)} - {0}
            actual_ids = set()
            for ann in grouped.get(image["id"], []):
                actual_ids.add(ann["instance_id"])
                rle = coco_mask.frPyObjects(ann["segmentation"], image["height"], image["width"])
                mask = coco_mask.decode(rle)
                if not np.array_equal(mask.astype(bool), ids == ann["instance_id"]):
                    raise ValueError("COCO mask differs from final visible IDs")
                area = int(mask.sum())
                bbox = [int(v) for v in coco_mask.toBbox(rle)]
                assert ann["area"] == area and ann["bbox"] == bbox
                assert 1 <= ann["category_id"] <= 4
                total_annotations += 1
            assert actual_ids == expected_ids, "Unlabelled visible object ID"
            total_images += 1
            if total_images % 100 == 0:
                print(f"Mask verification: {total_images} images", flush=True)
        print("Verified", split, len(coco["images"]), "images", flush=True)
    manifest = read(dataset / "manifest.json")
    validate("hall-photo-export-v1.0.json", manifest)
    assert total_images == manifest["images"] and total_annotations == manifest["annotations"]
    write(
        dataset / "verification.json",
        {
            "passed": True,
            "images": total_images,
            "annotations": total_annotations,
            "checks": [
                "Every exported RLE equals its final instance mask",
                "Every bounding box and area match its mask",
                "Every visible ID is labelled",
                "Reco basketball class ordering and one-based COCO IDs",
                "Every exported RGB checksum matches the render",
            ],
            "manifest_sha256": sha(dataset / "manifest.json"),
        },
    )
