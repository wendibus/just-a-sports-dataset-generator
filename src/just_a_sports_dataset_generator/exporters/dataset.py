from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import PIL

from just_a_sports_dataset_generator import __version__
from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.annotations.scene import Scene, load_frame, load_scene, mapping
from just_a_sports_dataset_generator.core import labels_for_sport
from just_a_sports_dataset_generator.core.taxonomy import RECO_REVISION
from just_a_sports_dataset_generator.plugins import PLUGIN_API_VERSION, PluginMetadata
from just_a_sports_dataset_generator.storage import new_directory, sha256_file, write_json

from .contracts import ExportResult

SPLITS = ("train", "val", "test")


def _yolo_axis(start: int, extent: int, size: int) -> tuple[float, float]:
    center, length = (start + extent / 2) / size, extent / size
    recovered = min(size, (center + length / 2) * size) - max(0, (center - length / 2) * size)
    if recovered < extent:
        # Reco rejects widths below one pixel, even by a rounding error. Expand
        # only by a few floating-point ULPs (far below one pixel), never round inward.
        length = min(1.0, length + 4 * math.ulp(center))
    return center, length


class LocalDatasetExporter:
    metadata = PluginMetadata(
        "org.just-a-sports-dataset-generator.local-exporter",
        "0.2.0",
        PLUGIN_API_VERSION,
        "exporter",
        ("coco", "yolo", "visible_boxes", "instance_masks"),
    )

    def export(
        self, source: Path, destination: Path, formats: tuple[str, ...] = ("coco", "yolo")
    ) -> ExportResult:
        self.metadata.validate()
        return export_dataset(source, destination, formats)


def scene_splits(scenes: tuple[Scene, ...]) -> dict[str, str]:
    """Allocate whole scenes, reserving each split when at least three exist."""
    ranked = sorted(
        scenes,
        key=lambda s: hashlib.sha256(f"split:v1:{s.seed}:{s.job_id}".encode()).digest(),
    )
    count = len(ranked)
    train = 1 if count < 3 else min(count - 2, max(1, int(count * 0.8)))
    val = (count - train) if count < 3 else max(1, int(count * 0.1))
    return {
        scene.job_id: "train" if index < train else "val" if index < train + val else "test"
        for index, scene in enumerate(ranked)
    }


def export_dataset(
    source: Path, destination: Path, formats: tuple[str, ...] = ("coco", "yolo")
) -> ExportResult:
    if not formats or len(formats) != len(set(formats)) or set(formats) - {"coco", "yolo"}:
        raise AnnotationError("choose coco, yolo, or both")
    source = source.resolve()
    destination = destination.absolute()
    if not source.is_dir():
        raise AnnotationError("render source must be a directory")
    resolved_destination = destination.resolve()
    if resolved_destination.is_relative_to(source) or source.is_relative_to(resolved_destination):
        raise AnnotationError("render source and export destination must be separate directories")
    paths = (
        [source / "scene.json"]
        if (source / "scene.json").is_file()
        else sorted(source.glob("*/scene.json"))
    )
    if not paths:
        raise AnnotationError("no scene.json renderer outputs found")
    if any(not path.resolve().is_relative_to(source) for path in paths):
        raise AnnotationError("scene manifest points outside the render source")
    scenes = tuple(load_scene(path) for path in paths)
    if len({scene.job_id for scene in scenes}) != len(scenes):
        raise AnnotationError("duplicate scene job IDs")
    if len({scene.sport for scene in scenes}) != 1:
        raise AnnotationError("export one sport per dataset")
    if len({scene.seed for scene in scenes}) != len(scenes):
        raise AnnotationError("repeated scene seeds would compromise independent splits")
    sport = scenes[0].sport
    labels = labels_for_sport(sport)
    splits = scene_splits(scenes)
    warnings: list[str] = []
    if len(scenes) < 3:
        warnings.append("Fewer than three scenes: some evaluation splits are empty.")
    # Reco imports all items in a folder as a synthetic pseudo-video and rebuilds its splits.
    warnings.append("Reco Trainer's label importer does not preserve these train/val/test splits.")
    run_provenance: dict[str, object] | None = None
    run_path = source / "run.json"
    if run_path.exists():
        if (
            not run_path.resolve().is_relative_to(source)
            or run_path.stat().st_size > 32 * 1024 * 1024
        ):
            raise AnnotationError("invalid run provenance file")
        run_provenance = mapping(json.loads(run_path.read_text(encoding="utf-8")))
    else:
        warnings.append("Legacy render source has no run.json; original recipe is unavailable.")

    coco: dict[str, dict[str, Any]] = {
        split: {
            "info": {"description": f"Synthetic {sport} reference dataset", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": index + 1, "name": label, "supercategory": sport}
                for index, label in enumerate(labels)
            ],
        }
        for split in SPLITS
    }
    image_count = 0
    annotation_count = 0
    hidden_count = 0
    source_files: dict[str, str] = {}
    pixel_splits: dict[str, str] = {}
    split_counts = {split: {"images": 0, "annotations": 0} for split in SPLITS}
    with new_directory(destination) as staging:
        for scene in scenes:
            split = splits[scene.job_id]
            native = staging / "native" / scene.job_id
            native.mkdir(parents=True)
            native_frames: list[dict[str, Any]] = []
            source_files[scene.path.relative_to(source).as_posix()] = sha256_file(scene.path)
            for frame in scene.frames:
                rgb, instances, annotations = load_frame(scene, frame)
                # Detect identical images even when encoders or metadata differ.
                digest = hashlib.sha256(
                    f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes()
                ).hexdigest()
                if digest in pixel_splits and pixel_splits[digest] != split:
                    raise AnnotationError(
                        "identical image pixels found in different dataset splits"
                    )
                pixel_splits[digest] = split
                for path in (frame.rgb, frame.instance_ids):
                    source_files[path.relative_to(source).as_posix()] = sha256_file(path)
                image_count += 1
                hidden_count += len(frame.objects) - len(annotations)
                name = f"{scene.job_id}-frame-{frame.index:06d}.png"
                mask_name = f"frame-{frame.index:06d}.instances.png"
                instances.save(native / mask_name, format="PNG")
                split_counts[split]["images"] += 1
                split_counts[split]["annotations"] += len(annotations)
                native_frames.append(
                    {
                        "frame_index": frame.index,
                        "image_name": name,
                        "instance_ids": mask_name,
                        "objects": [asdict(item) for item in frame.objects],
                        "visible_annotations": [asdict(item) for item in annotations],
                    }
                )
                if "coco" in formats:
                    coco_folder = staging / "coco" / ("valid" if split == "val" else split)
                    coco_folder.mkdir(parents=True, exist_ok=True)
                    rgb.save(coco_folder / name, format="PNG")
                    coco[split]["images"].append(
                        {
                            "id": image_count,
                            "file_name": name,
                            "width": scene.width,
                            "height": scene.height,
                        }
                    )
                if "yolo" in formats:
                    images = staging / "yolo" / "images" / split
                    txt = staging / "yolo" / "labels" / split
                    images.mkdir(parents=True, exist_ok=True)
                    txt.mkdir(parents=True, exist_ok=True)
                    rgb.save(images / name, format="PNG")
                    lines: list[str] = []
                    for item in annotations:
                        x, y, width, height = item.bbox_xywh
                        cx, bw = _yolo_axis(x, width, scene.width)
                        cy, bh = _yolo_axis(y, height, scene.height)
                        values = (cx, cy, bw, bh)
                        lines.append(f"{item.class_id} " + " ".join(repr(v) for v in values))
                    (txt / Path(name).with_suffix(".txt")).write_text(
                        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                    )
                for item in annotations:
                    annotation_count += 1
                    coco[split]["annotations"].append(
                        {
                            "id": annotation_count,
                            "image_id": image_count,
                            "category_id": item.class_id + 1,
                            "bbox": list(item.bbox_xywh),
                            "area": item.visible_area,
                            "iscrowd": 0,
                            "segmentation": {
                                "size": [scene.height, scene.width],
                                "counts": list(item.rle_counts),
                            },
                        }
                    )
            write_json(
                native / "annotations.json",
                {
                    "schema_version": "1.0",
                    "job_id": scene.job_id,
                    "sport": sport,
                    "labels": labels,
                    "seed": scene.seed,
                    "split": split,
                    "resolution": {"width": scene.width, "height": scene.height},
                    "bbox_convention": "visible pixels; top-left xywh; exclusive right/bottom",
                    "frames": native_frames,
                },
            )
        for split in SPLITS:
            if "coco" in formats:
                folder = staging / "coco" / ("valid" if split == "val" else split)
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / "_annotations.coco.json", coco[split])
            if "yolo" in formats:
                for kind in ("images", "labels"):
                    (staging / "yolo" / kind / split).mkdir(parents=True, exist_ok=True)
        if "yolo" in formats:
            # Omitting path makes the root relative to data.yaml in Ultralytics.
            yaml = "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
            yaml += "".join(f"  {index}: {label}\n" for index, label in enumerate(labels))
            (staging / "yolo" / "data.yaml").write_text(yaml, encoding="utf-8")
        if run_provenance is not None:
            source_files["run.json"] = sha256_file(run_path)
        write_json(
            staging / "report.json",
            {
                "schema_version": "1.0",
                "images": image_count,
                "annotations": annotation_count,
                "fully_occluded_or_offscreen_objects": hidden_count,
                "split_counts": split_counts,
                "scene_splits": splits,
                "warnings": warnings,
                "checks": [
                    "taxonomy",
                    "image/mask dimensions",
                    "instance identities",
                    "visible pixel bounds/area",
                    "whole-scene splits",
                    "cross-split duplicate pixels",
                ],
            },
        )
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "application_version": __version__,
            "exporter": asdict(LocalDatasetExporter.metadata),
            "sport": sport,
            "labels": labels,
            "class_mapping": [
                {"label": label, "yolo_id": index, "coco_id": index + 1}
                for index, label in enumerate(labels)
            ],
            "reco_trainer_revision": RECO_REVISION,
            "formats": sorted(formats),
            "scene_splits": splits,
            "sources_sha256": source_files,
            "run": run_provenance,
            "renderers": [
                {
                    "job_id": scene.job_id,
                    "id": scene.renderer,
                    "version": scene.renderer_version,
                    "seed": scene.seed,
                    "assets": scene.assets,
                }
                for scene in scenes
            ],
            "environment": {
                "python": platform.python_version(),
                "pillow": PIL.__version__,
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "files_sha256": {
                path.relative_to(staging).as_posix(): sha256_file(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
        }
        write_json(staging / "manifest.json", manifest)
    return ExportResult(destination, image_count, annotation_count, tuple(warnings))
