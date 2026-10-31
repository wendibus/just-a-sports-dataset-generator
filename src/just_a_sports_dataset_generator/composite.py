"""Paste generated person crops back into a render without touching the ground truth.

Labels are never re-detected. The default ``mask`` blend writes generated pixels only where
the instance-ID pass shows that person, so every box and mask stays exactly valid; only the
appearance changes. The ``inpaint`` blend also accepts the dilated band around the person,
but never pixels of another object (players, ball, hoops).

Quality checks reject a crop (keeping the rendered person) when the generator left the
person unchanged, when it changed the background band around the person strongly (a sign
of a shifted or oversized figure), or, if silhouettes are supplied, when their IoU with the
instance mask is too low. These are filters on generated pixels, not label sources.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.annotations.scene import (
    integer,
    load_scene,
    mapping,
    safe_file,
    sequence,
    string,
)
from just_a_sports_dataset_generator.storage import new_directory, sha256_file, write_json

BLENDS = ("mask", "inpaint")


@dataclass(frozen=True, slots=True)
class CompositeResult:
    output_directory: Path
    applied: int
    rejected: int
    missing: int


@dataclass(frozen=True, slots=True)
class _Crop:
    id: str
    job_id: str
    frame_index: int
    instance_id: int
    box: tuple[int, int, int, int]
    files: dict[str, Path]


def _load_crops(crops: Path) -> tuple[int, list[_Crop]]:
    path = crops / "crops.json"
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise AnnotationError("crops directory needs a crops.json up to 32 MiB")
    data = mapping(json.loads(path.read_text(encoding="utf-8")))
    if data.get("schema_version") != "1.0":
        raise AnnotationError("unsupported crops.json version")
    size = integer(data.get("size"), 64)
    result: list[_Crop] = []
    for raw in sequence(data.get("crops")):
        item = mapping(raw)
        box = _ints(item.get("crop_box_xyxy"))
        if len(box) != 4 or box[2] - box[0] != box[3] - box[1] or box[2] <= box[0]:
            raise AnnotationError("crop boxes must be non-empty squares")
        files = mapping(item.get("files"))
        result.append(
            _Crop(
                string(item.get("id")),
                string(item.get("job_id")),
                integer(item.get("frame_index")),
                integer(item.get("instance_id"), 1),
                (box[0], box[1], box[2], box[3]),
                {kind: safe_file(crops, files.get(kind)) for kind in ("rgb", "mask", "inpaint")},
            )
        )
    if len({crop.id for crop in result}) != len(result):
        raise AnnotationError("duplicate crop IDs")
    return size, result


def _ints(value: object) -> list[int]:
    items = sequence(value)
    if any(type(item) is not int for item in items):
        raise AnnotationError("expected an integer array")
    return [int(item) for item in items]  # pyright: ignore[reportArgumentType]


def _find(directory: Path | None, crop_id: str) -> list[Path]:
    """Files named ``<id>.png`` or ``<id>_<suffix>.png`` (ComfyUI appends counters)."""
    if directory is None:
        return []
    return sorted(
        path
        for path in directory.rglob(f"{crop_id}*.png")
        if path.is_file() and (path.stem == crop_id or path.stem.startswith(crop_id + "_"))
    )


def _binary(image: Image.Image) -> Image.Image:
    return image.convert("L").point([255 if value > 127 else 0 for value in range(256)])


def _mean_difference(a: Image.Image, b: Image.Image, region: Image.Image) -> float | None:
    """Mean absolute RGB difference in [0, 1] over the white pixels of ``region``."""
    if region.getbbox() is None:
        return None
    difference = ImageChops.difference(a, b).convert("L")
    return ImageStat.Stat(difference, region).mean[0] / 255


def _iou(a: Image.Image, b: Image.Image) -> float:
    union = ImageChops.lighter(a, b).histogram()[255]
    return ImageChops.darker(a, b).histogram()[255] / union if union else 0.0


def composite_crops(
    render: Path,
    crops: Path,
    generated: Path,
    destination: Path,
    *,
    blend: str = "mask",
    silhouettes: Path | None = None,
    min_iou: float = 0.85,
    max_spill: float = 0.15,
    min_change: float = 0.02,
    generator: str | None = None,
) -> CompositeResult:
    if blend not in BLENDS:
        raise ValueError(f"blend must be one of: {', '.join(BLENDS)}")
    for name, value in (("min_iou", min_iou), ("max_spill", max_spill), ("min_change", min_change)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    render, crops, generated = render.resolve(), crops.resolve(), generated.resolve()
    for folder in (render, crops, generated, *([silhouettes] if silhouettes else [])):
        if not folder.is_dir():
            raise AnnotationError(f"not a directory: {folder}")
    size, items = _load_crops(crops)
    by_frame: dict[tuple[str, int], list[_Crop]] = {}
    for crop in items:
        by_frame.setdefault((crop.job_id, crop.frame_index), []).append(crop)

    report: list[dict[str, object]] = []
    with new_directory(destination) as staging:
        shutil.copytree(render, staging, dirs_exist_ok=True, symlinks=False)
        scenes = {path.parent.name: load_scene(path) for path in staging.glob("*/scene.json")}
        if (staging / "scene.json").is_file():
            single = load_scene(staging / "scene.json")
            scenes = {single.job_id: single}
        applied_per_scene: dict[str, int] = {}
        for (job_id, frame_index), frame_crops in sorted(by_frame.items()):
            scene = scenes.get(job_id)
            frame = (
                next((f for f in scene.frames if f.index == frame_index), None) if scene else None
            )
            if scene is None or frame is None:
                raise AnnotationError(f"crops refer to {job_id} frame {frame_index}, not in render")
            with Image.open(frame.rgb) as source, Image.open(frame.instance_ids) as ids_image:
                original, ids = source.convert("RGB"), ids_image.copy()
            # Checks read the untouched render; overlapping crops paste into a working copy.
            rgb = original.copy()
            if ids.mode != "L" or ids.size != rgb.size:
                raise AnnotationError("instance-ID pass must be 8-bit and match the RGB size")
            changed = False
            for crop in frame_crops:
                side = crop.box[2] - crop.box[0]
                with Image.open(crop.files["rgb"]) as stored:
                    crop_rgb = stored.convert("RGB")
                expected = original.crop(crop.box).resize((size, size), Image.Resampling.LANCZOS)
                if ImageChops.difference(expected, crop_rgb).getbbox() is not None:
                    raise AnnotationError(f"crop {crop.id} was made from a different render")
                with Image.open(crop.files["mask"]) as m, Image.open(crop.files["inpaint"]) as p:
                    mask, inpaint = _binary(m), _binary(p)
                lookup = [255 if value in (0, crop.instance_id) else 0 for value in range(256)]
                own_or_background = (
                    ids.crop(crop.box).point(lookup).resize((size, size), Image.Resampling.NEAREST)
                )
                band = ImageChops.subtract(ImageChops.darker(inpaint, own_or_background), mask)
                entry: dict[str, object] = {"id": crop.id, "instance_id": crop.instance_id}
                candidates = _find(generated, crop.id)
                if not candidates:
                    report.append({**entry, "status": "missing"})
                    continue
                if len(candidates) > 1:
                    names = [path.relative_to(generated).as_posix() for path in candidates]
                    report.append(
                        {**entry, "status": "rejected", "reason": "ambiguous", "files": names}
                    )
                    continue
                with Image.open(candidates[0]) as raw_generated:
                    if raw_generated.width != raw_generated.height:
                        report.append({**entry, "status": "rejected", "reason": "not square"})
                        continue
                    result = raw_generated.convert("RGB").resize(
                        (size, size), Image.Resampling.LANCZOS
                    )
                change = _mean_difference(result, crop_rgb, mask) or 0.0
                spill = _mean_difference(result, crop_rgb, band)
                metrics: dict[str, object] = {
                    "generated": candidates[0].relative_to(generated).as_posix(),
                    "generated_sha256": sha256_file(candidates[0]),
                    "change": round(change, 4),
                    "spill": None if spill is None else round(spill, 4),
                }
                reason = None
                if change < min_change:
                    reason = "unchanged"
                elif spill is not None and spill > max_spill:
                    reason = "background band changed"
                silhouette_files = _find(silhouettes, crop.id)
                if silhouettes is not None:
                    if len(silhouette_files) != 1:
                        reason = reason or "no unique silhouette"
                    else:
                        with Image.open(silhouette_files[0]) as raw_silhouette:
                            outline = _binary(
                                raw_silhouette.resize((size, size), Image.Resampling.NEAREST)
                            )
                        metrics["iou"] = round(_iou(outline, mask), 4)
                        if _iou(outline, mask) < min_iou:
                            reason = reason or "silhouette differs"
                if reason:
                    report.append({**entry, **metrics, "status": "rejected", "reason": reason})
                    continue
                # Back to source pixels; paste only where this person (or its band) is.
                patch = result.resize((side, side), Image.Resampling.LANCZOS)
                if blend == "mask":
                    own = [255 if value == crop.instance_id else 0 for value in range(256)]
                    alpha = ids.crop(crop.box).point(own)
                else:
                    allowed = ids.crop(crop.box).point(lookup)
                    alpha = ImageChops.darker(
                        inpaint.resize((side, side), Image.Resampling.NEAREST), allowed
                    )
                rgb.paste(patch, crop.box[:2], alpha)
                changed = True
                applied_per_scene[job_id] = applied_per_scene.get(job_id, 0) + 1
                report.append({**entry, **metrics, "status": "applied"})
            if changed:
                if frame.rgb.suffix.lower() != ".png":
                    raise AnnotationError("composites are written as PNG; the render RGB is not")
                rgb.save(frame.rgb)
        for job_id, count in applied_per_scene.items():
            path = scenes[job_id].path
            data = mapping(json.loads(path.read_text(encoding="utf-8")))
            assets = sequence(data.get("assets"))
            assets.append(
                {
                    "id": "pose-guided-repaint",
                    "generator": generator or "unspecified",
                    "blend": blend,
                    "crops_applied": count,
                    "report": "composite.json",
                    "note": "Appearance of people replaced; labels unchanged from the render.",
                }
            )
            data["assets"] = assets
            write_json(path, data)
        crops_digest = hashlib.sha256((crops / "crops.json").read_bytes()).hexdigest()
        write_json(
            staging / "composite.json",
            {
                "schema_version": "1.0",
                "generator": generator or "unspecified",
                "blend": blend,
                "thresholds": {
                    "min_change": min_change,
                    "max_spill": max_spill,
                    "min_iou": min_iou if silhouettes is not None else None,
                },
                "crops_sha256": crops_digest,
                "items": report,
            },
        )
    count = {
        status: sum(item["status"] == status for item in report)
        for status in ("applied", "rejected", "missing")
    }
    return CompositeResult(destination, count["applied"], count["rejected"], count["missing"])
