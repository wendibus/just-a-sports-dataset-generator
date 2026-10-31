"""Read the versioned renderer handoff without trusting paths or label metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image

from just_a_sports_dataset_generator.core import labels_for_sport

from .model import AnnotationError, ObjectIdentity, VisibleAnnotation, visible_annotations

MAX_PIXELS = 16_777_216


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AnnotationError("expected a JSON object")
    return cast(dict[str, object], value)


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AnnotationError("expected a JSON array")
    return cast(list[object], value)


def integer(value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AnnotationError(f"expected an integer >= {minimum}")
    return value


def string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AnnotationError("expected a non-empty string")
    return value


def safe_file(parent: Path, name: object) -> Path:
    raw = string(name)
    relative = Path(raw)
    # Check the serialized spelling before Windows converts '/' to '\\'.
    if relative.is_absolute() or relative.drive or ".." in relative.parts or "\\" in raw:
        raise AnnotationError("scene file paths must be relative and stay inside the scene")
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()) or not path.is_file():
        raise AnnotationError("missing scene file or path outside its scene directory")
    return path


@dataclass(frozen=True, slots=True)
class SceneFrame:
    index: int
    rgb: Path
    instance_ids: Path
    objects: tuple[ObjectIdentity, ...]


@dataclass(frozen=True, slots=True)
class Scene:
    path: Path
    job_id: str
    sport: str
    seed: int
    width: int
    height: int
    renderer: str
    renderer_version: str
    assets: tuple[dict[str, object], ...]
    frames: tuple[SceneFrame, ...]


def load_scene(path: Path) -> Scene:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise AnnotationError("scene manifest exceeds 32 MiB")
    data = mapping(json.loads(path.read_text(encoding="utf-8")))
    if data.get("schema_version") not in ("1.0", "2.0"):
        raise AnnotationError("unsupported scene schema version")
    sport = string(data.get("sport"))
    if sequence(data.get("labels")) != list(labels_for_sport(sport)):
        raise AnnotationError("scene label order differs from Reco Trainer")
    job_id = string(data.get("job_id"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise AnnotationError("job ID must contain only letters, digits, underscores, or hyphens")
    resolution = mapping(data.get("resolution"))
    width, height = integer(resolution.get("width"), 1), integer(resolution.get("height"), 1)
    if width * height > MAX_PIXELS:
        raise AnnotationError("scene exceeds the current 16-megapixel export limit")
    frames: list[SceneFrame] = []
    indices: set[int] = set()
    for raw in sequence(data.get("frames")):
        frame = mapping(raw)
        index = integer(frame.get("frame_index"))
        if index in indices:
            raise AnnotationError("duplicate frame index")
        indices.add(index)
        identities: list[ObjectIdentity] = []
        for raw_object in sequence(frame.get("objects")):
            obj = mapping(raw_object)
            identities.append(
                ObjectIdentity(
                    integer(obj.get("instance_id"), 1),
                    integer(obj.get("class_id")),
                    string(obj.get("label")),
                )
            )
        frames.append(
            SceneFrame(
                index,
                safe_file(path.parent, frame.get("rgb")),
                safe_file(path.parent, frame.get("instance_ids")),
                tuple(identities),
            )
        )
    if not frames:
        raise AnnotationError("scene contains no frames")
    return Scene(
        path,
        job_id,
        sport,
        integer(data.get("seed")),
        width,
        height,
        string(data.get("renderer")),
        string(data.get("renderer_version")),
        tuple(mapping(asset) for asset in sequence(data.get("assets", []))),
        tuple(sorted(frames, key=lambda frame: frame.index)),
    )


def load_frame(
    scene: Scene, frame: SceneFrame
) -> tuple[Image.Image, Image.Image, tuple[VisibleAnnotation, ...]]:
    size = (scene.width, scene.height)
    with _open_image(frame.rgb) as source:
        if source.size != size or source.mode != "RGB":
            raise AnnotationError("RGB image mode or dimensions differ from the scene")
        rgb = source.copy()
    with _open_image(frame.instance_ids) as source:
        if source.size != size or source.mode != "L":
            raise AnnotationError("instance-ID image must be 8-bit grayscale of the scene size")
        instances = source.copy()
    annotations = visible_annotations(*size, instances.tobytes(), frame.objects, scene.sport)
    return rgb, instances, annotations


def _open_image(path: Path) -> Image.Image:
    try:
        return Image.open(path)
    except Image.DecompressionBombError as exc:
        raise AnnotationError("image dimensions exceed the safe decoding limit") from exc
