from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from just_a_sports_dataset_generator.core import (
    CAMERA_PRESETS,
    SPORTS,
    DatasetRecipe,
    SceneJob,
    labels_for_sport,
)

SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")
"""1.1 adds the optional ``camera`` preset; 1.0 recipes keep the broadcast camera."""
SUPPORTED_ANNOTATIONS = frozenset(
    {"bounding_boxes", "instance_masks", "semantic_masks", "keypoints", "depth"}
)


class RecipeError(ValueError):
    """Raised when a recipe cannot be parsed or validated."""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecipeError(f"{field} must be a positive integer")
    return value


def load_recipe(path: str | Path) -> DatasetRecipe:
    recipe_path = Path(path)
    try:
        raw = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecipeError(f"could not read recipe: {exc}") from exc
    if not isinstance(raw, dict):
        raise RecipeError("recipe root must be an object")
    raw = cast(dict[str, Any], raw)

    required = {
        "schema_version",
        "name",
        "sport",
        "seed",
        "scenes",
        "frames_per_scene",
        "resolution",
        "annotations",
    }
    if raw.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise RecipeError(f"unsupported schema_version: {raw.get('schema_version')!r}")
    schema_version = cast(str, raw["schema_version"])
    optional = {"camera"} if schema_version == "1.1" else set[str]()
    missing = sorted(required - raw.keys())
    unknown = sorted(raw.keys() - required - optional)
    if missing:
        raise RecipeError(f"missing fields: {', '.join(missing)}")
    if unknown:
        raise RecipeError(f"unknown fields: {', '.join(unknown)}")
    camera = raw.get("camera", "broadcast")
    if camera not in CAMERA_PRESETS:
        raise RecipeError(f"camera must be one of: {', '.join(CAMERA_PRESETS)}")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise RecipeError("name must be a non-empty string")
    if not isinstance(raw["sport"], str) or not raw["sport"].strip():
        raise RecipeError("sport must be a non-empty string")
    sport = raw["sport"].strip().lower()
    if sport not in SPORTS:
        raise RecipeError(f"sport must be a Reco Trainer sport: {', '.join(SPORTS)}")
    if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int) or raw["seed"] < 0:
        raise RecipeError("seed must be a non-negative integer")
    resolution = raw["resolution"]
    if not isinstance(resolution, dict):
        raise RecipeError("resolution must contain exactly width and height")
    resolution = cast(dict[str, Any], resolution)
    if set(resolution) != {"width", "height"}:
        raise RecipeError("resolution must contain exactly width and height")
    annotations = raw["annotations"]
    if not isinstance(annotations, list) or not annotations:
        raise RecipeError("annotations must be a non-empty list")
    if any(not isinstance(item, str) for item in cast(list[object], annotations)):
        raise RecipeError("each annotation must be a string")
    annotations = cast(list[str], annotations)
    unsupported = sorted(set(annotations) - SUPPORTED_ANNOTATIONS)
    if unsupported:
        raise RecipeError(f"unsupported annotations: {', '.join(unsupported)}")
    if len(set(annotations)) != len(annotations):
        raise RecipeError("annotations must not contain duplicates")

    return DatasetRecipe(
        schema_version=schema_version,
        name=raw["name"].strip(),
        sport=sport,
        seed=raw["seed"],
        scenes=_positive_int(raw["scenes"], "scenes"),
        frames_per_scene=_positive_int(raw["frames_per_scene"], "frames_per_scene"),
        width=_positive_int(resolution["width"], "resolution.width"),
        height=_positive_int(resolution["height"], "resolution.height"),
        annotations=tuple(annotations),
        labels=labels_for_sport(sport),
        camera=cast(str, camera),
    )


def _derive_seed(master_seed: int, scene_index: int) -> int:
    material = f"sports-dataset:v1:{master_seed}:{scene_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def plan_jobs(recipe: DatasetRecipe) -> tuple[SceneJob, ...]:
    return tuple(
        SceneJob(
            job_id=f"scene-{index:06d}",
            scene_index=index,
            frame_count=recipe.frames_per_scene,
            seed=_derive_seed(recipe.seed, index),
            sport=recipe.sport,
            width=recipe.width,
            height=recipe.height,
            annotations=recipe.annotations,
            labels=recipe.labels,
            camera=recipe.camera,
        )
        for index in range(recipe.scenes)
    )
