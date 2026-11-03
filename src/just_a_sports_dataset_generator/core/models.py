from dataclasses import dataclass

CAMERA_PRESETS = ("broadcast", "reco_panorama")
"""Camera presets. ``reco_panorama`` approximates a stitched Reco full-court panorama."""
EQUIPMENT_PROFILES = ("international", "germany")
"""Weightings of the ball and referee-kit catalogue (docs/EQUIPMENT.md)."""


@dataclass(frozen=True, slots=True)
class DatasetRecipe:
    schema_version: str
    name: str
    sport: str
    seed: int
    scenes: int
    frames_per_scene: int
    width: int
    height: int
    annotations: tuple[str, ...]
    labels: tuple[str, ...]
    camera: str = "broadcast"
    equipment_profile: str = "international"


@dataclass(frozen=True, slots=True)
class SceneJob:
    job_id: str
    scene_index: int
    frame_count: int
    seed: int
    sport: str
    width: int
    height: int
    annotations: tuple[str, ...]
    labels: tuple[str, ...]
    camera: str = "broadcast"
    equipment_profile: str = "international"
