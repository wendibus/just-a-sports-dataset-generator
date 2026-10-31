from dataclasses import dataclass


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

