from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from just_a_sports_dataset_generator.core import SceneJob
from just_a_sports_dataset_generator.plugins import PluginMetadata


@dataclass(frozen=True, slots=True)
class RenderResult:
    job_id: str
    output_directory: Path
    frames: int
    manifest: Path


class Renderer(Protocol):
    metadata: PluginMetadata

    def render(
        self, job: SceneJob, output_directory: Path, max_frames: int | None = None
    ) -> RenderResult:
        """Render a job after the caller has validated plugin metadata."""
        ...
