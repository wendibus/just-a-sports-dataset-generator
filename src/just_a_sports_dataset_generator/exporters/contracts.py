from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from just_a_sports_dataset_generator.plugins import PluginMetadata


@dataclass(frozen=True, slots=True)
class ExportResult:
    output_directory: Path
    images: int
    annotations: int
    warnings: tuple[str, ...]


class DatasetExporter(Protocol):
    metadata: PluginMetadata

    def export(
        self, source: Path, destination: Path, formats: tuple[str, ...] = ("coco", "yolo")
    ) -> ExportResult:
        """Validate rendered scene handoffs and publish a new dataset directory."""
        ...
