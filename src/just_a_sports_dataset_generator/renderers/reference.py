from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from just_a_sports_dataset_generator.core import SceneJob
from just_a_sports_dataset_generator.plugins import PLUGIN_API_VERSION, PluginMetadata

from .contracts import RenderResult

_COLORS = (
    (220, 55, 55),
    (55, 110, 220),
    (140, 90, 45),
    (40, 170, 85),
    (155, 80, 190),
)


class ReferenceRenderer:
    """Dependency-free deterministic renderer for contract and pipeline tests.

    It intentionally produces simple geometric scenes, not photorealistic data.
    RGB is written as PPM and instance IDs as 8-bit PGM so no image library is required.
    """

    metadata = PluginMetadata(
        plugin_id="org.just-a-sports-dataset-generator.reference-renderer",
        version="0.1.0",
        plugin_api=PLUGIN_API_VERSION,
        kind="renderer",
        capabilities=("rgb", "instance_ids", "bounding_boxes"),
    )

    @staticmethod
    def _number(seed: int, frame_index: int, object_index: int, axis: str) -> int:
        material = f"reference:v1:{seed}:{frame_index}:{object_index}:{axis}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def _objects(self, job: SceneJob, frame_index: int) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        for index, label in enumerate(job.labels):
            width = max(2, job.width // (9 if label in {"ball", "puck"} else 6))
            height = max(2, job.height // (9 if label in {"ball", "puck"} else 3))
            x_range = max(1, job.width - width)
            y_range = max(1, job.height - height)
            x = self._number(job.seed, frame_index, index, "x") % x_range
            y = self._number(job.seed, frame_index, index, "y") % y_range
            objects.append(
                {
                    "instance_id": index + 1,
                    "class_id": index,
                    "label": label,
                    "bbox_xywh": [x, y, width, height],
                }
            )
        return objects

    @staticmethod
    def _draw(job: SceneJob, objects: list[dict[str, Any]]) -> tuple[bytearray, bytearray]:
        rgb = bytearray((24, 86, 48)) * (job.width * job.height)
        instances = bytearray(job.width * job.height)
        for item in objects:
            x, y, width, height = item["bbox_xywh"]
            instance_id = item["instance_id"]
            color = _COLORS[item["class_id"] % len(_COLORS)]
            for row in range(y, min(y + height, job.height)):
                for column in range(x, min(x + width, job.width)):
                    pixel = row * job.width + column
                    rgb[pixel * 3 : pixel * 3 + 3] = bytes(color)
                    instances[pixel] = instance_id
        return rgb, instances

    @staticmethod
    def _write_image(path: Path, magic: str, width: int, height: int, pixels: bytearray) -> None:
        path.write_bytes(f"{magic}\n{width} {height}\n255\n".encode("ascii") + pixels)

    def render(self, job: SceneJob, output_directory: Path, max_frames: int | None = None) -> RenderResult:
        self.metadata.validate()
        job_directory = output_directory / job.job_id
        job_directory.mkdir(parents=True, exist_ok=True)
        count = job.frame_count if max_frames is None else min(job.frame_count, max_frames)
        frames: list[dict[str, Any]] = []
        for frame_index in range(count):
            objects = self._objects(job, frame_index)
            rgb, instances = self._draw(job, objects)
            stem = f"frame-{frame_index:06d}"
            rgb_name = f"{stem}.ppm"
            instance_name = f"{stem}.instances.pgm"
            self._write_image(job_directory / rgb_name, "P6", job.width, job.height, rgb)
            self._write_image(job_directory / instance_name, "P5", job.width, job.height, instances)
            frames.append(
                {
                    "frame_index": frame_index,
                    "rgb": rgb_name,
                    "instance_ids": instance_name,
                    "objects": objects,
                }
            )
        manifest = job_directory / "scene.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "renderer": self.metadata.plugin_id,
                    "renderer_version": self.metadata.version,
                    "job_id": job.job_id,
                    "sport": job.sport,
                    "seed": job.seed,
                    "labels": job.labels,
                    "resolution": {"width": job.width, "height": job.height},
                    "frames": frames,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return RenderResult(job.job_id, job_directory, count, manifest)

