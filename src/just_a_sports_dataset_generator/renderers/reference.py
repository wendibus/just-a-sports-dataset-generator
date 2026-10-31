from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from just_a_sports_dataset_generator.annotations import ObjectIdentity, visible_annotations
from just_a_sports_dataset_generator.annotations.scene import MAX_PIXELS
from just_a_sports_dataset_generator.core import SceneJob, labels_for_sport
from just_a_sports_dataset_generator.plugins import PLUGIN_API_VERSION, PluginMetadata
from just_a_sports_dataset_generator.storage import new_directory, write_json

from .contracts import RenderResult

_COLORS = ((220, 55, 55), (55, 110, 220), (140, 90, 45), (40, 170, 85), (155, 80, 190))


@dataclass(frozen=True, slots=True)
class _Rectangle:
    identity: ObjectIdentity
    bbox: tuple[int, int, int, int]


class ReferenceRenderer:
    """Deterministic geometric fixture, not a photorealistic render backend."""

    metadata = PluginMetadata(
        plugin_id="org.just-a-sports-dataset-generator.reference-renderer",
        version="0.2.0",
        plugin_api=PLUGIN_API_VERSION,
        kind="renderer",
        capabilities=("rgb", "instance_ids", "bounding_boxes", "instance_masks"),
    )

    @staticmethod
    def _number(seed: int, frame_index: int, object_index: int, axis: str) -> int:
        material = f"reference:v1:{seed}:{frame_index}:{object_index}:{axis}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def _objects(self, job: SceneJob, frame_index: int) -> tuple[_Rectangle, ...]:
        objects: list[_Rectangle] = []
        for index, label in enumerate(job.labels):
            width = min(job.width, max(1, job.width // (9 if label in {"ball", "puck"} else 6)))
            height = min(job.height, max(1, job.height // (9 if label in {"ball", "puck"} else 3)))
            x = self._number(job.seed, frame_index, index, "x") % (job.width - width + 1)
            y = self._number(job.seed, frame_index, index, "y") % (job.height - height + 1)
            objects.append(
                _Rectangle(ObjectIdentity(index + 1, index, label), (x, y, width, height))
            )
        return tuple(objects)

    @staticmethod
    def _draw(job: SceneJob, objects: tuple[_Rectangle, ...]) -> tuple[bytearray, bytearray]:
        rgb = bytearray((24, 86, 48)) * (job.width * job.height)
        instances = bytearray(job.width * job.height)
        for item in objects:
            x, y, width, height = item.bbox
            color = _COLORS[item.identity.class_id % len(_COLORS)]
            for row in range(y, y + height):
                for column in range(x, x + width):
                    pixel = row * job.width + column
                    rgb[pixel * 3 : pixel * 3 + 3] = bytes(color)
                    instances[pixel] = item.identity.instance_id
        return rgb, instances

    @staticmethod
    def _write_image(path: Path, magic: str, width: int, height: int, pixels: bytearray) -> None:
        path.write_bytes(f"{magic}\n{width} {height}\n255\n".encode("ascii") + pixels)

    def render(
        self, job: SceneJob, output_directory: Path, max_frames: int | None = None
    ) -> RenderResult:
        self.metadata.validate()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job.job_id):
            raise ValueError("invalid scene job ID")
        if job.labels != labels_for_sport(job.sport):
            raise ValueError("job label order differs from Reco Trainer")
        if job.camera != "broadcast":
            raise ValueError("the geometric reference renderer has no camera presets")
        for value in (job.width, job.height, job.frame_count):
            if type(value) is not int or value < 1:
                raise ValueError("dimensions and frame count must be positive integers")
        if type(job.seed) is not int or job.seed < 0:
            raise ValueError("scene seed must be a non-negative integer")
        if job.width * job.height > MAX_PIXELS:
            raise ValueError("reference renderer is limited to 16 megapixels")
        if set(job.annotations) - {"bounding_boxes", "instance_masks"}:
            raise ValueError("reference renderer supports only bounding_boxes and instance_masks")
        if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
            raise ValueError("max_frames must be a positive integer")
        job_directory = output_directory / job.job_id
        count = job.frame_count if max_frames is None else min(job.frame_count, max_frames)
        frames: list[dict[str, object]] = []
        with new_directory(job_directory) as staging:
            for frame_index in range(count):
                objects = self._objects(job, frame_index)
                rgb, instances = self._draw(job, objects)
                visible = {
                    item.instance_id: item
                    for item in visible_annotations(
                        job.width,
                        job.height,
                        bytes(instances),
                        tuple(obj.identity for obj in objects),
                        job.sport,
                    )
                }
                stem = f"frame-{frame_index:06d}"
                rgb_name, instance_name = f"{stem}.ppm", f"{stem}.instances.pgm"
                self._write_image(staging / rgb_name, "P6", job.width, job.height, rgb)
                self._write_image(staging / instance_name, "P5", job.width, job.height, instances)
                frames.append(
                    {
                        "frame_index": frame_index,
                        "rgb": rgb_name,
                        "instance_ids": instance_name,
                        "objects": [
                            {
                                "instance_id": obj.identity.instance_id,
                                "class_id": obj.identity.class_id,
                                "label": obj.identity.label,
                                "amodal_bbox_xywh": obj.bbox,
                                "bbox_xywh": visible[obj.identity.instance_id].bbox_xywh
                                if obj.identity.instance_id in visible
                                else None,
                                "visible_area": visible[obj.identity.instance_id].visible_area
                                if obj.identity.instance_id in visible
                                else 0,
                            }
                            for obj in objects
                        ],
                    }
                )
            write_json(
                staging / "scene.json",
                {
                    "schema_version": "2.0",
                    "renderer": self.metadata.plugin_id,
                    "renderer_version": self.metadata.version,
                    "job_id": job.job_id,
                    "sport": job.sport,
                    "seed": job.seed,
                    "labels": job.labels,
                    "resolution": {"width": job.width, "height": job.height},
                    "requested_annotations": job.annotations,
                    "annotation_semantics": "visible pixels; xywh; exclusive right/bottom",
                    "assets": [
                        {
                            "id": "builtin-geometric-fixture",
                            "license": "Apache-2.0",
                            "source": "renderers/reference.py",
                            "version": self.metadata.version,
                        }
                    ],
                    "frames": frames,
                },
            )
        return RenderResult(job.job_id, job_directory, count, job_directory / "scene.json")
