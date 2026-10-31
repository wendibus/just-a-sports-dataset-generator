"""Local Blender adapter. The isolated GPL worker exchanges JSON and image files."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from just_a_sports_dataset_generator.annotations import ObjectIdentity, visible_annotations
from just_a_sports_dataset_generator.annotations.scene import (
    MAX_PIXELS,
    integer,
    mapping,
    safe_file,
    sequence,
    string,
)
from just_a_sports_dataset_generator.core import SceneJob, labels_for_sport
from just_a_sports_dataset_generator.plugins import PLUGIN_API_VERSION, PluginMetadata
from just_a_sports_dataset_generator.storage import new_directory, sha256_file, write_json

from .contracts import RenderResult

WORKER = Path(__file__).with_name("blender_worker") / "basketball.py"


def find_blender(explicit: Path | None = None) -> Path:
    """Use an explicit executable, the user's installation, or PATH; never download."""
    candidates = (
        [explicit]
        if explicit is not None
        else [
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path(found) if (found := shutil.which("blender")) else None,
        ]
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise ValueError("Blender 5.x was not found. Install it locally or specify --blender PATH.")


class BlenderBasketballRenderer:
    """Procedural 3D basketball; realism/training value is not yet validated."""

    metadata = PluginMetadata(
        "org.just-a-sports-dataset-generator.blender-basketball",
        "0.4.0",
        PLUGIN_API_VERSION,
        "renderer",
        ("rgb", "instance_ids", "bounding_boxes", "instance_masks"),
    )

    def __init__(
        self, executable: Path | None = None, samples: int = 32, device: str = "cpu"
    ) -> None:
        if type(samples) is not int or not 1 <= samples <= 512:
            raise ValueError("samples must be an integer between 1 and 512")
        if device not in ("auto", "cpu"):
            raise ValueError("device must be auto or cpu")
        self.executable, self.samples, self.device = find_blender(executable), samples, device
        self.assets: list[dict[str, object]] = [
            {
                "id": "procedural-basketball-hall-and-athletes",
                "source": "renderers/blender_worker/basketball.py",
                "license": "GPL-3.0-or-later",
                "version": self.metadata.version,
                "sha256": sha256_file(WORKER),
                "external_assets": False,
            }
        ]

    def render(
        self, job: SceneJob, output_directory: Path, max_frames: int | None = None
    ) -> RenderResult:
        self.metadata.validate()
        if job.sport != "basketball" or job.labels != labels_for_sport("basketball"):
            raise ValueError("the Blender sport backend currently supports basketball only")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job.job_id):
            raise ValueError("invalid scene job ID")
        for value in (job.width, job.height, job.frame_count):
            if type(value) is not int or value < 1:
                raise ValueError("dimensions and frame count must be positive integers")
        if type(job.seed) is not int or job.seed < 0:
            raise ValueError("scene seed must be a non-negative integer")
        if job.width * job.height > MAX_PIXELS:
            raise ValueError("renderer is limited to 16 megapixels")
        if set(job.annotations) - {"bounding_boxes", "instance_masks"}:
            raise ValueError("basketball backend supports bounding_boxes and instance_masks")
        if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
            raise ValueError("max_frames must be positive")
        count = min(job.frame_count, max_frames) if max_frames is not None else job.frame_count
        destination = output_directory / job.job_id
        with new_directory(destination) as staging:
            request = staging / "request.json"
            write_json(
                request,
                {
                    "job": asdict(job),
                    "frames": count,
                    "samples": self.samples,
                    "device": self.device,
                },
            )
            command = [
                str(self.executable),
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "3",
                "--python",
                str(WORKER),
                "--",
                str(request.resolve()),
            ]
            env = os.environ.copy()
            # A frozen host's Python/library environment must not leak into Blender.
            for name in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
                env.pop(name, None)
                if name + "_ORIG" in env:
                    env[name] = env[name + "_ORIG"]
            started = time.monotonic()
            with (staging / "blender.log").open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        timeout=max(600, count * 300),
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ValueError("Blender exceeded the render time limit") from exc
            if result.returncode:
                tail = (staging / "blender.log").read_text(errors="replace")[-2500:]
                raise ValueError(
                    f"Blender could not render this scene (exit {result.returncode}):\n{tail}"
                )
            payload = mapping(json.loads(safe_file(staging, "worker-result.json").read_text()))
            frames = sequence(payload.get("frames"))
            if len(frames) != count:
                raise ValueError("Blender returned an unexpected number of frames")
            validated: list[dict[str, object]] = []
            for expected_index, raw in enumerate(frames):
                frame = mapping(raw)
                if integer(frame.get("frame_index")) != expected_index:
                    raise ValueError("Blender returned unordered frame indices")
                objects: list[ObjectIdentity] = []
                for raw_obj in sequence(frame.get("objects")):
                    obj = mapping(raw_obj)
                    objects.append(
                        ObjectIdentity(
                            integer(obj.get("instance_id"), 1),
                            integer(obj.get("class_id")),
                            string(obj.get("label")),
                        )
                    )
                with (
                    Image.open(safe_file(staging, frame.get("rgb"))) as rgb,
                    Image.open(safe_file(staging, frame.get("instance_ids"))) as mask,
                ):
                    if (
                        rgb.mode != "RGB"
                        or mask.mode != "L"
                        or rgb.size != (job.width, job.height)
                        or mask.size != rgb.size
                    ):
                        raise ValueError("Blender passes have invalid dimensions or color modes")
                    visible = {
                        item.instance_id: item
                        for item in visible_annotations(
                            job.width, job.height, mask.tobytes(), tuple(objects), job.sport
                        )
                    }
                validated.append(
                    {
                        "frame_index": expected_index,
                        "rgb": frame["rgb"],
                        "instance_ids": frame["instance_ids"],
                        "objects": [
                            {
                                **asdict(obj),
                                "bbox_xywh": visible[obj.instance_id].bbox_xywh
                                if obj.instance_id in visible
                                else None,
                                "visible_area": visible[obj.instance_id].visible_area
                                if obj.instance_id in visible
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
                    "annotation_semantics": "visible pixels from the same Cycles Object Index pass; xywh; exclusive right/bottom; no mask antialiasing",
                    "assets": self.assets
                    + [
                        {
                            "id": "blender-cycles",
                            "version": payload.get("blender_version"),
                            "license": "GPL-3.0-or-later",
                            "source": "https://www.blender.org/",
                            "samples": self.samples,
                            "device": payload.get("device"),
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        }
                    ],
                    "frames": validated,
                },
            )
        return RenderResult(job.job_id, destination, count, destination / "scene.json")
