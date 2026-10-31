"""Per-person crops for pose-conditioned image generation (e.g. Qwen-Image inpainting).

Each visible person becomes a square crop with four aligned images:

- ``rgb.png``      the rendered person and surroundings,
- ``mask.png``     the exact instance mask from the ID pass (white = this person),
- ``inpaint.png``  the mask dilated by a few pixels: the only region a generator may change,
- ``pose.png``     an OpenPose BODY_18 skeleton on black, for pose ControlNets.

``crops.json`` records the crop box in source pixels and the scale, so a generated crop can
be resized and pasted back exactly. The labels stay the renderer's ground truth; this module
never predicts or edits annotations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.annotations.poses import OPENPOSE_18, POSE_LABELS
from just_a_sports_dataset_generator.annotations.scene import (
    Scene,
    integer,
    load_frame,
    load_scene,
    mapping,
    sequence,
    string,
)
from just_a_sports_dataset_generator.storage import new_directory, write_json

# OpenPose BODY_18 limbs and colours, as used by the reference renderer of pose ControlNets.
LIMBS = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)
COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)


@dataclass(frozen=True, slots=True)
class CropResult:
    output_directory: Path
    crops: int
    skipped_small: int


def _poses(scene: Scene) -> dict[tuple[int, int], list[list[float]]]:
    path = scene.path.parent / "poses.json"
    if not path.is_file():
        raise AnnotationError(
            f"{scene.job_id} has no poses.json; render it again with render-basketball"
        )
    if path.stat().st_size > 32 * 1024 * 1024:
        raise AnnotationError("poses.json exceeds 32 MiB")
    data = mapping(json.loads(path.read_text(encoding="utf-8")))
    if data.get("schema_version") != "1.0" or data.get("format") != "openpose-18":
        raise AnnotationError("unsupported poses.json version or format")
    if string(data.get("job_id")) != scene.job_id:
        raise AnnotationError("poses.json belongs to a different scene")
    if sequence(data.get("keypoint_names")) != list(OPENPOSE_18):
        raise AnnotationError("poses.json keypoint order differs from OpenPose BODY_18")
    result: dict[tuple[int, int], list[list[float]]] = {}
    for raw_frame in sequence(data.get("frames")):
        frame = mapping(raw_frame)
        index = integer(frame.get("frame_index"))
        for raw_person in sequence(frame.get("people")):
            person = mapping(raw_person)
            points: list[list[float]] = []
            for raw_point in sequence(person.get("keypoints")):
                point = sequence(raw_point)
                if len(point) != 3 or any(type(v) not in (int, float) for v in point):
                    raise AnnotationError("keypoints must be [x, y, v]")
                x, y, v = (float(value) for value in point)  # pyright: ignore[reportArgumentType]
                if not (math.isfinite(x) and math.isfinite(y)) or v not in (0, 1, 2):
                    raise AnnotationError("invalid keypoint value")
                points.append([x, y, v])
            if len(points) != len(OPENPOSE_18):
                raise AnnotationError("each person needs 18 keypoints")
            result[(index, integer(person.get("instance_id"), 1))] = points
    return result


FACE = frozenset({0, 14, 15, 16, 17})


def draw_openpose(points: list[list[float]], size: int) -> Image.Image:
    """Render BODY_18 keypoints (crop pixel coordinates) on black.

    Body joints are drawn when inside the image (v >= 1), so a limb hidden behind another
    player still defines the pose. Face points are drawn only when visible (v == 2), as in
    OpenPose output: a person turned away shows no nose or eyes.
    """
    points = [
        [x, y, 0.0] if index in FACE and v < 2 else [x, y, v]
        for index, (x, y, v) in enumerate(points)
    ]
    canvas = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(canvas)
    stick, joint = max(2, round(size / 85)), max(2, round(size / 128))
    for (a, b), color in zip(LIMBS, COLORS, strict=False):
        if points[a][2] >= 1 and points[b][2] >= 1:
            dim = tuple(round(channel * 0.6) for channel in color)
            draw.line((points[a][0], points[a][1], points[b][0], points[b][1]), dim, stick)
    for (x, y, v), color in zip(points, COLORS, strict=True):
        if v >= 1:
            draw.ellipse((x - joint, y - joint, x + joint, y + joint), color)
    return canvas


def export_crops(
    source: Path,
    destination: Path,
    size: int = 1024,
    margin: float = 0.25,
    min_height: int = 24,
    dilate: int = 9,
) -> CropResult:
    if type(size) is not int or not 64 <= size <= 4096:
        raise ValueError("size must be an integer between 64 and 4096")
    if not 0 <= margin <= 2:
        raise ValueError("margin must be between 0 and 2")
    if type(min_height) is not int or min_height < 1:
        raise ValueError("min_height must be a positive integer")
    if type(dilate) is not int or dilate < 1 or dilate % 2 == 0:
        raise ValueError("dilate must be a positive odd integer (pixels at crop size)")
    source = source.resolve()
    paths = (
        [source / "scene.json"]
        if (source / "scene.json").is_file()
        else sorted(source.glob("*/scene.json"))
    )
    if not paths:
        raise AnnotationError("no scene.json renderer outputs found")
    items: list[dict[str, object]] = []
    skipped = 0
    with new_directory(destination) as staging:
        for path in paths:
            scene = load_scene(path)
            poses = _poses(scene)
            for frame in scene.frames:
                rgb, instances, annotations = load_frame(scene, frame)
                for annotation in annotations:
                    if annotation.label not in POSE_LABELS:
                        continue
                    x, y, width, height = annotation.bbox_xywh
                    if height < min_height:
                        skipped += 1
                        continue
                    points = poses.get((frame.index, annotation.instance_id))
                    if points is None:
                        raise AnnotationError("a visible person has no pose")
                    side = math.ceil(max(width, height) * (1 + 2 * margin))
                    left = round(x + width / 2 - side / 2)
                    top = round(y + height / 2 - side / 2)
                    box = (left, top, left + side, top + side)
                    scale = size / side
                    stem = f"{scene.job_id}-f{frame.index:06d}-i{annotation.instance_id:03d}"
                    folder = staging / scene.job_id
                    folder.mkdir(exist_ok=True)
                    # Out-of-image parts of the box are padded black (and masked out).
                    rgb.crop(box).resize((size, size), Image.Resampling.LANCZOS).save(
                        folder / f"{stem}.rgb.png"
                    )
                    identity = annotation.instance_id
                    table = [255 if value == identity else 0 for value in range(256)]
                    mask = (
                        instances.crop(box)
                        .point(table)
                        .resize((size, size), Image.Resampling.NEAREST)
                    )
                    mask.save(folder / f"{stem}.mask.png")
                    mask.filter(ImageFilter.MaxFilter(dilate)).save(folder / f"{stem}.inpaint.png")
                    local = [
                        [(px - left + 0.5) * scale - 0.5, (py - top + 0.5) * scale - 0.5, v]
                        if v
                        else [0.0, 0.0, 0.0]
                        for px, py, v in points
                    ]
                    draw_openpose(local, size).save(folder / f"{stem}.pose.png")
                    items.append(
                        {
                            "id": stem,
                            "job_id": scene.job_id,
                            "frame_index": frame.index,
                            "instance_id": identity,
                            "label": annotation.label,
                            "source_rgb": str(frame.rgb.relative_to(source)),
                            "bbox_xywh": list(annotation.bbox_xywh),
                            "visible_area": annotation.visible_area,
                            "crop_box_xyxy": list(box),
                            "scale": scale,
                            "files": {
                                kind: f"{scene.job_id}/{stem}.{kind}.png"
                                for kind in ("rgb", "mask", "inpaint", "pose")
                            },
                            "keypoints": [[round(a, 3), round(b, 3), int(v)] for a, b, v in local],
                        }
                    )
        write_json(
            staging / "crops.json",
            {
                "schema_version": "1.0",
                "size": size,
                "margin": margin,
                "min_height": min_height,
                "dilate_px": dilate,
                "pose_format": "openpose-18",
                "keypoint_names": list(OPENPOSE_18),
                "note": "Generated images may only change pixels inside inpaint.png; "
                "labels remain the renderer's ground truth.",
                "crops": items,
            },
        )
    return CropResult(destination, len(items), skipped)
