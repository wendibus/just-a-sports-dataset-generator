"""2D body keypoints from scene geometry, with visibility checked against the ID pass."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .model import AnnotationError, ObjectIdentity
from .scene import integer, mapping, sequence

# OpenPose BODY_18 order, the layout pose ControlNets are trained on.
OPENPOSE_18 = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
POSE_LABELS = frozenset({"player", "goalkeeper", "referee"})
NOT_IN_IMAGE, OCCLUDED, VISIBLE = 0, 1, 2
"""COCO visibility flags: outside the image, inside but hidden, inside and visible."""


def _point(value: object) -> tuple[float, float, bool] | None:
    """Parse ``null``, ``[x, y]`` or ``[x, y, faces_camera]`` from the worker."""
    if value is None:
        return None
    raw = sequence(value)
    if len(raw) not in (2, 3) or any(type(item) not in (int, float) for item in raw[:2]):
        raise AnnotationError("a keypoint must be null, [x, y] or [x, y, faces_camera]")
    if len(raw) == 3 and type(raw[2]) is not bool:
        raise AnnotationError("faces_camera must be a boolean")
    x, y = float(raw[0]), float(raw[1])  # pyright: ignore[reportArgumentType]
    if not (math.isfinite(x) and math.isfinite(y)):
        raise AnnotationError("keypoint coordinates must be finite")
    return x, y, raw[2] is not False if len(raw) == 3 else True


def _visibility(
    point: tuple[float, float, bool] | None, identity: int, width: int, height: int, ids: bytes
) -> int:
    if point is None:
        return NOT_IN_IMAGE
    column, row = round(point[0]), round(point[1])
    if not (0 <= column < width and 0 <= row < height):
        return NOT_IN_IMAGE
    if not point[2]:
        return OCCLUDED  # turned away: hidden by the person's own head
    # Joints sit on the body surface; a 3x3 neighbourhood absorbs rounding at thin limbs.
    for y in range(max(0, row - 1), min(height, row + 2)):
        for x in range(max(0, column - 1), min(width, column + 2)):
            if ids[y * width + x] == identity:
                return VISIBLE
    return OCCLUDED


def frame_poses(
    raw_poses: object,
    objects: Sequence[ObjectIdentity],
    width: int,
    height: int,
    ids: bytes,
) -> list[dict[str, object]]:
    """Validate worker keypoints and attach COCO visibility from the final ID pass."""
    if len(ids) != width * height:
        raise AnnotationError("instance-ID pass does not match the frame size")
    people: Mapping[int, ObjectIdentity] = {
        item.instance_id: item for item in objects if item.label in POSE_LABELS
    }
    result: list[dict[str, object]] = []
    seen: set[int] = set()
    for raw in sequence(raw_poses):
        entry = mapping(raw)
        identity = integer(entry.get("instance_id"), 1)
        if identity not in people or identity in seen:
            raise AnnotationError("pose refers to an unknown, non-person or repeated instance")
        seen.add(identity)
        points = [_point(value) for value in sequence(entry.get("keypoints"))]
        if len(points) != len(OPENPOSE_18):
            raise AnnotationError(f"expected {len(OPENPOSE_18)} keypoints per person")
        keypoints: list[list[float]] = []
        for point in points:
            flag = _visibility(point, identity, width, height, ids)
            x, y = (point[0], point[1]) if point and flag else (0.0, 0.0)
            keypoints.append([x, y, float(flag)])
        result.append(
            {
                "instance_id": identity,
                "label": people[identity].label,
                "keypoints": [[round(x, 3), round(y, 3), int(v)] for x, y, v in keypoints],
            }
        )
    return result
