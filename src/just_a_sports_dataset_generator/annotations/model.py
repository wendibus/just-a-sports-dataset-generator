from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from PIL import Image

from just_a_sports_dataset_generator.core import labels_for_sport


class AnnotationError(ValueError):
    """Invalid scene or annotation data; exporting must stop."""


@dataclass(frozen=True, slots=True)
class ObjectIdentity:
    instance_id: int
    class_id: int
    label: str


@dataclass(frozen=True, slots=True)
class VisibleAnnotation:
    instance_id: int
    class_id: int
    label: str
    bbox_xywh: tuple[int, int, int, int]
    visible_area: int
    # COCO uncompressed RLE: column-major, alternating background/foreground.
    rle_counts: tuple[int, ...]


def visible_annotations(
    width: int,
    height: int,
    instance_ids: bytes,
    objects: tuple[ObjectIdentity, ...],
    sport: str,
) -> tuple[VisibleAnnotation, ...]:
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise AnnotationError("image dimensions must be positive integers")
    if len(instance_ids) != width * height:
        raise AnnotationError("instance-ID buffer size differs from image dimensions")
    labels = labels_for_sport(sport)
    known: set[int] = set()
    for item in objects:
        if type(item.instance_id) is not int or not 1 <= item.instance_id <= 255:
            raise AnnotationError("instance IDs must be integers in 1..255 (0 is background)")
        if item.instance_id in known:
            raise AnnotationError("duplicate instance ID")
        known.add(item.instance_id)
        if (
            type(item.class_id) is not int
            or not 0 <= item.class_id < len(labels)
            or item.label != labels[item.class_id]
        ):
            raise AnnotationError("object class does not match the Reco Trainer taxonomy")
    image = Image.frombytes("L", (width, height), instance_ids)
    histogram = image.histogram()
    if any(count and index not in known for index, count in enumerate(histogram) if index):
        raise AnnotationError("instance-ID pass contains an object absent from scene metadata")
    result: list[VisibleAnnotation] = []
    for item in sorted(objects, key=lambda obj: obj.instance_id):
        if histogram[item.instance_id] == 0:
            continue  # Fully occluded or outside the image: no visible annotation.
        mask = image.point([int(index == item.instance_id) for index in range(256)])
        bbox = mask.getbbox()
        assert bbox is not None
        left, top, right, bottom = bbox
        column_major = mask.transpose(Image.Transpose.TRANSPOSE).tobytes()
        counts = [sum(1 for _ in group) for _, group in groupby(column_major)]
        if column_major[0]:
            counts.insert(0, 0)
        result.append(
            VisibleAnnotation(
                item.instance_id,
                item.class_id,
                item.label,
                (left, top, right - left, bottom - top),
                histogram[item.instance_id],
                tuple(counts),
            )
        )
    return tuple(result)
