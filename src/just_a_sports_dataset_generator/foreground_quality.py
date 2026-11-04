"""Conservative silhouette filters, not an anatomy or photorealism certificate."""

from __future__ import annotations

from PIL import Image, ImageChops, ImageFilter

MIN_FOREGROUND_COVERAGE = 0.90
MIN_LOCAL_COVERAGE = 0.55
EDGE_TOLERANCE_PX = 2


def coverage_metrics(reference: Image.Image, foreground: Image.Image) -> dict[str, float]:
    """Measure missing foreground globally and locally, tolerating two edge pixels.

    Local windows stop a missing hand/foot being hidden by a high whole-body IoU.
    Predicted segmentation can be wrong; passing still requires visual review.
    """
    if reference.size != foreground.size:
        raise ValueError("foreground and reference masks must have the same size")
    ref = reference.convert("L").point([255 if x >= 128 else 0 for x in range(256)])
    fg = foreground.convert("L").point([255 if x >= 128 else 0 for x in range(256)])
    box = ref.getbbox()
    total = ref.histogram()[255]
    if box is None or not total:
        return {"foreground_coverage": 0.0, "min_local_coverage": 0.0}
    tolerant = fg.filter(ImageFilter.MaxFilter(2 * EDGE_TOLERANCE_PX + 1))
    covered = ImageChops.darker(ref, tolerant)
    worst = 1.0
    left, top, right, bottom = box
    # Overlapping windows avoid depending on which tile contains a missing limb.
    width, height = max(4, (right - left) // 3), max(4, (bottom - top) // 5)
    for y in range(top, bottom, max(1, height // 2)):
        for x in range(left, right, max(1, width // 2)):
            region = (x, y, min(right, x + width), min(bottom, y + height))
            area = ref.crop(region).histogram()[255]
            if area >= max(8, total * 0.005):
                worst = min(worst, covered.crop(region).histogram()[255] / area)
    return {
        "foreground_coverage": round(covered.histogram()[255] / total, 4),
        "min_local_coverage": round(worst, 4),
    }
