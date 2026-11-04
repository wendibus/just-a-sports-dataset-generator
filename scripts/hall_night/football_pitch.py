"""Original CPU-only football pitch fixture and its explicit pinhole camera.

This procedural outdoor backplate is a neutral prototype, not a real venue photo.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import read, sha, validate, write

SIZE = (3200, 900)
CAMERA = np.array([52.5, -70.0, 22.0])
TARGET = np.array([52.5, 34.0, 0.0])
FORWARD = (TARGET - CAMERA) / np.linalg.norm(TARGET - CAMERA)
RIGHT = np.cross(FORWARD, np.array([0.0, 0.0, 1.0]))
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(RIGHT, FORWARD)
FOCAL = 2000.0
CENTER = (1600.0, 520.0)


def project(point):
    relative = np.asarray(point, dtype=float) - CAMERA
    depth = float(relative @ FORWARD)
    if depth <= 0:
        raise ValueError("Point is behind the football camera")
    return (
        CENTER[0] + FOCAL * float(relative @ RIGHT) / depth,
        CENTER[1] - FOCAL * float(relative @ UP) / depth,
    )


def ground_point(u, v):
    # Increasing compositor depth v means closer to the sideline camera.
    return tuple(round(c) for c in project((105 * u, 68 * (1 - v), 0)))


def unproject_ground(pixel):
    ray = FORWARD + (pixel[0] - CENTER[0]) / FOCAL * RIGHT - (pixel[1] - CENTER[1]) / FOCAL * UP
    if ray[2] >= 0:
        raise ValueError("Pixel does not intersect the ground")
    return CAMERA + (-CAMERA[2] / ray[2]) * ray


def pixels_per_metre(pixel):
    ground = unproject_ground(pixel)
    top = project(ground + np.array([0.0, 0.0, 1.0]))
    return abs(top[1] - pixel[1])


def build_pitch(root: Path, seed: int):
    """Write an empty background and two goals with matching alpha, without ML."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy = np.arange(SIZE[1])[:, None]
    noise = rng.normal(0, 2.8, (SIZE[1], SIZE[0]))
    base = np.empty((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    for channel, (sky, grass) in enumerate(((171, 55), (192, 100), (202, 44))):
        base[:, :, channel] = np.clip(np.where(yy < 285, sky + yy * 0.04, grass + noise), 0, 255)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    # Quiet, target-free park surroundings; no crowds or existing goal objects.
    for x in range(0, 3200, 38):
        height = int(rng.integers(25, 75))
        draw.ellipse((x - 22, 285 - height, x + 50, 320), fill=(43, int(rng.integers(66, 85)), 43))
    for stripe in range(14):
        y0, y1 = 68 * stripe / 14, 68 * (stripe + 1) / 14
        poly = [project(p) for p in ((0, y0, 0), (105, y0, 0), (105, y1, 0), (0, y1, 0))]
        draw.polygon(poly, fill=(58, 111, 49) if stripe % 2 else (66, 120, 54))
    # Seeded fine texture, still a deliberately procedural fixture.
    textured = np.asarray(image).astype(np.int16)
    field = Image.new("L", SIZE)
    ImageDraw.Draw(field).polygon(
        [project(p) for p in ((0, 0, 0), (105, 0, 0), (105, 68, 0), (0, 68, 0))], fill=255
    )
    pixels = np.asarray(field) > 0
    for channel in range(3):
        textured[:, :, channel] += np.where(pixels, noise, 0).astype(np.int16)
    image = Image.fromarray(np.clip(textured, 0, 255).astype("uint8"))
    draw = ImageDraw.Draw(image)

    def line(points, closed=False):
        points = [project((x, y, 0)) for x, y in points]
        if closed:
            points.append(points[0])
        draw.line(points, fill=(218, 225, 204), width=2, joint="curve")

    line([(0, 0), (105, 0), (105, 68), (0, 68)], True)
    line([(52.5, 0), (52.5, 68)])
    line(
        [
            (52.5 + 9.15 * math.cos(t), 34 + 9.15 * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, 180)
        ]
    )
    for end, direction in ((0, 1), (105, -1)):
        for length, width in ((16.5, 40.32), (5.5, 18.32)):
            line(
                [
                    (end, 34 - width / 2),
                    (end + direction * length, 34 - width / 2),
                    (end + direction * length, 34 + width / 2),
                    (end, 34 + width / 2),
                ]
            )
        x, y = project((end + direction * 11, 34, 0))
        draw.ellipse((x - 2, y - 1, x + 2, y + 1), fill=(220, 228, 205))
    image.save(root / "background.png")
    write(
        root / "camera.json",
        {
            "schema_version": "1.0",
            "field_m": [105, 68],
            "camera_position_m": CAMERA.tolist(),
            "look_at_m": TARGET.tolist(),
            "focal_px": FOCAL,
            "principal_point_px": list(CENTER),
            "resolution": list(SIZE),
            "source": "Original CPU procedural pitch, not photographic venue",
            "seed": seed,
        },
    )
    validate("football-camera-v1.0.json", read(root / "camera.json"))
    folder = root / "cache/goals"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for side, end, direction in (("left", 0, -1), ("right", 105, 1)):
        scale = 3
        rgba = Image.new("RGBA", (SIZE[0] * scale, SIZE[1] * scale))
        d = ImageDraw.Draw(rgba)

        def rod(a, b, width, color=(222, 226, 221, 255), d=d, scale=scale):
            p, q = project(a), project(b)
            d.line(
                [(p[0] * scale, p[1] * scale), (q[0] * scale, q[1] * scale)],
                fill=color,
                width=max(1, round(width * scale)),
                joint="curve",
            )

        back = end + direction * 1.8
        # 7.32 x 2.44 m clear aperture; all-white posts, original projected mesh.
        for y in (34 - 3.71, 34 + 3.71):
            rod((end, y, 0), (end, y, 2.54), 2.6)
            rod((end, y, 0), (back, y, 0), 1.0)
            rod((end, y, 2.49), (back, y, 2.49), 1.0)
        rod((end, 30.29, 2.49), (end, 37.71, 2.49), 2.6)
        for y in np.arange(30.3, 37.72, 0.18):
            rod((back, y, 0), (back, y, 2.49), 0.65, (181, 192, 180, 255))
            rod((end, y, 2.49), (back, y, 2.49), 0.65)
        for z in np.arange(0, 2.5, 0.18):
            rod((back, 30.29, z), (back, 37.71, z), 0.65, (181, 192, 180, 255))
            for y in (30.29, 37.71):
                rod((end, y, z), (back, y, z), 0.65)
        rgba = rgba.resize(SIZE, Image.Resampling.LANCZOS)
        bbox = rgba.getchannel("A").getbbox()
        if bbox is None:
            raise ValueError("Empty projected football goal")
        rgba = rgba.crop(bbox)
        path = folder / f"goal-{side}.png"
        rgba.save(path)
        records.append(
            {
                "id": f"goal-{side}",
                "file": path.relative_to(root / "cache").as_posix(),
                "sha256": sha(path),
                "position_pixel": list(bbox[:2]),
                "ground_pixel": list(project((end, 34, 0))),
                "opening_m": [7.32, 2.44],
                "depth_m": 1.8,
                "rights": "Original project procedural geometry; no external assets",
            }
        )
    write(root / "cache/goals.json", {"schema_version": "1.0", "assets": records})
    preview = image.convert("RGBA")
    for record in records:
        with Image.open(root / "cache" / record["file"]) as raw:
            preview.alpha_composite(raw, tuple(record["position_pixel"]))
    preview.convert("RGB").save(root / "Rasenplatz-Vorlage.jpg", quality=94)
    return records
