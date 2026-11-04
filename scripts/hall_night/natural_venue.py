"""Validate a local Cycles football venue and derive reusable visible goal layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .common import read, sha, validate, write
from .football_pitch import project


def prepare_layers(render: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(render / "with-goals.png") as raw:
        full = raw.convert("RGB")
    with Image.open(render / "without-goals.png") as raw:
        empty = raw.convert("RGB")
    with Image.open(render / "goal-ids.pgm") as raw:
        ids = np.asarray(raw).copy()
    if full.size != (3200, 900) or empty.size != full.size or ids.shape != (900, 3200):
        raise ValueError("Unexpected venue dimensions")
    if set(map(int, np.unique(ids))) != {0, 200, 201}:
        raise ValueError("Missing or unexpected goal IDs")
    errors = []
    for item in read(render / "camera-check.json"):
        error = max(abs(a - b) for a, b in zip(project(item["world"]), item["pixel"]))
        errors.append(error)
    if not errors or max(errors) > 0.02:
        raise ValueError("Venue camera does not match the placement calibration")
    # Preserve physically rendered goal shadows as background. Remove categorical
    # goal support and restore it through the exact RGB/ID-derived alpha layers.
    background = np.asarray(full).copy()
    support = ids > 0
    background[support] = np.asarray(empty)[support]
    Image.fromarray(background).save(output / "background.png")
    goals = output / "goals-natural"
    goals.mkdir(exist_ok=True)
    assets = []
    for identity, side, end in ((200, "left", 0), (201, "right", 105)):
        alpha = Image.fromarray(np.uint8(ids == identity) * 255)
        bbox = alpha.getbbox()
        if bbox is None or int((ids == identity).sum()) < 50:
            raise ValueError("Empty or implausibly small goal layer")
        rgba = full.convert("RGBA")
        rgba.putalpha(alpha)
        path = goals / f"goal-{side}.png"
        rgba.crop(bbox).save(path)
        assets.append(
            {
                "id": f"goal-{side}",
                "file": path.relative_to(output).as_posix(),
                "sha256": sha(path),
                "position_pixel": list(bbox[:2]),
                "ground_pixel": list(project((end, 34, 0))),
                "opening_m": [7.32, 2.44],
                "depth_m": 1.8,
                "shadow_baked": True,
                "rights": "Original local Blender render; CC0 ambientCG / Poly Haven textures; see venue sources",
            }
        )
    reconstructed = Image.fromarray(background).convert("RGBA")
    for a in assets:
        with Image.open(output / a["file"]) as raw:
            reconstructed.alpha_composite(raw, tuple(a["position_pixel"]))
    if not np.array_equal(np.asarray(reconstructed.convert("RGB")), np.asarray(full)):
        raise ValueError("RGB/goal layer reconstruction differs from original render")
    write(output / "goals.json", {"schema_version": "1.0", "assets": assets})
    write(
        output / "checks.json",
        {
            "schema_version": "1.0",
            "passed": True,
            "camera_max_pixel_error": max(errors),
            "goal_areas": {str(i): int((ids == i).sum()) for i in (200, 201)},
            "exact_rgb_reconstruction": True,
            "semantics": "Categorical Object Index support from the same Cycles render; antialiased edges approximate pixel coverage; physical goal shadows baked into background",
        },
    )
    validate("football-venue-checks-v1.0.json", read(output / "checks.json"))
    return assets


def assert_unstarted(root: Path):
    for name in (
        "run-fingerprint.json",
        "library-fingerprint.json",
        "renders",
        "dataset",
        "pilot",
        "cache/photos",
    ):
        if (root / name).exists():
            raise ValueError("Cannot replace the venue after generation started: " + name)
