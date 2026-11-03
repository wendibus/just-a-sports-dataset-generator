"""Give clean renders the look of a real sports camera without touching the ground truth.

Renders are too clean compared with footage from a Reco or a hall camera: no lost
resolution, no motion blur, no sensor noise, no compression, perfect white balance. This
module degrades only the RGB images of a copy of a render. Geometry is unchanged, so
every box, mask and keypoint stays the renderer's ground truth; ``export`` and
``preview`` accept the result unchanged.

Each frame draws its parameters from its own random stream (seed, scene, frame), so the
same command reproduces the same images. Pillow only; no extra dependencies.
"""

from __future__ import annotations

import io
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.annotations.scene import load_scene, mapping, sequence
from just_a_sports_dataset_generator.storage import new_directory, write_json

PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    # Reco: a stitched wide-angle panorama, strongly compressed and streamed.
    "reco": {
        "resolution": (0.55, 0.85),  # share of pixels per side that survives
        "motion_px": (0.0, 4.0),  # horizontal blur from panning or fast play
        "noise": (2.0, 6.0),  # sensor noise, in 8-bit steps
        "chroma_noise": (0.5, 2.0),
        "white_balance": (-0.08, 0.08),  # warm/cool cast
        "tint": (-0.04, 0.04),  # green/magenta cast
        "contrast": (0.9, 1.05),
        "saturation": (0.8, 1.05),
        "brightness": (0.88, 1.08),
        "vignette": (0.05, 0.25),
        "sharpen": (20.0, 60.0),  # in-camera sharpening (unsharp mask percent)
        "jpeg_quality": (40.0, 75.0),
    },
    # A lighter look, for broadcast-style material.
    "light": {
        "resolution": (0.7, 0.95),
        "motion_px": (0.0, 3.0),
        "noise": (1.0, 3.0),
        "chroma_noise": (0.0, 1.0),
        "white_balance": (-0.05, 0.05),
        "tint": (-0.02, 0.02),
        "contrast": (0.92, 1.05),
        "saturation": (0.9, 1.05),
        "brightness": (0.94, 1.06),
        "vignette": (0.0, 0.12),
        "sharpen": (10.0, 40.0),
        "jpeg_quality": (55.0, 85.0),
    },
}


@dataclass(frozen=True, slots=True)
class CameraLookResult:
    output_directory: Path
    frames: int


def sample(preset: str, seed: int, job_id: str, frame_index: int) -> dict[str, float]:
    """Parameters for one frame; deterministic for the same preset, seed, scene and frame."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of: {', '.join(PRESETS)}")
    rng = random.Random(f"camera-look:v1:{preset}:{seed}:{job_id}:{frame_index}")
    params = {
        name: round(rng.uniform(low, high), 4) for name, (low, high) in PRESETS[preset].items()
    }
    params["jpeg_quality"] = float(round(params["jpeg_quality"]))
    params["noise_seed"] = float(rng.randrange(2**31))
    return params


def _noise(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Deterministic, roughly Gaussian noise centred on 128 (sum of two uniform fields)."""
    count = size[0] * size[1]
    a = Image.frombytes("L", size, rng.randbytes(count))
    b = Image.frombytes("L", size, rng.randbytes(count))
    return ImageChops.add(a, b, scale=2.0)


def _scaled_noise(size: tuple[int, int], rng: random.Random, sigma: float) -> Image.Image:
    # The triangular field has a standard deviation of about 52; scale it to sigma.
    factor = sigma / 52.0
    table = [max(0, min(255, round(128 + (v - 128) * factor))) for v in range(256)]
    return _noise(size, rng).point(table)


def degrade(image: Image.Image, params: dict[str, float]) -> Image.Image:
    """Apply one frame's camera look to an RGB image of unchanged size."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    # 1. Lost resolution: stitching, scaling and the stream drop fine detail.
    scale = params["resolution"]
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = rgb.resize(small, Image.Resampling.BICUBIC).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    # 2. Horizontal motion blur: average of shifted copies.
    steps = int(params["motion_px"])
    if steps >= 1:
        blurred = rgb
        for shift in range(1, steps + 1):
            moved = ImageChops.offset(rgb, shift - (steps + 1) // 2, 0)
            blurred = Image.blend(blurred, moved, 1 / (shift + 1))
        rgb = blurred
    # 3. White balance, tint, contrast, saturation, brightness.
    warm, tint = params["white_balance"], params["tint"]
    gains = (1 + warm, 1 + tint, 1 - warm)
    rgb = Image.merge(
        "RGB",
        [
            channel.point([min(255, round(v * gain)) for v in range(256)])
            for channel, gain in zip(rgb.split(), gains, strict=True)
        ],
    )
    rgb = ImageEnhance.Contrast(rgb).enhance(params["contrast"])
    rgb = ImageEnhance.Color(rgb).enhance(params["saturation"])
    rgb = ImageEnhance.Brightness(rgb).enhance(params["brightness"])
    # 4. Vignetting of a wide-angle lens.
    if params["vignette"] > 0:
        gradient = Image.radial_gradient("L").resize((width, height), Image.Resampling.BILINEAR)
        strength = params["vignette"]
        # radial_gradient is 0 in the centre and 255 at the edge midpoints.
        falloff = gradient.point(
            [round(255 * (1 - strength * min(1.0, (v / 255) ** 2))) for v in range(256)]
        )
        rgb = ImageChops.multiply(rgb, Image.merge("RGB", (falloff, falloff, falloff)))
    # 5. Sensor noise: shared luma noise plus a little independent chroma noise.
    rng = random.Random(int(params["noise_seed"]))
    luma = _scaled_noise((width, height), rng, params["noise"])
    channels: list[Image.Image] = []
    for channel in rgb.split():
        noisy = ImageChops.add(channel, luma, scale=1.0, offset=-128)
        if params["chroma_noise"] > 0:
            chroma = _scaled_noise((width, height), rng, params["chroma_noise"])
            noisy = ImageChops.add(noisy, chroma, scale=1.0, offset=-128)
        channels.append(noisy)
    rgb = Image.merge("RGB", channels)
    # 6. In-camera sharpening, then 7. JPEG compression with 4:2:0 chroma subsampling.
    rgb = rgb.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=round(params["sharpen"]), threshold=2)
    )
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=int(params["jpeg_quality"]), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def apply_camera_look(
    render: Path, destination: Path, *, preset: str = "reco", seed: int = 0
) -> CameraLookResult:
    """Copy a render and replace every RGB frame with its degraded version."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of: {', '.join(PRESETS)}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    render = render.resolve()
    if not render.is_dir():
        raise AnnotationError(f"not a directory: {render}")
    report: list[dict[str, object]] = []
    with new_directory(destination) as staging:
        shutil.copytree(render, staging, dirs_exist_ok=True, symlinks=False)
        paths = (
            [staging / "scene.json"]
            if (staging / "scene.json").is_file()
            else sorted(staging.glob("*/scene.json"))
        )
        if not paths:
            raise AnnotationError("no scene.json renderer outputs found")
        for path in paths:
            scene = load_scene(path)
            for frame in scene.frames:
                if frame.rgb.suffix.lower() != ".png":
                    raise AnnotationError("camera look writes PNG; the render RGB is not")
                params = sample(preset, seed, scene.job_id, frame.index)
                with Image.open(frame.rgb) as source:
                    degraded = degrade(source, params)
                degraded.save(frame.rgb)
                report.append({"job_id": scene.job_id, "frame_index": frame.index, **params})
            data = mapping(json.loads(path.read_text(encoding="utf-8")))
            assets = sequence(data.get("assets"))
            assets.append(
                {
                    "id": "camera-look",
                    "preset": preset,
                    "seed": seed,
                    "report": "camera-look.json",
                    "note": "RGB degraded like a real camera; labels unchanged from the render.",
                }
            )
            data["assets"] = assets
            write_json(path, data)
        write_json(
            staging / "camera-look.json",
            {
                "schema_version": "1.0",
                "preset": preset,
                "seed": seed,
                "ranges": {name: list(bounds) for name, bounds in PRESETS[preset].items()},
                "frames": report,
            },
        )
    return CameraLookResult(destination, len(report))
