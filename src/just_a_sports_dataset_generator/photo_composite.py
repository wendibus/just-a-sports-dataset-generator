"""Rasterize camera-specific photo cutouts and derive final visible instance IDs.

This is a 2.5D composite, not a 3D simulation or a pose/keypoint ground-truth source.
A thresholded alpha is used for BOTH RGB and IDs, making their pixel support exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter

from just_a_sports_dataset_generator.annotations import ObjectIdentity, visible_annotations


@dataclass(frozen=True)
class PhotoLayer:
    identity: ObjectIdentity
    rgba: Image.Image
    foot: tuple[int, int]
    height: int


def place_layer(layer: PhotoLayer) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    if layer.rgba.mode != "RGBA":
        raise ValueError("photo layer must have an explicit RGBA foreground mask")
    if type(layer.height) is not int or not 16 <= layer.height <= 4096:
        raise ValueError("photo height must be an integer in 16..4096 pixels")
    alpha = layer.rgba.getchannel("A").point([255 if x >= 128 else 0 for x in range(256)])
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("empty photo foreground")
    rgb = layer.rgba.convert("RGB").crop(bbox)
    alpha = alpha.crop(bbox)
    width = max(1, round(rgb.width * layer.height / rgb.height))
    size = (width, layer.height)
    rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    alpha = alpha.resize(size, Image.Resampling.LANCZOS).point(
        [255 if x >= 128 else 0 for x in range(256)]
    )
    # The feet, rather than wide arms, determine the horizontal ground anchor.
    lower = alpha.crop((0, int(size[1] * 0.85), size[0], size[1])).getbbox()
    anchor = (lower[0] + lower[2]) / 2 if lower else size[0] / 2
    return rgb, alpha, (round(layer.foot[0] - anchor), layer.foot[1] - size[1])


def compose_photo_layers(
    background: Image.Image,
    instance_ids: Image.Image,
    base_objects: tuple[ObjectIdentity, ...],
    layers: list[PhotoLayer],
    *,
    shadows: bool = True,
) -> tuple[Image.Image, Image.Image]:
    if background.mode != "RGB" or instance_ids.mode != "L" or background.size != instance_ids.size:
        raise ValueError("background must be RGB and IDs must be matching 8-bit grayscale")
    objects = base_objects + tuple(layer.identity for layer in layers)
    # Validate taxonomy, duplicate IDs and background identities before any mutation.
    visible_annotations(*background.size, instance_ids.tobytes(), objects, "basketball")
    if any(layer.identity.instance_id in set(instance_ids.tobytes()) for layer in layers):
        raise ValueError("new photo IDs already exist in background")
    if len({layer.foot[1] for layer in layers}) != len(layers):
        raise ValueError("ambiguous equal-depth photo layers; use distinct ground depths")
    rgb = background.copy()
    ids = instance_ids.copy()
    if shadows:
        shadow = Image.new("L", rgb.size)
        draw = ImageDraw.Draw(shadow)
        for layer in layers:
            x, y = layer.foot
            rx = layer.height * 0.15
            ry = layer.height * 0.025
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=40)
        shadow = shadow.filter(ImageFilter.GaussianBlur(4))
        rgb = Image.composite(Image.new("RGB", rgb.size, (0, 0, 0)), rgb, shadow)
    for layer in sorted(layers, key=lambda layer: layer.foot[1]):
        patch, alpha, position = place_layer(layer)
        rgb.paste(patch, position, alpha)
        ids.paste(
            layer.identity.instance_id,
            (*position, position[0] + patch.width, position[1] + patch.height),
            alpha,
        )
    visible_annotations(*rgb.size, ids.tobytes(), objects, "basketball")
    return rgb, ids
