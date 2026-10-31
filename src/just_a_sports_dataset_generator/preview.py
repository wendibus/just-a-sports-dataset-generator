"""Offline RGB/mask previews, kept outside training data and raw ID masks."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from just_a_sports_dataset_generator.annotations import (
    AnnotationError,
    ObjectIdentity,
    visible_annotations,
)
from just_a_sports_dataset_generator.annotations.scene import (
    MAX_PIXELS,
    integer,
    mapping,
    safe_file,
    sequence,
    string,
)
from just_a_sports_dataset_generator.core import labels_for_sport
from just_a_sports_dataset_generator.storage import new_directory

_COLORS = ((225, 68, 83), (49, 104, 222), (183, 111, 34), (12, 150, 111), (143, 80, 204))
_BACKGROUND = (235, 239, 245)
_SPLIT_NAMES = {"train": "Training", "val": "Validierung", "test": "Test"}
_STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#182333;font:16px/1.5 system-ui,sans-serif}
main{max-width:1300px;margin:auto;padding:32px 24px}h1{font-size:30px;margin:0 0 8px}
.intro{max-width:900px;color:#415166}.badge{display:inline-block;padding:5px 12px;border-radius:20px;
background:#dcebea;color:#175a53;font-size:14px;font-weight:650}.legend{display:flex;gap:16px;
flex-wrap:wrap;margin:20px 0}.legend span{display:flex;align-items:center;gap:7px}.dot{width:12px;
height:12px;border-radius:50%;display:inline-block}.frame{margin:24px 0;padding:20px;
border:1px solid #d9e0e9;border-radius:16px;background:white}h2{font-size:19px;margin:0 0 14px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}figcaption{font-weight:650;
margin-bottom:8px}img{display:block;width:100%;height:auto;border-radius:8px;background:#ebeff5;
border:1px solid #d9e0e9}.note{font-size:14px;color:#526175}a{color:#1d54b7}
@media(max-width:720px){.pair{grid-template-columns:1fr}main{padding:20px 12px}}
"""


def _document(path: Path) -> dict[str, object]:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise AnnotationError("preview metadata exceeds 32 MiB")
    return mapping(json.loads(path.read_text(encoding="utf-8")))


def colorize_mask(
    mask: Image.Image, objects: tuple[ObjectIdentity, ...], sport: str
) -> Image.Image:
    """Return display colors without changing the numeric ID pass."""
    if mask.mode != "L" or mask.width * mask.height > MAX_PIXELS:
        raise AnnotationError("preview needs an 8-bit instance mask within the image size limit")
    visible_annotations(mask.width, mask.height, mask.tobytes(), objects, sport)
    palette = [int(channel) for channel in _BACKGROUND] * 256
    for obj in objects:
        palette[obj.instance_id * 3 : obj.instance_id * 3 + 3] = _COLORS[
            obj.class_id % len(_COLORS)
        ]
    result = mask.copy()
    result.putpalette(palette)
    return result.convert("RGB")


def create_preview(dataset: Path, destination: Path, limit: int = 48) -> Path:
    """Create an offline index.html and thumbnails, leaving dataset bytes untouched."""
    if type(limit) is not int or limit < 1:
        raise ValueError("preview limit must be positive")
    dataset, destination = dataset.resolve(), destination.absolute()
    if destination.resolve().is_relative_to(dataset) or dataset.is_relative_to(
        destination.resolve()
    ):
        raise AnnotationError("preview must be separate from the training dataset")
    manifest = _document(safe_file(dataset, "manifest.json"))
    if manifest.get("schema_version") not in ("1.0", "2.0"):
        raise AnnotationError("unsupported export manifest")
    sport = string(manifest.get("sport"))
    labels = labels_for_sport(sport)
    if sequence(manifest.get("labels")) != list(labels):
        raise AnnotationError("preview labels differ from Reco Trainer")
    renderers = sequence(manifest.get("renderers", []))
    basketball_3d = any(
        mapping(item).get("id") == "org.just-a-sports-dataset-generator.blender-basketball"
        for item in renderers
    )
    status = (
        "Lokal gerenderte 3D-Basketballszenen mit prozeduralen Figuren. "
        "Fotorealismus und Nutzen für das Modelltraining sind noch nicht validiert."
        if basketball_3d
        else "Der Referenzrenderer erzeugt geometrische Testbilder. Fotorealistische Sportszenen sind noch in Entwicklung."
    )
    cards: list[str] = []
    total = 0
    paths = sorted((dataset / "native").glob("*/annotations.json"))
    if not paths:
        raise AnnotationError("no native annotations available for preview")
    with new_directory(destination) as staging:
        images = staging / "images"
        images.mkdir()
        for candidate in paths:
            path = safe_file(dataset, candidate.relative_to(dataset).as_posix())
            scene = _document(path)
            if scene.get("schema_version") != "1.0" or scene.get("sport") != sport:
                raise AnnotationError("unsupported native annotations")
            split = string(scene.get("split"))
            if split not in _SPLIT_NAMES:
                raise AnnotationError("invalid preview split")
            dimensions = mapping(scene.get("resolution"))
            size = (integer(dimensions.get("width"), 1), integer(dimensions.get("height"), 1))
            if size[0] * size[1] > MAX_PIXELS:
                raise AnnotationError("preview image exceeds size limit")
            folder = "valid" if split == "val" else split
            for raw in sequence(scene.get("frames")):
                frame = mapping(raw)
                total += 1
                if len(cards) >= limit:
                    continue
                name = string(frame.get("image_name"))
                # Check both enclosing folders and final files against symlink escapes.
                rgb_path = safe_file(dataset, f"coco/{folder}/{name}")
                mask_path = safe_file(path.parent, frame.get("instance_ids"))
                objects: list[ObjectIdentity] = []
                for raw_object in sequence(frame.get("objects")):
                    obj = mapping(raw_object)
                    objects.append(
                        ObjectIdentity(
                            integer(obj.get("instance_id"), 1),
                            integer(obj.get("class_id")),
                            string(obj.get("label")),
                        )
                    )
                try:
                    with Image.open(rgb_path) as original, Image.open(mask_path) as mask:
                        if original.mode != "RGB" or original.size != size or mask.size != size:
                            raise AnnotationError(
                                "preview image dimensions/mode do not match metadata"
                            )
                        rgb = original.copy()
                        colored = colorize_mask(mask, tuple(objects), sport)
                except Image.DecompressionBombError as exc:
                    raise AnnotationError("preview image exceeds decoding limit") from exc
                rgb.thumbnail((640, 360), Image.Resampling.LANCZOS)
                colored.thumbnail((640, 360), Image.Resampling.NEAREST)
                stem = f"{len(cards) + 1:04d}"
                rgb.save(images / f"{stem}-rgb.png")
                colored.save(images / f"{stem}-mask.png")
                original_url = quote(
                    os.path.relpath(rgb_path, destination.resolve()).replace(os.sep, "/"), safe="/"
                )
                cards.append(
                    f'<section class="frame"><h2>Bild {len(cards) + 1} · {_SPLIT_NAMES[split]}</h2>'
                    '<div class="pair"><figure><figcaption>Farbbild für RF-DETR</figcaption>'
                    f'<a href="{escape(original_url, quote=True)}">'
                    f'<img src="images/{stem}-rgb.png" alt="Originales Farbbild" loading="lazy"></a>'
                    '<p class="note">Anklicken öffnet das Originalbild.</p></figure>'
                    "<figure><figcaption>Objektbereiche · farbige Vorschau</figcaption>"
                    f'<img src="images/{stem}-mask.png" alt="Farbig dargestellte Objektmaske" loading="lazy">'
                    '<p class="note">Nur zur Kontrolle; die numerischen Labels bleiben erhalten.</p>'
                    "</figure></div></section>"
                )
        legend = "".join(
            f'<span><i class="dot" style="background:rgb{_COLORS[index % len(_COLORS)]}"></i>'
            f"{escape(label)}</span>"
            for index, label in enumerate(labels)
        )
        html = (
            '<!doctype html><html lang="de"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Bilder ansehen · {escape(sport)}</title><style>{_STYLE}</style><body><main>"
            '<span class="badge">Lokal · RF-DETR · Bildkontrolle</span>'
            "<h1>Deine erzeugten Bilder</h1>"
            '<p class="intro">Links stehen die Farbbilder für das Training. Rechts werden die '
            "Objektbereiche zur Kontrolle eingefärbt. Rohe Maskendateien können fast schwarz "
            "aussehen: Sie speichern Objektnummern statt sichtbarer Farben.</p>"
            f'<p class="intro">{status}</p>'
            f'<p class="note">Vorschau: {len(cards)} von {total} Bildern. '
            "Für Reco Trainer den Ordner dataset/coco importieren.</p>"
            f'<div class="legend">{legend}</div>' + "".join(cards) + "</main></body></html>"
        )
        (staging / "index.html").write_text(html, encoding="utf-8")
    return destination / "index.html"
