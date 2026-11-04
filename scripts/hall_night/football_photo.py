"""Offline staging of the selected real venue; deliberately not a batch renderer."""

from __future__ import annotations

import html
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from .common import read, sha, validate, write

SOURCE_SHA = "185d938420695a1f8f4010abd60d2dfc39e323d771c1d776d095db338a76015f"
SOURCE_SIZE = (7670, 3407)
CROP = (0, 1250, 7670, 3407)
SIZE = (3200, 900)
LICENSE_URL = "https://creativecommons.org/licenses/by/2.0/"
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Subsidiary_Football_Yard_(43019524354).jpg"
# Manually inspected full-resolution boxes, including net/support extent.
# These are detection annotations, never substitutes for fine instance masks.
GOALS = (
    ("near-white", (2648, 2400, 4807, 3098)),
    ("far-white", (3335, 2545, 3895, 2725)),
    ("blue-training", (1300, 2640, 1700, 2715)),
    ("red-mini", (1020, 2650, 1125, 2705)),
    ("red-mini-behind-net", (3170, 2658, 3315, 2715)),
)


def transform(point: tuple[float, float]) -> tuple[float, float]:
    return (
        (point[0] - CROP[0]) * SIZE[0] / (CROP[2] - CROP[0]),
        (point[1] - CROP[1]) * SIZE[1] / (CROP[3] - CROP[1]),
    )


def calibration() -> dict:
    """A scale hypothesis from the near aperture, not a measured pinhole camera."""
    top = sum(transform(p)[1] for p in ((2735, 2417), (4685, 2410))) / 2
    ground = sum(transform(p)[1] for p in ((2720, 3044), (4710, 3050))) / 2
    ppm = (ground - top) / 2.44
    return {
        "method": "estimated local upright scale from presumed 2.44m near goal aperture",
        "metric_ground_truth": False,
        "assumed_goal_height_m": 2.44,
        "assumed_camera_height_m": 1.65,
        "goal_ground_y_px": ground,
        "goal_pixels_per_metre": ppm,
        "horizon_y_px": ground - ppm * 1.65,
        "placement_bounds_xyxy": [150, 800, 3050, 875],
        "restriction": "People and balls only in front of the near net; no behind-net placement",
        "limitations": "Single panorama photograph, unmeasured lens/level and goal size; local scale hypothesis only",
    }


def person_height_px(ground_y: float, height_m: float, camera: dict) -> float:
    if not math.isfinite(ground_y) or not math.isfinite(height_m) or height_m <= 0:
        raise ValueError("Finite ground coordinate and positive height required")
    if ground_y <= camera["horizon_y_px"]:
        raise ValueError("Ground must be below the estimated horizon")
    return (ground_y - camera["horizon_y_px"]) / camera["assumed_camera_height_m"] * height_m


def pilot_requests() -> list[dict]:
    requests = []
    for i in range(10):
        gender = "men" if i < 4 else "women" if i < 8 else "mixed"
        ball_count = 1 if i < 7 else (2 if i == 7 else 4) if i < 9 else 0
        requests.append(
            {
                "id": f"ground-photo-pilot-{i:02d}",
                "gender_mode": gender,
                "ball_count": ball_count,
                "field_players": 4,
                "goalkeepers": 1,
                "referees": 1,
                "view": "full foreground" if i < 5 else "closer crop",
                "review": [
                    "adult anatomy and varied skin tones/heights",
                    "football boots and kit",
                    "ground-level pose perspective",
                    "warm diffuse light and contact shadows",
                    "partial player/ball overlap; no fully hidden ball counted as visible",
                ],
            }
        )
    return requests


def prepare(source: Path, output: Path) -> Path:
    """Verify one approved photograph and publish a new, non-runnable staging folder."""
    source, output = source.absolute(), output.absolute()
    if output.exists():
        raise ValueError("Output already exists; staging never overwrites a venue")
    if sha(source) != SOURCE_SHA:
        raise ValueError("Selected photograph checksum mismatch")
    with Image.open(source) as image:
        if image.size != SOURCE_SIZE or image.getexif().get(274, 1) != 1:
            raise ValueError("Unexpected source dimensions/orientation")
        background = image.convert("RGB").crop(CROP).resize(SIZE, Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(exist_ok=False)
    try:
        shutil.copy2(source, staging / "source.jpg")
        background.save(staging / "background.png")
        background.save(staging / "Vorschau.jpg", quality=95)
        boxes = []
        for i, (name, box) in enumerate(GOALS, 1):
            x0, y0 = transform(box[:2])
            x1, y1 = transform(box[2:])
            boxes.append(
                {
                    "id": i,
                    "name": name,
                    "source_xyxy": list(box),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                }
            )
        camera = calibration()
        manifest = {
            "schema_version": "1.0",
            "sport": "football",
            "stage": "photo_selected_scale_estimated",
            "production_enabled": False,
            "requested_count": 10009,
            "image_size": list(SIZE),
            "source_size": list(SOURCE_SIZE),
            "source_crop_xyxy": list(CROP),
            "source_sha256": SOURCE_SHA,
            "background_sha256": sha(staging / "background.png"),
            "rights": {
                "author": "Dejan Krsmanovic",
                "title": "Subsidiary Football Yard",
                "source": SOURCE_PAGE,
                "license": "CC BY 2.0",
                "license_url": LICENSE_URL,
                "changes": "Sky cropped and resized; original pitch, goals, buildings and light retained",
            },
            "camera": camera,
            "labels": ["ball", "player", "goalkeeper", "referee", "goal"],
            "goals": boxes,
            "annotation_policy": "Manually inspected detection boxes, no segmentation or net-thread masks; overlapping boxes allowed",
            "pending": [
                "ground-level photographic person assets",
                "labelled composite and occlusion review",
                "photo-aware rendering/export integration before enabling production",
            ],
        }
        validate("football-photo-preparation-v1.0.json", manifest)
        write(staging / "preparation.json", manifest)
        write(
            staging / "pilot-requests.json",
            {"schema_version": "1.0", "status": "not_run", "requests": pilot_requests()},
        )
        # A detection-only COCO review fixture, deliberately outside any dataset/coco folder.
        write(
            staging / "goal-box-review.json",
            {
                "info": {
                    "description": "Background detection-box review only; not a generated training dataset"
                },
                "licenses": [{"id": 1, "name": "CC BY 2.0", "url": LICENSE_URL}],
                "images": [
                    {
                        "id": 1,
                        "file_name": "background.png",
                        "width": SIZE[0],
                        "height": SIZE[1],
                        "license": 1,
                    }
                ],
                "categories": [
                    {"id": i, "name": label} for i, label in enumerate(manifest["labels"], 1)
                ],
                "annotations": [
                    {
                        "id": b["id"],
                        "image_id": 1,
                        "category_id": 5,
                        "bbox": b["bbox"],
                        "area": b["bbox"][2] * b["bbox"][3],
                        "iscrowd": 0,
                    }
                    for b in boxes
                ],
            },
        )
        labelled = background.copy()
        draw = ImageDraw.Draw(labelled)
        for b in boxes:
            x, y, w, h = b["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="orange", width=3)
            draw.text((x, y - 15), "goal: " + b["name"], fill="orange")
        labelled.save(staging / "Torboxen-Pruefung.jpg", quality=95)
        attribution = f"# Image attribution\n\nSubsidiary Football Yard — Dejan Krsmanovic.\n\nSource: {SOURCE_PAGE}\n\nLicense: CC BY 2.0 ({LICENSE_URL}).\n\nChanges: sky crop, resizing; separate detection-box overlay.\nThe photo's license is separate from the Apache-2.0 project code.\nRetain attribution and a change notice in future composite/dataset handoffs.\n"
        (staging / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
        (staging / "venue-rejected.json").write_text(
            '{"reason":"Photo selected; person composite and photo-aware renderer still pending. Production remains disabled."}\n',
            encoding="utf-8",
        )
        (staging / "index.html").write_text(
            '<!doctype html><html lang="de"><meta charset="utf-8"><title>Fußball – echte Fotovorlage B</title>'
            "<style>body{background:#202322;color:#eee;font:18px system-ui;margin:32px}img{width:100%;height:auto}a{color:#9de}p{max-width:1000px}</style>"
            "<h1>Fußball – Amateurplatz aus Bodennähe</h1><p>Echte Fotovorlage B. Himmel beschnitten; Platz und Tore unverändert.</p>"
            '<img src="Vorschau.jpg" alt="Echter Amateurplatz mit Toren und Rasen">'
            "<p>Vorbereitung für 10.009 Bilder. Noch kein Personenpilot und kein freigegebener Trainingsdatensatz. "
            "Das laufende Reco-Training wird nicht gesteuert. Keine Modelle wurden gestartet.</p>"
            "<p>Geplant: 40 % Männer, 40 % Frauen, 20 % gemischt; Erwachsene. "
            "70 % ein Ball, 20 % zwei bis vier Bälle, 10 % ohne sichtbaren Ball. "
            "Wenige Spieler vor dem nahen Netz; unterschiedliche Hautfarben, Größen, Ausschnitte und Überdeckungen.</p>"
            "<p>Größenmaßstab aus angenommener Torhöhe und Kamerahöhe geschätzt; keine vermessene Kalibrierung. "
            "Fünf Torbereiche manuell erfasst. Netzmasken fehlen noch. "
            '<a href="Torboxen-Pruefung.jpg">Torboxen prüfen</a> · <a href="preparation.json">Metadaten</a> · '
            '<a href="pilot-requests.json">Zehn Pilotaufträge (nicht gestartet)</a></p>'
            f'<p>Foto: Dejan Krsmanovic · <a href="{html.escape(SOURCE_PAGE)}">Original</a> · '
            f'<a href="{LICENSE_URL}">CC BY 2.0</a> · Bearbeitung: Ausschnitt/Skalierung.</p></html>',
            encoding="utf-8",
        )
        # Read back publication-critical metadata and recheck the published pixels.
        validate("football-photo-preparation-v1.0.json", read(staging / "preparation.json"))
        if sha(staging / "source.jpg") != SOURCE_SHA:
            raise ValueError("Copied source checksum mismatch")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output
