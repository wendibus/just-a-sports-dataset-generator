"""Build one auditable 5-v-5 example from approved/local photographic cutouts.

All visible masks come from the FINAL raster, including cutout occlusions. This
is a camera-specific 2.5D demonstration, not simulated match or keypoint truth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from just_a_sports_dataset_generator.annotations import ObjectIdentity, visible_annotations
from just_a_sports_dataset_generator.core import labels_for_sport
from just_a_sports_dataset_generator.exporters import export_dataset
from just_a_sports_dataset_generator.photo_composite import PhotoLayer, compose_photo_layers
from just_a_sports_dataset_generator.storage import new_directory, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--approved", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    approved = args.approved.resolve()
    # IDs 10..14 yellow, 15..19 red. Every athlete uses a distinct source cutout.
    specs = [
        (10, approved, "scene-000049-f000000-i014", (880, 650), 190, "yellow"),
        (11, root / "player-assets", "scene-000006-f000000-i011", (760, 810), 205, "yellow"),
        (12, root / "player-assets", "scene-000021-f000000-i013", (550, 710), 185, "yellow"),
        (13, root / "player-assets", "scene-000030-f000000-i014", (1270, 560), 165, "yellow"),
        (14, root / "player-assets", "scene-000037-f000000-i013", (1260, 835), 210, "yellow"),
        (15, approved, "scene-000058-f000000-i010", (1100, 755), 200, "red"),
        (16, root / "player-assets", "scene-000002-f000000-i015", (860, 590), 175, "red"),
        (17, root / "player-assets", "scene-000004-f000000-i013", (690, 555), 160, "red"),
        (18, root / "player-assets", "scene-000053-f000000-i015", (980, 535), 150, "red"),
        (19, root / "player-assets", "scene-000056-f000000-i018", (1410, 700), 190, "red"),
        (30, root / "referee-assets", "scene-000000-f000000-i030", (1550, 940), 280, "official"),
    ]
    layers = []
    assets = []
    placements = []
    for identity, folder, name, foot, height, team in specs:
        path = folder / (name + "-cutout.png")
        with Image.open(path) as raw:
            image = raw.copy()
        obj = ObjectIdentity(
            identity, 2 if identity == 30 else 1, "referee" if identity == 30 else "player"
        )
        layers.append(PhotoLayer(obj, image, foot, height))
        assets.append(
            {
                "id": f"photo-{identity}",
                "source": str(path),
                "sha256": sha256_file(path),
                "generation_manifest_sha256": sha256_file(folder / "photos/generation.json"),
                "foreground_manifest_sha256": sha256_file(
                    folder / "silhouettes" / (name + ".json")
                ),
                "model": "Qwen-Image-Edit-2511 + local BiRefNet",
                "model_licenses": "Apache-2.0 (Qwen); MIT (BiRefNet)",
            }
        )
        placements.append(
            {
                "instance_id": identity,
                "source_asset": f"photo-{identity}",
                "team": team,
                "foot_pixel": list(foot),
                "height_pixels": height,
            }
        )
    with Image.open(root / "background/frame-000000.png") as raw:
        background = raw.convert("RGB")
    with Image.open(root / "background/frame-000000.instances.pgm") as raw:
        base_ids = raw.copy()
    base = (
        ObjectIdentity(1, 0, "ball"),
        ObjectIdentity(40, 3, "hoop"),
        ObjectIdentity(41, 3, "hoop"),
    )
    rgb, ids = compose_photo_layers(background, base_ids, base, layers)
    objects = base + tuple(layer.identity for layer in layers)
    annotations = visible_annotations(*rgb.size, ids.tobytes(), objects, "basketball")
    counts = Counter(a.label for a in annotations)
    if counts != {"ball": 1, "player": 10, "referee": 1, "hoop": 1}:
        raise ValueError(f"Unexpected visibility: {dict(counts)}")
    assets.append(
        {
            "id": "blender-background",
            "source": "background/request.json",
            "request_sha256": sha256_file(root / "background/request.json"),
            "rgb_sha256": sha256_file(root / "background/frame-000000.png"),
            "ids_sha256": sha256_file(root / "background/frame-000000.instances.pgm"),
        }
    )
    with new_directory(root / "render") as stage:
        rgb.save(stage / "example.png")
        ids.save(stage / "example.instances.png")
        write_json(
            stage / "scene.json",
            {
                "schema_version": "2.0",
                "renderer": "local.photo-layer-composite",
                "renderer_version": "1.0",
                "job_id": "basketball-photo-example",
                "sport": "basketball",
                "seed": 2026,
                "labels": labels_for_sport("basketball"),
                "resolution": {"width": rgb.width, "height": rgb.height},
                "requested_annotations": ["bounding_boxes", "instance_masks"],
                "annotation_semantics": "Final visible pixels. Thresholded cutout alpha >=128 is used identically for RGB and IDs; geometry masks from clean Blender render. Boxes xywh, exclusive right/bottom.",
                "assets": assets,
                "frames": [
                    {
                        "frame_index": 0,
                        "rgb": "example.png",
                        "instance_ids": "example.instances.png",
                        "objects": [
                            {
                                **asdict(obj),
                                "bbox_xywh": next(
                                    (
                                        list(a.bbox_xywh)
                                        for a in annotations
                                        if a.instance_id == obj.instance_id
                                    ),
                                    None,
                                ),
                                "visible_area": next(
                                    (
                                        a.visible_area
                                        for a in annotations
                                        if a.instance_id == obj.instance_id
                                    ),
                                    0,
                                ),
                            }
                            for obj in objects
                        ],
                    }
                ],
            },
        )
    write_json(
        root / "placements.json",
        {
            "schema_version": "1.0",
            "layers": placements,
            "depth_order": "ground pixel y, back to front",
            "shadows": "approximate painted contact shadows, background ID 0",
            "limitations": [
                "2.5D composition, not match simulation",
                "No original Blender keypoints reused",
                "Single example, not a validated training corpus",
            ],
        },
    )
    from jsonschema import Draft202012Validator

    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/scene-v2.0.json").read_text()
    )
    Draft202012Validator(schema).validate(json.loads((root / "render/scene.json").read_text()))
    exported = export_dataset(root / "render", root / "dataset")
    if exported.images != 1 or exported.annotations != 13:
        raise AssertionError("COCO export count mismatch")
    # Independently decode every exported RLE and compare its support with the final IDs.
    import numpy as np
    from pycocotools import mask as coco_mask

    coco_path = root / "dataset/coco/train/_annotations.coco.json"
    coco = json.loads(coco_path.read_text())
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    assert [categories[i] for i in sorted(categories)] == list(labels_for_sport("basketball"))
    expected = [(a.class_id + 1, a.bbox_xywh, a.visible_area) for a in annotations]
    for ann in coco["annotations"]:
        rle = ann["segmentation"]
        if isinstance(rle["counts"], list):
            rle = coco_mask.frPyObjects(rle, rgb.height, rgb.width)
        decoded = coco_mask.decode(rle)
        ys, xs = np.where(decoded > 0)
        bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
        assert bbox == tuple(ann["bbox"]) and int(decoded.sum()) == ann["area"]
        match = [
            a for a in annotations if a.class_id + 1 == ann["category_id"] and a.bbox_xywh == bbox
        ]
        assert len(match) == 1
        original = np.asarray(ids) == match[0].instance_id
        assert np.array_equal(decoded.astype(bool), original)
        assert (ann["category_id"], bbox, ann["area"]) in expected
    overlay = rgb.copy()
    draw = ImageDraw.Draw(overlay)
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font = (
        ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    )
    colors = [(255, 70, 90), (45, 150, 255), (255, 180, 30), (50, 230, 130)]
    masks = Image.new("RGB", ids.size, (25, 28, 35))
    for ann in annotations:
        mask = ids.point([255 if v == ann.instance_id else 0 for v in range(256)])
        color = colors[ann.class_id]
        masks.paste(color, (0, 0), mask)
        x, y, w, h = ann.bbox_xywh
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=3)
        label = f"{ann.label} #{ann.instance_id}"
        text_box = draw.textbbox((x, max(0, y - 24)), label, font=font)
        draw.rectangle(text_box, fill=(15, 18, 24))
        draw.text((x, max(0, y - 24)), label, fill=color, font=font)
    overlay.save(root / "labelled-example.png")
    masks.save(root / "visible-masks.png")
    write_json(
        root / "annotation-check.json",
        {
            "schema_version": "1.0",
            "passed": True,
            "counts": dict(counts),
            "image_size": list(rgb.size),
            "annotations": len(annotations),
            "checks": [
                "COCO category order equals Reco basketball labels",
                "all 13 RLE masks decode identically to final instance IDs",
                "every box is the tight extent of visible pixels",
                "areas equal mask pixel counts",
                "ten distinct player assets",
            ],
            "rgb_sha256": sha256_file(root / "render/example.png"),
            "instance_ids_sha256": sha256_file(root / "render/example.instances.png"),
            "coco_sha256": sha256_file(coco_path),
        },
    )
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Basketball photo example</title><style>body{font:16px system-ui;background:#17191d;color:#eee;margin:24px}img{width:100%;max-width:1920px}a{color:#8cf}</style><h1>Basketball photographic composite</h1><p>10 players, 1 referee, 1 ball, 1 visible hoop. Visible-pixel masks and COCO boxes verified. Camera-specific 2.5D demonstration.</p><p><a href="dataset/coco/train/_annotations.coco.json">COCO annotations</a> · <a href="annotation-check.json">Verification</a></p><h2>Image</h2><img src="render/example.png"><h2>Labels</h2><img src="labelled-example.png"><h2>Visible masks</h2><img src="visible-masks.png">',
        encoding="utf-8",
    )
    print("Example:", root / "index.html", flush=True)


if __name__ == "__main__":
    main()
