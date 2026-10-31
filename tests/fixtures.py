"""Small, generated fixtures. No third-party images or assets."""

import json
from pathlib import Path

from PIL import Image

from just_a_sports_dataset_generator.core import labels_for_sport
from just_a_sports_dataset_generator.storage import write_json


def fixture(root: Path, sport="basketball", scenes=3, frames=2) -> Path:
    root.mkdir()
    labels = labels_for_sport(sport)
    for scene in range(scenes):
        folder = root / f"scene-{scene:06d}"
        folder.mkdir()
        entries = []
        for index in range(frames):
            # Odd dimensions exercise floating-point box roundtrips, including 1px boxes.
            rgb = Image.new("RGB", (17, 13), (scene * 40, index * 50, 37))
            mask = Image.new("L", rgb.size)
            objects = []
            for class_id, label in enumerate(labels):
                iid = class_id + 1
                x, y = class_id * 3, 1 + index
                box = (x, y, 1 if class_id == 0 else 2, 3)
                for dx in range(box[2]):
                    for dy in range(box[3]):
                        mask.putpixel((x + dx, y + dy), iid)
                        rgb.putpixel((x + dx, y + dy), (iid * 30, 100, 160))
                objects.append(
                    {
                        "instance_id": iid,
                        "class_id": class_id,
                        "label": label,
                        "bbox_xywh": list(box),
                        "amodal_bbox_xywh": list(box),
                        "visible_area": box[2] * box[3],
                    }
                )
            rgb.save(folder / f"{index}.ppm")
            mask.save(folder / f"{index}.pgm")
            entries.append(
                {
                    "frame_index": index,
                    "rgb": f"{index}.ppm",
                    "instance_ids": f"{index}.pgm",
                    "objects": objects,
                }
            )
        write_json(
            folder / "scene.json",
            {
                "schema_version": "2.0",
                "job_id": folder.name,
                "sport": sport,
                "seed": 100 + scene,
                "labels": labels,
                "resolution": {"width": 17, "height": 13},
                "renderer": "tests.fixture",
                "renderer_version": "1.0",
                "frames": entries,
                "requested_annotations": ["bounding_boxes", "instance_masks"],
                "annotation_semantics": "visible pixels; xywh; exclusive right/bottom",
                "assets": [
                    {
                        "id": "test-fixture",
                        "license": "Apache-2.0",
                        "source": "tests/fixtures.py",
                        "version": "1.0",
                    }
                ],
            },
        )
    write_json(root / "run.json", {"schema_version": "1.0", "fixture": True})
    return root


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
