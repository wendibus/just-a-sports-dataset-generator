import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fixtures import read_json
from PIL import Image
from test_crops import render

from just_a_sports_dataset_generator.camera_look import (
    PRESETS,
    apply_camera_look,
    degrade,
    sample,
)
from just_a_sports_dataset_generator.cli import main


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CameraLookTests(unittest.TestCase):
    def test_parameters_are_deterministic_and_inside_the_preset(self):
        first = sample("reco", 3, "scene-000001", 2)
        self.assertEqual(first, sample("reco", 3, "scene-000001", 2))
        self.assertNotEqual(first, sample("reco", 3, "scene-000001", 3))
        for name, (low, high) in PRESETS["reco"].items():
            self.assertTrue(low <= first[name] <= high, name)
        with self.assertRaises(ValueError):
            sample("film", 0, "x", 0)

    def test_degrade_keeps_size_and_changes_pixels_reproducibly(self):
        image = Image.new("RGB", (120, 80), (200, 120, 60))
        image.paste((20, 40, 200), (40, 20, 80, 60))
        params = sample("reco", 0, "scene-000000", 0)
        out = degrade(image, params)
        self.assertEqual(out.size, image.size)
        self.assertNotEqual(out.tobytes(), image.tobytes())
        self.assertEqual(out.tobytes(), degrade(image, params).tobytes())

    def test_render_copy_keeps_every_label_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = render(root)
            result = apply_camera_look(source, root / "look", seed=5)
            self.assertEqual(result.frames, 1)
            for path in source.rglob("*"):
                if not path.is_file() or path.name in ("frame.png", "scene.json"):
                    continue
                copy = root / "look" / path.relative_to(source)
                self.assertEqual(digest(copy), digest(path), path.name)  # ids, poses, layout
            frame = next(source.rglob("frame.png"))
            changed = root / "look" / frame.relative_to(source)
            self.assertNotEqual(digest(changed), digest(frame))
            scene = read_json(next((root / "look").rglob("scene.json")))
            self.assertEqual(scene["assets"][-1]["id"], "camera-look")
            report = read_json(root / "look/camera-look.json")
            self.assertEqual(report["seed"], 5)
            self.assertEqual(len(report["frames"]), 1)
            # The degraded copy exports like any render.
            self.assertEqual(
                main(["export", str(root / "look"), "--output", str(root / "dataset")]), 0
            )
            boxes = sum(
                len(json.loads(path.read_text())["annotations"])
                for path in (root / "dataset/coco").rglob("_annotations.coco.json")
            )
            self.assertGreater(boxes, 0)


if __name__ == "__main__":
    unittest.main()
