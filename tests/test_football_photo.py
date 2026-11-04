import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hall_night import football_photo as photo
from hall_night.common import sha, validate


class FootballPhotoTests(unittest.TestCase):
    def test_estimated_scale_increases_with_foreground_depth(self):
        camera = photo.calibration()
        self.assertFalse(camera["metric_ground_truth"])
        self.assertAlmostEqual(
            photo.person_height_px(camera["goal_ground_y_px"], 2.44, camera),
            camera["goal_pixels_per_metre"] * 2.44,
        )
        self.assertLess(
            photo.person_height_px(800, 1.8, camera), photo.person_height_px(875, 1.8, camera)
        )
        self.assertGreater(photo.person_height_px(800, 1.8, camera), 200)
        self.assertLess(photo.person_height_px(875, 2.1, camera), 410)
        for ground_y, height in ((float("nan"), 1.8), (800, -1), (camera["horizon_y_px"], 1.8)):
            with self.assertRaises(ValueError):
                photo.person_height_px(ground_y, height, camera)
        self.assertGreater(camera["placement_bounds_xyxy"][1], photo.transform((4807, 3098))[1])

    def test_wrong_source_is_refused_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "wrong.jpg"
            Image.new("RGB", (20, 20)).save(source)
            with self.assertRaisesRegex(ValueError, "checksum"):
                photo.prepare(source, root / "output")
            self.assertFalse((root / "output").exists())
            self.assertFalse((root / "output.partial").exists())

    def test_staging_preserves_pixels_provenance_and_detection_only_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fixture.png"
            # A local fixture at the native dimensions; no third-party photo in Git.
            im = Image.new("RGB", photo.SOURCE_SIZE, (31, 83, 46))
            im.putpixel((200, 200), (255, 20, 0))  # This sky pixel must be cropped away.
            im.save(source)
            output = root / "selected"
            with patch.object(photo, "SOURCE_SHA", sha(source)):
                photo.prepare(source, output)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    photo.prepare(source, output)
            data = json.loads((output / "preparation.json").read_text())
            validate("football-photo-preparation-v1.0.json", data)
            self.assertFalse(data["production_enabled"])
            self.assertTrue((output / "venue-rejected.json").exists())
            self.assertFalse((output / "config.json").exists())
            with Image.open(output / "background.png") as bg:
                self.assertEqual(bg.size, (3200, 900))
                self.assertEqual(bg.getextrema(), ((31, 31), (83, 83), (46, 46)))
            coco = json.loads((output / "goal-box-review.json").read_text())
            self.assertEqual(coco["images"][0]["license"], 1)
            self.assertEqual(coco["licenses"][0]["name"], "CC BY 2.0")
            self.assertEqual(len(coco["annotations"]), 5)
            for a in coco["annotations"]:
                x, y, w, h = a["bbox"]
                self.assertTrue(0 <= x < x + w <= 3200 and 0 <= y < y + h <= 900)
                self.assertEqual(a["category_id"], 5)
                self.assertNotIn("segmentation", a)
            self.assertIn("Dejan Krsmanovic", (output / "ATTRIBUTION.md").read_text())
            self.assertEqual(sha(source), sha(output / "source.jpg"))
            self.assertFalse((root / "selected.partial").exists())

    def test_failed_validation_cleans_only_owned_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fixture.png"
            Image.new("RGB", photo.SOURCE_SIZE).save(source)
            with (
                patch.object(photo, "SOURCE_SHA", sha(source)),
                patch.object(photo, "validate", side_effect=ValueError("bad contract")),
                self.assertRaisesRegex(ValueError, "bad contract"),
            ):
                photo.prepare(source, root / "output")
            self.assertTrue(source.exists())
            self.assertFalse((root / "output.partial").exists())
            (root / "output.partial").mkdir()
            sentinel = root / "output.partial" / "keep.txt"
            sentinel.write_text("keep")
            with patch.object(photo, "SOURCE_SHA", sha(source)), self.assertRaises(FileExistsError):
                photo.prepare(source, root / "output")
            self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
