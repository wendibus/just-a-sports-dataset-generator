import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from test_composite import CROP_ID, setup

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "photo_candidates", ROOT / "scripts/qwen_photo_candidates.py"
)
assert spec and spec.loader
photo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(photo)


class PhotoCandidateTests(unittest.TestCase):
    def test_neutral_reference_and_new_alpha_without_reusing_labels(self):
        self.check_candidate("player", 768)

    def test_referee_prompt_resolution_and_interpreter(self):
        self.check_candidate("referee", 640)

    def check_candidate(self, role, resolution):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, source = setup(root)
            out = root / "photos"
            original = (source / "crops.json").read_bytes()

            def generate(inputs, destination, **options):
                self.assertEqual(options["mode"], "edit")
                self.assertEqual(options["resolution"], resolution)
                self.assertIn(
                    "LONG BLACK TROUSERS" if role == "referee" else "basketball jersey",
                    options["prompt"],
                )
                self.assertFalse(options["confirm"](None))
                index = json.loads((inputs / "crops.json").read_text())
                self.assertEqual(index["size"], resolution)
                with Image.open(inputs / index["crops"][0]["files"]["rgb"]) as im:
                    self.assertEqual(im.size, (resolution, resolution))
                    self.assertEqual(im.getpixel((0, 0)), (190, 190, 190))
                destination.mkdir()
                Image.new("RGB", (resolution, resolution), (100, 50, 20)).save(
                    destination / f"{CROP_ID}.png"
                )

            def masks(command, **options):
                self.assertEqual(command[0], str(Path(sys.executable).absolute()))
                dest = Path(command[command.index("--output") + 1])
                dest.mkdir()
                mask = Image.new("L", (resolution, resolution))
                mask.paste(255, (100, 100, resolution - 100, resolution - 80))
                mask.save(dest / f"{CROP_ID}.png")

            argv = [
                "candidate",
                str(root),
                "--role",
                role,
                "--resolution",
                str(resolution),
                "--output",
                str(out),
                "--ids",
                CROP_ID,
                "--review-python",
                sys.executable,
                "--foreground-model",
                str(root / "model"),
            ]
            with (
                patch("sys.argv", argv),
                patch.object(photo, "generate_crops", side_effect=generate),
                patch.object(photo.subprocess, "run", side_effect=masks),
            ):
                photo.main()
            self.assertEqual((source / "crops.json").read_bytes(), original)
            self.assertTrue((out / "index.html").is_file())
            with Image.open(out / f"{CROP_ID}-cutout.png") as im:
                self.assertEqual(im.mode, "RGBA")
                self.assertEqual(
                    im.getchannel("A").getbbox(), (100, 100, resolution - 100, resolution - 80)
                )
            self.assertFalse((out / "scene.json").exists())


if __name__ == "__main__":
    unittest.main()
