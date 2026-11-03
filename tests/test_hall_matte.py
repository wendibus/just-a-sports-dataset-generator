"""Regression checks in the already-installed optional matte environment."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

MATTE_PYTHON = os.environ.get("HALL_MATTE_PYTHON")


@unittest.skipUnless(MATTE_PYTHON, "set HALL_MATTE_PYTHON to the existing OpenCV matte environment")
class HallMatteTests(unittest.TestCase):
    def test_attached_ball_is_flagged_but_clean_person_and_skin_are_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "photos").mkdir()
            (root / "masks").mkdir()
            for extra in (False, True):
                image = Image.new("RGB", (640, 640), (190, 190, 190))
                mask = Image.new("L", (640, 640))
                draw = ImageDraw.Draw(image)
                alpha = ImageDraw.Draw(mask)
                for shape, box in [
                    ("ellipse", (285, 55, 345, 120)),
                    ("rectangle", (275, 120, 355, 350)),
                    ("rectangle", (275, 350, 303, 560)),
                    ("rectangle", (327, 350, 355, 560)),
                    ("rectangle", (350, 180, 414, 197)),
                ]:
                    getattr(draw, shape)(box, fill=(160, 25, 35))
                    getattr(alpha, shape)(box, fill=255)
                # A brown hand must not be treated as a basketball.
                draw.ellipse((395, 180, 414, 197), fill=(130, 75, 38))
                if extra:
                    draw.ellipse(
                        (407, 170, 467, 230), fill=(210, 110, 35), outline=(45, 30, 20), width=3
                    )
                    draw.line((417, 180, 457, 220), fill=(45, 30, 20), width=2)
                    alpha.ellipse((407, 170, 467, 230), fill=255)
                identity = "with-ball" if extra else "no-ball"
                image.save(root / "photos" / (identity + ".png"))
                mask.save(root / "masks" / (identity + ".png"))
            subprocess.run(
                [
                    MATTE_PYTHON,
                    str(Path(__file__).resolve().parents[1] / "scripts/hall_night/matte_checks.py"),
                    str(root),
                    "--ids",
                    "no-ball",
                    "with-ball",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            report = json.loads((root / "matte-checks.json").read_text())
            self.assertFalse(report["no-ball"]["possible_extra_balls"])
            self.assertTrue(report["with-ball"]["possible_extra_balls"])
            for identity in report:
                with Image.open(root / "masks" / (identity + ".clean.png")) as mask:
                    self.assertGreater(mask.getbbox()[3], 550)


if __name__ == "__main__":
    unittest.main()
