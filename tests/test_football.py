import random
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hall_night.common import read, sha, write
from hall_night.export import export_and_verify
from hall_night.football import asset_plan, build_balls, choose_roster, make_plans, prompt_for
from hall_night.football_pitch import (
    build_pitch,
    ground_point,
    pixels_per_metre,
    project,
    unproject_ground,
)
from hall_night.render import render_scene


class FootballTests(unittest.TestCase):
    def test_exact_odd_budget_and_full_rosters_are_split_safe(self):
        plans = make_plans(10009, 42)
        self.assertEqual(sum(p["frame_count"] for p in plans), 10009)
        self.assertEqual(plans, make_plans(10009, 42))
        self.assertTrue(all(p["sport"] == "football" for p in plans))
        self.assertEqual(len({p["seed"] for p in plans}), len(plans))
        assets = asset_plan(42)
        self.assertEqual(
            Counter(a["label"] for a in assets), {"player": 288, "goalkeeper": 24, "referee": 24}
        )
        self.assertEqual(len({a["id"] for a in assets}), 336)
        for split in ("train", "val", "test"):
            for gender in ("men", "women", "mixed"):
                roster = choose_roster(
                    assets, {"split": split, "gender_mode": gender}, random.Random(9), False
                )
                self.assertEqual(
                    Counter(a["label"] for a in roster),
                    {"player": 20, "goalkeeper": 2, "referee": 1},
                )
                self.assertTrue(all(a["split"] == split for a in roster))
                self.assertEqual(len({a["id"] for a in roster}), 23)
        for a in assets:
            prompt = prompt_for(a)
            self.assertIn("outdoor", prompt)
            self.assertIn("grass", prompt)
            self.assertNotIn("FUTSAL", prompt)
        self.assertIn("gloves", prompt_for(next(a for a in assets if a["label"] == "goalkeeper")))

    def test_camera_round_trip_and_metric_ball_size(self):
        for x, y in ((0, 0), (105, 68), (52.5, 34), (17, 42)):
            point = np.array([x, y, 0.0])
            self.assertTrue(np.allclose(unproject_ground(project(point)), point))
        far, near = ground_point(0.5, 0.1), ground_point(0.5, 0.9)
        self.assertGreater(near[1], far[1])
        self.assertGreater(pixels_per_metre(near), pixels_per_metre(far))
        self.assertGreater(pixels_per_metre(near) * 0.22, 3)

    def test_pitch_goals_are_original_repeatable_and_have_real_net_holes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a"
            b = root / "b"
            goals = build_pitch(a, 42)
            build_pitch(b, 42)
            self.assertEqual(sha(a / "background.png"), sha(b / "background.png"))
            self.assertEqual(len(goals), 2)
            for goal in goals:
                self.assertEqual(goal["opening_m"], [7.32, 2.44])
                with Image.open(a / "cache" / goal["file"]) as rgba:
                    hist = rgba.getchannel("A").histogram()
                    self.assertGreater(hist[0], 100)
                    self.assertGreater(sum(hist[128:]), 50)
            self.assertEqual(read(a / "camera.json")["field_m"], [105, 68])

    def test_ball_chart_crop_does_not_include_chart_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "chart.png"
            image = Image.new("RGB", (1536, 1024), "white")
            d = ImageDraw.Draw(image)
            for y in (230, 652):
                for x in (175, 480, 787, 1092, 1375):
                    d.ellipse((x - 112, y - 112, x + 112, y + 112), fill="red")
            image.save(source)
            balls = build_balls({"ball_reference": str(source)}, root)
            self.assertEqual(len(balls), 10)
            for ball in balls:
                with Image.open(root / ball["file"]) as rgba:
                    self.assertEqual(rgba.getpixel((113, 113))[:3], (255, 0, 0))

    def test_full_scene_export_preserves_five_classes_for_odd_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_pitch(root, 42)
            assets = asset_plan(42)
            cache = root / "cache"
            (cache / "cutouts").mkdir()
            for i, a in enumerate(assets):
                image = Image.new("RGBA", (64, 128))
                d = ImageDraw.Draw(image)
                color = ((37 * i) % 180 + 50, (63 * i) % 180 + 50, (17 * i) % 180 + 50, 255)
                d.ellipse((22, 4, 42, 24), fill=color)
                d.rectangle((18, 23, 46, 88), fill=color)
                d.rectangle((8, 31, 55, 43), fill=color)
                d.rectangle((18, 86, 28, 123), fill=color)
                d.rectangle((36, 86, 46, 123), fill=color)
                path = cache / "cutouts" / f"{a['id']}.png"
                image.save(path)
                a.update(file=path.relative_to(cache).as_posix(), sha256=sha(path))
            (cache / "balls").mkdir()
            ball = Image.new("RGBA", (64, 64))
            ImageDraw.Draw(ball).ellipse((1, 1, 62, 62), fill="white")
            ball.save(cache / "balls/ball-test.png")
            balls = [
                {
                    "id": "ball-test",
                    "file": "balls/ball-test.png",
                    "style": "white",
                    "sha256": sha(cache / "balls/ball-test.png"),
                }
            ]
            write(cache / "balls.json", {"assets": balls})
            config = {
                "sport": "football",
                "background": str(root / "background.png"),
                "background_sha256": sha(root / "background.png"),
                "output_width": 3200,
            }
            plans = make_plans(9, 67)
            for plan in plans:
                render_scene(config, root, plan, assets, balls, {}, False, time.time() + 120)
            target = export_and_verify(root, config, plans)
            manifest = read(target / "manifest.json")
            self.assertEqual(manifest["images"], 9)
            self.assertEqual(manifest["sport"], "football")
            self.assertEqual(manifest["schema_version"], "3.0")
            self.assertEqual(
                manifest["labels"], ["ball", "player", "goalkeeper", "referee", "goal"]
            )
            self.assertGreater(manifest["counts"]["goal"], 0)
            self.assertGreater(manifest["counts"]["goalkeeper"], 0)
            self.assertTrue(read(target / "verification.json")["passed"])


if __name__ == "__main__":
    unittest.main()
