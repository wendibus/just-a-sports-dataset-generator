import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hall_night.assets import asset_plan
from hall_night.common import read, sha, write
from hall_night.export import export_and_verify, verify
from hall_night.render import Layer, make_plans, pixels_per_metre, rasterize, render_scene

from just_a_sports_dataset_generator.annotations import ObjectIdentity


class HallPlanningTests(unittest.TestCase):
    def test_2000_images_have_exact_approved_balances_and_stable_groups(self):
        plans = make_plans(2000, 42)
        self.assertEqual(plans, make_plans(2000, 42))
        self.assertEqual(len(plans), 1000)
        self.assertEqual(
            Counter(p["split"] for p in plans), {"train": 800, "val": 100, "test": 100}
        )
        self.assertEqual(
            Counter(p["gender_mode"] for p in plans), {"men": 400, "women": 400, "mixed": 200}
        )
        self.assertEqual(
            Counter(p["ball_mode"] for p in plans), {"single": 700, "multi": 200, "none": 100}
        )
        self.assertEqual(len({p["seed"] for p in plans}), 1000)
        self.assertEqual(len({p["id"] for p in plans}), 1000)
        for invalid in (0, 1, 3, -2, True):
            with self.assertRaises(ValueError):
                make_plans(invalid, 1)
        with self.assertRaises(ValueError):
            make_plans(100, 1, {"men": 0.5, "women": 0.6, "mixed": 0})

    def test_disjoint_asset_strata_cover_each_gender_kit_and_split(self):
        assets = asset_plan(42)
        self.assertEqual(len(assets), 156)
        self.assertEqual(len({a["id"] for a in assets}), 156)
        counts = Counter(
            (a["gender"], a["kit"], a["split"]) for a in assets if a["label"] == "player"
        )
        for gender in ("man", "woman"):
            for kit in ("red", "yellow"):
                for split, n in (("train", 24), ("val", 6), ("test", 6)):
                    self.assertEqual(counts[gender, kit, split], n)
        self.assertEqual(len({a["skin"] for a in assets}), 3)
        self.assertTrue(all(a["height_m"] >= 1.58 for a in assets))
        self.assertGreater(pixels_per_metre((1600, 800)), pixels_per_metre((1600, 550)))
        self.assertGreater(pixels_per_metre((2700, 750)), pixels_per_metre((1600, 550)))

    def test_a_ball_behind_a_player_has_only_visible_pixels(self):
        background = Image.new("RGB", (3200, 900), "white")
        ball = Image.new("L", (30, 30))
        ImageDraw.Draw(ball).ellipse((0, 0, 29, 29), fill=255)
        body = Image.new("L", (20, 60), 255)
        layers = [
            Layer(
                ObjectIdentity(1, 0, "ball"),
                Image.new("RGB", (30, 30), "orange"),
                ball,
                (1000, 600),
                0.4,
                (1015, 630),
                "ball",
                {},
            ),
            Layer(
                ObjectIdentity(10, 1, "player"),
                Image.new("RGB", (20, 60), "red"),
                body,
                (1005, 580),
                0.5,
                (1015, 640),
                "person",
                {},
            ),
        ]
        rgb, ids, ratios = rasterize(background, layers)
        self.assertGreater(ratios[1], 0)
        self.assertLess(ratios[1], 1)
        self.assertEqual(ids.getpixel((1010, 610)), 10)
        self.assertEqual(rgb.getpixel((1010, 610)), (255, 0, 0))
        self.assertEqual(ids.getpixel((1028, 615)), 1)


def fixture(root):
    cache = root / "cache"
    (cache / "cutouts").mkdir(parents=True)
    (cache / "balls").mkdir()
    assets = asset_plan(123)
    for i, asset in enumerate(assets):
        image = Image.new("RGBA", (64, 128))
        draw = ImageDraw.Draw(image)
        color = ((37 * i) % 180 + 50, (63 * i) % 180 + 50, (17 * i) % 180 + 50, 255)
        draw.ellipse((22, 4, 42, 24), fill=color)
        draw.rectangle((18, 23, 46, 88), fill=color)
        draw.rectangle((8, 31, 55, 43), fill=color)
        draw.rectangle((18, 86, 28, 123), fill=color)
        draw.rectangle((36, 86, 46, 123), fill=color)
        # Make RGB distinct even when exposure/crop is otherwise the same.
        draw.line((20, 25, 44, 86), fill=(i, 255 - i, 80, 255), width=3)
        path = cache / "cutouts" / f"{asset['id']}.png"
        image.save(path)
        asset.update(file=str(path.relative_to(cache)), sha256=sha(path))
    ball = Image.new("RGBA", (64, 64))
    ImageDraw.Draw(ball).ellipse((1, 1, 62, 62), fill=(220, 100, 20, 255))
    ball.save(cache / "balls/ball-fixture.png")
    balls = [
        {
            "id": "ball-fixture",
            "file": "balls/ball-fixture.png",
            "sha256": sha(cache / "balls/ball-fixture.png"),
            "style": "orange",
        }
    ]
    write(cache / "balls.json", {"assets": balls})
    background = Image.new("RGB", (3200, 900), (135, 145, 150))
    ImageDraw.Draw(background).rectangle((0, 450, 3199, 899), fill=(168, 122, 70))
    background.save(root / "background.png")
    config = {
        "background": str(root / "background.png"),
        "background_sha256": sha(root / "background.png"),
        "output_width": 1000,
    }
    return config, assets, balls


class HallEndToEndTests(unittest.TestCase):
    def test_mask_export_splits_occlusions_and_resume_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, assets, balls = fixture(root)
            plans = make_plans(20, 123)
            # Six images exercise all visibility buckets in one fast fixture.
            for i, plan in enumerate(plans):
                plan.update(ball_mode=("single", "multi", "none")[i % 3], occlude_ball=i % 3 < 2)
            for plan in plans:
                render_scene(config, root, plan, assets, balls, {}, False, time.time() + 120)
            target = export_and_verify(root, config, plans)
            manifest = read(target / "manifest.json")
            self.assertEqual(manifest["images"], 20)
            self.assertEqual(manifest["ball_modes"], {"single": 8, "multi": 6, "none": 6})
            self.assertTrue(read(target / "verification.json")["passed"])
            self.assertGreater(manifest["images_with_partially_occluded_people"], 0)
            self.assertGreater(manifest["images_with_partially_occluded_balls"], 0)
            for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
                self.assertFalse(
                    set(manifest["person_asset_ids_by_split"][first])
                    & set(manifest["person_asset_ids_by_split"][second])
                )
            before = (root / "renders" / plans[0]["id"] / "done.json").read_bytes()
            render_scene(config, root, plans[0], assets, balls, {}, False, time.time() + 10)
            self.assertEqual(before, (root / "renders" / plans[0]["id"] / "done.json").read_bytes())
            expired = dict(plans[0], id="hall-expired")
            with self.assertRaises(TimeoutError):
                render_scene(config, root, expired, assets, balls, {}, False, 0)
            self.assertFalse((root / "renders/hall-expired").exists())
            # A modified mask must never be accepted as a valid resumed/exported dataset.
            native = next((target / "native").glob("*.png"))
            with Image.open(native) as raw:
                changed = raw.copy()
            changed.putpixel((0, 0), 254)
            changed.save(native)
            with self.assertRaises((ValueError, AssertionError)):
                verify(target)
            scene = root / "renders" / plans[0]["id"] / "frame-000.png"
            scene.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                render_scene(config, root, plans[0], assets, balls, {}, False, time.time() + 10)


if __name__ == "__main__":
    unittest.main()
