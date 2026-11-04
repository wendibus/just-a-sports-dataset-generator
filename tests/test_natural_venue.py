import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hall_night.common import read, sha, write
from hall_night.football_pitch import project
from hall_night.natural_venue import assert_unstarted, prepare_layers
from hall_night.render import Layer, rasterize
from upgrade_football_venue import install

from just_a_sports_dataset_generator.annotations import ObjectIdentity


def render_fixture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    empty = Image.new("RGB", (3200, 900), (65, 120, 70))
    full = empty.copy()
    ids = Image.new("L", full.size)
    for i, x in ((200, 500), (201, 2600)):
        ImageDraw.Draw(full).rectangle((x, 500, x + 100, 570), outline=(220, 225, 215), width=3)
        ImageDraw.Draw(ids).rectangle((x, 500, x + 100, 570), outline=i, width=3)
    # Physical shadow is background and must survive removal of goal support.
    ImageDraw.Draw(full).rectangle((500, 575, 600, 580), fill=(35, 70, 40))
    empty.save(folder / "without-goals.png")
    full.save(folder / "with-goals.png")
    ids.save(folder / "goal-ids.pgm")
    write(
        folder / "camera-check.json",
        [
            {"world": list(p), "pixel": list(project(p))}
            for p in ((0, 0, 0), (105, 68, 0), (52.5, 34, 0))
        ],
    )


class NaturalVenueTests(unittest.TestCase):
    def test_render_layers_reconstruct_exactly_and_keep_physical_shadows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_fixture(root / "render")
            assets = prepare_layers(root / "render", root / "out")
            self.assertEqual(len(assets), 2)
            self.assertTrue(all(a["shadow_baked"] for a in assets))
            self.assertTrue(read(root / "out/checks.json")["exact_rgb_reconstruction"])
            with Image.open(root / "out/background.png") as bg:
                self.assertEqual(bg.getpixel((550, 578)), (35, 70, 40))
                self.assertEqual(bg.getpixel((500, 500)), (65, 120, 70))
            for a in assets:
                self.assertEqual(sha(root / "out" / a["file"]), a["sha256"])
                with Image.open(root / "out" / a["file"]) as rgba:
                    self.assertEqual(rgba.getchannel("A").getpixel((50, 30)), 0)

    def test_camera_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_fixture(root / "render")
            data = read(root / "render/camera-check.json")
            data[0]["pixel"][0] += 1
            write(root / "render/camera-check.json", data)
            with self.assertRaisesRegex(ValueError, "camera"):
                prepare_layers(root / "render", root / "out")

    def test_rejected_venue_cannot_pass_preflight(self):
        from run_hall_night import preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "venue-rejected.json", {"status": "rejected_by_user"})
            with self.assertRaisesRegex(ValueError, "Venue rejected"):
                preflight(root, {})

    def test_started_roots_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assert_unstarted(root)
            (root / "run-fingerprint.json").write_text("{}")
            with self.assertRaises(ValueError):
                assert_unstarted(root)

    def test_baked_goal_shadows_are_not_drawn_twice(self):
        bg = Image.new("RGB", (100, 100), "white")
        alpha = Image.new("L", (20, 30), 255)
        layer = Layer(
            ObjectIdentity(200, 4, "goal"),
            Image.new("RGB", alpha.size, "grey"),
            alpha,
            (40, 40),
            0.5,
            (50, 70),
            "goal",
            {"shadow_baked": True},
        )
        rgb, ids, _ = rasterize(bg, [layer], "football")
        self.assertEqual(rgb.getpixel((35, 70)), (255, 255, 255))
        self.assertEqual(set(np.unique(np.asarray(ids))), {0, 200})

    def test_install_rolls_back_config_engine_and_goal_manifest_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            render = Path(tmp) / "render"
            render_fixture(render)
            (root / "engine").mkdir()
            (root / "engine/original.txt").write_text("old engine")
            (root / "cache").mkdir()
            (root / "Referenzen.html").write_text("<main>old references</main>")
            write(root / "cache/goals.json", {"original": True})
            config = {
                "schema_version": "3.0",
                "sport": "football",
                "seed": 1,
                "count": 10009,
                "max_hours": 14,
                "resolution": 640,
                "background": str(root / "background.png"),
                "background_sha256": "0" * 64,
                "source_crops": "crops",
                "ball_reference": "balls",
                "referee_reference": "refs",
                "qwen_python": "python",
                "matte_python": "python",
                "matte_model": "matte",
                "model": {"base_dir": "base", "gguf": "model", "lora": "lora"},
                "generator_root": str(root / "engine"),
                "output_width": 3200,
                "gender_weights": {"men": 0.4, "women": 0.4, "mixed": 0.2},
                "ball_weights": {"single": 0.7, "multi": 0.2, "none": 0.1},
                "goals": str(root / "cache/goals.json"),
                "background_cleanup": False,
            }
            write(root / "config.json", config)
            write(
                root / "prepared.json",
                {"engine_sha256": {"original.txt": sha(root / "engine/original.txt")}},
            )
            write(root / "camera.json", {"principal_point_px": [1600, 520]})
            assets = Path(tmp) / "assets"
            assets.mkdir()
            write(assets / "sources.json", {"assets": []})

            def fake_snapshot(repo, target):
                target.mkdir()
                (target / "new.txt").write_text("new")

            with (
                patch("upgrade_football_venue.snapshot", fake_snapshot),
                patch(
                    "upgrade_football_venue.dashboard", side_effect=RuntimeError("publish failed")
                ),
                self.assertRaisesRegex(RuntimeError, "publish failed"),
            ):
                install(root, render, assets)
            self.assertEqual(read(root / "config.json"), config)
            self.assertEqual((root / "engine/original.txt").read_text(), "old engine")
            self.assertEqual(read(root / "cache/goals.json"), {"original": True})
            self.assertFalse((root / "venue-natural").exists())
            self.assertFalse((root / "cache/goals-natural").exists())
            self.assertFalse((root / "Rasenplatz-fotorealistisch.png").exists())


if __name__ == "__main__":
    unittest.main()
