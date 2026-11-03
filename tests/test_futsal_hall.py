import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hall_night.common import read, sha, write
from hall_night.export import export_and_verify
from hall_night.futsal import asset_plan, choose_roster, make_plans, prompt_for
from hall_night.render import Layer, rasterize, render_scene

from just_a_sports_dataset_generator.annotations import ObjectIdentity


class FutsalPlanningTests(unittest.TestCase):
    def test_exact_10009_frame_balances_and_group_integrity(self):
        plans = make_plans(10009, 42)
        self.assertEqual(plans, make_plans(10009, 42))
        self.assertEqual(sum(p["frame_count"] for p in plans), 10009)
        for field, expected in [
            ("gender_mode", {"men": 4004, "women": 4003, "mixed": 2002}),
            ("ball_mode", {"single": 7006, "multi": 2002, "none": 1001}),
            ("split", {"train": 8007, "val": 1001, "test": 1001}),
        ]:
            counts = Counter()
            for p in plans:
                counts[p[field]] += p["frame_count"]
            self.assertEqual(counts, expected)
        self.assertEqual(len({p["seed"] for p in plans}), len(plans))
        self.assertTrue(all(p["sport"] == "futsal" and p["frame_count"] in (1, 2) for p in plans))
        for invalid in (0, 1, -2, True):
            with self.assertRaises(ValueError):
                make_plans(invalid, 1)

    def test_rosters_include_separate_goalkeepers_and_football_kits(self):
        import random

        assets = asset_plan(42)
        self.assertEqual(
            Counter(a["label"] for a in assets), {"player": 144, "goalkeeper": 24, "referee": 24}
        )
        self.assertEqual(len({a["id"] for a in assets}), 192)
        for split in ("train", "val", "test"):
            for mode in ("men", "women", "mixed"):
                roster = choose_roster(
                    assets, {"split": split, "gender_mode": mode}, random.Random(42), False
                )
                counts = Counter(a["label"] for a in roster)
                self.assertEqual(counts["player"], 8)
                self.assertEqual(counts["goalkeeper"], 2)
                self.assertGreaterEqual(counts["referee"], 1)
                self.assertTrue(all(a["split"] == split for a in roster))
                self.assertEqual(len({a["id"] for a in roster}), len(roster))
        referee = next(a for a in assets if a["label"] == "referee")
        self.assertIn("BLACK SHORTS", prompt_for(referee))
        self.assertIn(
            "LONG-SLEEVED", prompt_for(next(a for a in assets if a["label"] == "goalkeeper"))
        )

    def test_net_holes_keep_background_and_players_occlude_posts(self):
        bg = Image.new("RGB", (100, 100), "grey")
        net = Image.new("L", (30, 40))
        ImageDraw.Draw(net).rectangle((0, 0, 29, 39), outline=255, width=2)
        layers = [
            Layer(
                ObjectIdentity(200, 4, "goal"),
                Image.new("RGB", net.size, "white"),
                net,
                (20, 20),
                0.5,
                (35, 60),
                "goal-left",
                {},
            ),
            Layer(
                ObjectIdentity(30, 2, "goalkeeper"),
                Image.new("RGB", (5, 20), "green"),
                Image.new("L", (5, 20), 255),
                (19, 30),
                0.6,
                (21, 50),
                "keeper",
                {},
            ),
        ]
        _, ids, _ = rasterize(bg, layers, "futsal")
        self.assertEqual(ids.getpixel((30, 35)), 0)
        self.assertEqual(ids.getpixel((20, 35)), 30)
        self.assertEqual(ids.getpixel((20, 25)), 200)
        self.assertEqual(set(ids.getdata()) - {0}, {30, 200})


def fixture(root):
    cache = root / "cache"
    (cache / "cutouts").mkdir(parents=True)
    (cache / "balls").mkdir()
    (cache / "goals").mkdir()
    assets = asset_plan(42)
    for i, a in enumerate(assets):
        image = Image.new("RGBA", (64, 128))
        d = ImageDraw.Draw(image)
        color = ((37 * i) % 180 + 50, (63 * i) % 180 + 50, (17 * i) % 180 + 50, 255)
        d.ellipse((22, 4, 42, 24), fill=color)
        d.rectangle((18, 23, 46, 88), fill=color)
        d.rectangle((8, 31, 55, 43), fill=color)
        d.rectangle((18, 86, 28, 123), fill=color)
        d.rectangle((36, 86, 46, 123), fill=color)
        d.line((20, 25, 44, 86), fill=(i, 255 - i, 80, 255), width=3)
        path = cache / "cutouts" / f"{a['id']}.png"
        image.save(path)
        a.update(file=path.relative_to(cache).as_posix(), sha256=sha(path))
    image = Image.new("RGBA", (64, 64))
    ImageDraw.Draw(image).ellipse((1, 1, 62, 62), fill="white")
    path = cache / "balls/ball-fixture.png"
    image.save(path)
    balls = [
        {
            "id": "ball-fixture",
            "file": "balls/ball-fixture.png",
            "sha256": sha(path),
            "style": "white-pattern",
        }
    ]
    goals = []
    for side in ("left", "right"):
        image = Image.new("RGBA", (240, 160))
        d = ImageDraw.Draw(image)
        d.rectangle((2, 2, 237, 157), outline="white", width=5)
        for x in range(5, 238, 12):
            d.line((x, 5, x, 155), fill=(220, 220, 220, 255), width=1)
        for y in range(5, 157, 12):
            d.line((5, y, 235, y), fill=(220, 220, 220, 255), width=1)
        path = cache / "goals" / f"goal-{side}.png"
        image.save(path)
        goals.append(
            {"id": f"goal-{side}", "file": path.relative_to(cache).as_posix(), "sha256": sha(path)}
        )
    write(cache / "goals.json", {"assets": goals})
    write(cache / "balls.json", {"assets": balls})
    bg = Image.new("RGB", (3200, 900), (135, 145, 150))
    ImageDraw.Draw(bg).rectangle((0, 450, 3199, 899), fill=(168, 122, 70))
    bg.save(root / "background.png")
    return (
        {
            "sport": "futsal",
            "background": str(root / "background.png"),
            "background_sha256": sha(root / "background.png"),
            "output_width": 1000,
        },
        assets,
        balls,
    )


class FutsalEndToEndTests(unittest.TestCase):
    def test_odd_export_all_five_classes_and_no_identity_leaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, assets, balls = fixture(root)
            plans = make_plans(21, 123)
            for plan in plans:
                render_scene(config, root, plan, assets, balls, {}, False, time.time() + 120)
            target = export_and_verify(root, config, plans)
            manifest = read(target / "manifest.json")
            self.assertEqual(manifest["images"], 21)
            self.assertEqual(
                manifest["labels"], ["ball", "player", "goalkeeper", "referee", "goal"]
            )
            self.assertGreater(manifest["counts"]["goalkeeper"], 0)
            self.assertGreater(manifest["counts"]["goal"], 0)
            self.assertNotIn("hoop", manifest["counts"])
            self.assertTrue(read(target / "verification.json")["passed"])
            self.assertTrue(
                all(
                    not any(a.startswith("goal-") for a in ids)
                    for ids in manifest["person_asset_ids_by_split"].values()
                )
            )
            group = root / "renders" / plans[0]["id"]
            before = (group / "done.json").read_bytes()
            render_scene(config, root, plans[0], assets, balls, {}, False, time.time() + 1)
            self.assertEqual(before, (group / "done.json").read_bytes())


class FutsalResourceTests(unittest.TestCase):
    def test_wait_checks_queue_before_starting(self):
        from unittest.mock import patch

        from hall_night.resources import wait_for_other_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("hall_night.resources.active_runs", side_effect=[["basketball"], []]),
                patch("hall_night.resources.time.sleep") as sleep,
            ):
                wait_for_other_runs(root)
                sleep.assert_called_once_with(30)

    def test_deferred_background_publish_preserves_original_on_failure(self):
        from unittest.mock import patch

        from hall_night.resources import finish_background

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "background.png"
            Image.new("RGB", (12, 12), "grey").save(original)
            original_sha = sha(original)
            write(root / "prepared.json", {"engine_sha256": {}})
            config = {
                "background": str(original),
                "background_sha256": original_sha,
                "background_cleanup": True,
                "model_hashes": {},
                "model": {},
                "generator_root": str(root),
                "count": 10009,
                "qwen_python": sys.executable,
            }

            def succeed(command, log):
                self.assertEqual(sha(original), original_sha)
                path = root / "background-preparation/background.png"
                path.parent.mkdir(exist_ok=True)
                Image.new("RGB", (12, 12), "white").save(path)

            with patch("hall_night.resources.validate"):
                with self.assertRaises(RuntimeError):
                    finish_background(
                        root,
                        config,
                        lambda *a: (_ for _ in ()).throw(RuntimeError("interrupted")),
                        lambda *a: None,
                    )
                self.assertTrue(config["background_cleanup"])
                self.assertEqual(sha(original), original_sha)
                finish_background(root, config, succeed, lambda *a: None)
            self.assertFalse(read(root / "config.json")["background_cleanup"])
            self.assertEqual(
                Path(config["background"]), root / "background-preparation/background.png"
            )
            self.assertEqual(sha(original), original_sha)

    @unittest.skipIf(sys.platform == "win32", "POSIX flock regression")
    def test_active_sibling_is_detected_but_stale_lock_is_not(self):
        import fcntl

        from hall_night.resources import active_runs

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            sibling = parent / "basketball"
            sibling.mkdir()
            write(sibling / "config.json", {})
            with (sibling / ".night.lock").open("w") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(active_runs(parent / "futsal"), ["basketball"])
                self.assertEqual(active_runs(sibling), [])
            self.assertEqual(active_runs(parent / "futsal"), [])


if __name__ == "__main__":
    unittest.main()
