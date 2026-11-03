import copy
import json
import random
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from just_a_sports_dataset_generator.scenario.planner import RecipeError, load_recipe
from just_a_sports_dataset_generator.sports import equipment
from just_a_sports_dataset_generator.sports.basketball import plan_scene
from just_a_sports_dataset_generator.sports.equipment import (
    EquipmentError,
    catalogue,
    choose_ball,
    choose_referee_kit,
    linear,
    parse,
)

ROOT = Path(__file__).parents[1]
DATA = Path(equipment.__file__).with_name("equipment.json")


class EquipmentTests(unittest.TestCase):
    def test_catalogue_matches_its_schema_and_covers_three_sports(self):
        data = json.loads(DATA.read_text())
        schema = json.loads((ROOT / "schemas/sport-equipment-v1.0.json").read_text())
        Draft202012Validator(schema).validate(data)
        self.assertEqual(set(catalogue()), {"basketball", "football", "futsal"})

    def test_regulation_balls(self):
        sizes = {sport: e.regulation.size for sport, e in catalogue().items()}
        self.assertEqual(sizes, {"basketball": 7, "football": 5, "futsal": 4})
        # The planner's basketball radius lies inside the regulation range.
        self.assertAlmostEqual(catalogue()["basketball"].regulation.radius_m, 0.1216, places=3)
        self.assertEqual(catalogue()["futsal"].regulation.circumference_cm, (62.0, 64.0))
        self.assertTrue(catalogue()["football"].regulation.verified)

    def test_linear_rgb_conversion(self):
        self.assertEqual(linear("#FFFFFF"), (1.0, 1.0, 1.0))
        self.assertEqual(linear("#000000"), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(linear("#808080")[0], 0.2159, places=3)

    def test_referee_kit_differs_from_both_teams(self):
        orange_team, black_team = linear("#F26A1B"), linear("#26272A")
        for seed in range(200):
            kit = choose_referee_kit(random.Random(seed), "basketball", [orange_team, black_team])
            self.assertNotIn(kit.id, {"orange_black_sleeves", "black_neon"})

    def test_malformed_catalogues_are_rejected(self):
        data = json.loads(DATA.read_text())
        broken = []
        bad_color = copy.deepcopy(data)
        bad_color["sports"]["futsal"]["referee"]["kits"][0]["shirt"] = "yellow"
        broken.append(bad_color)
        missing_sport = copy.deepcopy(data)
        del missing_sport["sports"]["football"]
        broken.append(missing_sport)
        bad_range = copy.deepcopy(data)
        bad_range["sports"]["football"]["ball"]["regulation"]["mass_g"] = [450, 410]
        broken.append(bad_range)
        duplicate = copy.deepcopy(data)
        kits = duplicate["sports"]["basketball"]["referee"]["kits"]
        kits.append(copy.deepcopy(kits[0]))
        broken.append(duplicate)
        for case in broken:
            with self.subTest(), self.assertRaises(EquipmentError):
                parse(case)

    def test_basketball_layout_carries_equipment_deterministically(self):
        first, again = plan_scene(7, 1), plan_scene(7, 1)
        self.assertEqual(first, again)
        self.assertEqual(first["schema_version"], "1.2")
        gear = first["equipment"]
        assert isinstance(gear, dict)
        ids = {b.id for b in catalogue()["basketball"].balls}
        self.assertIn(gear["ball"]["design"], ids)
        referees = [a for a in first["athletes"] if a["team"] == 2]
        self.assertTrue(referees)
        for referee in referees:
            self.assertEqual(referee["referee_kit"], gear["referee_kit"])
            self.assertEqual(referee["kit"], gear["referee_kit"]["shirt"])
        seen = {plan_scene(seed, 1)["equipment"]["ball"]["design"] for seed in range(60)}
        self.assertEqual(seen, ids)  # every design appears

    def test_germany_profile_reweights_the_same_looks(self):
        rng = random.Random(3)
        balls = [choose_ball(rng, "basketball", "germany").id for _ in range(400)]
        self.assertGreater(balls.count("orange_ivory"), 250)  # the common DBB ball look
        kits = {choose_referee_kit(rng, "basketball", [], profile="germany").id for _ in range(400)}
        self.assertNotIn("black_white_stripes", kits)  # weight 0 in this profile
        with self.assertRaises(ValueError):
            choose_ball(rng, "basketball", "mars")
        bad = json.loads(DATA.read_text())
        bad["sports"]["futsal"]["profiles"]["germany"]["balls"] = {"no_such_ball": 1}
        with self.assertRaises(EquipmentError):
            parse(bad)

    def test_recipe_1_2_selects_the_profile(self):
        import tempfile

        base = {
            "name": "x",
            "sport": "basketball",
            "seed": 1,
            "scenes": 1,
            "frames_per_scene": 1,
            "resolution": {"width": 64, "height": 48},
            "annotations": ["bounding_boxes"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r.json"
            path.write_text(
                json.dumps({**base, "schema_version": "1.2", "equipment_profile": "germany"})
            )
            self.assertEqual(load_recipe(path).equipment_profile, "germany")
            path.write_text(json.dumps({**base, "schema_version": "1.2"}))
            self.assertEqual(load_recipe(path).equipment_profile, "international")
            for version, profile in (("1.2", "mars"), ("1.1", "germany")):
                path.write_text(
                    json.dumps({**base, "schema_version": version, "equipment_profile": profile})
                )
                with self.subTest(version=version), self.assertRaises(RecipeError):
                    load_recipe(path)
        layout = plan_scene(5, 1, "germany")
        self.assertEqual(layout["equipment"]["profile"], "germany")

    def test_synthetic_halls_have_no_spectators(self):
        for seed in range(40):
            self.assertEqual(plan_scene(seed, 1)["hall"]["spectator_density"], 0.0)


if __name__ == "__main__":
    unittest.main()
