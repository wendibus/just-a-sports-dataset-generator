import copy
import json
import random
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from just_a_sports_dataset_generator.sports import equipment
from just_a_sports_dataset_generator.sports.basketball import plan_scene
from just_a_sports_dataset_generator.sports.equipment import (
    EquipmentError,
    catalogue,
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


if __name__ == "__main__":
    unittest.main()
