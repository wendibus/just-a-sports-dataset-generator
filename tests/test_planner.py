import json
import tempfile
import unittest
from pathlib import Path

from just_a_sports_dataset_generator.core import SPORTS, labels_for_sport
from just_a_sports_dataset_generator.scenario import RecipeError, load_recipe, plan_jobs


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe_path = Path(__file__).parents[1] / "examples" / "basketball.json"

    def test_example_is_valid_and_plans_requested_scenes(self) -> None:
        recipe = load_recipe(self.recipe_path)
        jobs = plan_jobs(recipe)
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].job_id, "scene-000000")
        self.assertEqual(jobs[2].frame_count, 12)
        self.assertEqual(jobs[0].labels, ("ball", "player", "referee", "hoop"))

    def test_planning_is_deterministic(self) -> None:
        recipe = load_recipe(self.recipe_path)
        self.assertEqual(plan_jobs(recipe), plan_jobs(recipe))
        self.assertNotEqual(plan_jobs(recipe)[0].seed, plan_jobs(recipe)[1].seed)

    def test_unknown_fields_fail_closed(self) -> None:
        data = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        data["surprise"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RecipeError, "unknown fields"):
                load_recipe(path)

    def test_unsupported_annotations_are_rejected(self) -> None:
        data = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        data["annotations"] = ["magic"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RecipeError, "unsupported annotations"):
                load_recipe(path)

    def _write(self, directory: str, data: dict[str, object]) -> Path:
        path = Path(directory) / "recipe.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_reco_panorama_camera_reaches_every_job(self) -> None:
        recipe = load_recipe(self.recipe_path.with_name("basketball-reco.json"))
        self.assertEqual(recipe.schema_version, "1.1")
        self.assertEqual(recipe.camera, "reco_panorama")
        self.assertTrue(all(job.camera == "reco_panorama" for job in plan_jobs(recipe)))

    def test_camera_defaults_to_broadcast(self) -> None:
        self.assertEqual(load_recipe(self.recipe_path).camera, "broadcast")
        data = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        data["schema_version"] = "1.1"
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_recipe(self._write(directory, data)).camera, "broadcast")

    def test_camera_requires_schema_1_1_and_a_known_preset(self) -> None:
        data = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        data["camera"] = "reco_panorama"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RecipeError, "unknown fields: camera"):
                load_recipe(self._write(directory, data))
            data["schema_version"] = "1.1"
            data["camera"] = "drone"
            with self.assertRaisesRegex(RecipeError, "camera must be one of"):
                load_recipe(self._write(directory, data))
            data["schema_version"] = "2.0"
            with self.assertRaisesRegex(RecipeError, "unsupported schema_version"):
                load_recipe(self._write(directory, data))

    def test_taxonomy_matches_reco_trainer_class_order(self) -> None:
        expected = {
            "basketball": ("ball", "player", "referee", "hoop"),
            "football": ("ball", "player", "goalkeeper", "referee", "goal"),
            "futsal": ("ball", "player", "goalkeeper", "referee", "goal"),
            "handball": ("ball", "player", "goalkeeper", "referee", "goal"),
            "hockey": ("puck", "player", "goalkeeper", "referee", "goal"),
            "rugby": ("ball", "player", "referee", "goalpost"),
            "lacrosse": ("ball", "player", "goalkeeper", "referee", "goal"),
            "american_football": ("ball", "player", "referee", "goalpost"),
        }
        self.assertEqual(set(SPORTS), set(expected))
        for sport, labels in expected.items():
            self.assertEqual(labels_for_sport(sport), labels)

    def test_unknown_sport_is_rejected(self) -> None:
        data = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        data["sport"] = "chess_boxing"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RecipeError, "Reco Trainer sport"):
                load_recipe(path)


if __name__ == "__main__":
    unittest.main()
