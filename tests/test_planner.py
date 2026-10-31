import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

