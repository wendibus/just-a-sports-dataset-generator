import importlib.util
import unittest
from pathlib import Path

from just_a_sports_dataset_generator.generators import qwen

WORKER = Path(qwen.__file__).with_name("qwen_worker.py")
spec = importlib.util.spec_from_file_location("qwen_worker", WORKER)
assert spec is not None and spec.loader is not None
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)  # stdlib imports only at module level; torch loads in main()


class PromptTests(unittest.TestCase):
    def test_host_and_worker_agree_on_the_prompt_version(self):
        self.assertEqual(qwen.PROMPT_VERSION, worker.PROMPT_VERSION)

    def test_player_prompt_describes_a_real_jersey_and_the_pose(self):
        prompt = worker.build_prompt({"label": "player", "pose_family": "defend"})
        self.assertIn("tank-top jersey", prompt)
        self.assertIn("shoulder panels", prompt)
        self.assertIn("colours of image 1", prompt)
        self.assertIn("knees deeply bent", prompt)
        self.assertIn("skeleton in image 2", prompt)
        # The word that produced overall straps in the first Mac test must not reappear.
        self.assertNotIn("strap", prompt.lower())

    def test_referee_and_unknown_families(self):
        referee = worker.build_prompt({"label": "referee", "pose_family": "signal"})
        self.assertIn("referee shirt", referee)
        self.assertNotIn("tank-top", referee)
        self.assertIn("raising one arm", referee)
        plain = worker.build_prompt({"label": "player", "pose_family": None})
        self.assertNotIn("exactly as in image 1. ", plain)
        self.assertEqual(
            set(worker.POSE_HINTS),
            {"stand", "run", "defend", "dribble", "shoot", "pass", "catch", "rebound", "signal"},
        )

    def test_negative_prompt_targets_the_observed_failures(self):
        for word in ("straps", "buckles", "overalls", "backpack", "standing upright"):
            self.assertIn(word, worker.NEGATIVE_PROMPT)


if __name__ == "__main__":
    unittest.main()
