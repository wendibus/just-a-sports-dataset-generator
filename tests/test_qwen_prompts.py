import importlib.util
import unittest
from pathlib import Path

from PIL import Image

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
        self.assertIn("sleeveless basketball jersey", prompt)
        self.assertIn("colours of image 1", prompt)
        self.assertIn("knees deeply bent", prompt)
        self.assertIn("not a 3D render", prompt)
        self.assertNotIn("image 2", prompt)  # no skeleton image by default
        self.assertIn("skeleton in image 2", worker.build_prompt({"label": "player"}, True))
        # Words that produced overall straps (test 1) and shoulder wings (test 2).
        self.assertNotIn("strap", prompt.lower())
        self.assertNotIn("shoulder panel", prompt.lower())

    def test_referee_and_unknown_families(self):
        referee = worker.build_prompt({"label": "referee", "pose_family": "signal"})
        self.assertIn("referee shirt", referee)
        self.assertNotIn("sleeveless", referee)
        self.assertIn("raising one arm", referee)
        plain = worker.build_prompt({"label": "player", "pose_family": None})
        self.assertNotIn("exactly as in image 1. ", plain)
        self.assertEqual(
            set(worker.POSE_HINTS),
            {"stand", "run", "defend", "dribble", "shoot", "pass", "catch", "rebound", "signal"},
        )

    def test_negative_prompt_targets_the_observed_failures(self):
        for word in ("straps", "buckles", "overalls", "shoulder pads", "standing upright"):
            self.assertIn(word, worker.NEGATIVE_PROMPT)

    def test_isolate_keeps_the_person_and_blurs_the_rest(self):
        rgb = Image.new("RGB", (96, 96), (0, 0, 0))
        rgb.paste((255, 255, 255), (60, 0, 96, 96))  # another player on the right
        rgb.paste((200, 20, 20), (20, 20, 40, 80))  # the person
        mask = Image.new("L", (96, 96), 0)
        mask.paste(255, (20, 20, 40, 80))
        out = worker.isolate(rgb, mask)
        self.assertEqual(out.getpixel((30, 50)), (200, 20, 20))
        edge = out.getpixel((60, 50))
        self.assertTrue(30 < edge[0] < 225, edge)  # the other player's edge is smeared

    def test_inpaint_prompt_keeps_the_render_and_forbids_lettering(self):
        prompt = worker.inpaint_prompt({"label": "player", "pose_family": "run"})
        self.assertIn("Keep exactly its pose, outline", prompt)
        self.assertIn("no other lettering", prompt)
        self.assertIn("running mid-stride", prompt)
        self.assertIn("referee shirt", worker.inpaint_prompt({"label": "referee"}))
        named = worker.inpaint_prompt({"label": "player", "kit": "white"})
        self.assertIn("a white sleeveless basketball jersey and white shorts", named)
        kit = "a plain grey short-sleeved shirt with black sleeves and black long trousers"
        self.assertIn(kit, worker.inpaint_prompt({"label": "referee", "kit": kit}))

    def test_condition_size_follows_the_work_resolution(self):
        worker.set_condition_size(512)  # no diffusers here: must be a no-op, not an error
        worker.set_inpaint_area(512)


if __name__ == "__main__":
    unittest.main()
