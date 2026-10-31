import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from just_a_sports_dataset_generator.plugins import PluginCompatibilityError, PluginMetadata
from just_a_sports_dataset_generator.renderers import ReferenceRenderer
from just_a_sports_dataset_generator.scenario import load_recipe, plan_jobs


class ReferenceRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        recipe = load_recipe(Path(__file__).parents[1] / "examples" / "basketball.json")
        self.job = plan_jobs(recipe)[0]

    def test_renderer_writes_rgb_instance_ids_and_reco_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = ReferenceRenderer().render(self.job, Path(directory), max_frames=1)
            manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertEqual(result.frames, 1)
            self.assertEqual(manifest["labels"], ["ball", "player", "referee", "hoop"])
            self.assertTrue((result.output_directory / manifest["frames"][0]["rgb"]).exists())
            self.assertTrue(
                (result.output_directory / manifest["frames"][0]["instance_ids"]).exists()
            )
            self.assertEqual(
                [item["class_id"] for item in manifest["frames"][0]["objects"]],
                [0, 1, 2, 3],
            )

    def test_camera_presets_are_rejected_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no camera presets"):
                ReferenceRenderer().render(
                    replace(self.job, camera="reco_panorama"), Path(directory)
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_output_is_byte_for_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = ReferenceRenderer().render(self.job, Path(first), max_frames=1)
            two = ReferenceRenderer().render(self.job, Path(second), max_frames=1)
            for name in ("frame-000000.ppm", "frame-000000.instances.pgm", "scene.json"):
                first_hash = hashlib.sha256((one.output_directory / name).read_bytes()).digest()
                second_hash = hashlib.sha256((two.output_directory / name).read_bytes()).digest()
                self.assertEqual(first_hash, second_hash)

    def test_incompatible_plugin_api_fails_before_use(self) -> None:
        metadata = PluginMetadata("example.renderer", "1.0.0", 999, "renderer", ("rgb",))
        with self.assertRaises(PluginCompatibilityError):
            metadata.validate()


if __name__ == "__main__":
    unittest.main()
