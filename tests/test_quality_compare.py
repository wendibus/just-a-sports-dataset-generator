import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator
from PIL import Image
from test_composite import CROP_ID, RED, setup

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "quality_compare", ROOT / "scripts/qwen_quality_compare.py"
)
assert spec and spec.loader
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ComparisonTests(unittest.TestCase):
    def test_comparison_writes_review_page_and_refuses_changed_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, crops = setup(root)
            output = root / "comparison"
            review_python = root / "review-python"
            review_python.symlink_to(sys.executable)
            (root / "model").mkdir()
            (root / "model/model.safetensors").write_bytes(b"stand-in")

            def generate(source, destination, **options):
                self.assertEqual(options["seed"], 0)
                self.assertEqual(options["resolution"], 512)
                self.assertFalse(options["confirm"](None))
                destination.mkdir(exist_ok=True)
                with Image.open(crops / f"scene-000000/{CROP_ID}.rgb.png") as raw:
                    image = raw.convert("RGB")
                with Image.open(crops / f"scene-000000/{CROP_ID}.mask.png") as mask:
                    image.paste(RED, (0, 0, *image.size), mask)
                image.save(destination / f"{CROP_ID}.png")

            def silhouettes(command, **options):
                self.assertEqual(command[0], str(review_python.absolute()))
                masks = Path(command[command.index("--output") + 1])
                masks.mkdir(exist_ok=True)
                with Image.open(crops / f"scene-000000/{CROP_ID}.mask.png") as mask:
                    mask.save(masks / f"{CROP_ID}.png")

            argv = [
                "compare",
                str(root),
                "--output",
                str(output),
                "--ids",
                CROP_ID,
                "--strengths",
                ".75",
                "--review-python",
                str(review_python),
                "--foreground-model",
                str(root / "model"),
            ]
            with (
                patch.object(compare, "generate_crops", side_effect=generate) as generator,
                patch.object(compare.subprocess, "run", side_effect=silhouettes),
                patch("sys.argv", argv),
            ):
                compare.main()
                report = json.loads((output / "quality-report.json").read_text())
                schema = json.loads((ROOT / "schemas/quality-review-report-v1.0.json").read_text())
                Draft202012Validator(schema).validate(report)
                self.assertEqual(report["variants"][0]["items"][0]["status"], "needs_visual_review")
                self.assertTrue((output / "index.html").is_file())
                compare.main()  # Same plan may resume.
                self.assertEqual(generator.call_count, 2)
                with (
                    patch("sys.argv", [*argv[:-1], str(root / "model"), "--strengths", ".85"]),
                    self.assertRaisesRegex(ValueError, "different comparison"),
                ):
                    compare.main()


if __name__ == "__main__":
    unittest.main()
