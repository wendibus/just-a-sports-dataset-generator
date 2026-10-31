import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from fixtures import read_json
from jsonschema import Draft202012Validator

from just_a_sports_dataset_generator.cli import main

ROOT = Path(__file__).parents[1]


class SchemaTests(unittest.TestCase):
    def test_render_and_export_files_follow_published_contracts(self):
        schemas = {}
        for path in (ROOT / "schemas").glob("*.json"):
            schema = read_json(path)
            Draft202012Validator.check_schema(schema)
            schemas[path.stem] = Draft202012Validator(schema)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "render-reference",
                            str(ROOT / "examples/basketball.json"),
                            "--output",
                            str(root / "render"),
                            "--max-scenes",
                            "1",
                            "--max-frames",
                            "1",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(["export", str(root / "render"), "--output", str(root / "dataset")]), 0
                )
            cases = {
                "render-run-v1.0": root / "render/run.json",
                "scene-v2.0": root / "render/scene-000000/scene.json",
                "native-annotations-v1.0": root / "dataset/native/scene-000000/annotations.json",
                "export-manifest-v1.0": root / "dataset/manifest.json",
                "export-report-v1.0": root / "dataset/report.json",
            }
            for name, path in cases.items():
                with self.subTest(schema=name):
                    schemas[name].validate(read_json(path))
