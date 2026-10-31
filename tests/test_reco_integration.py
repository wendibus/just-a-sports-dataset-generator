"""Run the real, locally checked-out Reco Trainer importer; no mock or network."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fixtures import fixture, read_json
from PIL import Image

from just_a_sports_dataset_generator.core import SPORTS, labels_for_sport
from just_a_sports_dataset_generator.core.taxonomy import RECO_REVISION
from just_a_sports_dataset_generator.exporters import export_dataset
from just_a_sports_dataset_generator.storage import sha256_file, write_json

RECO_PATH = os.environ.get("RECO_TRAINER_PATH")


@unittest.skipUnless(RECO_PATH, "set RECO_TRAINER_PATH to a local Reco Trainer checkout")
class RecoImporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkout = Path(RECO_PATH).resolve()
        cls.worker_path = cls.checkout / "cross-platform/ml_worker.py"
        if not cls.worker_path.is_file():
            raise AssertionError("RECO_TRAINER_PATH must contain cross-platform/ml_worker.py")
        revision = subprocess.run(
            ["git", "-C", str(cls.checkout), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout.strip()
        if revision != RECO_REVISION:
            raise AssertionError(
                f"Update the compatibility baseline deliberately: {revision} != {RECO_REVISION}"
            )
        spec = importlib.util.spec_from_file_location("reco_compatibility_worker", cls.worker_path)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def invoke(self, source, project, format_name):
        result = subprocess.run(
            [
                sys.executable,
                str(self.worker_path),
                "import-labels",
                "--project",
                str(project),
                "--source",
                str(source),
                "--format",
                format_name,
                "--language",
                "en",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_taxonomy_matches_upstream_function(self):
        for sport in SPORTS:
            with self.subTest(sport=sport):
                self.assertEqual(
                    tuple(self.worker.sport_categories(sport)), labels_for_sport(sport)
                )

    def test_all_sports_both_importers_preserve_images_boxes_and_classes(self):
        for sport in SPORTS:
            for format_name in ("coco", "yolo"):
                with (
                    self.subTest(sport=sport, format=format_name),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    source = fixture(root / "render", sport=sport)
                    with contextlib.redirect_stdout(io.StringIO()):
                        exported = export_dataset(source, root / "dataset")
                    project = root / "project"
                    project.mkdir()
                    write_json(project / "project.json", {"sport": sport, "frames": []})
                    report = self.invoke(root / "dataset" / format_name, project, format_name)
                    self.assertEqual(report["imported"], 6)
                    self.assertEqual(report["annotations"], exported.annotations)
                    self.assertEqual(report["skippedClasses"], {})
                    for field in (
                        "invalidLines",
                        "unreadableImages",
                        "imagesWithoutLabels",
                        "duplicates",
                    ):
                        self.assertEqual(report[field], 0, field)
                    imported = read_json(project / "project.json")["frames"]
                    originals = {
                        sha256_file(path): path
                        for path in (root / "dataset" / format_name).rglob("*.png")
                    }
                    for frame in imported:
                        copied = project / frame["relativePath"]
                        self.assertIn(sha256_file(copied), originals)
                        original = originals[sha256_file(copied)]
                        index = int(original.stem.rsplit("-", 1)[1])
                        self.assertEqual(
                            [a["category"] for a in frame["annotations"]],
                            list(labels_for_sport(sport)),
                        )
                        for class_id, annotation in enumerate(frame["annotations"]):
                            expected = (class_id * 3, 1 + index, 1 if class_id == 0 else 2, 3)
                            actual = tuple(
                                annotation[field] for field in ("x", "y", "width", "height")
                            )
                            for value, target in zip(actual, expected, strict=True):
                                self.assertAlmostEqual(value, target, places=9)
                    again = self.invoke(root / "dataset" / format_name, project, format_name)
                    self.assertEqual((again["imported"], again["duplicates"]), (0, 6))

    def test_background_frames_import_without_spurious_boxes(self):
        for format_name in ("coco", "yolo"):
            with self.subTest(format=format_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = fixture(root / "render", scenes=1, frames=1)
                Image.new("L", (17, 13)).save(source / "scene-000000/0.pgm")
                export_dataset(source, root / "dataset")
                project = root / "project"
                project.mkdir()
                write_json(project / "project.json", {"sport": "basketball", "frames": []})
                report = self.invoke(root / "dataset" / format_name, project, format_name)
                self.assertEqual((report["imported"], report["annotations"]), (1, 0))
                self.assertEqual(
                    read_json(project / "project.json")["frames"][0]["annotations"], []
                )
