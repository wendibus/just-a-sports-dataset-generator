import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fixtures import fixture, read_json
from PIL import Image
from pycocotools.coco import COCO

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.cli import main
from just_a_sports_dataset_generator.exporters import export_dataset
from just_a_sports_dataset_generator.renderers import ReferenceRenderer
from just_a_sports_dataset_generator.scenario import load_recipe, plan_jobs
from just_a_sports_dataset_generator.storage import sha256_file, write_json

RECIPE = Path(__file__).parents[1] / "examples/basketball.json"


class ExportTests(unittest.TestCase):
    def test_rf_detr_coco_masks_boxes_mapping_splits_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root / "source")
            result = export_dataset(source, root / "dataset")
            self.assertEqual((result.images, result.annotations), (6, 24))
            report = read_json(root / "dataset/report.json")
            self.assertEqual(set(report["scene_splits"].values()), {"train", "val", "test"})
            manifest = read_json(root / "dataset/manifest.json")
            self.assertEqual(
                manifest["class_mapping"][0], {"label": "ball", "coco_id": 1, "class_id": 0}
            )
            self.assertEqual(
                manifest["training_target"],
                {"framework": "RF-DETR", "dataset_dir": "coco", "format": "coco"},
            )
            self.assertEqual(manifest["formats"], ["coco"])
            self.assertFalse((root / "dataset/yolo").exists())
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(sha256_file(root / "dataset" / name), digest)
            for folder in ("train", "valid", "test"):
                path = root / "dataset/coco" / folder / "_annotations.coco.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    coco = COCO(str(path))
                self.assertEqual(len(coco.imgs), 2)
                self.assertEqual([c["name"] for c in coco.cats.values()], manifest["labels"])
                for annotation in coco.anns.values():
                    mask = coco.annToMask(annotation)
                    self.assertEqual(int(mask.sum()), annotation["area"])
                    ys, xs = mask.nonzero()
                    self.assertEqual(
                        annotation["bbox"],
                        [
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max() - xs.min() + 1),
                            int(ys.max() - ys.min() + 1),
                        ],
                    )
                for entry in coco.imgs.values():
                    name = entry["file_name"]
                    job_id, frame_id = name.rsplit("-frame-", 1)
                    original = source / job_id / f"{int(Path(frame_id).stem)}.ppm"
                    with Image.open(path.parent / name) as exported, Image.open(original) as rgb:
                        self.assertEqual(exported.tobytes(), rgb.tobytes())

    def test_export_is_deterministic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root / "source")
            for name in ("one", "two"):
                export_dataset(source, root / name)
            files = [p.relative_to(root / "one") for p in (root / "one").rglob("*") if p.is_file()]
            for name in files:
                self.assertEqual(
                    (root / "one" / name).read_bytes(), (root / "two" / name).read_bytes()
                )
            with self.assertRaises(FileExistsError):
                export_dataset(source, root / "one")
            self.assertEqual(read_json(root / "one/report.json")["images"], 6)

    def test_background_and_legacy_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root / "source", scenes=1, frames=1)
            path = source / "scene-000000/scene.json"
            data = read_json(path)
            data["schema_version"] = "1.0"
            write_json(path, data)
            Image.new("L", (17, 13)).save(path.parent / "0.pgm")
            (source / "run.json").unlink()
            result = export_dataset(source, root / "dataset")
            self.assertEqual(result.annotations, 0)
            self.assertEqual(len(result.warnings), 3)
            coco = read_json(root / "dataset/coco/train/_annotations.coco.json")
            self.assertEqual(len(coco["images"]), 1)
            self.assertEqual(coco["annotations"], [])
            native = read_json(root / "dataset/native/scene-000000/annotations.json")
            self.assertEqual(native["frames"][0]["visible_annotations"], [])
            self.assertEqual(
                read_json(root / "dataset/report.json")["fully_occluded_or_offscreen_objects"], 4
            )

    def test_invalid_source_leaves_no_partial_export(self):
        mutations = {
            "path": lambda d: d["frames"][0].update(rgb="../outside.ppm"),
            "labels": lambda d: d.update(labels=list(reversed(d["labels"]))),
            "duplicate_frame": lambda d: d["frames"].append(d["frames"][0]),
            "class": lambda d: d["frames"][0]["objects"][0].update(label="goal"),
            "schema": lambda d: d.update(schema_version=[]),
            "size": lambda d: d["resolution"].update(width=19),
            "unknown_instance": lambda d: d["frames"][0].update(objects=[]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = fixture(root / "source", scenes=1)
                path = source / "scene-000000/scene.json"
                data = read_json(path)
                mutate(data)
                write_json(path, data)
                with self.assertRaises(AnnotationError):
                    export_dataset(source, root / "dataset")
                self.assertFalse((root / "dataset").exists())
                self.assertFalse(list(root.glob(".sports-staging-*")))

    def test_duplicate_images_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root / "source", scenes=3, frames=1)
            image = (source / "scene-000000/0.ppm").read_bytes()
            (source / "scene-000001/0.ppm").write_bytes(image)
            with self.assertRaisesRegex(AnnotationError, "identical image"):
                export_dataset(source, root / "dataset")
            self.assertFalse((root / "dataset").exists())

    def test_output_cannot_be_inside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = fixture(Path(directory) / "source", scenes=1)
            with self.assertRaisesRegex(AnnotationError, "separate"):
                export_dataset(source, source / "export")

    def test_oversized_image_header_and_escaping_symlink_fail_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root / "source", scenes=1)
            rgb = source / "scene-000000/0.ppm"
            rgb.write_bytes(b"P6\n20000 20000\n255\n")
            with self.assertRaisesRegex(AnnotationError, "safe decoding limit"):
                export_dataset(source, root / "dataset")
            self.assertFalse((root / "dataset").exists())
            rgb.unlink()
            outside = root / "outside.ppm"
            Image.new("RGB", (17, 13)).save(outside)
            try:
                rgb.symlink_to(outside)
            except OSError:
                self.skipTest("creating symlinks is not permitted on this platform")
            with self.assertRaisesRegex(AnnotationError, "outside"):
                export_dataset(source, root / "dataset")
            self.assertFalse((root / "dataset").exists())

    def test_non_coco_formats_are_rejected_before_output(self):
        for formats in (("yolo",), ("coco", "yolo"), ()):
            with self.subTest(formats=formats), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = fixture(root / "source", scenes=1)
                with self.assertRaisesRegex(AnnotationError, "RF-DETR"):
                    export_dataset(source, root / "dataset", formats)
                self.assertFalse((root / "dataset").exists())
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    main(
                        [
                            "export",
                            str(source),
                            "--output",
                            str(root / "dataset"),
                            "--format",
                            "yolo",
                        ]
                    )
                self.assertEqual(error.exception.code, 2)
                self.assertFalse((root / "dataset").exists())

    def test_cli_pipeline_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                patch(
                    "socket.socket",
                    side_effect=AssertionError("pipeline must not open network sockets"),
                ),
            ):
                self.assertEqual(
                    main(
                        [
                            "render-reference",
                            str(RECIPE),
                            "--output",
                            str(root / "render"),
                            "--max-scenes",
                            "3",
                            "--max-frames",
                            "1",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(["export", str(root / "render"), "--output", str(root / "export")]), 0
                )
                self.assertEqual(
                    main(["export", str(root / "render"), "--output", str(root / "export")]), 2
                )
                self.assertEqual(
                    main(["export", str(root / "missing"), "--output", str(root / "other")]), 2
                )
            manifest = read_json(root / "export/manifest.json")
            self.assertEqual(manifest["run"]["recipe"]["seed"], 42)
            self.assertEqual(manifest["run"]["assets"][0]["license"], "Apache-2.0")

    def test_renderer_capabilities_tiny_images_and_no_overwrite(self):
        job = replace(plan_jobs(load_recipe(RECIPE))[0], width=1, height=1, frame_count=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = ReferenceRenderer().render(job, root)
            data = read_json(result.manifest)
            self.assertEqual(
                [o["visible_area"] for o in data["frames"][0]["objects"]], [0, 0, 0, 1]
            )
            self.assertIsNone(data["frames"][0]["objects"][0]["bbox_xywh"])
            with self.assertRaises(FileExistsError):
                ReferenceRenderer().render(job, root)
            with self.assertRaisesRegex(ValueError, "supports only"):
                ReferenceRenderer().render(replace(job, annotations=("keypoints",)), root / "bad")
            self.assertFalse((root / "bad").exists())
