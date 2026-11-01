import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.core import SceneJob, labels_for_sport
from just_a_sports_dataset_generator.crops import export_crops
from just_a_sports_dataset_generator.exporters import export_dataset
from just_a_sports_dataset_generator.renderers.blender import (
    BlenderBasketballRenderer,
    find_blender,
)


def job():
    return SceneJob(
        "scene-000000",
        0,
        1,
        1234,
        "basketball",
        320,
        180,
        ("bounding_boxes", "instance_masks"),
        labels_for_sport("basketball"),
    )


class BlenderAdapterTests(unittest.TestCase):
    def test_validation_before_start_and_explicit_missing_blender(self):
        renderer = BlenderBasketballRenderer(Path(sys.executable))
        with tempfile.TemporaryDirectory() as directory, patch("subprocess.run") as run:
            root = Path(directory)
            for invalid in (
                replace(job(), sport="football"),
                replace(job(), job_id="../outside"),
                replace(job(), width=0),
                replace(job(), seed=-1),
                replace(job(), annotations=("depth",)),
                replace(job(), camera="drone"),
            ):
                with self.subTest(job=invalid), self.assertRaises(ValueError):
                    renderer.render(invalid, root)
            with self.assertRaises(ValueError):
                find_blender(root / "missing-blender")
            with self.assertRaises(ValueError):
                renderer.render(job(), root, max_frames=0)
            run.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])
        for samples in (0, 513, True):
            with self.assertRaises(ValueError):
                BlenderBasketballRenderer(Path(sys.executable), samples)

    def test_worker_failure_leaves_no_partial_scene(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("subprocess.run", return_value=subprocess.CompletedProcess([], 3)),
        ):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Blender could not render"):
                BlenderBasketballRenderer(Path(sys.executable)).render(job(), root)
            self.assertEqual(list(root.iterdir()), [])

    def test_worker_passes_are_validated_before_publication(self):
        def worker(command, **kwargs):
            output = Path(command[-1]).parent
            rgb = Image.new("RGB", (320, 180), (80, 120, 180))
            rgb.save(output / "frame.png")
            mask = Image.new("L", (320, 180), 0)
            mask.paste(1, (12, 20, 18, 25))
            mask.save(output / "ids.png")
            (output / "worker-result.json").write_text(
                json.dumps(
                    {
                        "blender_version": "5.0.1",
                        "device": "CPU",
                        "frames": [
                            {
                                "frame_index": 0,
                                "rgb": "frame.png",
                                "instance_ids": "ids.png",
                                "objects": [{"instance_id": 1, "class_id": 0, "label": "ball"}],
                            }
                        ],
                    }
                )
            )
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renderer = BlenderBasketballRenderer(Path(sys.executable))
            with patch("subprocess.run", side_effect=worker):
                rendered = renderer.render(job(), root / "render")
            scene = json.loads(rendered.manifest.read_text())
            self.assertEqual(scene["frames"][0]["objects"][0]["bbox_xywh"], [12, 20, 6, 5])
            self.assertEqual(scene["frames"][0]["objects"][0]["visible_area"], 30)
            exported = export_dataset(root / "render", root / "dataset")
            self.assertEqual((exported.images, exported.annotations), (1, 1))
            with self.assertRaises(FileExistsError):
                renderer.render(job(), root / "render")

            # The renderer must reject IDs not described by the worker.
            def corrupt(command, **kwargs):
                result = worker(command, **kwargs)
                Image.new("L", (320, 180), 99).save(Path(command[-1]).parent / "ids.png")
                return result

            with patch("subprocess.run", side_effect=corrupt), self.assertRaises(AnnotationError):
                renderer.render(replace(job(), job_id="invalid"), root / "render")
            self.assertFalse((root / "render/invalid").exists())


@unittest.skipUnless(
    os.environ.get("BLENDER_EXECUTABLE"), "set BLENDER_EXECUTABLE for local 3D integration"
)
def ball_projection_matches(test, result, frame, camera):
    """The independently projected ball centre lies in the ball's visible box, if any."""
    ball = next(o for o in frame["objects"] if o["label"] == "ball")
    if ball["bbox_xywh"] is None:  # hidden behind a player or outside the frame
        return
    bx, by, bw, bh = ball["bbox_xywh"]
    x, y = camera["ball_pixel"]
    test.assertTrue(bx - 2 <= x <= bx + bw + 2 and by - 2 <= y <= by + bh + 2, (x, y, ball))


class BlenderIntegrationTests(unittest.TestCase):
    def test_real_cycles_scene_export_and_mask_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = BlenderBasketballRenderer(
                Path(os.environ["BLENDER_EXECUTABLE"]), samples=1
            ).render(job(), root / "render")
            scene = json.loads(result.manifest.read_text())
            layout = json.loads((result.output_directory / "layout.json").read_text())
            frame = scene["frames"][0]
            planned = {a["instance_id"] for a in layout["athletes"]}
            # Every planned person is labelled, whether or not the camera sees it.
            self.assertEqual({o["instance_id"] for o in frame["objects"]}, planned | {1, 40, 41})
            with Image.open(result.output_directory / frame["instance_ids"]) as mask:
                ids = {value for _, value in mask.getcolors()}
            self.assertIn(0, ids)
            self.assertLessEqual(ids - {0}, planned | {1, 40, 41})
            self.assertGreaterEqual(len(ids & set(range(10, 20))), 6)
            self.assertGreater(export_dataset(root / "render", root / "dataset").annotations, 6)
            details = json.loads((result.output_directory / "worker-result.json").read_text())
            self.assertEqual(details["situation"], layout["situation"])
            ball_projection_matches(self, result, frame, details["cameras"][0])

    def test_reco_panorama_scene_and_mask_alignment(self):
        panorama = replace(job(), width=960, height=270, camera="reco_panorama")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = BlenderBasketballRenderer(
                Path(os.environ["BLENDER_EXECUTABLE"]), samples=1
            ).render(panorama, root / "render")
            frame = json.loads(result.manifest.read_text())["frames"][0]
            layout = json.loads((result.output_directory / "layout.json").read_text())
            people_ids = {a["instance_id"] for a in layout["athletes"]}
            with Image.open(result.output_directory / frame["instance_ids"]) as mask:
                ids = {value for _, value in mask.getcolors()}
            details = json.loads((result.output_directory / "worker-result.json").read_text())
            camera = details["cameras"][0]
            self.assertEqual(camera["projection"], "equirectangular")
            # The whole court is in view: both hoops and almost every athlete.
            self.assertTrue({40, 41} <= ids)
            self.assertGreaterEqual(len(ids & people_ids), len(people_ids) - 2)
            ball_projection_matches(self, result, frame, camera)
            self.assertGreater(export_dataset(root / "render", root / "dataset").annotations, 10)
            poses = json.loads((result.output_directory / "poses.json").read_text())
            people = poses["frames"][0]["people"]
            self.assertEqual({p["instance_id"] for p in people}, people_ids)
            # Body joints project onto the rendered athletes, not beside them.
            well_seen = [p for p in people if sum(k[2] == 2 for k in p["keypoints"][1:14]) >= 9]
            self.assertGreaterEqual(len(well_seen), 6)
            crops = export_crops(root / "render", root / "crops", size=256)
            self.assertGreaterEqual(crops.crops, 8)
