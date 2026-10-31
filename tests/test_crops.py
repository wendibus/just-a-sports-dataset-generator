import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fixtures import read_json
from jsonschema import Draft202012Validator
from PIL import Image

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.annotations.poses import OPENPOSE_18
from just_a_sports_dataset_generator.core import SceneJob, labels_for_sport
from just_a_sports_dataset_generator.crops import export_crops
from just_a_sports_dataset_generator.renderers.blender import BlenderBasketballRenderer

ROOT = Path(__file__).parents[1]
JOB = SceneJob(
    "scene-000000",
    0,
    1,
    7,
    "basketball",
    64,
    48,
    ("bounding_boxes", "instance_masks"),
    labels_for_sport("basketball"),
)


def keypoints() -> list[object]:
    points: list[object] = [[25.0, 25.0]] * len(OPENPOSE_18)
    points[0] = [25.0, 12.0, True]  # nose, facing the camera
    points[14] = [24.0, 11.0, False]  # right eye, turned away
    points[4] = [80.0, 20.0]  # right wrist outside the image
    return points


def fake_worker(
    poses: bool = True,
    ball: tuple[int, int, int, int] = (40, 40, 43, 43),
    color: tuple[int, int, int] = (90, 70, 50),
    second_player: bool = False,
):
    def worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[-1]).parent
        Image.new("RGB", (64, 48), color).save(output / "frame.png")
        mask = Image.new("L", (64, 48), 0)
        mask.paste(10, (20, 10, 30, 40))  # player: 10 x 30 px
        mask.paste(1, ball)
        if second_player:
            mask.paste(11, (32, 12, 38, 40))  # inside the first player's crop box
        mask.save(output / "ids.png")
        frame: dict[str, object] = {
            "frame_index": 0,
            "rgb": "frame.png",
            "instance_ids": "ids.png",
            "objects": [
                {"instance_id": 1, "class_id": 0, "label": "ball"},
                {"instance_id": 10, "class_id": 1, "label": "player"},
            ],
        }
        if second_player:
            objects = frame["objects"]
            assert isinstance(objects, list)
            objects.append({"instance_id": 11, "class_id": 1, "label": "player"})
        if poses:
            frame["poses"] = [{"instance_id": 10, "keypoints": keypoints()}]
            if second_player:
                frame["poses"].append({"instance_id": 11, "keypoints": [[35.0, 25.0]] * 18})
        result = {"blender_version": "5.0.1", "device": "CPU", "frames": [frame]}
        (output / "worker-result.json").write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0)

    return worker


def render(root: Path, poses: bool = True, **worker: object) -> Path:
    with patch("subprocess.run", side_effect=fake_worker(poses, **worker)):  # type: ignore[arg-type]
        BlenderBasketballRenderer(Path(sys.executable)).render(JOB, root / "render")
    return root / "render"


class CropExportTests(unittest.TestCase):
    def test_renderer_writes_poses_with_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            source = render(Path(directory))
            poses = read_json(source / "scene-000000/poses.json")
            schema = read_json(ROOT / "schemas/poses-v1.0.json")
            Draft202012Validator(schema).validate(poses)
            [person] = poses["frames"][0]["people"]
            flags = [point[2] for point in person["keypoints"]]
            self.assertEqual((flags[0], flags[1], flags[4], flags[14]), (2, 2, 0, 1))

    def test_crops_are_aligned_and_reversible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = render(root)
            result = export_crops(source, root / "crops", size=90, margin=0.25)
            self.assertEqual((result.crops, result.skipped_small), (1, 0))
            index = read_json(root / "crops/crops.json")
            Draft202012Validator(read_json(ROOT / "schemas/crops-v1.0.json")).validate(index)
            [crop] = index["crops"]
            # bbox 10x30 centred at (25, 25); side = ceil(30 * 1.5) = 45; scale = 90 / 45.
            self.assertEqual(crop["bbox_xywh"], [20, 10, 10, 30])
            self.assertEqual(crop["crop_box_xyxy"], [2, 2, 47, 47])
            self.assertEqual(crop["scale"], 2.0)
            self.assertEqual(crop["keypoints"][1], [46.5, 46.5, 2])
            self.assertEqual(crop["keypoints"][4], [0.0, 0.0, 0])
            with Image.open(root / "crops" / crop["files"]["mask"]) as mask:
                self.assertEqual(mask.size, (90, 90))
                self.assertEqual(mask.histogram()[255], 10 * 30 * 4)
                exact = mask.tobytes()
            with Image.open(root / "crops" / crop["files"]["inpaint"]) as inpaint:
                grown = inpaint.tobytes()
            self.assertTrue(all(g >= e for g, e in zip(grown, exact, strict=True)))
            self.assertGreater(sum(grown), sum(exact))
            with Image.open(root / "crops" / crop["files"]["pose"]) as pose:
                # Nose at (25, 12) -> crop (46.5, 20.5): drawn in the OpenPose nose colour.
                self.assertEqual(pose.getpixel((46, 20)), (255, 0, 0))
                # The eye is turned away: no face point is drawn there.
                eye = crop["keypoints"][14]
                self.assertEqual(eye[2], 1)
                self.assertEqual(pose.getpixel((round(eye[0]), round(eye[1]))), (0, 0, 0))
                self.assertEqual(pose.getpixel((0, 0)), (0, 0, 0))

    def test_small_people_are_skipped_and_poses_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = render(root)
            self.assertEqual(export_crops(source, root / "few", min_height=31).crops, 0)
            for invalid in ({"size": 32}, {"margin": -1.0}, {"dilate": 4}, {"min_height": 0}):
                with self.subTest(invalid), self.assertRaises(ValueError):
                    export_crops(source, root / "invalid", **invalid)  # type: ignore[arg-type]
            self.assertFalse((root / "invalid").exists())
            legacy = render(root / "legacy", poses=False)
            self.assertFalse((legacy / "scene-000000/poses.json").exists())
            with self.assertRaisesRegex(AnnotationError, "no poses.json"):
                export_crops(legacy, root / "legacy-crops")
            self.assertFalse((root / "legacy-crops").exists())


if __name__ == "__main__":
    unittest.main()
