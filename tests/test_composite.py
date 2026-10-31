import json
import tempfile
import unittest
from pathlib import Path

from fixtures import read_json
from jsonschema import Draft202012Validator
from PIL import Image
from test_crops import render

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.composite import composite_crops
from just_a_sports_dataset_generator.crops import export_crops
from just_a_sports_dataset_generator.exporters import export_dataset

ROOT = Path(__file__).parents[1]
CROP_ID = "scene-000000-f000000-i010"
RED = (255, 0, 0)


def painted(crops: Path, region: str | None, name: str, target: Path) -> Path:
    """A stand-in for a generator: the crop with ``region`` (mask/inpaint/all) painted red."""
    with Image.open(crops / f"scene-000000/{CROP_ID}.rgb.png") as source:
        image = source.convert("RGB")
    if region == "all":
        image = Image.new("RGB", image.size, RED)
    elif region is not None:
        with Image.open(crops / f"scene-000000/{CROP_ID}.{region}.png") as mask:
            image.paste(RED, (0, 0, *image.size), mask)
    target.mkdir(parents=True, exist_ok=True)
    image.save(target / name)
    return target


def setup(root: Path, **worker: object) -> tuple[Path, Path]:
    source = render(root, **worker)
    export_crops(source, root / "crops", size=90)
    return source, root / "crops"


def pixels(path: Path) -> list[tuple[int, int, int]]:
    with Image.open(path) as image:
        data = image.convert("RGB").tobytes()
    return [(data[i], data[i + 1], data[i + 2]) for i in range(0, len(data), 3)]


def ids(path: Path) -> bytes:
    with Image.open(path) as image:
        return image.tobytes()


class CompositeTests(unittest.TestCase):
    def test_mask_blend_changes_only_the_person_and_keeps_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, crops = setup(root)
            # ComfyUI-style counter suffix must be found.
            generated = painted(crops, "mask", f"{CROP_ID}_00001_.png", root / "generated")
            result = composite_crops(
                source, crops, generated, root / "out", generator="test painter"
            )
            self.assertEqual((result.applied, result.rejected, result.missing), (1, 0, 0))
            frame = "scene-000000/frame.png"
            before, after = pixels(source / frame), pixels(root / "out" / frame)
            identity = ids(source / "scene-000000/ids.png")
            self.assertEqual(after[25 * 64 + 25], RED)
            for index, value in enumerate(identity):
                if value != 10:
                    self.assertEqual(after[index], before[index])
            report = read_json(root / "out/composite.json")
            schema = read_json(ROOT / "schemas/composite-report-v1.0.json")
            Draft202012Validator(schema).validate(report)
            self.assertEqual(report["items"][0]["status"], "applied")
            scene = read_json(root / "out/scene-000000/scene.json")
            self.assertEqual(scene["assets"][-1]["generator"], "test painter")
            # Ground truth is unchanged: identical COCO annotations before and after.
            export_dataset(source, root / "before")
            export_dataset(root / "out", root / "after")
            for split in ("train", "valid", "test"):
                old = json.loads((root / f"before/coco/{split}/_annotations.coco.json").read_text())
                new = json.loads((root / f"after/coco/{split}/_annotations.coco.json").read_text())
                self.assertEqual(old["annotations"], new["annotations"])

    def test_overlapping_crops_are_checked_against_the_untouched_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, crops = setup(root, second_player=True)
            generated = root / "generated"
            for identity in (10, 11):
                crop_id = f"scene-000000-f000000-i{identity:03d}"
                with Image.open(crops / f"scene-000000/{crop_id}.rgb.png") as image:
                    painted_crop = image.convert("RGB")
                with Image.open(crops / f"scene-000000/{crop_id}.mask.png") as mask:
                    painted_crop.paste(RED, (0, 0, *painted_crop.size), mask)
                generated.mkdir(exist_ok=True)
                painted_crop.save(generated / f"{crop_id}.png")
            result = composite_crops(source, crops, generated, root / "out")
            self.assertEqual((result.applied, result.rejected), (2, 0))
            after = pixels(root / "out/scene-000000/frame.png")
            for red, green, blue in (after[25 * 64 + 25], after[25 * 64 + 35]):
                self.assertTrue(red > 240 and green < 15 and blue < 15)

    def test_inpaint_blend_never_paints_other_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, crops = setup(root, ball=(30, 20, 33, 23))  # ball touches the player
            generated = painted(crops, "inpaint", f"{CROP_ID}.png", root / "generated")
            result = composite_crops(
                source, crops, generated, root / "out", blend="inpaint", max_spill=1.0
            )
            self.assertEqual(result.applied, 1)
            before = pixels(source / "scene-000000/frame.png")
            after = pixels(root / "out/scene-000000/frame.png")
            red, green, blue = after[30 * 64 + 31]  # background band next to the player
            self.assertTrue(red > 200 and green < 40 and blue < 40)
            for y in range(20, 23):
                for x in range(30, 33):
                    self.assertEqual(after[y * 64 + x], before[y * 64 + x])

    def test_quality_checks_keep_the_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, crops = setup(root)
            cases = {
                "unchanged": (None, {}),
                "background band changed": ("all", {}),
                "silhouette differs": ("mask", {"silhouettes": "shifted"}),
            }
            shifted = root / "shifted"
            shifted.mkdir()
            with Image.open(crops / f"scene-000000/{CROP_ID}.mask.png") as mask:
                mask.transform(mask.size, Image.Transform.AFFINE, (1, 0, 30, 0, 1, 0)).save(
                    shifted / f"{CROP_ID}.png"
                )
            for reason, (region, options) in cases.items():
                with self.subTest(reason):
                    generated = painted(crops, region, f"{CROP_ID}.png", root / reason / "gen")
                    extra = {"silhouettes": shifted} if options else {}
                    result = composite_crops(
                        source, crops, generated, root / reason / "out", **extra
                    )
                    self.assertEqual((result.applied, result.rejected), (0, 1))
                    item = read_json(root / reason / "out/composite.json")["items"][0]
                    self.assertEqual(item["reason"], reason)
                    self.assertEqual(
                        pixels(root / reason / "out/scene-000000/frame.png"),
                        pixels(source / "scene-000000/frame.png"),
                    )
            # A matching silhouette passes the IoU check.
            good = root / "good-silhouettes"
            good.mkdir()
            Image.open(crops / f"scene-000000/{CROP_ID}.mask.png").save(good / f"{CROP_ID}.png")
            generated = painted(crops, "mask", f"{CROP_ID}.png", root / "ok" / "gen")
            result = composite_crops(source, crops, generated, root / "ok/out", silhouettes=good)
            self.assertEqual(result.applied, 1)
            self.assertEqual(read_json(root / "ok/out/composite.json")["items"][0]["iou"], 1.0)

    def test_missing_ambiguous_and_mismatched_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, crops = setup(root)
            empty = root / "empty"
            empty.mkdir()
            self.assertEqual(composite_crops(source, crops, empty, root / "a").missing, 1)
            twice = painted(crops, "mask", f"{CROP_ID}_1.png", root / "twice")
            painted(crops, "mask", f"{CROP_ID}_2.png", twice)
            result = composite_crops(source, crops, twice, root / "b")
            self.assertEqual(result.rejected, 1)
            self.assertEqual(
                read_json(root / "b/composite.json")["items"][0]["reason"], "ambiguous"
            )
            other = render(root / "other", color=(10, 200, 10))
            with self.assertRaisesRegex(AnnotationError, "different render"):
                composite_crops(other, crops, twice, root / "c")
            self.assertFalse((root / "c").exists())
            with self.assertRaises(ValueError):
                composite_crops(source, crops, twice, root / "d", blend="soft")


if __name__ == "__main__":
    unittest.main()
