import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

from fixtures import fixture, read_json
from PIL import Image

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.exporters import export_dataset
from just_a_sports_dataset_generator.preview import create_preview
from just_a_sports_dataset_generator.storage import sha256_file


class _Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "img":
            self.sources.append(attributes["src"])
        if tag == "a":
            self.links.append(attributes["href"])


class PreviewTests(unittest.TestCase):
    def test_bright_preview_keeps_raw_ids_and_dataset_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root / "render")
            export_dataset(source, root / "dataset")
            before = {p: sha256_file(p) for p in (root / "dataset").rglob("*") if p.is_file()}
            index = create_preview(root / "dataset", root / "Vorschau")
            for path, checksum in before.items():
                self.assertEqual(sha256_file(path), checksum)
            raw = root / "dataset/native/scene-000000/frame-000000.instances.png"
            with (
                Image.open(raw) as original,
                Image.open(root / "Vorschau/images/0001-mask.png") as colored,
            ):
                self.assertEqual(original.mode, "L")
                self.assertEqual(original.getextrema(), (0, 4))
                self.assertEqual(colored.mode, "RGB")
                self.assertEqual(len(colored.getcolors()), 5)
                for _, color in colored.getcolors():
                    self.assertGreater(max(color), 120)
            page = index.read_text(encoding="utf-8")
            self.assertIn("6 von 6 Bildern", page)
            self.assertIn("Farbbild für RF-DETR", page)
            self.assertIn("Objektbereiche", page)
            parser = _Images()
            parser.feed(page)
            self.assertEqual(len(parser.sources), 12)
            self.assertEqual(len(parser.links), 6)
            for url in parser.sources + parser.links:
                self.assertTrue((index.parent / unquote(url)).resolve().is_file())
            with (
                Image.open(root / "Vorschau/images/0001-rgb.png") as thumb,
                Image.open((index.parent / unquote(parser.links[0])).resolve()) as original,
            ):
                self.assertEqual(thumb.tobytes(), original.tobytes())
            self.assertFalse(list((root / "dataset/coco").rglob("*mask*")))

    def test_preview_limit_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_dataset(fixture(root / "render"), root / "dataset")
            index = create_preview(root / "dataset", root / "Vorschau", limit=2)
            self.assertIn("2 von 6 Bildern", index.read_text())
            self.assertEqual(len(list((index.parent / "images").glob("*.png"))), 4)
            with self.assertRaises(FileExistsError):
                create_preview(root / "dataset", index.parent)
            with self.assertRaises(AnnotationError):
                create_preview(root / "dataset", root / "dataset/preview")

    def test_invalid_mask_does_not_publish_partial_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_dataset(fixture(root / "render", scenes=1), root / "dataset")
            raw = root / "dataset/native/scene-000000/frame-000000.instances.png"
            Image.new("L", (17, 13), 99).save(raw)
            with self.assertRaises(AnnotationError):
                create_preview(root / "dataset", root / "Vorschau")
            self.assertFalse((root / "Vorschau").exists())
            self.assertEqual(
                read_json(root / "dataset/manifest.json")["training_target"]["framework"], "RF-DETR"
            )
