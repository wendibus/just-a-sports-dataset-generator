import tempfile
import unittest
from pathlib import Path

from just_a_sports_dataset_generator.annotations import AnnotationError
from just_a_sports_dataset_generator.annotations.scene import safe_file


class ScenePathTests(unittest.TestCase):
    def test_nested_portable_paths_and_rejected_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested folder" / "image.png"
            nested.parent.mkdir()
            nested.write_bytes(b"fixture")
            self.assertEqual(safe_file(root, "nested folder/image.png"), nested.resolve())
            for invalid in (
                "../image.png",
                "nested folder/../image.png",
                "nested folder\\image.png",
                str(nested.resolve()),
            ):
                with self.subTest(path=invalid), self.assertRaises(AnnotationError):
                    safe_file(root, invalid)
