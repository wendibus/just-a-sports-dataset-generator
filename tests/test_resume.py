import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from just_a_sports_dataset_generator.cli import main
from just_a_sports_dataset_generator.renderers import ReferenceRenderer

RECIPE = Path(__file__).parents[1] / "examples" / "basketball.json"
original_render = ReferenceRenderer.render


def run(output: Path, *extra: str) -> tuple[int, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(
            ["render-reference", str(RECIPE), "--output", str(output), "--max-frames", "1", *extra]
        )
    return code, stdout.getvalue() + stderr.getvalue()


def digest(folder: Path) -> dict[str, str]:
    return {
        path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


class ResumeTests(unittest.TestCase):
    def test_an_interrupted_render_resumes_without_redoing_finished_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[str] = []

            def failing(self, job, output, max_frames=None):
                calls.append(job.job_id)
                if job.job_id == "scene-000001":
                    raise ValueError("simulated interruption")
                return original_render(self, job, output, max_frames)

            with patch.object(ReferenceRenderer, "render", failing):
                code, text = run(root / "out")
            self.assertEqual(code, 2, text)
            self.assertFalse((root / "out").exists())
            partial = root / "out.partial"
            self.assertTrue((partial / "scene-000000/scene.json").is_file())
            self.assertTrue((partial / "resume.json").is_file())
            # A scene interrupted mid-way leaves only a staging folder, which is discarded.
            (partial / ".sports-staging-leftover").mkdir()

            calls.clear()

            def counting(self, job, output, max_frames=None):
                calls.append(job.job_id)
                return original_render(self, job, output, max_frames)

            with patch.object(ReferenceRenderer, "render", counting):
                code, text = run(root / "out")
            self.assertEqual(code, 0, text)
            self.assertIn("fortgesetzt", text)
            self.assertEqual(calls, ["scene-000001", "scene-000002"])
            self.assertFalse(partial.exists())
            self.assertFalse((root / "out/resume.json").exists())

            code, text = run(root / "fresh")
            self.assertEqual(code, 0, text)
            # Byte-identical to an uninterrupted run, including run.json.
            self.assertEqual(digest(root / "out"), digest(root / "fresh"))

    def test_leftovers_with_other_settings_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def failing(self, job, output, max_frames=None):
                raise ValueError("simulated interruption")

            with patch.object(ReferenceRenderer, "render", failing):
                run(root / "out")
            code, text = run(root / "out", "--max-scenes", "2")
            self.assertEqual(code, 2)
            self.assertIn("unfinished run with other settings", text)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
