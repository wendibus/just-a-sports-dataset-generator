"""Mac starter logic that does not need macOS dialogs: resumable runs and their state."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from just_a_sports_dataset_generator.renderers import ReferenceRenderer

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("launcher", ROOT / "macos/launcher.py")
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
original_render = ReferenceRenderer.render


class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"SPORTS_DATASET_HOME": str(self.root / "home")})
        self.env.start()
        self.recipe = ROOT / "examples/basketball.json"

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def test_an_interrupted_run_resumes_and_is_published_once(self):
        destination = self.root / "Ausgaben/Durchlauf-1"
        rendered: list[str] = []

        def interrupted(self, job, output, max_frames=None):
            rendered.append(job.job_id)
            if job.job_id == "scene-000001":
                raise ValueError("Blender was stopped")  # what a cancelled render looks like
            return original_render(self, job, output, max_frames)

        launcher.remember_unfinished(destination)
        with patch.object(ReferenceRenderer, "render", interrupted), self.assertRaises(ValueError):
            launcher.run_dataset(self.recipe, destination, True, reference=True)
        self.assertFalse(destination.exists())
        found = launcher.unfinished_run()
        assert found is not None
        self.assertEqual(found[0], destination)
        self.assertEqual(found[1]["reference"], True)

        rendered.clear()

        def counting(self, job, output, max_frames=None):
            rendered.append(job.job_id)
            return original_render(self, job, output, max_frames)

        messages: list[str] = []
        with patch.object(ReferenceRenderer, "render", counting):
            report = launcher.run_dataset(
                self.recipe, destination, True, reference=True, progress=messages.append
            )
        self.assertEqual(rendered, ["scene-000001", "scene-000002"])
        self.assertTrue(any("fortgesetzt" in m for m in messages))
        self.assertEqual(report["images"], 6)
        self.assertTrue((destination / "Vorschau/index.html").is_file())
        self.assertTrue((destination / "dataset/coco").is_dir())
        self.assertFalse(launcher.work_folder(destination).exists())
        self.assertFalse((destination / "launcher.json").exists())
        self.assertIsNone(launcher.unfinished_run())  # published runs are not offered again
        launcher.forget_unfinished()
        self.assertFalse(launcher.state_file().exists())

    def test_a_work_folder_with_other_settings_is_refused(self):
        destination = self.root / "Ausgaben/Durchlauf-2"

        def failing(self, job, output, max_frames=None):
            raise ValueError("stopped")

        with patch.object(ReferenceRenderer, "render", failing), self.assertRaises(ValueError):
            launcher.run_dataset(self.recipe, destination, True, reference=True)
        with self.assertRaisesRegex(FileExistsError, "other settings"):
            launcher.run_dataset(self.recipe, destination, False, reference=True)

    @unittest.skipIf(sys.platform == "win32" or not Path("/usr/bin/pgrep").exists(), "needs pgrep")
    def test_cancelling_stops_child_processes_but_not_the_dialog(self):
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        dialog = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            launcher.stop_children(exclude=dialog.pid)
            worker.wait(timeout=10)
            self.assertNotEqual(worker.returncode, 0)
            time.sleep(0.2)
            self.assertIsNone(dialog.poll())
        finally:
            for process in (worker, dialog):
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    unittest.main()
