import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import PIL
from fixtures import read_json
from PIL import Image
from test_crops import render

from just_a_sports_dataset_generator.cli import main
from just_a_sports_dataset_generator.crops import export_crops
from just_a_sports_dataset_generator.generators.qwen import generate_crops
from just_a_sports_dataset_generator.generators.qwen_setup import SetupError

FAKE_TORCH = """
bfloat16 = "bfloat16"
class _Flag:
    @staticmethod
    def is_available():
        return False
class backends:
    mps = _Flag
cuda = _Flag
class Generator:
    def __init__(self, device="cpu"):
        self.device = device
    def manual_seed(self, seed):
        self.seed = seed
        return self
"""
FAKE_DIFFUSERS = """
import json, os
from types import SimpleNamespace
from PIL import Image, ImageDraw

def _log(entry):
    with open(os.environ["FAKE_QWEN_LOG"], "a") as stream:
        stream.write(json.dumps(entry) + "\\n")

class FlowMatchEulerDiscreteScheduler:
    @classmethod
    def from_config(cls, config):
        _log({"scheduler": config["time_shift_type"]})
        return cls()

class GGUFQuantizationConfig:
    def __init__(self, compute_dtype):
        self.compute_dtype = compute_dtype

class QwenImageTransformer2DModel:
    @classmethod
    def from_single_file(cls, path, **kwargs):
        _log({"gguf": os.path.basename(path), "subfolder": kwargs["subfolder"]})
        return cls()

class QwenImageEditPlusPipeline:
    @classmethod
    def from_pretrained(cls, path, transformer=None, torch_dtype=None):
        assert os.path.isfile(os.path.join(path, "model_index.json"))
        assert os.environ.get("HF_HUB_OFFLINE") == "1", "generation must run offline"
        pipe = cls()
        pipe.scheduler = None
        return pipe
    def load_lora_weights(self, folder, weight_name):
        if os.environ.get("FAKE_QWEN_LORA_FAIL"):
            raise RuntimeError("LoRA not supported on this transformer")
        _log({"lora": weight_name})
    def to(self, device):
        _log({"device": device})
        return self
    def set_progress_bar_config(self, **kwargs):
        pass
    def __call__(self, image, prompt, num_inference_steps, true_cfg_scale, height, width,
                 generator, **kwargs):
        if os.environ.get("FAKE_QWEN_CRASH_AFTER") and len(open(os.environ["FAKE_QWEN_LOG"]).read().split("call")) > int(os.environ["FAKE_QWEN_CRASH_AFTER"]):
            raise SystemExit(7)
        _log({"call": len(image), "steps": num_inference_steps, "cfg": true_cfg_scale,
              "size": [width, height], "seed": generator.seed, "referee": "referee" in prompt})
        out = image[0].resize((width, height))
        ImageDraw.Draw(out).rectangle((width // 3, height // 4, 2 * width // 3, 3 * height // 4), fill=(200, 30, 30))
        return SimpleNamespace(images=[out])
"""


def fake_runtime(home: Path) -> None:
    """An isolated venv with stand-in torch/diffusers, plus a matching install manifest."""
    qwen = home / "qwen"
    runtime = qwen / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(runtime)], check=True)
    python = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    site = Path(
        subprocess.run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    (site / "torch.py").write_text(FAKE_TORCH)
    (site / "diffusers.py").write_text(FAKE_DIFFUSERS)
    (site / "pillow_from_host.pth").write_text(str(Path(PIL.__file__).parents[1]) + "\n")
    base = qwen / "models/base"
    base.mkdir(parents=True)
    (base / "model_index.json").write_text("{}")
    (qwen / "models/q.gguf").write_bytes(b"gguf")
    (qwen / "models/lora.safetensors").write_bytes(b"lora")
    manifest = {
        "schema_version": "1.0",
        "model": "Qwen-Image-Edit-2511",
        "quant": "Q4_K_M",
        "packages": ["fake"],
        "python": str(python),
        "base_dir": str(base),
        "gguf": str(qwen / "models/q.gguf"),
        "lora": str(qwen / "models/lora.safetensors"),
        "components": [],
    }
    (qwen / "installed.json").write_text(json.dumps(manifest))


def never(_plan: object) -> bool:
    raise AssertionError("an installed runtime must not ask for consent")


class QwenGenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        fake_runtime(self.home)
        self.log = self.root / "fake.log"
        self.env = patch.dict(
            os.environ, {"SPORTS_DATASET_HOME": str(self.home), "FAKE_QWEN_LOG": str(self.log)}
        )
        self.env.start()
        self.source = render(self.root / "scene", second_player=True)
        export_crops(self.source, self.root / "crops", size=100)

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def entries(self) -> list[dict[str, object]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def calls(self) -> list[dict[str, object]]:
        return [entry for entry in self.entries() if "call" in entry]

    def test_generates_every_crop_with_lightning_and_resumes(self):
        result = generate_crops(
            self.root / "crops", self.root / "gen", confirm=never, progress=lambda _: None
        )
        self.assertEqual((result.generated, result.total), (2, 2))
        self.assertIn("Lightning", result.generator)
        entries = self.entries()
        self.assertIn({"lora": "lora.safetensors"}, entries)
        self.assertIn({"scheduler": "exponential"}, entries)
        self.assertIn({"device": "cpu"}, entries)
        calls = self.calls()
        self.assertEqual([c["steps"] for c in calls], [4, 4])
        self.assertEqual([c["cfg"] for c in calls], [1.0, 1.0])
        self.assertEqual(calls[0]["call"], 2)  # RGB crop and pose skeleton
        self.assertEqual(calls[0]["size"], [256, 256])  # at least 256, multiple of 16
        with Image.open(self.root / "gen/scene-000000-f000000-i010.png") as image:
            self.assertEqual(image.size, (100, 100))  # back at crop size
        records = [
            json.loads(line)
            for line in (self.root / "gen/generation.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            {r["id"] for r in records}, {"scene-000000-f000000-i010", "scene-000000-f000000-i011"}
        )
        header = read_json(self.root / "gen/generation.json")
        self.assertEqual(header["settings"]["steps"], 4)
        # Resume: finished crops are skipped; a deleted one is regenerated with the same seed.
        first_seed = calls[0]["seed"]
        (self.root / "gen/scene-000000-f000000-i010.png").unlink()
        generate_crops(
            self.root / "crops", self.root / "gen", confirm=never, progress=lambda _: None
        )
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(self.calls()[-1]["seed"], first_seed)
        with self.assertRaises(FileExistsError):
            generate_crops(
                self.root / "crops",
                self.root / "gen",
                confirm=never,
                steps=8,
                progress=lambda _: None,
            )

    def test_lora_failure_falls_back_to_the_full_schedule(self):
        with patch.dict(os.environ, {"FAKE_QWEN_LORA_FAIL": "1"}):
            messages: list[str] = []
            generate_crops(
                self.root / "crops", self.root / "gen", confirm=never, progress=messages.append
            )
        self.assertEqual({c["steps"] for c in self.calls()}, {30})
        self.assertEqual({c["cfg"] for c in self.calls()}, {4.0})
        self.assertTrue(any("Lightning LoRA not usable" in m for m in messages))

    def test_worker_failure_keeps_finished_crops(self):
        crash = patch.dict(os.environ, {"FAKE_QWEN_CRASH_AFTER": "1"})
        with crash, self.assertRaisesRegex(SetupError, "finished crops are kept"):
            generate_crops(
                self.root / "crops", self.root / "gen", confirm=never, progress=lambda _: None
            )
        self.assertEqual(len(list((self.root / "gen").glob("scene-*.png"))), 1)
        generate_crops(
            self.root / "crops", self.root / "gen", confirm=never, progress=lambda _: None
        )
        self.assertEqual(len(list((self.root / "gen").glob("scene-*.png"))), 2)

    def test_repaint_command_runs_crops_generation_and_paste_back(self):
        output = self.root / "repainted"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = main(["repaint", str(self.source), "--output", str(output), "--size", "100"])
        self.assertEqual(code, 0, stdout.getvalue())
        report = read_json(output / "render/composite.json")
        self.assertIn("Qwen-Image-Edit-2511", report["generator"])
        self.assertEqual(len(report["items"]), 2)
        self.assertTrue((output / "render/scene-000000/scene.json").is_file())


if __name__ == "__main__":
    unittest.main()
