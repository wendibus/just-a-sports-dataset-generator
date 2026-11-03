"""Repaint person crops with a local Qwen-Image-Edit-2511, installing it on first use.

The heavy model runs in an isolated Python environment (see :mod:`.qwen_setup`) through a
subprocess and a JSON/file boundary, like the Blender worker. If the runtime is missing, the
caller's ``confirm`` callback sees the exact download plan first; nothing is installed
without its approval. Results land in the output folder as ``<crop id>.png``; an
interrupted run resumes and skips finished crops.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from just_a_sports_dataset_generator.storage import write_json

from .qwen_setup import (
    InstallPlan,
    SetupError,
    install,
    installed,
    plan_install,
    update_packages,
)

WORKER = Path(__file__).with_name("qwen_worker.py")
PROMPT_VERSION = 5
"""Must match qwen_worker.PROMPT_VERSION; part of a run's fingerprint so runs never mix."""


@dataclass(frozen=True, slots=True)
class GenerateResult:
    output_directory: Path
    generated: int
    total: int
    generator: str


def ensure_installed(
    confirm: Callable[[InstallPlan], bool],
    quant: str = "Q4_K_M",
    home: Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Return the install manifest, installing first if the user approves the plan."""
    manifest = installed(home)
    if manifest is not None:
        return update_packages(home, progress)
    plan = plan_install(quant, home)
    if not confirm(plan):
        raise SetupError("Qwen is not installed and the installation was declined")
    install(plan, progress)
    manifest = installed(home)
    if manifest is None:
        raise SetupError("the Qwen installation did not complete")
    return manifest


def generate_crops(
    crops: Path,
    output: Path,
    *,
    confirm: Callable[[InstallPlan], bool],
    steps: int = 4,
    seed: int = 0,
    device: str = "auto",
    prompt: str | None = None,
    quant: str = "Q4_K_M",
    resolution: int = 768,
    pose_image: bool = False,
    mode: str = "inpaint",
    strength: float = 0.75,
    time_limit: float | None = None,
    home: Path | None = None,
    progress: Callable[[str], None] = print,
) -> GenerateResult:
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer between 1 and 100")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(resolution) is not int or not 256 <= resolution <= 2048:
        raise ValueError("resolution must be an integer between 256 and 2048")
    if mode not in ("inpaint", "edit"):
        raise ValueError("mode must be inpaint or edit")
    if not 0.1 <= strength <= 1.0:
        raise ValueError("strength must be between 0.1 and 1.0")
    if time_limit is not None and not time_limit > 0:
        raise ValueError("time_limit must be a positive number of minutes")
    if device not in ("auto", "mps", "cuda", "cpu"):
        raise ValueError("device must be auto, mps, cuda or cpu")
    crops, output = crops.resolve(), output.absolute()
    index_path = crops / "crops.json"
    if not index_path.is_file():
        raise ValueError(f"no crops.json in {crops}; run export-crops first")
    crops_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    total = len(json.loads(index_path.read_text(encoding="utf-8"))["crops"])
    header_path = output / "generation.json"
    settings = {
        "steps": steps,
        "seed": seed,
        "prompt": prompt,
        "prompt_version": PROMPT_VERSION,
        "quant": quant,
        "resolution": resolution,
        "pose_image": pose_image,
        "mode": mode,
        "strength": strength if mode == "inpaint" else None,
    }
    if output.exists():
        try:
            header = json.loads(header_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileExistsError(f"output exists and is not a generation run: {output}") from exc
        if header.get("crops_sha256") != crops_sha256 or header.get("settings") != settings:
            raise FileExistsError(f"{output} belongs to different crops or settings")
        progress("Resuming the existing run …")
    manifest = ensure_installed(confirm, quant, home, progress)
    generator = f"Qwen-Image-Edit-2511 {manifest.get('quant')} + Lightning LoRA, {mode} (local)"
    if not output.exists():
        output.mkdir(parents=True)
        write_json(
            header_path,
            {
                "schema_version": "1.0",
                "generator": generator,
                "crops_sha256": crops_sha256,
                "settings": settings,
                "install": {k: manifest[k] for k in ("model", "quant", "packages", "components")},
            },
        )
    request = output / ".request.json"
    write_json(
        request,
        {
            "crops_dir": str(crops),
            "output_dir": str(output),
            "base_dir": manifest["base_dir"],
            "gguf": manifest["gguf"],
            "lora": manifest["lora"],
            "steps": steps,
            "seed": seed,
            "device": device,
            "prompt": prompt,
            "resolution": resolution,
            "pose_image": pose_image,
            "mode": mode,
            "strength": strength,
            # Not part of the settings: a resumed run may get a different budget.
            "deadline": time.time() + time_limit * 60 if time_limit else None,
        },
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # rare unsupported ops run on CPU instead of failing
    # Generation is offline: every file was verified at setup; never fetch anything now.
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
    with (output / "qwen.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(manifest["python"]), "-I", str(WORKER), str(request)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if line.startswith("QWEN_"):
                progress(_message(line.strip()))
        code = process.wait()
    if code:
        tail = (output / "qwen.log").read_text(encoding="utf-8", errors="replace")[-2500:]
        raise SetupError(f"Qwen worker failed (exit {code}); finished crops are kept:\n{tail}")
    done = sum(1 for path in output.glob("*.png") if not path.name.startswith("."))
    return GenerateResult(output, done, total, generator)


def _message(line: str) -> str:
    kind, _, rest = line.partition(" ")
    if kind == "QWEN_CROP":
        count, crop, *seconds = rest.split(" ")
        took = f", {seconds[0]} s" if seconds else ""
        return f"Person {count} done ({crop}{took})"
    if kind == "QWEN_READY":
        return f"Model loaded ({rest})"
    if kind == "QWEN_TODO":
        return f"To generate: {rest} people"
    return rest or line
