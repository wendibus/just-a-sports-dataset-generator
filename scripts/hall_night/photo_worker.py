"""Runs only inside the existing Qwen environment. No downloads or hosted calls."""

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
request = json.loads(Path(sys.argv[1]).read_text())
root = Path(request["root"])
spec = importlib.util.spec_from_file_location("approved_qwen_worker", request["worker"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import torch
from PIL import Image

settings = request["model"]
settings.update(mode="edit", device=module.pick_device(torch, "auto"))
todo = [a for a in request["assets"] if not (root / "photos" / (a["id"] + ".png")).exists()]
print(f"Photographs remaining: {len(todo)}", flush=True)
if not todo:
    sys.exit(0)
pipe, lightning = module.load_pipeline(torch, settings, print)
if not lightning:
    raise RuntimeError("Approved 4-step Lightning weights are required for the overnight budget")
module.set_condition_size(request["resolution"])
(root / "photos").mkdir(exist_ok=True)
for asset in todo:
    if time.time() >= request["deadline"]:
        break
    began = time.monotonic()
    reference_path = root / "references" / (asset["id"] + ".png")
    if (
        asset.get("reference_sha256")
        and hashlib.sha256(reference_path.read_bytes()).hexdigest() != asset["reference_sha256"]
    ):
        raise ValueError("Reference checksum mismatch")
    module.set_condition_size(request["resolution"])
    with Image.open(reference_path) as raw:
        source = raw.convert("RGB")
    work = request["resolution"]
    for attempt in range(3):
        try:
            result = pipe(
                prompt=asset["prompt"],
                negative_prompt=" ",
                true_cfg_scale=1.0,
                guidance_scale=1.0,
                num_inference_steps=4,
                height=work,
                width=work,
                image=[source],
                generator=torch.Generator(device="cpu").manual_seed(asset["seed"]),
            ).images[0]
            break
        except Exception as exc:
            if not module.is_out_of_memory(exc) or attempt == 2:
                raise
            module.release_memory(torch)
            work = module.smaller(work)
            module.set_condition_size(work)
            print("Memory retry:", asset["id"], work, flush=True)
    target = root / "photos" / (asset["id"] + ".png")
    partial = target.with_suffix(".partial.png")
    result.save(partial)
    os.replace(partial, target)
    with (root / "generation.jsonl").open("a") as log:
        log.write(
            json.dumps(
                {
                    "id": asset["id"],
                    "seed": asset["seed"],
                    "resolution": work,
                    "seconds": round(time.monotonic() - began, 2),
                    "prompt_sha256": hashlib.sha256(asset["prompt"].encode()).hexdigest(),
                    "photo_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
            + "\n"
        )
    print("Generated", asset["id"], round(time.monotonic() - began), "s", flush=True)
    del source, result
    module.release_memory(torch)
