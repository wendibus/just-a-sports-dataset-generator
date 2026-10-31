# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit-2511 worker, run inside the isolated runtime created by qwen_setup.

Invoked by the host with ``python qwen_worker.py REQUEST.json``. Reads crops.json, repaints
each person crop from its RGB crop and OpenPose skeleton, and writes ``<crop id>.png`` into
the output folder. Existing results are skipped, so an interrupted run resumes. The host
validates and composites the results; this worker never touches labels.
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

PROMPTS = {
    "player": (
        "Repaint the person in image 1 as a photorealistic basketball player photographed in "
        "an indoor sports hall. Use exactly the body pose of the skeleton in image 2. Keep the "
        "person's position, size and outline, the camera angle and the background unchanged. "
        "Keep the jersey and shorts colours and the jersey number of image 1. Natural skin, "
        "realistic hair and fabric, sharp sports photography."
    ),
    "goalkeeper": None,  # same as player
    "referee": (
        "Repaint the person in image 1 as a photorealistic basketball referee in a grey and "
        "black striped shirt and black trousers, photographed in an indoor sports hall. Use "
        "exactly the body pose of the skeleton in image 2. Keep the person's position, size "
        "and outline, the camera angle and the background unchanged. Natural skin, realistic "
        "hair and fabric, sharp sports photography."
    ),
}
# Scheduler published for the Qwen-Image Lightning LoRAs (4-step, no CFG).
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def crop_seed(base_seed, crop_id):
    digest = hashlib.sha256(f"qwen:v1:{base_seed}:{crop_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def pick_device(torch, requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(torch, request, log):
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        GGUFQuantizationConfig,
        QwenImageEditPlusPipeline,
        QwenImageTransformer2DModel,
    )

    dtype = torch.bfloat16
    transformer = QwenImageTransformer2DModel.from_single_file(
        request["gguf"],
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        config=request["base_dir"],
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        request["base_dir"], transformer=transformer, torch_dtype=dtype
    )
    lightning = False
    if request.get("lora"):
        try:
            lora = Path(request["lora"])
            pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
            lightning = True
        except Exception as exc:  # noqa: BLE001 - fall back to the slower base schedule
            log(f"QWEN_WARNING Lightning LoRA not usable, falling back to 30 steps: {exc}")
    pipe.to(request["device"])
    pipe.set_progress_bar_config(disable=True)
    return pipe, lightning


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    def log(message):
        print(message, flush=True)

    import torch
    from PIL import Image

    crops_dir, output = Path(request["crops_dir"]), Path(request["output_dir"])
    index = json.loads((crops_dir / "crops.json").read_text(encoding="utf-8"))
    crops = [c for c in index["crops"] if not request.get("ids") or c["id"] in request["ids"]]
    todo = [c for c in crops if not (output / f"{c['id']}.png").is_file()]
    log(f"QWEN_TODO {len(todo)}/{len(crops)}")
    if not todo:
        return
    request["device"] = pick_device(torch, request.get("device", "auto"))
    started = time.monotonic()
    pipe, lightning = load_pipeline(torch, request, log)
    steps = request["steps"] if lightning else max(request["steps"], 30)
    cfg = 1.0 if lightning else 4.0
    log(
        f"QWEN_READY device={request['device']} lightning={lightning} "
        f"load_seconds={time.monotonic() - started:.0f}"
    )
    size = index["size"]
    work = max(256, size // 16 * 16)  # the transformer needs multiples of 16
    records = output / "generation.jsonl"
    for number, crop in enumerate(todo, 1):
        began = time.monotonic()
        with (
            Image.open(crops_dir / crop["files"]["rgb"]) as rgb,
            Image.open(crops_dir / crop["files"]["pose"]) as pose,
        ):
            images = [rgb.convert("RGB"), pose.convert("RGB")]
        prompt = request.get("prompt") or PROMPTS.get(crop["label"]) or PROMPTS["player"]
        seed = crop_seed(request["seed"], crop["id"])
        result = pipe(
            image=images,
            prompt=prompt,
            negative_prompt=" ",
            true_cfg_scale=cfg,
            guidance_scale=1.0,
            num_inference_steps=steps,
            height=work,
            width=work,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        if result.size != (size, size):
            result = result.resize((size, size), Image.Resampling.LANCZOS)
        partial = output / f".{crop['id']}.partial.png"
        result.save(partial)
        os.replace(partial, output / f"{crop['id']}.png")
        with records.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "id": crop["id"],
                        "seed": seed,
                        "steps": steps,
                        "true_cfg_scale": cfg,
                        "lightning": lightning,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "seconds": round(time.monotonic() - began, 1),
                    }
                )
                + "\n"
            )
        log(f"QWEN_CROP {number}/{len(todo)} {crop['id']}")


if __name__ == "__main__":
    main()
