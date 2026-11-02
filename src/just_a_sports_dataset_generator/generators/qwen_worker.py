# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit-2511 worker, run inside the isolated runtime created by qwen_setup.

Invoked by the host with ``python qwen_worker.py REQUEST.json``. Reads crops.json, repaints
each person crop from its RGB crop and OpenPose skeleton, and writes ``<crop id>.png`` into
the output folder. Existing results are skipped, so an interrupted run resumes. The host
validates and composites the results; this worker never touches labels.
"""

import contextlib
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

PROMPT_VERSION = 3
"""2: tank-top jersey description, pose hint per pose family, colours taken from image 1.
3: no "shoulder panels" (Qwen drew pads and wings), real-photo wording against the 3D look,
   pose skeleton optional, background outside the person blurred before generation."""
JERSEY = (
    "a thin, loose sleeveless basketball jersey of lightweight fabric with simple round "
    "armholes, a scoop neckline and the number printed on the chest, and loose knee-length "
    "basketball shorts"
)
REFEREE_KIT = (
    "a grey short-sleeved referee shirt with thin black vertical stripes, a collar, and black "
    "trousers"
)
POSE_HINTS = {
    "stand": "standing relaxed",
    "run": "running mid-stride with arms swinging",
    "defend": "in a low defensive stance, knees deeply bent, feet wide apart, arms spread out",
    "dribble": "dribbling the ball, knees bent, body low, one hand pushing the ball down",
    "shoot": "shooting, both arms extended above the head",
    "pass": "passing, both arms pushing forward from the chest",
    "catch": "reaching forward with both hands to catch the ball",
    "rebound": "jumping for a rebound with both arms stretched upwards",
    "signal": "raising one arm high to signal",
}
NEGATIVE_PROMPT = (
    "shoulder pads, wings, straps, buckles, suspenders, overalls, apron, backpack, square neckline, vest, standing "
    "upright, straight legs, extra limbs, extra fingers, deformed hands, cartoon, 3D render, "
    "plastic skin, doll, action figure, second person, blurry, watermark, text artifacts"
)


def build_prompt(crop, pose_image=False):
    """Prompt for one crop: kit description, explicit pose hint, keep colours and outline."""
    referee = crop.get("label") == "referee"
    skeleton = " and of the skeleton in image 2" if pose_image else ""
    hint = POSE_HINTS.get(crop.get("pose_family") or "")
    pose = (
        f"The {'referee' if referee else 'player'} is {hint}, exactly as in image 1. "
        if hint
        else ""
    )
    kit = (
        f"wears {REFEREE_KIT}"
        if referee
        else f"wears {JERSEY}, in exactly the colours of image 1 and with the same number"
    )
    return (
        f"Repaint the 3D figure in image 1 as a photorealistic basketball "
        f"{'referee' if referee else 'player'} in an indoor sports hall. {pose}"
        f"Keep exactly the body pose of image 1{skeleton}: the same knee bend, arm positions, "
        "body lean, position, size and outline, and the same camera angle. "
        f"The {'referee' if referee else 'player'} {kit}. Only one person, the background "
        "stays out of focus. A real photograph of a real athlete, not a 3D render, not a doll "
        "or action figure: natural skin texture and hair, anatomically correct hands, matte "
        "fabric with natural folds, sharp sports photography."
    )


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


def release_memory(torch):
    """Return cached accelerator memory after each crop.

    PyTorch keeps freed MPS/CUDA blocks in its own pool. Over several crops that pool grew
    until a Mac run failed with "MPS backend out of memory" on the third person.
    """
    gc.collect()
    for backend in ("mps", "cuda"):
        module = getattr(torch, backend, None)
        if module is not None and hasattr(module, "empty_cache"):
            with contextlib.suppress(Exception):  # backend present but not initialised
                module.empty_cache()


def is_out_of_memory(exc):
    return isinstance(exc, (RuntimeError, MemoryError)) and "out of memory" in str(exc).lower()


def smaller(work):
    """Next work resolution after an out-of-memory error (about 30 % fewer pixels)."""
    return max(256, int(work * 0.84) // 16 * 16)


def isolate(rgb, inpaint):
    """Blur everything outside the person's inpaint mask.

    Other players inside the crop were repainted as a second, distorted person; blurred they
    only provide light and colour context. The host pastes back the person only.
    """
    from PIL import Image, ImageFilter

    blurred = rgb.filter(ImageFilter.GaussianBlur(max(2, rgb.width // 24)))
    return Image.composite(rgb, blurred, inpaint)


def set_condition_size(work):
    """Encode the input images at the work resolution instead of diffusers' fixed 1024 px.

    QwenImageEditPlusPipeline turns every input image into 1024 x 1024 latents (4096 tokens
    each); with a 768 px output and two inputs the transformer saw ~10,500 tokens per step.
    """
    try:
        from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as module
    except ImportError:
        return
    if hasattr(module, "VAE_IMAGE_SIZE"):
        module.VAE_IMAGE_SIZE = work * work


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
    # Work at most at the requested resolution: cost grows with the pixel count, and a
    # repainted person is downscaled for the dataset anyway. Multiples of 16 are required.
    work = max(256, min(size, request.get("resolution") or size) // 16 * 16)
    records = output / "generation.jsonl"
    for number, crop in enumerate(todo, 1):
        began = time.monotonic()
        pose_image = bool(request.get("pose_image"))
        with Image.open(crops_dir / crop["files"]["rgb"]) as rgb:
            source = rgb.convert("RGB")
        inpaint_path = crops_dir / crop["files"].get("inpaint", "")
        if inpaint_path.is_file():
            with Image.open(inpaint_path) as inpaint:
                source = isolate(source, inpaint.convert("L"))
        images = [source]
        if pose_image:
            with Image.open(crops_dir / crop["files"]["pose"]) as pose:
                images.append(pose.convert("RGB"))
        prompt = request.get("prompt") or build_prompt(crop, pose_image)
        seed = crop_seed(request["seed"], crop["id"])
        # On "out of memory": free the cache and retry at the same size, then smaller.
        attempt_work, attempts = work, 0
        while True:
            try:
                set_condition_size(attempt_work)
                result = pipe(
                    image=images,
                    prompt=prompt,
                    # Only used with classifier-free guidance (the 30-step fallback); the
                    # 4-step Lightning schedule runs without it (true_cfg_scale 1).
                    negative_prompt=NEGATIVE_PROMPT if cfg > 1 else " ",
                    true_cfg_scale=cfg,
                    guidance_scale=1.0,
                    num_inference_steps=steps,
                    height=attempt_work,
                    width=attempt_work,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                ).images[0]
                break
            except Exception as exc:
                attempts += 1
                if not is_out_of_memory(exc) or attempts > 3 or attempt_work == 256:
                    raise
                release_memory(torch)
                if attempts > 1:
                    attempt_work = smaller(attempt_work)
                log(
                    f"QWEN_WARNING Speicher voll bei {crop['id']}, neuer Versuch mit "
                    f"{attempt_work} px (andere GPU-Programme beenden hilft)"
                )
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
                        "resolution": attempt_work,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "seconds": round(time.monotonic() - began, 1),
                    }
                )
                + "\n"
            )
        seconds = time.monotonic() - began
        del result, images
        release_memory(torch)
        log(f"QWEN_CROP {number}/{len(todo)} {crop['id']} {seconds:.0f}")


if __name__ == "__main__":
    main()
