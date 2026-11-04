"""Local masked removal of the two folded wall goals before placing floor goals."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path


def main():
    import torch
    from PIL import Image, ImageDraw, ImageFilter

    request = json.loads(Path(sys.argv[1]).read_text())
    root = Path(request["output"])
    root.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("local_qwen_worker", request["worker"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = {**request["model"], "mode": "inpaint", "device": module.pick_device(torch, "auto")}
    image = Image.open(request["background"]).convert("RGB")
    jobs = [
        ("left", (75, 95, 615, 615), [(135, 130, 565, 540)]),
        ("right", (2925, 50, 3200, 600), [(2970, 95, 3200, 495)]),
    ]
    pipe = None
    records = []
    for name, box, rects in jobs:
        start = time.monotonic()
        source = image.crop(box)
        mask = Image.new("L", source.size)
        d = ImageDraw.Draw(mask)
        for x, y, r, b in rects:
            d.rectangle((x - box[0], y - box[1], r - box[0], b - box[1]), fill=255)
        generated_path = root / f"{name}-generated.png"
        if not generated_path.exists():
            if pipe is None:
                pipe, lightning = module.load_pipeline(torch, settings, print)
                if not lightning:
                    raise ValueError("The already approved four-step Lightning model is required")
            module.set_inpaint_area(640)
            prompt = "Remove the red-and-white folded handball goal frame AND its entire hanging net from the wall. Reconstruct the plain empty gym wall behind: light upper wall, grey lower wall. Preserve the straight horizontal grey paint boundary, original lighting, nearby basketball backboards, doors and floor. Absolutely no goal, net or new object. Empty realistic sports hall photograph."
            result = pipe(
                prompt=prompt,
                negative_prompt=" ",
                true_cfg_scale=1.0,
                guidance_scale=1.0,
                num_inference_steps=4,
                height=640,
                width=640,
                image=source.resize((640, 640), Image.Resampling.LANCZOS),
                mask_image=mask.resize((640, 640), Image.Resampling.NEAREST),
                strength=1.0,
                generator=torch.Generator(device="cpu").manual_seed(20261104),
            ).images[0]
            result.save(generated_path)
        generated = (
            Image.open(generated_path).convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
        )
        image.paste(
            Image.composite(generated, source, mask.filter(ImageFilter.GaussianBlur(2))), box[:2]
        )
        mask.save(root / f"{name}-mask.png")
        records.append({"side": name, "crop": box, "seconds": round(time.monotonic() - start, 2)})
        print("Cleaned folded wall goal:", name, flush=True)
    image.save(root / "background.png")
    (root / "cleanup.json").write_text(
        json.dumps(
            {
                "method": "local Qwen masked inpainting, only wall-goal patches changed",
                "jobs": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
