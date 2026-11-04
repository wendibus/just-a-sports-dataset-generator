"""Experimental local photographic cutouts. These do NOT retain Blender ground truth."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from just_a_sports_dataset_generator.annotations.scene import safe_file
from just_a_sports_dataset_generator.generators.qwen import generate_crops
from just_a_sports_dataset_generator.storage import sha256_file, write_json

PROMPT = (
    "Create a genuine documentary sports photograph of ONE adult professional basketball player. "
    "Use the body pose, uniform colours and viewing direction in image 1 as reference. "
    "Keep the knee bend and arm directions, but replace the mannequin with a real athletic human "
    "with natural anatomical proportions. A realistic face, subtle uneven skin texture, natural "
    "hair, realistic hands and fingers, natural muscle contours and a slight sheen of perspiration. "
    "A lightweight sleeveless basketball jersey and loose matching shorts, fine woven mesh, "
    "realistic seams and fabric folds, socks and detailed basketball shoes with laces. "
    "Natural photographic contrast and colour, soft indoor sports lighting. Complete full body: "
    "head, both hands and both shoes inside the picture, with empty space around the person. "
    "Plain seamless neutral grey background. No court, other people, ball, lettering or logos. "
    "A real camera photograph, not a computer rendered character or a plastic doll."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--resolution", type=int, default=768, choices=(512, 640, 768, 1024))
    parser.add_argument("--role", choices=("player", "referee"), default="player")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--review-python", type=Path, required=True)
    parser.add_argument("--foreground-model", type=Path, required=True)
    args = parser.parse_args()
    resolution = args.resolution
    prompt = PROMPT
    if args.role == "referee":
        prompt = (
            "A genuine documentary full-body sports photograph of ONE adult basketball referee. "
            "Use the pose and viewing direction in image 1 as guidance. A natural human face, "
            "realistic skin, hair, hands and anatomical proportions. The referee wears a plain "
            "grey short-sleeved shirt with black sleeves, LONG BLACK TROUSERS and black shoes. "
            "Natural cloth texture and folds. Keep both hands, the complete head and both shoes "
            "inside the picture. Neutral grey seamless background, soft indoor sports lighting. "
            "No ball, other people, logos, lettering or extra equipment. Real camera photography."
        )
    source = args.run.resolve() / "crops"
    out = args.output.resolve()
    index = json.loads((source / "crops.json").read_text())
    by_id = {c["id"]: c for c in index["crops"]}
    if len(set(args.ids)) != len(args.ids) or set(args.ids) - by_id.keys():
        parser.error("Unknown/duplicate crop ID")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", identity) for identity in args.ids):
        parser.error("Unsafe crop ID")
    chosen = [json.loads(json.dumps(by_id[i])) for i in args.ids]
    plan = {
        "source_sha256": sha256_file(source / "crops.json"),
        "ids": args.ids,
        "resolution": resolution,
        "prompt": prompt,
        "seed": 2026,
    }
    if out.exists() and (
        not (out / "plan.json").exists() or json.loads((out / "plan.json").read_text()) != plan
    ):
        raise ValueError("Use a new output folder for changed settings")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "plan.json", plan)
    inputs = out / "references"
    inputs.mkdir(exist_ok=True)
    for crop in chosen:
        with Image.open(safe_file(source, crop["files"]["rgb"])) as raw:
            rgb = raw.convert("RGB")
        with Image.open(safe_file(source, crop["files"]["mask"])) as raw:
            mask = raw.convert("L")
        neutral = Image.composite(rgb, Image.new("RGB", rgb.size, (190, 190, 190)), mask)
        for kind, name in crop["files"].items():
            original = safe_file(source, name)
            dest = inputs / name
            if not dest.resolve().is_relative_to(inputs.resolve()):
                raise ValueError("Unsafe crop path")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(original) as raw:
                image = raw.copy()
            if kind == "rgb":
                image = neutral
            if kind == "inpaint":
                image = Image.new("L", rgb.size, 255)
            image.resize((resolution, resolution), Image.Resampling.LANCZOS).save(dest)
        factor = resolution / index["size"]
        crop["scale"] *= factor
        for point in crop.get("keypoints", []):
            point[0] *= factor
            point[1] *= factor
    write_json(
        inputs / "crops.json",
        {
            **index,
            "size": resolution,
            "crops": chosen,
            "note": "Neutral-background photo reference only; not a renderer ground-truth crop.",
        },
    )
    generate_crops(
        inputs,
        out / "photos",
        confirm=lambda _: False,
        mode="edit",
        resolution=resolution,
        seed=2026,
        steps=4,
        prompt=prompt,
        pose_image=False,
    )
    subprocess.run(
        [
            str(args.review_python.absolute()),
            str(Path(__file__).with_name("foreground_masks.py")),
            "--generated",
            str(out / "photos"),
            "--output",
            str(out / "silhouettes"),
            "--model-dir",
            str(args.foreground_model.resolve()),
            "--ids",
            *args.ids,
        ],
        check=True,
    )
    cells = []
    for crop in chosen:
        identity = crop["id"]
        with Image.open(out / "photos" / (identity + ".png")) as raw:
            photo = raw.convert("RGBA")
        with Image.open(out / "silhouettes" / (identity + ".png")) as raw:
            alpha = raw.convert("L")
        photo.putalpha(alpha)
        photo.save(out / (identity + "-cutout.png"))
        bg = Image.new("RGBA", photo.size, (35, 37, 42, 255))
        bg.alpha_composite(photo)
        bg.convert("RGB").save(out / (identity + "-dark.png"))
        cells.append(
            f'<figure><img src="references/{html.escape(quote(crop["files"]["rgb"]))}"><img src="photos/{identity}.png"><img src="{identity}-dark.png"><figcaption>{identity}: reference / photographic edit / extracted foreground</figcaption></figure>'
        )
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Photographic player candidates</title><style>body{font:15px system-ui;background:#17191d;color:#eee}img{width:32%;max-width:768px}figure{margin:20px}</style><h1>Photographic player candidates</h1><p>Experimental cutouts. Check pose and anatomy. Original Blender labels are invalid for these new silhouettes.</p>'
        + "".join(cells),
        encoding="utf-8",
    )
    print("Candidates:", out / "index.html", flush=True)


if __name__ == "__main__":
    main()
