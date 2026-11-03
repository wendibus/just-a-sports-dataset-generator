"""Compare Qwen inpainting strengths on a few crops of an earlier run.

    python scripts/qwen_inpaint_vergleich.py ../qwen-nacht [--strengths 0.6 0.75 0.9] [--count 12]

Copies ``--count`` crops (spread over pose families and referees) from CROPS_RUN/crops into
CROPS_RUN/vergleich/crops, runs ``generate-crops --mode inpaint`` once per strength and writes
CROPS_RUN/vergleich/vergleich.jpg: per crop the render and, per strength, the result pasted
back with the exact instance mask, as ``composite-crops --blend mask`` would do it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def pick(crops: list[dict], count: int) -> list[dict]:
    """Round-robin over (label, pose family) so every kind of person appears."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for crop in crops:
        groups.setdefault((crop["label"], crop.get("pose_family") or ""), []).append(crop)
    chosen: list[dict] = []
    while len(chosen) < count and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop(0))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="folder with crops/ from qwen-nacht.sh")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    source = args.run.resolve() / "crops"
    target = args.run.resolve() / "vergleich"
    crops_dir = target / "crops"
    index = json.loads((source / "crops.json").read_text())
    if not (crops_dir / "crops.json").is_file():
        chosen = pick(index["crops"], args.count)
        for crop in chosen:
            for name in crop["files"].values():
                (crops_dir / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, crops_dir / name)
        (crops_dir / "crops.json").write_text(json.dumps({**index, "crops": chosen}, indent=2))
    chosen = json.loads((crops_dir / "crops.json").read_text())["crops"]

    for strength in args.strengths:
        output = target / f"staerke-{strength:g}"
        print(f"== Stärke {strength:g} -> {output}", flush=True)
        code = subprocess.call(
            [
                sys.executable,
                "-m",
                "just_a_sports_dataset_generator",
                "generate-crops",
                str(crops_dir),
                "--output",
                str(output),
                "--mode",
                "inpaint",
                "--strength",
                str(strength),
                "--resolution",
                str(args.resolution),
            ]
        )
        if code:
            print(f"Stärke {strength:g} fehlgeschlagen (exit {code}); weiter mit der nächsten")

    size = 200
    columns = 1 + len(args.strengths)
    sheet = Image.new("RGB", (columns * size, len(chosen) * (size + 16)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for row, crop in enumerate(chosen):
        y = row * (size + 16)
        with Image.open(crops_dir / crop["files"]["rgb"]) as raw:
            rgb = raw.convert("RGB")
        with Image.open(crops_dir / crop["files"]["mask"]) as raw:
            mask = raw.convert("L")
        sheet.paste(rgb.resize((size, size)), (0, y + 16))
        draw.text((4, y + 2), f"{crop['label']} {crop.get('pose_family')}", fill=(255, 255, 0))
        for column, strength in enumerate(args.strengths, 1):
            result = target / f"staerke-{strength:g}" / f"{crop['id']}.png"
            draw.text((column * size + 4, y + 2), f"Stärke {strength:g}", fill=(255, 255, 0))
            if result.is_file():
                with Image.open(result) as raw:
                    generated = raw.convert("RGB").resize(rgb.size)
                pasted = Image.composite(generated, rgb, mask)
                sheet.paste(pasted.resize((size, size)), (column * size, y + 16))
    sheet.save(target / "vergleich.jpg", quality=88)
    print(f"Übersicht: {target / 'vergleich.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
