from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .common import read, seed_for, sha, validate, write

POSES = {
    "run": ("running mid-stride with clearly bent knees and swinging arms", 0.93),
    "defend": ("a deep low defensive stance, wide feet, knees bent, arms spread sideways", 0.78),
    "dribble": (
        "a low forward-leaning athletic stance, one open hand reaching down beside the knee, empty hands",
        0.85,
    ),
    "shoot": ("a basketball jump-shot follow-through, both hands high above the head", 1.18),
    "pass": ("a chest-pass follow-through, both arms reaching forward, knees slightly bent", 0.96),
    "rebound": ("jumping for a rebound, both arms reaching overhead, knees bent in midair", 1.19),
}
SKINS = ("light skin", "medium brown skin", "deep brown skin")
KITS = ("red", "yellow")


def asset_plan(seed: int):
    records = []
    for gender in ("woman", "man"):
        for kit in KITS:
            order = list(range(36))
            random.Random(seed_for(seed, gender + kit)).shuffle(order)
            splits = {
                slot: "train" if rank < 24 else "val" if rank < 30 else "test"
                for rank, slot in enumerate(order)
            }
            for i in range(36):
                identity = f"p-{gender}-{kit}-{i:02d}"
                rng = random.Random(seed_for(seed, identity))
                pose = list(POSES)[(i // 3) % 6]
                records.append(
                    {
                        "id": identity,
                        "label": "player",
                        "gender": gender,
                        "kit": kit,
                        "skin": SKINS[i % 3],
                        "pose": pose,
                        "extent": POSES[pose][1],
                        "view": "front three-quarter" if i < 18 else "rear three-quarter",
                        "height_m": round(
                            rng.uniform(1.58, 2.00)
                            if gender == "woman"
                            else rng.uniform(1.68, 2.14),
                            3,
                        ),
                        "split": splits[i],
                        "seed": seed_for(seed, identity),
                    }
                )
    for i in range(12):
        records.append(
            {
                "id": f"r-{i:02d}",
                "label": "referee",
                "gender": "woman" if i % 2 == 0 else "man",
                "kit": (
                    "grey-black",
                    "grey-black",
                    "orange-black",
                    "black-white-stripes",
                    "orange-black",
                    "black-neon",
                    "grey-black",
                    "grey-black",
                    "grey-black",
                    "grey-black",
                    "black-white-stripes",
                    "black-neon",
                )[i],
                "skin": SKINS[(i // 2) % 3],
                "pose": "signal" if i % 3 == 0 else "run" if i % 3 == 1 else "stand",
                "extent": 1.16 if i % 3 == 0 else 0.97,
                "view": "rear three-quarter" if i % 2 else "front three-quarter",
                "height_m": 1.68 + (i % 6) * 0.04,
                "split": "train" if i < 8 else "val" if i < 10 else "test",
                "seed": seed_for(seed, f"r-{i:02d}"),
            }
        )
    return records


def prompt_for(asset):
    gender = asset["gender"]
    body = (
        "lean athletic build"
        if asset["seed"] % 3 == 0
        else "powerful athletic build"
        if asset["seed"] % 3 == 1
        else "compact athletic build"
    )
    if asset["label"] == "player":
        role = f"adult {gender} basketball player"
        clothing = f"a plain {asset['kit']} sleeveless basketball jersey and matching loose knee-length shorts, socks and basketball shoes"
        pose = POSES[asset["pose"]][0]
    else:
        role = f"adult {gender} basketball referee"
        clothing = {
            "grey-black": "a light grey short-sleeved shirt with black sleeves",
            "orange-black": "an orange short-sleeved referee shirt with black side panels",
            "black-white-stripes": "a referee shirt with vertical black-and-white stripes",
            "black-neon": "a dark black referee shirt with narrow neon-yellow collar and sleeve trim",
        }[asset["kit"]]
        clothing += ", LONG BLACK TROUSERS and black sports shoes"
        pose = {
            "signal": "raising one hand overhead to signal",
            "run": "jogging, knees slightly bent",
            "stand": "standing relaxed",
        }[asset["pose"]]
    return (
        f"Create a genuine documentary photograph of ONE {role}, {asset['skin']}, {body}, "
        f"approximately {asset['height_m']:.2f} metres tall. {pose}. Use the reference body pose as guidance. "
        f"A {asset['view']} view, a slightly elevated sideline camera. Wearing {clothing}. "
        "Natural adult facial features and hair, anatomically realistic hands, real skin texture, natural muscle contours, "
        "soft sweat sheen, thin woven jersey fabric, seams and natural cloth folds. "
        "Match the warm wooden indoor gym in the reference: large frosted windows behind the athlete, "
        "soft diffuse backlight and gentle bounced fill on the front, restrained contrast and natural colour. "
        "COMPLETE FULL BODY, including all fingers, the whole head and both shoes, with empty margin around every limb. "
        "Only ONE person. Absolutely NO ball, no extra people, no text, no logos. "
        "A real sports photograph, not a mannequin, illustration, 3D character or plastic doll."
    )


def prepare_references(config, root: Path, plan):
    source = Path(config["source_crops"])
    index = read(source / "crops.json")
    chosen = root / "references"
    chosen.mkdir(parents=True, exist_ok=True)
    with Image.open(config["background"]) as raw:
        # No auxiliary basket in this low, central wall/floor lighting reference.
        context = (
            raw.convert("RGB")
            .crop((1300, 380, 1820, 900))
            .resize((640, 640), Image.Resampling.LANCZOS)
        )
    for asset in plan:
        destination = chosen / (asset["id"] + ".png")
        options = [
            c
            for c in index["crops"]
            if c["label"] == asset.get("reference_label", asset["label"])
            and c.get("pose_family") == asset.get("reference_pose", asset["pose"])
        ]
        if not options:
            raise ValueError(f"No pose reference for {asset['id']}")
        crop = sorted(options, key=lambda c: c["id"])[asset["seed"] % len(options)]
        from just_a_sports_dataset_generator.annotations.scene import safe_file

        with Image.open(safe_file(source, crop["files"]["rgb"])) as raw:
            rgb = raw.convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)
        with Image.open(safe_file(source, crop["files"]["mask"])) as raw:
            alpha = raw.convert("L").resize((640, 640), Image.Resampling.LANCZOS)
        if not destination.exists():
            reference = Image.composite(rgb, context, alpha)
            reference.save(destination)
        asset["reference_crop"] = crop["id"]
        asset["reference_sha256"] = sha(destination)
    write(root / "asset-plan.json", {"schema_version": "1.0", "assets": plan})


def build_balls(config, root: Path):
    folder = root / "balls"
    folder.mkdir(parents=True, exist_ok=True)
    path = Path(config["ball_reference"])
    with Image.open(path) as raw:
        source = raw.convert("RGB")
    # The supplied 1536x1024 reference, normalized to its illustrated layout.
    source = source.resize((1536, 1024), Image.Resampling.LANCZOS)
    records = []
    for row in range(2):
        for col, x in enumerate((171, 466, 764, 1062, 1363)):
            y = 229 if row == 0 else 629
            radius = 116
            box = (x - radius, y - radius, x + radius, y + radius)
            rgb = source.crop(box).convert("RGBA")
            alpha = Image.new("L", rgb.size)
            ImageDraw.Draw(alpha).ellipse((2, 2, 229, 229), fill=255)
            rgb.putalpha(alpha)
            identity = f"ball-{row}-{col}"
            rgb.save(folder / f"{identity}.png")
            records.append(
                {
                    "id": identity,
                    "file": f"balls/{identity}.png",
                    "sha256": sha(folder / f"{identity}.png"),
                    "source_sha256": sha(path),
                    "source": str(path),
                    "rights": "User-supplied reference; redistribution rights not asserted",
                    "style": (
                        ("orange", "dark-brown", "orange-cream", "orange-black", "orange-cream"),
                        ("dark-brown", "orange-cream", "orange", "orange", "dark-brown"),
                    )[row][col],
                }
            )
    write(root / "balls.json", {"schema_version": "1.0", "assets": records})
    return records


def dominant_fraction(alpha):
    small = alpha.resize((96, 96), Image.Resampling.NEAREST)
    pixels = np.asarray(small) >= 128
    visited = np.zeros_like(pixels)
    counts = []
    for y, x in np.argwhere(pixels):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        count = 0
        while stack:
            a, b = stack.pop()
            count += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                ny, nx = a + dy, b + dx
                if 0 <= ny < 96 and 0 <= nx < 96 and pixels[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        counts.append(count)
    return max(counts, default=0) / max(1, int(pixels.sum()))


def finalize_assets(config, root: Path, plan):
    ids = [a["id"] for a in plan if (root / "photos" / (a["id"] + ".png")).exists()]
    if not ids:
        raise ValueError("No locally generated photographs")
    command = [
        config["matte_python"],
        str(Path(__file__).parents[1] / "foreground_masks.py"),
        "--generated",
        str(root / "photos"),
        "--output",
        str(root / "masks"),
        "--model-dir",
        config["matte_model"],
        "--ids",
        *ids,
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            config["matte_python"],
            str(Path(__file__).with_name("matte_checks.py")),
            str(root),
            "--sport",
            config.get("sport", "basketball"),
            "--ids",
            *ids,
        ],
        check=True,
    )
    matte_checks = read(root / "matte-checks.json")
    accepted, rejected = [], []
    ledger = {}
    import json

    for line in (root / "generation.jsonl").read_text().splitlines():
        record = json.loads(line)
        ledger[record["id"]] = record
    cutouts = root / "cutouts"
    cutouts.mkdir(exist_ok=True)
    for asset in plan:
        if asset["id"] not in ids:
            continue
        with Image.open(root / "photos" / (asset["id"] + ".png")) as raw:
            rgba = raw.convert("RGBA")
        with Image.open(root / "masks" / (asset["id"] + ".clean.png")) as raw:
            alpha = raw.convert("L").point([255 if v >= 128 else 0 for v in range(256)])
        bbox = alpha.getbbox()
        reasons = []
        if matte_checks[asset["id"]]["possible_extra_balls"]:
            reasons.append("possible unlabelled ball in generated person")
        if bbox is None:
            reasons.append("empty foreground")
        else:
            x, y, r, b = bbox
            if min(x, y, rgba.width - r, rgba.height - b) < 4:
                reasons.append("foreground reaches image border")
            if b - y < rgba.height * 0.36 or r - x < rgba.width * 0.09:
                reasons.append("foreground too small")
            if dominant_fraction(alpha) < 0.95:
                reasons.append("multiple significant foreground components")
        if reasons:
            rejected.append({"id": asset["id"], "reasons": reasons})
            continue
        rgba.putalpha(alpha)
        output = cutouts / (asset["id"] + ".png")
        rgba.save(output)
        accepted.append(
            {
                **asset,
                "generation": ledger.get(asset["id"], {}),
                "matte_provenance": read(root / "masks" / (asset["id"] + ".json")),
                "file": output.relative_to(root).as_posix(),
                "sha256": sha(output),
                "matte_checks": matte_checks[asset["id"]],
                "quality": "automatic matte/framing checks passed; anatomy and requested traits need visual review",
            }
        )
    write(
        root / "library.json",
        {
            "schema_version": "2.0" if config.get("sport") == "futsal" else "1.0",
            "assets": accepted,
            "rejected": rejected,
            "generation": "Qwen-Image-Edit-2511 Q4_K_M + Lightning (Apache-2.0); BiRefNet (MIT), local",
            "traits": "Prompted attributes, not independently inferred demographic ground truth",
        },
    )
    version = "2.0" if config.get("sport") == "futsal" else "1.0"
    validate(f"hall-photo-library-v{version}.json", read(root / "library.json"))
    contact_sheet(root, accepted)
    return accepted, rejected


def contact_sheet(root: Path, assets):
    tiles = []
    for asset in assets:
        with Image.open(root / asset["file"]) as raw:
            image = raw.convert("RGBA")
            bg = Image.new("RGBA", image.size, (70, 70, 70, 255))
            bg.alpha_composite(image)
            thumb = ImageOps.contain(bg.convert("RGB"), (150, 150))
        tile = Image.new("RGB", (180, 190), (28, 30, 34))
        tile.paste(thumb, ((180 - thumb.width) // 2, 0))
        ImageDraw.Draw(tile).text((4, 154), asset["id"], fill="white")
        ImageDraw.Draw(tile).text((4, 170), asset["split"] + " / " + asset["skin"], fill="white")
        tiles.append(tile)
    for page in range(math.ceil(len(tiles) / 48)):
        chunk = tiles[page * 48 : (page + 1) * 48]
        sheet = Image.new("RGB", (8 * 180, math.ceil(len(chunk) / 8) * 190), (28, 30, 34))
        for i, tile in enumerate(chunk):
            sheet.paste(tile, ((i % 8) * 180, (i // 8) * 190))
        sheet.save(root / f"asset-review-{page + 1}.jpg", quality=92)
