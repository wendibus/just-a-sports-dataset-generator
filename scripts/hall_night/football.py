"""Experimental outdoor football adapter, prepared without running Qwen."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from . import futsal
from .common import seed_for, sha, write
from .football_pitch import ground_point, pixels_per_metre


def make_plans(count, seed, gender_weights=None, ball_weights=None):
    plans = futsal.make_plans(count, seed, gender_weights, ball_weights)
    for p in plans:
        p["id"] = p["id"].replace("futsal-", "football-")
        p["sport"] = "football"
        p["seed"] = seed_for(seed, p["id"])
    if len({p["seed"] for p in plans}) != len(plans):
        raise ValueError("Scene seed collision")
    return plans


def asset_plan(seed):
    base = futsal.asset_plan(seed)
    assets = []
    for a in base:
        if a["label"] != "player":
            assets.append(a)
            continue
        for copy in range(2):
            item = dict(a)
            slot = int(a["id"].rsplit("-", 1)[1]) + 36 * copy
            item["id"] = a["id"].rsplit("-", 1)[0] + f"-{slot:02d}"
            item["seed"] = seed_for(seed, item["id"])
            assets.append(item)
    # Every sex/kit stratum has 48 train and 12 val/test identities, enough for 11-a-side.
    for gender in ("man", "woman"):
        for kit in ("red", "yellow"):
            pool = [
                a
                for a in assets
                if a["label"] == "player" and a["gender"] == gender and a["kit"] == kit
            ]
            random.Random(seed_for(seed, gender + kit + "football-split")).shuffle(pool)
            for i, a in enumerate(pool):
                a["split"] = "train" if i < 48 else "val" if i < 60 else "test"
    colors = (
        "fluorescent blue",
        "fluorescent yellow",
        "black",
        "turquoise",
        "red",
        "light grey",
        "orange",
        "yellow",
    )
    for a in assets:
        if a["label"] == "referee":
            a["kit"] = colors[(int(a["id"].split("-")[1]) // 2) % len(colors)]
    return assets


def prompt_for(a):
    if a["label"] == "referee":
        clothes = f"a plain {a['kit']} short-sleeved referee shirt with two small chest pockets, black shorts above the knees, long socks matching the shirt or black, black outdoor football boots with studs"
        action = {
            "signal": "raising one arm to signal",
            "run": "jogging while observing play",
            "stand": "standing alert",
        }[a["pose"]]
    elif a["label"] == "goalkeeper":
        clothes = f"a plain {a['kit']} long-sleeved goalkeeper jersey, matching shorts, long socks, padded goalkeeper gloves and outdoor football boots with studs"
        action = (
            "a low goalkeeper ready position, open gloved hands, knees bent"
            if a["pose"] == "ready"
            else "a goalkeeper blocking stance, arms extended, feet planted on grass"
        )
    else:
        clothes = f"a plain {a['kit']} short-sleeved football jersey, matching shorts, knee-length football socks, shin guards under the socks and outdoor football boots with studs"
        action = futsal.POSES[a["pose"]][0].replace("indoor shoe", "football boot")
    return (
        f"A genuine sports press PHOTOGRAPH of ONE adult {a['gender']} association FOOTBALL {a['label']}, {a['skin']}, athletic build, around {a['height_m']:.2f} metres tall. "
        f"{action}. A {a['view']} view from an elevated sideline broadcast camera. Wearing {clothes}. "
        "An empty natural green outdoor grass football pitch, soft overcast daylight, neutral white balance, gentle natural contact shadow under the boots, realistic grass texture. "
        "Match the outdoor pitch in the reference; use its human silhouette only as rough pose guidance and replace all original clothing. "
        "Real adult anatomy, natural hair, realistic hands, true skin texture, woven sports fabric, seams and creases. "
        "COMPLETE FULL BODY, entire head, all hands and both boots visible, generous empty margin around every limb. "
        "One person only, no other humans, no balls, empty hands, no text, badges or logos. Natural documentary photography."
    )


def choose_roster(assets, plan, rng, pilot):
    candidates = [a for a in assets if pilot or a["split"] == plan["split"]]
    genders = (
        ["man"]
        if plan["gender_mode"] == "men"
        else ["woman"]
        if plan["gender_mode"] == "women"
        else ["man", "woman"]
    )
    roster = []
    for kit in ("red", "yellow"):
        for gender in genders:
            pool = [
                a
                for a in candidates
                if a["label"] == "player" and a["kit"] == kit and a["gender"] == gender
            ]
            n = 1 if pilot else 10 // len(genders)
            if len(pool) < n:
                raise ValueError(f"Missing football player stratum {plan['split']}/{kit}/{gender}")
            roster.extend(rng.sample(pool, n))
    for i, kit in enumerate(("green", "blue")):
        pool = [
            a
            for a in candidates
            if a["label"] == "goalkeeper"
            and a["kit"] == kit
            and a["gender"] == genders[i % len(genders)]
        ]
        if not pool:
            raise ValueError("Missing football goalkeeper")
        roster.append(rng.choice(pool))
    refs = [a for a in candidates if a["label"] == "referee"]
    if not refs:
        raise ValueError("Missing football referee")
    roster.append(rng.choice(refs))
    return roster


def matched_person(rgba, ground, rng):
    rgb = ImageEnhance.Brightness(ImageEnhance.Contrast(rgba.convert("RGB")).enhance(0.97)).enhance(
        rng.uniform(0.94, 1.02)
    )
    rgb = rgb.filter(ImageFilter.GaussianBlur(0.25))
    rgb.putalpha(rgba.getchannel("A"))
    return rgb


def make_layers(plan, assets, balls, cache, images, pilot, attempt):
    return futsal.make_layers(
        plan,
        assets,
        balls,
        cache,
        images,
        pilot,
        attempt,
        profile={
            "choose_roster": choose_roster,
            "ground_point": ground_point,
            "pixels_per_metre": pixels_per_metre,
            "matched_person": matched_person,
            "ball_radius_m": 0.110,
            "min_ball_radius": 2,
            "overlap_u": 0.003,
            "overlap_v": 0.025,
        },
    )


def build_balls(config, cache: Path):
    folder = cache / "balls"
    folder.mkdir(parents=True, exist_ok=True)
    path = Path(config["ball_reference"])
    with Image.open(path) as raw:
        source = raw.convert("RGB").resize((1536, 1024), Image.Resampling.LANCZOS)
    styles = (
        (
            "white-blue-gold-stars",
            "white-red-blue-green",
            "white-red-blue",
            "white-magenta-black",
            "yellow-purple-black",
        ),
        (
            "white-cyan-magenta",
            "white-black-red",
            "white-blue-yellow",
            "white-red-blue",
            "white-black-yellow",
        ),
    )
    records = []
    for row, y in enumerate((230, 652)):
        for col, x in enumerate((175, 480, 787, 1092, 1375)):
            radius = 113
            rgba = source.crop((x - radius, y - radius, x + radius, y + radius)).convert("RGBA")
            alpha = Image.new("L", rgba.size)
            ImageDraw.Draw(alpha).ellipse((1, 1, 224, 224), fill=255)
            rgba.putalpha(alpha)
            identity = f"ball-{row}-{col}"
            target = folder / f"{identity}.png"
            rgba.save(target)
            records.append(
                {
                    "id": identity,
                    "file": target.relative_to(cache).as_posix(),
                    "sha256": sha(target),
                    "source_sha256": sha(path),
                    "style": styles[row][col],
                    "rights": "User-supplied illustrative chart; local use, redistribution rights not asserted",
                }
            )
    write(cache / "balls.json", {"schema_version": "1.0", "assets": records})
    return records
