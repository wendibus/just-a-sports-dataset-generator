"""Futsal photographic adapter: exact frame budgets, kits, goals and Reco identities."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from just_a_sports_dataset_generator.annotations import ObjectIdentity
from just_a_sports_dataset_generator.photo_composite import PhotoLayer, place_layer

from .assets import SKINS
from .assets import asset_plan as basketball_assets
from .common import asset_path, read, seed_for, sha, write
from .render import Layer, allocate, ground_point, matched_person, person_height, pixels_per_metre

POSES = {
    "run": ("running after a pass, arms naturally swinging", "run", 0.94),
    "defend": ("defending in a low stance, knees bent, arms balanced", "defend", 0.82),
    "control": (
        "controlling an imaginary ball with the sole of one indoor shoe, empty hands",
        "stand",
        0.95,
    ),
    "kick": (
        "striking an imaginary ball with the instep, kicking leg extended low forward, other foot planted",
        "pass",
        0.88,
    ),
    "pass": (
        "a short inside-foot passing action, kicking foot turned outward close to the floor",
        "run",
        0.92,
    ),
    "sprint": ("sprinting, leaning forward, bent knees, arms driving", "run", 0.91),
}
REF_COLORS = ("turquoise", "fluorescent yellow", "black", "light grey", "orange", "red")


def make_plans(count, seed, gender_weights=None, ball_weights=None):
    """Allocate individual images first; pair only matching strata within a split."""
    if type(count) is not int or count < 2:
        raise ValueError("Image count must be an integer >= 2")
    weights = [
        gender_weights or {"men": 0.4, "women": 0.4, "mixed": 0.2},
        ball_weights or {"single": 0.7, "multi": 0.2, "none": 0.1},
        {"train": 0.8, "val": 0.1, "test": 0.1},
    ]
    for values, keys in zip(
        weights, ({"men", "women", "mixed"}, {"single", "multi", "none"}, {"train", "val", "test"})
    ):
        if (
            set(values) != keys
            or any(v < 0 for v in values.values())
            or abs(sum(values.values()) - 1) > 1e-9
        ):
            raise ValueError("Invalid composition weights")
    columns = []
    for i, values in enumerate(weights):
        values = [key for key, n in allocate(count, values).items() for _ in range(n)]
        random.Random(seed_for(seed, f"futsal-strata-{i}")).shuffle(values)
        columns.append(values)
    buckets = defaultdict(int)
    for key in zip(*columns):
        buckets[key] += 1
    plans = []
    for (gender, ball, split), n in sorted(buckets.items()):
        for first in range(0, n, 2):
            index = len(plans)
            identity = f"futsal-{index:06d}"
            plans.append(
                {
                    "id": identity,
                    "index": index,
                    "seed": seed_for(seed, identity),
                    "sport": "futsal",
                    "gender_mode": gender,
                    "ball_mode": ball,
                    "split": split,
                    "frame_count": min(2, n - first),
                    "occlude_ball": ball != "none"
                    and seed_for(seed, identity + "occlusion") % 100 < 55,
                }
            )
    if len({p["seed"] for p in plans}) != len(plans):
        raise ValueError("Scene seed collision")
    return plans


def asset_plan(seed):
    assets = [a for a in basketball_assets(seed) if a["label"] == "player"]
    for a in assets:
        slot = int(a["id"].rsplit("-", 1)[1])
        a["pose"] = list(POSES)[(slot // 3) % 6]
        a["extent"] = POSES[a["pose"]][2]
        a["reference_label"] = "player"
        a["reference_pose"] = POSES[a["pose"]][1]
        a["height_m"] = round(
            1.55 + (a["seed"] % 470) / 1000
            if a["gender"] == "woman"
            else 1.65 + (a["seed"] % 450) / 1000,
            3,
        )
    for gender in ("woman", "man"):
        for kit in ("green", "blue"):
            for i in range(6):
                identity = f"k-{gender}-{kit}-{i:02d}"
                assets.append(
                    {
                        "id": identity,
                        "label": "goalkeeper",
                        "gender": gender,
                        "kit": kit,
                        "skin": SKINS[i % 3],
                        "pose": "save" if i % 2 else "ready",
                        "extent": 0.85 if i % 2 else 0.94,
                        "reference_label": "player",
                        "reference_pose": "defend",
                        "view": "front three-quarter" if i % 2 else "rear three-quarter",
                        "height_m": 1.64 + i * 0.04,
                        "split": "train" if i < 4 else "val" if i == 4 else "test",
                        "seed": seed_for(seed, identity),
                    }
                )
    for i in range(24):
        identity = f"r-{i:02d}"
        assets.append(
            {
                "id": identity,
                "label": "referee",
                "gender": "woman" if i % 2 == 0 else "man",
                "kit": REF_COLORS[(i // 2) % 6],
                "skin": SKINS[(i // 4) % 3],
                "pose": "signal" if i % 3 == 0 else "run" if i % 3 == 1 else "stand",
                "reference_label": "referee",
                "reference_pose": "signal" if i % 3 == 0 else "run" if i % 3 == 1 else "stand",
                "extent": 1.16 if i % 3 == 0 else 0.97,
                "view": "front three-quarter" if i < 12 else "rear three-quarter",
                "height_m": 1.62 + (i % 7) * 0.035,
                "split": "train" if i < 16 else "val" if i < 20 else "test",
                "seed": seed_for(seed, identity),
            }
        )
    return assets


def prompt_for(a):
    role = a["label"]
    if role == "referee":
        clothes = f"a plain {a['kit']} short-sleeved futsal referee shirt with two small breast pockets, BLACK SHORTS above the knees, knee-length socks, black flat indoor court shoes"
        pose = {
            "signal": "raising one hand to signal",
            "run": "jogging along the touchline",
            "stand": "standing alert",
        }[a["pose"]]
    elif role == "goalkeeper":
        clothes = f"a plain {a['kit']} LONG-SLEEVED goalkeeper jersey, matching shorts, knee pads, long socks and flat indoor futsal shoes, contrasting with red and yellow field-player teams"
        pose = (
            "a low goalkeeper ready stance, hands open beside the thighs"
            if a["pose"] == "ready"
            else "a goalkeeper blocking stance, knees bent and arms extended sideways"
        )
    else:
        clothes = f"a plain {a['kit']} SHORT-SLEEVED football jersey with sleeves covering the shoulders, matching above-knee shorts, knee-length socks and flat non-marking indoor futsal shoes"
        pose = POSES[a["pose"]][0]
    return (
        f"Create a real documentary sports photograph of ONE adult {a['gender']} FUTSAL {role}, {a['skin']}, "
        f"athletic body, approximately {a['height_m']:.2f} metres tall. {pose}. A {a['view']} view. Wearing {clothes}. "
        "This is indoor FOOTBALL: natural football movement, realistic football clothing. No sleeveless basketball jersey, no basketball gesture, no outdoor studded boots. "
        "Match the reference hall: warm wooden floor, diffuse daylight from tall frosted windows behind, gentle bounced front fill, natural restrained contrast, slightly elevated sideline camera. "
        "Real skin pores, natural hair and facial features, realistic hands and legs, fabric seams and folds, authentic sports photography. "
        "COMPLETE FULL BODY including entire head, both shoes, all limbs and fingers; generous empty margin. "
        "Only one human. EMPTY HANDS and empty floor: NO BALL of any colour, no extra object or extra person, no text or logos. "
        "Use the reference only for body pose and hall lighting; replace its clothes with the specified futsal kit."
    )


def build_balls(config, cache):
    folder = cache / "balls"
    folder.mkdir(parents=True, exist_ok=True)
    path = Path(config["ball_reference"])
    with Image.open(path) as raw:
        source = raw.convert("RGB").resize((1280, 853), Image.Resampling.LANCZOS)
    records = []
    styles = (
        (
            "white-purple-yellow",
            "white-blue-lime",
            "white-black-lime",
            "white-red-yellow",
            "white-blue-lime",
        ),
        (
            "white-red-blue",
            "white-purple-lime",
            "white-black-yellow",
            "white-blue-red",
            "white-blue",
        ),
    )
    for row, y in enumerate((184, 522)):
        for col, x in enumerate((143, 395, 647, 897, 1146)):
            # Conservative inner disc excludes printed titles and the chart's drop shadow.
            radius = 87
            rgba = source.crop((x - radius, y - radius, x + radius, y + radius)).convert("RGBA")
            alpha = Image.new("L", rgba.size)
            ImageDraw.Draw(alpha).ellipse((1, 1, 172, 172), fill=255)
            rgba.putalpha(alpha)
            identity = f"ball-{row}-{col}"
            p = folder / f"{identity}.png"
            rgba.save(p)
            records.append(
                {
                    "id": identity,
                    "file": p.relative_to(cache).as_posix(),
                    "sha256": sha(p),
                    "source_sha256": sha(path),
                    "style": styles[row][col],
                    "rights": "User-supplied illustration; local use, redistribution rights not asserted",
                }
            )
    write(cache / "balls.json", {"schema_version": "1.0", "assets": records})
    return records


def choose_roster(assets, plan, rng, pilot):
    candidates = [a for a in assets if pilot or a["split"] == plan["split"]]
    roster = []
    genders = (
        ["man"]
        if plan["gender_mode"] == "men"
        else ["woman"]
        if plan["gender_mode"] == "women"
        else ["man", "woman"]
    )
    for kit in ("red", "yellow"):
        for gender in genders:
            pool = [
                a
                for a in candidates
                if a["label"] == "player" and a["kit"] == kit and a["gender"] == gender
            ]
            n = 1 if pilot else 4 // len(genders)
            if len(pool) < n:
                raise ValueError(f"Missing futsal {plan['split']}/{kit}/{gender} cutouts")
            roster.extend(rng.sample(pool, n))
    for i, kit in enumerate(("green", "blue")):
        gender = genders[i % len(genders)]
        pool = [
            a
            for a in candidates
            if a["label"] == "goalkeeper" and a["kit"] == kit and a["gender"] == gender
        ]
        if not pool:
            raise ValueError("Missing goalkeeper stratum")
        roster.append(rng.choice(pool))
    pool = [a for a in candidates if a["label"] == "referee"]
    if not pool:
        raise ValueError("Missing futsal referee")
    roster.extend(rng.sample(pool, min(len(pool), rng.randint(1, 2))))
    return roster


def make_layers(plan, assets, balls, cache, images, pilot, attempt):
    rng = random.Random(seed_for(plan["seed"], f"placement-{attempt}"))
    people = []
    counters = {"player": 10, "goalkeeper": 30, "referee": 40}
    classes = {"player": 1, "goalkeeper": 2, "referee": 3}
    previous = None
    for a in choose_roster(assets, plan, rng, pilot):
        role = a["label"]
        ident = counters[role]
        counters[role] += 1
        if role == "goalkeeper":
            u, v = (0.07 if ident == 30 else 0.93), rng.uniform(0.36, 0.61)
        elif role == "referee":
            u, v = rng.uniform(0.12, 0.88), rng.choice((0.12, 0.93))
        else:
            u, v = rng.uniform(0.16, 0.84), rng.uniform(0.22, 0.85)
            if ident == 11 and previous is not None:
                u, v = previous[0] + 0.012, min(0.88, previous[1] + 0.045)
            if ident == 10:
                previous = (u, v)
        ground = ground_point(u, v)
        if a["id"] not in images:
            with Image.open(asset_path(cache, a["file"])) as raw:
                images[a["id"]] = raw.copy()
        rgba = matched_person(images[a["id"]], ground, rng)
        identity = ObjectIdentity(ident, classes[role], role)
        rgb, alpha, pos = place_layer(PhotoLayer(identity, rgba, ground, person_height(a, ground)))
        people.append(
            Layer(
                identity,
                rgb,
                alpha,
                pos,
                v,
                ground,
                a["id"],
                {
                    "height_m": a["height_m"],
                    "pose": a["pose"],
                    "requested_gender": a["gender"],
                    "requested_skin": a["skin"],
                    "kit": a["kit"],
                    "ground_uv": [u, v],
                },
            )
        )
    layers = list(people)
    # Goal net/post alpha participates in depth ordering, including see-through mesh.
    goals = read(cache / "goals.json")["assets"]
    for i, a in enumerate(goals):
        ground = ground_point(0.025 if i == 0 else 0.975, 0.50)
        if a["id"] not in images:
            with Image.open(asset_path(cache, a["file"])) as raw:
                images[a["id"]] = raw.copy()
        rgba = images[a["id"]]
        bbox = rgba.getchannel("A").getbbox()
        if not bbox:
            raise ValueError("Empty goal asset")
        rgba = rgba.crop(bbox)
        height = round(pixels_per_metre(ground) * 2.08)
        rgba = rgba.resize(
            (round(rgba.width * height / rgba.height), height), Image.Resampling.LANCZOS
        )
        alpha = rgba.getchannel("A").point([255 if v >= 128 else 0 for v in range(256)])
        layers.append(
            Layer(
                ObjectIdentity(200 + i, 4, "goal"),
                rgba.convert("RGB"),
                alpha,
                (ground[0] - rgba.width // 2, ground[1] - height),
                0.50,
                ground,
                a["id"],
                {
                    "opening_m": [3, 2],
                    "depth_m": 1,
                    "model": "procedural goal; projected visual scale",
                },
            )
        )
    n = (
        0
        if plan["ball_mode"] == "none"
        else 1
        if plan["ball_mode"] == "single"
        else rng.randint(2, 4)
    )
    for i in range(n):
        person = rng.choice([p for p in people if p.identity.label in ("player", "goalkeeper")])
        radius = max(5, round(pixels_per_metre(person.ground) * 0.1003))
        hidden = plan["occlude_ball"] and i == 0
        airborne = rng.random() < 0.15
        if hidden:
            row = min(
                person.alpha.height - 2,
                round(person.alpha.height * (rng.uniform(0.35, 0.6) if airborne else 0.91)),
            )
            band = (
                person.alpha.crop((0, row - 1, person.alpha.width, row + 2)).getbbox()
                or person.alpha.getbbox()
            )
            side = rng.choice((-1, 1))
            edge = band[0] if side < 0 else band[2] - 1
            center = (
                round(person.position[0] + edge + side * radius * 0.15),
                person.position[1] + row,
            )
            depth = person.depth - 0.0001
        else:
            center = (
                person.ground[0] + rng.choice((-1, 1)) * rng.randint(radius * 2, radius * 5),
                person.ground[1] - radius - (rng.randint(25, 100) if airborne else 0),
            )
            depth = person.depth + 0.0001
        a = rng.choice(balls)
        if a["id"] not in images:
            with Image.open(asset_path(cache, a["file"])) as raw:
                images[a["id"]] = raw.copy()
        rgba = (
            images[a["id"]]
            .resize((radius * 2, radius * 2), Image.Resampling.LANCZOS)
            .rotate(rng.uniform(-180, 180), Image.Resampling.BICUBIC)
        )
        alpha = rgba.getchannel("A").point([255 if v >= 128 else 0 for v in range(256)])
        layers.append(
            Layer(
                ObjectIdentity(i + 1, 0, "ball"),
                rgba.convert("RGB"),
                alpha,
                (center[0] - radius, center[1] - radius),
                depth,
                (center[0], person.ground[1] if airborne else center[1] + radius),
                a["id"],
                {
                    "type": a["style"],
                    "state": "in_flight" if airborne else "floor",
                    "partial_requested": hidden,
                },
            )
        )
    return layers
