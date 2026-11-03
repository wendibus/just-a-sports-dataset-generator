from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from just_a_sports_dataset_generator.annotations import ObjectIdentity, visible_annotations
from just_a_sports_dataset_generator.core import labels_for_sport
from just_a_sports_dataset_generator.photo_composite import PhotoLayer, place_layer
from just_a_sports_dataset_generator.storage import new_directory

from .common import asset_path, read, seed_for, sha, write

# Manually traced board/rim/net envelopes in the user's final 3200x900 hall.
# Supports remain background; photographic net holes are approximate, not simulated GT.
HOOPS = (
    (
        (622, 153),
        (788, 176),
        (788, 305),
        (748, 305),
        (746, 315),
        (722, 313),
        (720, 303),
        (622, 299),
    ),
    (
        (2692, 160),
        (2933, 103),
        (2931, 268),
        (2801, 282),
        (2794, 289),
        (2762, 289),
        (2755, 284),
        (2692, 289),
    ),
    ((1144, 280), (1192, 280), (1192, 338), (1144, 337)),
    ((1644, 280), (1696, 280), (1696, 337), (1644, 336)),
    ((2167, 281), (2215, 281), (2215, 339), (2167, 338)),
)


def allocate(total, weights):
    keys = list(weights)
    raw = [total * weights[k] for k in keys]
    counts = [math.floor(v) for v in raw]
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return dict(zip(keys, counts))


def make_plans(count: int, seed: int, gender_weights=None, ball_weights=None):
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError("Image count must be a positive even integer >=2")
    gender_weights = gender_weights or {"men": 0.4, "women": 0.4, "mixed": 0.2}
    ball_weights = ball_weights or {"single": 0.7, "multi": 0.2, "none": 0.1}
    for weights, expected in (
        (gender_weights, {"men", "women", "mixed"}),
        (ball_weights, {"single", "multi", "none"}),
    ):
        if (
            set(weights) != expected
            or any(v < 0 for v in weights.values())
            or abs(sum(weights.values()) - 1) > 1e-9
        ):
            raise ValueError("Invalid composition weights")
    plans = [
        {"id": f"hall-{i:06d}", "seed": seed_for(seed, f"scene-{i}"), "index": i}
        for i in range(count // 2)
    ]
    if len({p["seed"] for p in plans}) != len(plans):
        raise ValueError("Scene seed collision")
    ranked = sorted(
        plans, key=lambda p: hashlib.sha256(f"split:v1:{p['seed']}:{p['id']}".encode()).digest()
    )
    sizes = allocate(len(plans), {"train": 0.8, "val": 0.1, "test": 0.1})
    start = 0
    for split, size in sizes.items():
        batch = ranked[start : start + size]
        start += size
        genders = [mode for mode, n in allocate(size, gender_weights).items() for _ in range(n)]
        balls = [mode for mode, n in allocate(size, ball_weights).items() for _ in range(n)]
        random.Random(seed_for(seed, split + "genders")).shuffle(genders)
        random.Random(seed_for(seed, split + "balls")).shuffle(balls)
        for p, gender, ball in zip(batch, genders, balls):
            p.update(
                split=split,
                gender_mode=gender,
                ball_mode=ball,
                occlude_ball=ball != "none" and seed_for(seed, p["id"] + "occlusion") % 100 < 55,
            )
    return plans


@dataclass
class Layer:
    identity: ObjectIdentity
    rgb: Image.Image
    alpha: Image.Image
    position: tuple[int, int]
    depth: float
    ground: tuple[int, int]
    source: str
    metadata: dict


def ground_point(u, v):
    # Four visually calibrated court corners; image-based placement, not metric calibration.
    far = (960 + (2450 - 960) * u, 515 + (490 - 515) * u)
    near = (60 + (3140 - 60) * u, 880 + (900 - 880) * u)
    return round(far[0] * (1 - v) + near[0] * v), round(far[1] * (1 - v) + near[1] * v)


def pixels_per_metre(ground):
    # Estimated 2.4 m sideline camera height; invert the known final panorama's
    # column correction for a position-dependent horizon. This is visual scale
    # calibration, not measured camera/extrinsic ground truth.
    rx = (ground[0] - 1600) / 780.3721417053783
    curve = (
        ((0.004435064875310612 * rx - 0.004248342107059682) * rx - 0.011566516174042798) * rx
        + 0.04132514839294399
    ) * rx + 0.391242885591407
    horizon = 350 - 1100 * (curve - 0.38854453180938087)
    return max(65, min(215, (ground[1] - horizon) / 2.4))


def person_height(asset, ground):
    return round(pixels_per_metre(ground) * asset["height_m"] * asset["extent"])


def matched_person(rgba, ground, rng):
    # Mild warm bounced fill, consistent with the hall; never changes silhouette.
    rgb = rgba.convert("RGB")
    gain = 0.91 + rng.uniform(-0.025, 0.025) + 0.06 * ground[1] / 900
    rgb = ImageEnhance.Brightness(ImageEnhance.Contrast(rgb).enhance(0.96)).enhance(gain)
    rgb = rgb.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.45)))
    channels = rgb.split()
    rgb = Image.merge(
        "RGB",
        (
            channels[0].point(lambda v: min(255, round(v * 1.025))),
            channels[1],
            channels[2].point(lambda v: round(v * 0.97)),
        ),
    )
    rgb.putalpha(rgba.getchannel("A"))
    return rgb


def choose_roster(assets, plan, rng, pilot):
    candidates = [a for a in assets if pilot or a["split"] == plan["split"]]
    roster = []
    gender_mode = plan["gender_mode"]
    if gender_mode in ("men", "women"):
        gender = "man" if gender_mode == "men" else "woman"
        by_kit = {
            kit: [
                a
                for a in candidates
                if a["label"] == "player" and a["gender"] == gender and a["kit"] == kit
            ]
            for kit in ("red", "yellow")
        }
        limit = min(len(v) for v in by_kit.values())
        if limit < (1 if pilot else 5):
            raise ValueError(
                f"Not enough distinct {gender}/{plan['split']} cutouts; repair the library first"
            )
        team_size = min(limit, rng.randint(3, 5))
        for pool in by_kit.values():
            roster.extend(rng.sample(pool, team_size))
    else:
        for kit in ("red", "yellow"):
            pools = [
                [
                    a
                    for a in candidates
                    if a["label"] == "player" and a["gender"] == gender and a["kit"] == kit
                ]
                for gender in ("man", "woman")
            ]
            if min(map(len, pools)) < (1 if pilot else 3):
                raise ValueError("Missing mixed-scene stratum")
            n = 1 if pilot else 2
            for pool in pools:
                roster.extend(rng.sample(pool, n))
    referees = [a for a in candidates if a["label"] == "referee"]
    if not referees:
        raise ValueError("No referee cutouts in split")
    roster.extend(rng.sample(referees, min(len(referees), rng.randint(1, 2))))
    assert len({a["id"] for a in roster}) == len(roster)
    return roster


def make_layers(plan, assets, balls, cache, images, pilot, attempt):
    rng = random.Random(seed_for(plan["seed"], f"placement-{attempt}"))
    roster = choose_roster(assets, plan, rng, pilot)
    people = []
    player_index, ref_index = 10, 40
    previous = None
    for asset in roster:
        if asset["label"] == "player":
            identity = ObjectIdentity(player_index, 1, "player")
            player_index += 1
            u, v = rng.uniform(0.1, 0.9), rng.uniform(0.20, 0.85)
            if previous is not None and player_index == 12:
                # One intentional partial human-human overlap, at similar ground depth.
                u, v = previous[0] + rng.choice((-1, 1)) * 0.012, min(0.88, previous[1] + 0.045)
            if previous is None:
                previous = (u, v)
        else:
            identity = ObjectIdentity(ref_index, 2, "referee")
            ref_index += 1
            u, v = (
                rng.uniform(0.10, 0.9),
                rng.choice((rng.uniform(0.08, 0.18), rng.uniform(0.88, 0.95))),
            )
        ground = ground_point(u, v)
        elevation = rng.uniform(8, 24) if asset["pose"] in ("rebound", "shoot") else 0
        foot = (ground[0], round(ground[1] - elevation))
        if asset["id"] not in images:
            with Image.open(asset_path(cache, asset["file"])) as raw:
                images[asset["id"]] = raw.copy()
        rgba = matched_person(images[asset["id"]], ground, rng)
        rgb, alpha, position = place_layer(
            PhotoLayer(identity, rgba, foot, person_height(asset, ground))
        )
        people.append(
            Layer(
                identity,
                rgb,
                alpha,
                position,
                v,
                ground,
                asset["id"],
                {
                    "height_m": asset["height_m"],
                    "pose": asset["pose"],
                    "requested_gender": asset["gender"],
                    "requested_skin": asset["skin"],
                    "kit": asset["kit"],
                    "ground_uv": [u, v],
                },
            )
        )
    layers = list(people)
    n = (
        0
        if plan["ball_mode"] == "none"
        else 1
        if plan["ball_mode"] == "single"
        else rng.randint(2, 4)
    )
    for i in range(n):
        person = rng.choice([p for p in people if p.identity.label == "player"])
        radius = max(5, round(pixels_per_metre(person.ground) * rng.choice((0.116, 0.122))))
        hidden = plan["occlude_ball"] and i == 0
        airborne = rng.random() < 0.2
        if hidden:
            row = round(person.alpha.height * (rng.uniform(0.32, 0.6) if airborne else 0.91))
            row = min(person.alpha.height - 2, max(1, row))
            band = person.alpha.crop((0, row - 1, person.alpha.width, row + 2)).getbbox()
            if band is None:
                band = person.alpha.getbbox()
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
                person.ground[1] - radius - rng.randint(25, 150)
                if airborne
                else person.ground[1] - radius,
            )
            depth = person.depth + 0.0001
        asset = rng.choice(balls)
        key = asset["id"]
        if key not in images:
            with Image.open(asset_path(cache, asset["file"])) as raw:
                images[key] = raw.copy()
        rgba = (
            images[key]
            .resize((radius * 2, radius * 2), Image.Resampling.LANCZOS)
            .rotate(rng.uniform(-55, 55), Image.Resampling.BICUBIC)
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
                key,
                {
                    "type": asset["style"],
                    "state": "in_flight" if airborne else "floor",
                    "partial_requested": hidden,
                },
            )
        )
    return layers


def rasterize(background, layers):
    rgb = background.copy()
    ids = Image.new("L", rgb.size)
    draw = ImageDraw.Draw(ids)
    for i, polygon in enumerate(HOOPS):
        draw.polygon(polygon, fill=200 + i)
    shadow = Image.new("L", rgb.size)
    d = ImageDraw.Draw(shadow)
    for layer in layers:
        x, y = layer.ground
        rx = (
            layer.alpha.height * 0.14
            if layer.identity.label != "ball"
            else layer.alpha.width * 0.48
        )
        ry = max(2, rx * 0.15)
        d.ellipse(
            (x - rx, y - ry, x + rx, y + ry), fill=35 if layer.identity.label != "ball" else 22
        )
    shadow = shadow.filter(ImageFilter.GaussianBlur(4))
    rgb = Image.composite(Image.new("RGB", rgb.size), rgb, shadow)
    amodal = {}
    for layer in sorted(layers, key=lambda l: (l.depth, l.identity.instance_id)):
        rgb.paste(layer.rgb, layer.position, layer.alpha)
        x, y = layer.position
        ids.paste(
            layer.identity.instance_id,
            (x, y, x + layer.alpha.width, y + layer.alpha.height),
            layer.alpha,
        )
        clipped = Image.new("L", rgb.size)
        clipped.paste(layer.alpha, layer.position)
        amodal[layer.identity.instance_id] = clipped.histogram()[255]
    hist = ids.histogram()
    ratios = {identity: hist[identity] / max(1, area) for identity, area in amodal.items()}
    return rgb, ids, ratios


def visible_ball_count(ids):
    hist = ids.histogram()
    return sum(hist[i] > 0 for i in range(1, 5))


def valid_ball_count(number, mode):
    return number == 0 if mode == "none" else number == 1 if mode == "single" else 2 <= number <= 4


def select_crop(ids, layers, plan, rng, shot):
    width, height = ids.size
    balls = [l for l in layers if l.identity.label == "ball"]
    people = [l for l in layers if l.identity.label == "player"]
    for _ in range(45):
        if shot == "full":
            box = (0, 0, width, height)
        else:
            crop_width = (
                rng.randint(1580, 1950)
                if shot == "half"
                else rng.randint(1100, 1570)
                if shot == "medium"
                else rng.randint(650, 1050)
            )
            crop_height = (
                height
                if shot == "half"
                else rng.randint(650, 900)
                if shot == "medium"
                else rng.randint(360, 680)
            )
            referees = [l for l in layers if l.identity.label == "referee"]
            focus = rng.choice(
                referees if not balls and referees and rng.random() < 0.35 else balls or people
            )
            fx = focus.position[0] + focus.alpha.width * 0.5
            fy = focus.position[1] + focus.alpha.height * 0.5
            x = max(0, min(width - crop_width, round(fx - crop_width * rng.uniform(0.2, 0.8))))
            y = max(0, min(height - crop_height, round(fy - crop_height * rng.uniform(0.25, 0.8))))
            box = (x, y, x + crop_width, y + crop_height)
        cropped = ids.crop(box)
        if valid_ball_count(visible_ball_count(cropped), plan["ball_mode"]):
            return box, shot
    return (0, 0, width, height), "full-fallback"


def preview(rgb, annotations, path):
    output = rgb.copy()
    draw = ImageDraw.Draw(output)
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font = (
        ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    )
    colors = ((255, 70, 100), (60, 180, 255), (255, 190, 45), (70, 245, 145))
    for a in annotations:
        x, y, w, h = a.bbox_xywh
        color = colors[a.class_id]
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=2)
        text = f"{a.label} {a.instance_id}"
        box = draw.textbbox((x, max(0, y - 18)), text, font=font)
        draw.rectangle(box, fill=(10, 15, 20))
        draw.text((x, max(0, y - 18)), text, font=font, fill=color)
    output.thumbnail((1280, 800))
    output.save(path, quality=92)


def render_scene(config, root, plan, assets, balls, images, pilot, deadline):
    destination = root / "renders" / plan["id"]
    cache = Path(config.get("cache", root / "cache"))
    if (destination / "done.json").exists():
        done = read(destination / "done.json")
        for name, digest in done["files"].items():
            if sha(destination / name) != digest:
                raise ValueError("Existing scene checksum mismatch: " + str(destination / name))
        return done["statistics"]
    if destination.exists():
        raise ValueError("Incomplete non-atomic scene directory: " + str(destination))
    if time.time() >= deadline:
        raise TimeoutError("Night budget exhausted; resume to finish the exact image count")
    with Image.open(config["background"]) as raw:
        background = raw.convert("RGB")
    for attempt in range(35):
        layers = make_layers(plan, assets, balls, cache, images, pilot, attempt)
        rgb, ids, ratios = rasterize(background, layers)
        person_ratios = [ratio for ident, ratio in ratios.items() if 10 <= ident < 40]
        if not all(r >= 0.08 for r in ratios.values()):
            continue
        if not valid_ball_count(visible_ball_count(ids), plan["ball_mode"]):
            continue
        if plan["occlude_ball"] and not any(0.12 < ratios[i] < 0.92 for i in ratios if i < 10):
            continue
        if not any(0.1 < r < 0.94 for r in person_ratios):
            continue
        break
    else:
        raise ValueError(f"Could not place valid, partially occluded objects for {plan['id']}")
    objects = tuple(l.identity for l in layers) + tuple(
        ObjectIdentity(200 + i, 3, "hoop") for i in range(5)
    )
    rng = random.Random(seed_for(plan["seed"], "views"))
    shots = ("full", "half", "medium", "tight")
    shot0 = rng.choices(shots, weights=(0.2, 0.35, 0.30, 0.15))[0]
    shot1 = rng.choice([s for s in shots if s != shot0])
    frames = []
    statistics = []
    previews = root / "previews"
    previews.mkdir(exist_ok=True)
    with new_directory(destination) as stage:
        for frame_index, shot in enumerate((shot0, shot1)):
            box, actual_shot = select_crop(ids, layers, plan, rng, shot)
            frame_rgb = rgb.crop(box)
            frame_ids = ids.crop(box)
            target = min(config["output_width"], frame_rgb.width)
            scale = target / frame_rgb.width
            size = (target, round(frame_rgb.height * scale))
            frame_rgb = frame_rgb.resize(size, Image.Resampling.LANCZOS)
            frame_ids = frame_ids.resize(size, Image.Resampling.NEAREST)
            # Global sensor/exposure augmentation, after compositing the full scene.
            frame_rgb = ImageEnhance.Brightness(frame_rgb).enhance(rng.uniform(0.94, 1.06))
            frame_rgb = ImageEnhance.Color(frame_rgb).enhance(rng.uniform(0.95, 1.04))
            annotations = visible_annotations(*size, frame_ids.tobytes(), objects, "basketball")
            if not valid_ball_count(sum(a.label == "ball" for a in annotations), plan["ball_mode"]):
                # Retain native crop resolution when a very small visible ball would
                # otherwise disappear. Never leave an unlabelled visible ball in RGB.
                frame_rgb = rgb.crop(box)
                frame_ids = ids.crop(box)
                size = frame_rgb.size
                annotations = visible_annotations(*size, frame_ids.tobytes(), objects, "basketball")
                if not valid_ball_count(
                    sum(a.label == "ball" for a in annotations), plan["ball_mode"]
                ):
                    raise ValueError("Invalid ball visibility even at native crop resolution")
            rgb_name = f"frame-{frame_index:03d}.png"
            ids_name = f"frame-{frame_index:03d}.instances.png"
            frame_rgb.save(stage / rgb_name, compress_level=2)
            frame_ids.save(stage / ids_name)
            by_id = {a.instance_id: a for a in annotations}
            frames.append(
                {
                    "frame_index": frame_index,
                    "rgb": rgb_name,
                    "instance_ids": ids_name,
                    "objects": [
                        {
                            **asdict(o),
                            "bbox_xywh": list(by_id[o.instance_id].bbox_xywh)
                            if o.instance_id in by_id
                            else None,
                            "visible_area": by_id[o.instance_id].visible_area
                            if o.instance_id in by_id
                            else 0,
                        }
                        for o in objects
                    ],
                }
            )
            stats = {
                "scene": plan["id"],
                "frame": frame_index,
                "split": plan["split"],
                "gender_mode": plan["gender_mode"],
                "ball_mode": plan["ball_mode"],
                "shot": actual_shot,
                "crop": list(box),
                "counts": dict(Counter(a.label for a in annotations)),
                "occlusion_full_scene": {str(k): round(1 - v, 4) for k, v in ratios.items()},
                "visible_area": {str(a.instance_id): a.visible_area for a in annotations},
            }
            # Compare each final mask with the same layer alone in this crop.
            # Border clipping is therefore distinct from inter-object occlusion.
            view_occlusion = {}
            for layer in layers:
                solo = Image.new("L", ids.size)
                solo.paste(layer.alpha, layer.position)
                area = solo.crop(box).resize(size, Image.Resampling.NEAREST).histogram()[255]
                visible = (
                    by_id[layer.identity.instance_id].visible_area
                    if layer.identity.instance_id in by_id
                    else 0
                )
                if area:
                    view_occlusion[str(layer.identity.instance_id)] = round(1 - visible / area, 4)
            stats["occlusion_in_view"] = view_occlusion
            statistics.append(stats)
            if plan["index"] < 20 or plan["index"] % 50 == 0:
                preview(frame_rgb, annotations, previews / f"{plan['id']}-{frame_index}.jpg")
        write(
            stage / "view-metadata.json",
            {
                "schema_version": "1.0",
                "plan": plan,
                "statistics": statistics,
                "layers": [
                    {
                        "id": l.identity.instance_id,
                        "asset": l.source,
                        "ground_pixel": list(l.ground),
                        "position_pixel": list(l.position),
                        **l.metadata,
                    }
                    for l in layers
                ],
                "annotation_semantics": "Photo cutout alpha and crop clipping; background hoops use manual board/rim/net envelopes.",
                "scale_calibration": {
                    "estimated_camera_height_m": 2.4,
                    "method": "foot-point distance from the rectified image horizon",
                    "metric_ground_truth": False,
                },
            },
        )
        # Per-view scenes are inside the group; a dedicated export wrapper preserves
        # group split rather than asking the stock exporter to randomize each crop.
        for frame in frames:
            with Image.open(stage / frame["rgb"]) as image:
                size = image.size
            write(
                stage / f"scene-{frame['frame_index']:03d}.json",
                {
                    "schema_version": "2.0",
                    "renderer": "local.hall-photo-composite",
                    "renderer_version": "1.0",
                    "job_id": plan["id"] + f"-v{frame['frame_index']}",
                    "sport": "basketball",
                    "seed": plan["seed"],
                    "labels": list(labels_for_sport("basketball")),
                    "resolution": {"width": size[0], "height": size[1]},
                    "requested_annotations": ["bounding_boxes", "instance_masks"],
                    "annotation_semantics": "Visible photo-alpha pixels; manually annotated background hoops. No original Blender human labels/keypoints reused.",
                    "assets": [
                        {"id": "hall-background", "sha256": config["background_sha256"]},
                        *[
                            {"id": a["id"], "sha256": a["sha256"], "split": a["split"]}
                            for a in assets
                            if a["id"] in {l.source for l in layers}
                        ],
                    ],
                    "frames": [frame],
                },
            )
        done = {
            "files": {p.name: sha(p) for p in stage.iterdir() if p.is_file()},
            "statistics": statistics,
        }
        write(stage / "done.json", done)
    return statistics
