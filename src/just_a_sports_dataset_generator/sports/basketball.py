"""Deterministic basketball scene planning, independent of any renderer.

The planner turns a scene seed into a complete, JSON-serializable layout: game situation,
hall and lighting variation, every athlete's skeleton (joint positions), position and
heading per frame, and the ball's world position per frame. A render backend only builds
geometry from the layout, so the sport rules are testable without Blender and every
renderer sees identical scenes.

Coordinates: metres, court centre at the origin, x along the 28 m length, y across the
15 m width, z up. Athlete-local coordinates face -y with the right side at -x; a root
transform (position, heading about z, height scale on z) maps them to the world.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

LAYOUT_SCHEMA = "1.0"
COURT_HALF_LENGTH, COURT_HALF_WIDTH = 14.0, 7.5
RIM_X, RIM_Z, BALL_RADIUS = 12.425, 3.05, 0.12
FRAME_INTERVAL = 0.2
"""Seconds between consecutive frames of a scene."""

SITUATIONS = ("half_court", "transition", "free_throw", "jump_ball")
SITUATION_WEIGHTS = (0.5, 0.25, 0.15, 0.1)

# Standing skeleton of a ~1.95 m athlete (athlete-local, metres).
THIGH, SHIN, UPPER_ARM, FOREARM = 0.45, 0.40, 0.30, 0.27
HIP_Z, ANKLE_Z = 0.97, 0.13
UPPER_BODY = {
    "neck": (0.0, 0.0, 1.58),
    "head": (0.0, -0.005, 1.80),
    "nose": (0.0, -0.102, 1.80),
    "right_eye": (-0.040, -0.092, 1.83),
    "left_eye": (0.040, -0.092, 1.83),
    "right_ear": (-0.102, 0.0, 1.79),
    "left_ear": (0.102, 0.0, 1.79),
    "right_shoulder": (-0.25, 0.0, 1.48),
    "left_shoulder": (0.25, 0.0, 1.48),
}
OPENPOSE_18 = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)

Vec = tuple[float, float, float]


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k, a[2] * k)


def _length(a: Vec) -> float:
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def _unit(a: Vec) -> Vec:
    length = _length(a)
    return _scale(a, 1 / length) if length > 1e-9 else (0.0, 0.0, 0.0)


def _arm_direction(side: int, elevation: float, azimuth: float) -> Vec:
    """Unit vector from elevation (deg from straight down) and azimuth (0 out, 90 forward)."""
    e, a = math.radians(elevation), math.radians(azimuth)
    return (side * math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), -math.cos(e))


def _knee(hip: Vec, ankle: Vec) -> tuple[Vec, Vec]:
    """Two-bone IK in the plane of the leg, knee bending forward (-y). Returns knee, ankle."""
    delta = _sub(ankle, hip)
    distance = _length(delta)
    if distance >= THIGH + SHIN - 1e-6:  # out of reach: straight leg towards the target
        ankle = _add(hip, _scale(_unit(delta), THIGH + SHIN - 1e-4))
        delta, distance = _sub(ankle, hip), THIGH + SHIN - 1e-4
    along = (THIGH**2 - SHIN**2 + distance**2) / (2 * distance)
    height = math.sqrt(max(THIGH**2 - along**2, 0.0))
    axis = _unit(delta)
    forward: Vec = (0.0, -1.0, 0.0)
    dot = sum(f * a for f, a in zip(forward, axis, strict=True))
    bend = _unit(_sub(forward, _scale(axis, dot)))
    return _add(_add(hip, _scale(axis, along)), _scale(bend, height)), ankle


@dataclass(frozen=True, slots=True)
class Pose:
    family: str
    crouch: float  # metres the upper body sinks; knees bend to keep the feet down
    stance: float  # lateral distance of each ankle from the centre line
    stride: float  # right foot forward (+) or back (-), left foot opposite
    arms: tuple[tuple[float, float, float], tuple[float, float, float]]
    """(elevation, azimuth, elbow bend) in degrees for the right and left arm."""
    ball_hand: str | None  # "right", "left", "both" or None


def sample_pose(rng: random.Random, family: str) -> Pose:
    u = rng.uniform
    if family == "stand":
        arms = ((u(8, 25), u(-10, 30), u(5, 35)), (u(8, 25), u(-10, 30), u(5, 35)))
        return Pose(family, u(0, 0.05), u(0.12, 0.2), u(-0.12, 0.12), arms, None)
    if family == "run":
        swing = u(20, 50)
        stride = rng.choice((-1, 1)) * u(0.25, 0.45)
        forward = 80 if stride < 0 else -60  # arm swings opposite to the leading leg
        arms = ((swing, forward, u(60, 100)), (swing, -forward, u(60, 100)))
        return Pose(family, u(0.03, 0.09), u(0.1, 0.16), stride, arms, None)
    if family == "defend":
        arms = ((u(40, 100), u(10, 60), u(15, 60)), (u(40, 100), u(10, 60), u(15, 60)))
        return Pose(family, u(0.12, 0.22), u(0.28, 0.4), u(-0.15, 0.15), arms, None)
    if family == "dribble":
        hand = rng.choice(("right", "left"))
        low = (u(20, 35), u(55, 85), u(30, 60))
        free = (u(25, 60), u(0, 50), u(30, 80))
        arms = (low, free) if hand == "right" else (free, low)
        return Pose(family, u(0.06, 0.16), u(0.15, 0.26), u(-0.3, 0.3), arms, hand)
    if family == "shoot":
        bend = u(30, 80)
        arms = ((u(140, 170), u(75, 90), bend), (u(130, 160), u(60, 85), bend))
        return Pose(family, u(0.0, 0.08), u(0.14, 0.22), u(-0.1, 0.15), arms, "both")
    if family == "pass":
        arms = ((u(70, 95), u(75, 90), u(5, 35)), (u(70, 95), u(75, 90), u(5, 35)))
        return Pose(family, u(0.03, 0.1), u(0.15, 0.24), u(0.0, 0.35), arms, "both")
    if family == "catch":
        arms = ((u(60, 100), u(60, 90), u(20, 50)), (u(60, 100), u(60, 90), u(20, 50)))
        return Pose(family, u(0.04, 0.12), u(0.15, 0.25), u(-0.2, 0.2), arms, "both")
    if family == "rebound":
        arms = ((u(150, 178), u(60, 90), u(0, 25)), (u(150, 178), u(60, 90), u(0, 25)))
        return Pose(family, u(0.0, 0.05), u(0.14, 0.22), u(-0.1, 0.1), arms, None)
    if family == "signal":  # referee arm up
        arms = ((u(150, 175), u(0, 40), u(0, 20)), (u(10, 30), u(0, 30), u(5, 30)))
        return Pose(family, 0.0, u(0.12, 0.18), u(-0.1, 0.1), arms, None)
    raise ValueError(f"unknown pose family {family!r}")


def skeleton(pose: Pose) -> dict[str, Vec]:
    """Athlete-local joints for a pose: feet on the floor, constant bone lengths."""
    drop: Vec = (0.0, 0.0, -pose.crouch)
    joints = {name: _add(position, drop) for name, position in UPPER_BODY.items()}
    for side, prefix, arm in ((-1, "right", pose.arms[0]), (1, "left", pose.arms[1])):
        elevation, azimuth, bend = arm
        shoulder = joints[f"{prefix}_shoulder"]
        elbow = _add(shoulder, _scale(_arm_direction(side, elevation, azimuth), UPPER_ARM))
        lower = _arm_direction(side, min(elevation + bend, 179.0), azimuth)
        joints[f"{prefix}_elbow"] = elbow
        joints[f"{prefix}_wrist"] = _add(elbow, _scale(lower, FOREARM))
        hip: Vec = (side * 0.12, 0.0, HIP_Z - pose.crouch)
        step = pose.stride / 2 * (1 if side == -1 else -1)
        knee, ankle = _knee(hip, (side * pose.stance, -step, ANKLE_Z))
        joints |= {f"{prefix}_hip": hip, f"{prefix}_knee": knee, f"{prefix}_ankle": ankle}
    return joints


@dataclass(frozen=True, slots=True)
class Root:
    x: float
    y: float
    z: float
    heading: float
    height_scale: float

    def world(self, local: Vec) -> Vec:
        lx, ly, lz = local[0], local[1], local[2] * self.height_scale
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (self.x + c * lx - s * ly, self.y + s * lx + c * ly, self.z + lz)


def facing(dx: float, dy: float) -> float:
    """Heading that turns the athlete's local -y towards the direction (dx, dy)."""
    return math.atan2(dy, dx) + math.pi / 2


def _held_ball(pose: Pose, joints: dict[str, Vec], phase: float) -> Vec:
    """Athlete-local ball centre for a ball in this pose's hands."""
    if pose.ball_hand in ("right", "left"):
        wrist = joints[f"{pose.ball_hand}_wrist"]
        depth = 0.16 + phase * max(wrist[2] - 0.16 - BALL_RADIUS, 0.0)  # dribble bounce
        return (wrist[0], wrist[1] - 0.05, max(wrist[2] - depth, BALL_RADIUS))
    right, left = joints["right_wrist"], joints["left_wrist"]
    middle = _scale(_add(right, left), 0.5)
    return (middle[0], middle[1] - 0.1, middle[2])


def _parabola(start: Vec, end: Vec, apex: float, progress: float) -> Vec:
    progress = min(max(progress, 0.0), 1.0)
    point = _add(start, _scale(_sub(end, start), progress))
    lift = 4 * apex * progress * (1 - progress)
    return (point[0], point[1], point[2] + lift)


def _palette(rng: random.Random) -> list[tuple[float, float, float]]:
    kits = [
        (0.025, 0.13, 0.62),
        (0.86, 0.09, 0.035),
        (0.9, 0.9, 0.88),
        (0.02, 0.02, 0.025),
        (0.05, 0.45, 0.12),
        (0.95, 0.62, 0.05),
        (0.42, 0.06, 0.45),
        (0.0, 0.5, 0.62),
    ]
    rng.shuffle(kits)
    return kits[:2]


def _spread(
    rng: random.Random,
    spots: list[tuple[float, float]],
    jitter: float,
    taken: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Jitter spots while keeping at least 0.65 m between athletes."""
    placed: list[tuple[float, float]] = []
    for sx, sy in spots:
        x, y = sx, sy
        for attempt in range(40):
            scale = jitter * (1 - attempt / 50)
            x, y = sx + rng.uniform(-scale, scale), sy + rng.uniform(-scale, scale)
            if all(math.hypot(x - ox, y - oy) >= 0.65 for ox, oy in taken + placed):
                break
        placed.append((x, y))
    return placed


def _situation(
    rng: random.Random, name: str, d: int
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    """Offense and defense floor spots for attack direction d, plus a typical speed (m/s)."""
    basket = (d * RIM_X, 0.0)
    if name == "half_court":
        spots = [
            (d * 4.9, 0.0),
            (d * 7.0, 4.6),
            (d * 7.0, -4.6),
            (d * 12.8, 6.6),
            (d * 12.8, -6.6),
            (d * 11.2, 2.3),
            (d * 11.2, -2.3),
            (d * 8.2, 1.2),
        ]
        rng.shuffle(spots)
        offense = spots[:5]
        defense: list[tuple[float, float]] = []
        for ox, oy in offense:
            k = rng.uniform(0.18, 0.35)
            defense.append((ox + (basket[0] - ox) * k, oy + (basket[1] - oy) * k))
        return offense, defense, 1.5
    if name == "transition":
        offense = [(d * rng.uniform(-6, 10), rng.uniform(-6.5, 6.5)) for _ in range(5)]
        defense = [(x + d * rng.uniform(0.8, 3.5), y + rng.uniform(-1.5, 1.5)) for x, y in offense]
        return offense, defense, 4.5
    if name == "free_throw":
        lane = [
            (d * 9.0, 2.75),
            (d * 9.0, -2.75),
            (d * 10.2, 2.75),
            (d * 10.2, -2.75),
            (d * 11.3, 2.75),
            (d * 11.3, -2.75),
        ]
        rng.shuffle(lane)
        offense = [(d * 8.2, 0.0), lane[0], lane[1], (d * 4.5, 3.0), (d * 4.5, -3.0)]
        defense = [lane[2], lane[3], lane[4], lane[5], (d * 3.5, 0.0)]
        return offense, defense, 0.2
    # jump ball at the centre circle
    offense = [
        (-0.45 * d, 0.0),
        (-3.0 * d, 2.2),
        (-3.0 * d, -2.2),
        (-2.0 * d, 4.5),
        (-2.0 * d, -4.5),
    ]
    defense = [(0.45 * d, 0.0), (2.9 * d, 2.0), (2.9 * d, -2.0), (2.2 * d, 4.6), (2.2 * d, -4.6)]
    return offense, defense, 0.3


def plan_scene(seed: int, frames: int) -> dict[str, object]:
    """Plan one basketball scene. Same seed and frame count, same layout."""
    rng = random.Random(f"basketball-layout:v1:{seed}")
    situation = rng.choices(SITUATIONS, SITUATION_WEIGHTS)[0]
    d = rng.choice((-1, 1))
    offense_spots, defense_spots, speed = _situation(rng, situation, d)
    jitter = {"half_court": 0.8, "transition": 0.6, "free_throw": 0.15, "jump_ball": 0.25}[
        situation
    ]
    taken: list[tuple[float, float]] = []
    offense = _spread(rng, offense_spots, jitter, taken)
    taken += offense
    defense = _spread(rng, defense_spots, jitter, taken)
    taken += defense
    kits = _palette(rng)
    holder = 0  # spot order is shuffled, so the first offense athlete is a random spot
    ball_event = _ball_event(rng, situation)
    athletes: list[dict[str, object]] = []
    roots: dict[int, list[Root]] = {}
    poses: dict[int, Pose] = {}
    skeletons: dict[int, dict[str, Vec]] = {}
    basket = (d * RIM_X, 0.0)
    receiver = 1 + rng.randrange(4)
    numbers = rng.sample([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 23, 33], 10)
    for index, (x, y) in enumerate(offense + defense):
        team = 0 if index < 5 else 1
        identity = 10 + index
        family = _family(rng, situation, index, index == holder, index == receiver, ball_event)
        pose = sample_pose(rng, family)
        if index < 5:
            target = basket if situation != "jump_ball" else (0.0, 0.0)
        else:
            mate = offense[index - 5]
            target = (mate[0], mate[1])
        heading = facing(target[0] - x, target[1] - y) + rng.uniform(-0.35, 0.35)
        velocity = _velocity(rng, situation, d, speed, family)
        jump = (
            rng.uniform(0.15, 0.5)
            if family == "rebound" or (family == "shoot" and rng.random() < 0.5)
            else 0.0
        )
        scale = rng.uniform(0.95, 1.08)
        roots[identity] = [
            _move(Root(x, y, jump, heading, scale), velocity, frame * FRAME_INTERVAL)
            for frame in range(frames)
        ]
        poses[identity], skeletons[identity] = pose, skeleton(pose)
        athletes.append(
            _athlete(
                rng,
                identity,
                team,
                numbers[index],
                pose,
                skeletons[identity],
                roots[identity],
                kits,
            )
        )
    referees = 1 + (rng.random() < 0.6)
    for k in range(referees):
        identity = 30 + k
        side = -1 if k == 0 else 1
        x = rng.uniform(-10, 10) if situation != "jump_ball" else rng.uniform(-1.5, 1.5)
        y = side * rng.uniform(6.2, 8.0)
        family = "signal" if rng.random() < 0.15 else rng.choice(("stand", "run"))
        pose = sample_pose(rng, family)
        heading = facing(-x * 0.3, -y) + rng.uniform(-0.6, 0.6)
        velocity = _velocity(rng, situation, d, speed * 0.6, family)
        roots[identity] = [
            _move(Root(x, y, 0.0, heading, rng.uniform(0.95, 1.05)), velocity, f * FRAME_INTERVAL)
            for f in range(frames)
        ]
        poses[identity], skeletons[identity] = pose, skeleton(pose)
        athletes.append(
            _athlete(rng, identity, 2, 0, pose, skeletons[identity], roots[identity], kits)
        )
    ball = _ball_frames(
        rng, ball_event, frames, 10 + holder, 10 + receiver, d, roots, poses, skeletons
    )
    xs = [r[0].x for r in roots.values()]
    return {
        "schema_version": LAYOUT_SCHEMA,
        "seed": seed,
        "situation": situation,
        "attack_direction": d,
        "frame_interval_s": FRAME_INTERVAL,
        "focus_x": sum(xs) / len(xs),
        "hall": _hall(rng),
        "athletes": athletes,
        "ball": ball,
    }


def _family(
    rng: random.Random, situation: str, index: int, holder: bool, receiver: bool, event: str
) -> str:
    team = 0 if index < 5 else 1
    if situation == "jump_ball":  # both centres (offense 0, defense 5) jump for the ball
        return "rebound" if index in (0, 5) else rng.choice(("stand", "defend"))
    if holder:
        return (
            {"held": "dribble", "pass": "pass", "shot": "shoot", "rim": "shoot", "floor": "run"}[
                event
            ]
            if situation != "free_throw"
            else "shoot"
        )
    if receiver and event == "pass":
        return "catch"
    if event in ("rim", "shot") and rng.random() < 0.35:
        return "rebound"
    if situation == "transition":
        return "run"
    if situation == "free_throw":
        return rng.choice(("stand", "defend"))
    return (
        rng.choice(("defend", "defend", "stand", "run"))
        if team == 1
        else rng.choice(("stand", "run", "defend"))
    )


def _ball_event(rng: random.Random, situation: str) -> str:
    if situation == "jump_ball":
        return "toss"
    if situation == "free_throw":
        return rng.choice(("held", "shot", "shot"))
    return rng.choices(("held", "pass", "shot", "rim", "floor"), (0.45, 0.22, 0.15, 0.08, 0.1))[0]


def _velocity(
    rng: random.Random, situation: str, d: int, speed: float, family: str
) -> tuple[float, float]:
    if family in ("shoot", "pass", "catch", "signal") or situation in ("free_throw", "jump_ball"):
        return (0.0, 0.0)
    magnitude = rng.uniform(0.3, 1.0) * speed
    if situation == "transition":
        angle = (0.0 if d > 0 else math.pi) + rng.uniform(-0.6, 0.6)
    else:
        angle = rng.uniform(0, math.tau)
    return (magnitude * math.cos(angle), magnitude * math.sin(angle))


def _move(root: Root, velocity: tuple[float, float], seconds: float) -> Root:
    x = min(max(root.x + velocity[0] * seconds, -15.0), 15.0)
    y = min(max(root.y + velocity[1] * seconds, -8.6), 8.6)
    return Root(x, y, root.z, root.heading, root.height_scale)


def _athlete(
    rng: random.Random,
    identity: int,
    team: int,
    number: int,
    pose: Pose,
    joints: dict[str, Vec],
    roots: list[Root],
    kits: list[tuple[float, float, float]],
) -> dict[str, object]:
    skins = [
        (0.40, 0.20, 0.10),
        (0.18, 0.075, 0.035),
        (0.68, 0.40, 0.23),
        (0.33, 0.15, 0.07),
        (0.55, 0.32, 0.18),
    ]
    return {
        "instance_id": identity,
        "team": team,
        "label": "referee" if team == 2 else "player",
        "number": number,
        "kit": list(kits[team]) if team < 2 else [0.5, 0.52, 0.53],
        "skin": list(rng.choice(skins)),
        "hair": rng.choice(("short", "short", "bald", "long")),
        "pose": {"family": pose.family, "crouch": pose.crouch},
        "joints": {name: [round(v, 5) for v in position] for name, position in joints.items()},
        "frames": [
            {
                "position": [round(r.x, 4), round(r.y, 4), round(r.z, 4)],
                "heading": round(r.heading, 5),
                "height_scale": round(r.height_scale, 4),
            }
            for r in roots
        ],
    }


def _ball_frames(
    rng: random.Random,
    event: str,
    frames: int,
    holder: int,
    receiver: int,
    d: int,
    roots: dict[int, list[Root]],
    poses: dict[int, Pose],
    skeletons: dict[int, dict[str, Vec]],
) -> dict[str, object]:
    phase0 = rng.uniform(0, math.tau)

    def held(identity: int, frame: int) -> Vec:
        phase = abs(math.sin(phase0 + frame * FRAME_INTERVAL * math.tau / 0.7))
        local = _held_ball(poses[identity], skeletons[identity], phase)
        x, y, z = roots[identity][frame].world(local)
        return (x, y, max(z, BALL_RADIUS))  # height scaling must not push the ball into the floor

    rim: Vec = (d * RIM_X, 0.0, RIM_Z + BALL_RADIUS + 0.02)
    duration = {"pass": rng.uniform(0.5, 0.9), "shot": rng.uniform(1.0, 1.4)}.get(event, 1.0)
    start = rng.uniform(-0.6, 0.6) * duration  # the event may already be under way at frame 0
    floor_start = (roots[holder][0].x + rng.uniform(-3, 3), rng.uniform(-6, 6))
    floor_velocity = (rng.uniform(-3, 3), rng.uniform(-2, 2))
    toss_peak = rng.uniform(3.3, 3.8)
    pass_apex = rng.uniform(0.2, 0.7)  # one arc per scene, so all frames share a trajectory
    positions: list[list[float]] = []
    states: list[str] = []
    for frame in range(frames):
        t = frame * FRAME_INTERVAL
        progress = (t - start) / duration
        if event == "held":
            point, state = held(holder, frame), "held"
        elif event == "pass":
            if progress <= 0:
                point, state = held(holder, frame), "held"
            elif progress >= 1:
                point, state = held(receiver, frame), "held"
            else:
                point = _parabola(held(holder, frame), held(receiver, frame), pass_apex, progress)
                state = "air"
        elif event == "shot":
            if progress <= 0:
                point, state = held(holder, frame), "held"
            elif progress >= 1:
                drop = min((progress - 1) * duration * 4.0, RIM_Z)
                point, state = (rim[0], rim[1], max(rim[2] - drop, BALL_RADIUS)), "air"
            else:
                apex = max(4.3 - held(holder, 0)[2], 0.8)
                point, state = _parabola(held(holder, frame), rim, apex, progress), "air"
        elif event == "rim":
            wobble = 0.1 * math.sin(phase0 + t * 9)
            point, state = (
                (rim[0] - d * 0.15, wobble, rim[2] + 0.05 + 0.25 * abs(math.sin(t * 5))),
                "air",
            )
        elif event == "floor":
            x = floor_start[0] + floor_velocity[0] * t
            y = floor_start[1] + floor_velocity[1] * t
            bounce = 0.6 * math.exp(-t) * abs(math.sin(phase0 + t * 6))
            point, state = (x, y, BALL_RADIUS + bounce), "floor"
        else:  # toss at a jump ball
            p = min(max(0.35 + (t - start * 0.3) / 1.2, 0.0), 1.0)
            point, state = (0.0, 0.0, 2.2 + 4 * (toss_peak - 2.2) * p * (1 - p)), "air"
        positions.append([round(v, 4) for v in point])
        states.append(state)
    return {
        "event": event,
        "holder": holder,
        "receiver": receiver,
        "positions": positions,
        "states": states,
    }


def _hall(rng: random.Random) -> dict[str, object]:
    u = rng.uniform
    wood = rng.choice(((0.65, 0.39, 0.16), (0.72, 0.5, 0.26), (0.55, 0.3, 0.12), (0.78, 0.6, 0.38)))
    key = rng.choice(
        ((0.018, 0.055, 0.10), (0.45, 0.04, 0.03), (0.02, 0.2, 0.08), (0.3, 0.3, 0.32), wood)
    )
    wall = rng.choice(
        ((0.28, 0.30, 0.32), (0.55, 0.55, 0.52), (0.12, 0.18, 0.3), (0.62, 0.6, 0.55))
    )
    return {
        "floor_color": [round(c * u(0.9, 1.1), 4) for c in wood],
        "floor_gloss": round(u(0.2, 0.45), 3),
        "key_color": list(key),
        "line_color": list(rng.choice(((0.9, 0.91, 0.86), (0.95, 0.85, 0.2), (0.05, 0.05, 0.05)))),
        "wall_color": list(wall),
        "seat_colors": [
            list(rng.choice(((0.035, 0.12, 0.22), (0.5, 0.06, 0.05), (0.08, 0.3, 0.12)))),
            [0.46, 0.49, 0.51],
        ],
        "light_energy": round(u(1100, 2600), 1),
        "light_kelvin": round(u(3800, 6500), 0),
        "light_rows": rng.choice((2, 3)),
        "spectator_density": round(rng.choice((0.0, u(0.05, 0.3), u(0.3, 0.8))), 3),
        "spectator_seed": rng.randrange(2**31),
    }
