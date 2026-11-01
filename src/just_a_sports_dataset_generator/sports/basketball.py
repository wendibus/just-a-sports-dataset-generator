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
from dataclasses import dataclass, replace

LAYOUT_SCHEMA = "1.1"
"""1.1: per-frame joints (animated poses, real jump arcs) and team benches."""
COURT_HALF_LENGTH, COURT_HALF_WIDTH = 14.0, 7.5
RIM_X, RIM_Z, BALL_RADIUS = 12.425, 3.05, 0.12
FRAME_INTERVAL = 0.2
GRAVITY = 9.81
DRIBBLE_PERIOD = 0.7  # seconds per bounce
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
class Motion:
    """How a pose changes over time. Times are seconds after the first frame."""

    phase0: float
    cadence: float  # cycles per second of the gait or sway
    speed: float  # ground speed in m/s
    jump: float  # peak height of a jump in metres, 0 for none
    takeoff: float  # moment the feet leave the floor (may be negative: already airborne)
    action: float  # moment of the release of a shot or pass


def _lerp(a: float, b: float, q: float) -> float:
    return a + (b - a) * q


def _unit_interval(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def dribble_phase(motion: Motion, t: float) -> float:
    """0 with the ball in the hand, 1 with the ball at the floor."""
    return abs(math.sin(motion.phase0 + t * math.pi / DRIBBLE_PERIOD))


def jump_height(motion: Motion, t: float) -> float:
    """Ballistic jump: height of the feet above the floor at time t."""
    if motion.jump <= 0:
        return 0.0
    air = 2 * math.sqrt(2 * motion.jump / GRAVITY)
    q = (t - motion.takeoff) / air
    return 4 * motion.jump * q * (1 - q) if 0 < q < 1 else 0.0


def animate(pose: Pose, motion: Motion, t: float) -> Pose:
    """The pose at time t: gait cycles, dribbling arm, shot and pass motions, sway."""
    phase = motion.phase0 + math.tau * motion.cadence * t
    s = math.sin(phase)
    (re, ra, rb), (le, la, lb) = pose.arms
    if pose.family == "run" and motion.speed >= 0.3:
        amplitude = min(max(0.08 + 0.09 * motion.speed, 0.1), 0.5)
        swing = max(re, le)
        # Each arm swings forward while the opposite leg leads.
        return replace(
            pose,
            stride=amplitude * s,
            crouch=pose.crouch + 0.02 * abs(math.cos(phase)),
            arms=((swing, 10 - 70 * s, rb), (swing, 10 + 70 * s, lb)),
        )
    if pose.family == "defend":
        return replace(
            pose,
            stance=pose.stance + 0.04 * s,
            stride=pose.stride + 0.06 * math.sin(phase / 2),
            arms=((re + 8 * s, ra, rb), (le - 8 * s, la, lb)),
        )
    if pose.family == "dribble":
        straighten = 1 - 0.5 * dribble_phase(motion, t)  # the arm pushes the ball down
        right = (re, ra, rb * straighten) if pose.ball_hand == "right" else (re, ra, rb)
        left = (le, la, lb * straighten) if pose.ball_hand == "left" else (le, la, lb)
        step = 0.1 * s if motion.speed >= 0.3 else 0.0
        return replace(pose, arms=(right, left), stride=pose.stride + step)
    if pose.family == "shoot":  # from the set point below the forehead to the release
        q = _unit_interval((t - motion.action + 0.4) / 0.5)
        return replace(
            pose,
            crouch=pose.crouch + 0.15 * (1 - q),
            arms=(
                (_lerp(110, re, q), ra, _lerp(95, 15, q)),
                (_lerp(105, le, q), la, _lerp(95, 15, q)),
            ),
        )
    if pose.family == "pass":  # ball at the chest, then arms extend towards the receiver
        q = _unit_interval((t - motion.action + 0.3) / 0.35)
        return replace(
            pose,
            crouch=pose.crouch + 0.05 * (1 - q),
            arms=(
                (_lerp(60, re, q), ra, _lerp(95, rb, q)),
                (_lerp(60, le, q), la, _lerp(95, lb, q)),
            ),
        )
    if pose.family == "rebound":  # load before the take-off, arms reach up in the air
        q = _unit_interval((t - motion.takeoff + 0.2) / 0.3)
        return replace(
            pose,
            crouch=pose.crouch + 0.18 * (1 - q),
            arms=((_lerp(60, re, q), ra, rb), (_lerp(60, le, q), la, lb)),
        )
    # stand, catch, signal and slow movement: small natural sway
    return replace(
        pose, stride=pose.stride + 0.02 * s, arms=((re + 4 * s, ra, rb), (le - 4 * s, la, lb))
    )


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
    rng = random.Random(f"basketball-layout:v1.1:{seed}")
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
    # Release of a pass or shot; it may already have happened before the first frame.
    duration = {"pass": rng.uniform(0.5, 0.9), "shot": rng.uniform(1.0, 1.4)}.get(ball_event, 1.0)
    release = rng.uniform(-0.6, 0.6) * duration
    times = [frame * FRAME_INTERVAL for frame in range(frames)]
    athletes: list[dict[str, object]] = []
    roots: dict[int, list[Root]] = {}
    frame_poses: dict[int, list[Pose]] = {}
    frame_joints: dict[int, list[dict[str, Vec]]] = {}
    motions: dict[int, Motion] = {}
    basket = (d * RIM_X, 0.0)
    receiver = 1 + rng.randrange(4)
    numbers = rng.sample([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 23, 33], 10)

    def add(
        identity: int,
        team: int,
        number: int,
        spot: tuple[float, float],
        family: str,
        target: tuple[float, float],
        spread: float,
        run_speed: float,
        holds: bool,
    ) -> None:
        x, y = spot
        pose = sample_pose(rng, family)
        velocity = _velocity(rng, situation, d, run_speed, family)
        moving = math.hypot(*velocity)
        if family == "run" and moving > 1.0:  # runners face where they run
            heading = facing(*velocity) + rng.uniform(-0.2, 0.2)
        else:
            heading = facing(target[0] - x, target[1] - y) + rng.uniform(-spread, spread)
        jumps = family == "rebound" or (
            family == "shoot" and situation != "free_throw" and rng.random() < 0.5
        )
        action = release if holds else rng.uniform(-0.2, 0.6)
        motion = Motion(
            phase0=rng.uniform(0, math.tau),
            cadence=rng.uniform(1.2, 1.7) if family == "run" else rng.uniform(0.3, 1.0),
            speed=moving,
            jump=rng.uniform(0.15, 0.45) if jumps else 0.0,
            takeoff=action - 0.25 if family == "shoot" else rng.uniform(-0.3, 0.5),
            action=action,
        )
        scale = rng.uniform(0.95, 1.08) if team < 2 else rng.uniform(0.95, 1.05)
        start = Root(x, y, 0.0, heading, scale)
        roots[identity] = [
            replace(_move(start, velocity, t), z=jump_height(motion, t)) for t in times
        ]
        frame_poses[identity] = [animate(pose, motion, t) for t in times]
        frame_joints[identity] = [skeleton(frame_pose) for frame_pose in frame_poses[identity]]
        motions[identity] = motion
        athletes.append(
            _athlete(
                rng,
                identity,
                team,
                number,
                frame_poses[identity],
                frame_joints[identity],
                roots[identity],
                kits,
            )
        )

    for index, spot in enumerate(offense + defense):
        family = _family(rng, situation, index, index == holder, index == receiver, ball_event)
        if index < 5:
            target = basket if situation != "jump_ball" else (0.0, 0.0)
        else:
            target = offense[index - 5]
        team = 0 if index < 5 else 1
        add(10 + index, team, numbers[index], spot, family, target, 0.35, speed, index == holder)
    referees = 1 + (rng.random() < 0.6)
    for k in range(referees):
        side = -1 if k == 0 else 1
        x = rng.uniform(-10, 10) if situation != "jump_ball" else rng.uniform(-1.5, 1.5)
        y = side * rng.uniform(6.2, 8.0)
        family = "signal" if rng.random() < 0.15 else rng.choice(("stand", "run"))
        add(30 + k, 2, 0, (x, y), family, (x * 0.7, 0.0), 0.6, speed * 0.6, False)
    ball = _ball_frames(
        rng,
        ball_event,
        times,
        release,
        duration,
        (10 + holder, 10 + receiver),
        d,
        roots,
        frame_poses,
        frame_joints,
        motions,
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
        "benches": _benches(rng, kits),
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


def _joints_json(joints: dict[str, Vec]) -> dict[str, list[float]]:
    return {name: [round(v, 5) for v in position] for name, position in joints.items()}


def _athlete(
    rng: random.Random,
    identity: int,
    team: int,
    number: int,
    poses: list[Pose],
    joints: list[dict[str, Vec]],
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
        "pose": {"family": poses[0].family, "crouch": round(poses[0].crouch, 5)},
        "joints": _joints_json(joints[0]),
        "frames": [
            {
                "position": [round(r.x, 4), round(r.y, 4), round(r.z, 4)],
                "heading": round(r.heading, 5),
                "height_scale": round(r.height_scale, 4),
                "crouch": round(pose.crouch, 5),
                "joints": _joints_json(frame_joints),
            }
            for r, pose, frame_joints in zip(roots, poses, joints, strict=True)
        ],
    }


def _ball_frames(
    rng: random.Random,
    event: str,
    times: list[float],
    start: float,
    duration: float,
    players: tuple[int, int],
    d: int,
    roots: dict[int, list[Root]],
    poses: dict[int, list[Pose]],
    joints: dict[int, list[dict[str, Vec]]],
    motions: dict[int, Motion],
) -> dict[str, object]:
    holder, receiver = players
    phase0 = rng.uniform(0, math.tau)

    def held(identity: int, frame: int) -> Vec:
        phase = dribble_phase(motions[identity], times[frame])
        local = _held_ball(poses[identity][frame], joints[identity][frame], phase)
        x, y, z = roots[identity][frame].world(local)
        return (x, y, max(z, BALL_RADIUS))  # height scaling must not push the ball into the floor

    rim: Vec = (d * RIM_X, 0.0, RIM_Z + BALL_RADIUS + 0.02)
    floor_start = (roots[holder][0].x + rng.uniform(-3, 3), rng.uniform(-6, 6))
    floor_velocity = (rng.uniform(-3, 3), rng.uniform(-2, 2))
    toss_peak = rng.uniform(3.3, 3.8)
    pass_apex = rng.uniform(0.2, 0.7)  # one arc per scene, so all frames share a trajectory
    positions: list[list[float]] = []
    states: list[str] = []
    for frame, t in enumerate(times):
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


BENCH_Y = 9.3
"""Team benches sit in front of the far stands, outside the 2 m zone behind the sideline."""
SKINS = ((0.40, 0.20, 0.10), (0.18, 0.075, 0.035), (0.68, 0.40, 0.23), (0.55, 0.32, 0.18))
SUITS = ((0.03, 0.03, 0.04), (0.15, 0.17, 0.22), (0.5, 0.5, 0.52))


def _benches(rng: random.Random, kits: list[tuple[float, float, float]]) -> list[dict[str, object]]:
    """Seated substitutes and standing coaches beside the court.

    They are rendered as background (instance 0): only people on the court are labelled,
    so benches act as hard negatives for people in team kit.
    """
    people: list[dict[str, object]] = []
    for team, (low, high) in enumerate(((-9.5, -2.5), (2.5, 9.5))):
        if rng.random() >= 0.75:
            continue
        seats = rng.sample(range(int((high - low) / 0.6)), rng.randint(3, 7))
        for seat in sorted(seats):
            people.append(
                {
                    "kind": "substitute",
                    "position": [round(low + 0.3 + seat * 0.6, 3), BENCH_Y],
                    "kit": list(kits[team]),
                    "skin": list(rng.choice(SKINS)),
                    "lean": round(rng.uniform(-0.15, 0.25), 3),
                }
            )
        if rng.random() < 0.6:
            people.append(
                {
                    "kind": "coach",
                    "position": [round(rng.uniform(low, high), 3), round(rng.uniform(8.3, 8.8), 3)],
                    "kit": list(rng.choice(SUITS)),
                    "skin": list(rng.choice(SKINS)),
                    "lean": round(rng.uniform(-0.1, 0.1), 3),
                }
            )
    return people


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
