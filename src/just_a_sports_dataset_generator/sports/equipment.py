"""Ball and referee appearance catalogue (``equipment.json``), independent of any renderer.

The catalogue holds regulation ball sizes and the looks of match balls and referee kits for
basketball, football and futsal, compiled from datasheets for the 2025/26 season. Planners
sample from it with their own random stream; renderers only receive plain colours. Brand
and league names are provenance and never reach a rendered image. See docs/EQUIPMENT.md.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import cast

from just_a_sports_dataset_generator.core import EQUIPMENT_PROFILES

SPORTS = ("basketball", "football", "futsal")
PROFILES = EQUIPMENT_PROFILES
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
RGB = tuple[float, float, float]


class EquipmentError(ValueError):
    """The equipment catalogue is malformed."""


@dataclass(frozen=True, slots=True)
class BallRegulation:
    size: int
    circumference_cm: tuple[float, float]
    mass_g: tuple[float, float]
    source: str
    verified: bool

    @property
    def radius_m(self) -> float:
        """Radius of a ball in the middle of the allowed circumference range."""
        mean = sum(self.circumference_cm) / 2
        return mean / 100 / math.tau


@dataclass(frozen=True, slots=True)
class BallDesign:
    id: str
    base: str
    panels: str | None
    seams: str | None
    accents: tuple[str, ...]
    weight: float
    observed_in: tuple[str, ...]

    def render_spec(self) -> dict[str, object]:
        """Linear-RGB colours for a renderer; no names of brands or leagues."""
        return {
            "design": self.id,
            "base": list(linear(self.base)),
            "panels": list(linear(self.panels)) if self.panels else None,
            "seams": list(linear(self.seams)) if self.seams else None,
            "accents": [list(linear(color)) for color in self.accents],
        }


@dataclass(frozen=True, slots=True)
class RefereeKit:
    id: str
    shirt: str
    sleeves: str
    pattern: str
    accent: str | None
    lower: str
    lower_color: str
    socks: str
    weight: float
    observed_in: tuple[str, ...]

    def render_spec(self) -> dict[str, object]:
        return {
            "kit": self.id,
            "shirt": list(linear(self.shirt)),
            "sleeves": list(linear(self.sleeves)),
            "pattern": self.pattern,
            "accent": list(linear(self.accent)) if self.accent else None,
            "lower": self.lower,
            "lower_color": list(linear(self.lower_color)),
            "socks": list(linear(self.socks)),
        }


@dataclass(frozen=True, slots=True)
class SportEquipment:
    sport: str
    regulation: BallRegulation
    balls: tuple[BallDesign, ...]
    referee_kits: tuple[RefereeKit, ...]
    profiles: dict[str, dict[str, dict[str, float]]]
    """Weight overrides: profile -> "balls"/"referee_kits" -> id -> weight."""

    def weight(self, profile: str, group: str, identity: str, default: float) -> float:
        return self.profiles.get(profile, {}).get(group, {}).get(identity, default)


PATTERNS = frozenset({"plain", "vertical_stripes", "chest_pockets"})
LOWER = frozenset({"trousers", "shorts"})


def linear(color: str) -> RGB:
    """sRGB hex to linear RGB, the colour space of Blender material inputs."""

    def channel(value: int) -> float:
        c = value / 255
        return round(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4, 4)

    return (
        channel(int(color[1:3], 16)),
        channel(int(color[3:5], 16)),
        channel(int(color[5:7], 16)),
    )


def _object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EquipmentError(f"{where}: expected a JSON object")
    return cast(dict[str, object], value)


def _array(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise EquipmentError(f"{where}: expected a JSON array")
    return cast(list[object], value)


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise EquipmentError(f"{where}: expected a non-empty string")
    return value


def _color(value: object, where: str) -> str:
    if not isinstance(value, str) or not HEX.match(value):
        raise EquipmentError(f"{where}: expected an sRGB hex colour like #A1B2C3")
    return value


def _optional_color(value: object, where: str) -> str | None:
    return None if value is None else _color(value, where)


def _number(value: object, where: str) -> float:
    if type(value) not in (int, float):
        raise EquipmentError(f"{where}: expected a number")
    return float(cast(float, value))


def _range(value: object, where: str) -> tuple[float, float]:
    items = _array(value, where)
    if len(items) != 2:
        raise EquipmentError(f"{where}: expected [min, max]")
    low, high = _number(items[0], where), _number(items[1], where)
    if not 0 < low <= high:
        raise EquipmentError(f"{where}: expected 0 < min <= max")
    return (low, high)


def _common(raw: dict[str, object], where: str) -> tuple[str, float, tuple[str, ...]]:
    identity = _text(raw.get("id"), where)
    if not identity.isidentifier():
        raise EquipmentError(f"{where}: id must be an identifier")
    weight = _number(raw.get("weight"), where)
    if weight <= 0:
        raise EquipmentError(f"{where}: weight must be positive")
    observed = tuple(_text(o, where) for o in _array(raw.get("observed_in", []), where))
    return identity, weight, observed


def _ball(raw: dict[str, object], where: str) -> BallDesign:
    identity, weight, observed = _common(raw, where)
    return BallDesign(
        identity,
        _color(raw.get("base"), where),
        _optional_color(raw.get("panels"), where),
        _optional_color(raw.get("seams"), where),
        tuple(_color(a, where) for a in _array(raw.get("accents", []), where)),
        weight,
        observed,
    )


def _kit(raw: dict[str, object], where: str) -> RefereeKit:
    identity, weight, observed = _common(raw, where)
    pattern, lower = _text(raw.get("pattern"), where), _text(raw.get("lower"), where)
    if pattern not in PATTERNS or lower not in LOWER:
        raise EquipmentError(f"{where}: unknown pattern or lower garment")
    return RefereeKit(
        identity,
        _color(raw.get("shirt"), where),
        _color(raw.get("sleeves"), where),
        pattern,
        _optional_color(raw.get("accent"), where),
        lower,
        _color(raw.get("lower_color"), where),
        _color(raw.get("socks"), where),
        weight,
        observed,
    )


def _profiles(
    value: object, sport: str, balls: list[BallDesign], kits: list[RefereeKit]
) -> dict[str, dict[str, dict[str, float]]]:
    known = {"balls": {b.id for b in balls}, "referee_kits": {k.id for k in kits}}
    defaults = {
        "balls": {b.id: b.weight for b in balls},
        "referee_kits": {k.id: k.weight for k in kits},
    }
    result: dict[str, dict[str, dict[str, float]]] = {}
    for profile, raw in _object(value, f"{sport} profiles").items():
        where = f"{sport} profile {profile}"
        if profile not in PROFILES or profile == "international":
            raise EquipmentError(f"{where}: unknown profile")
        groups: dict[str, dict[str, float]] = {}
        for group, weights in _object(raw, where).items():
            if group not in known:
                raise EquipmentError(f"{where}: unknown group {group}")
            parsed = {key: _number(v, where) for key, v in _object(weights, where).items()}
            if set(parsed) - known[group] or any(v < 0 for v in parsed.values()):
                raise EquipmentError(f"{where}: unknown id or negative weight in {group}")
            if not any(parsed.get(key, w) > 0 for key, w in defaults[group].items()):
                raise EquipmentError(f"{where}: every {group} weight is zero")
            groups[group] = parsed
        result[profile] = groups
    return result


def parse(data: object) -> dict[str, SportEquipment]:
    """Validate the catalogue; raises EquipmentError with the offending entry."""
    root = _object(data, "equipment")
    if root.get("schema_version") != "1.0":
        raise EquipmentError("unsupported equipment schema version")
    if root.get("default_profile") != "international":
        raise EquipmentError("default_profile must be international")
    sports = _object(root.get("sports"), "sports")
    if set(sports) != set(SPORTS):
        raise EquipmentError(f"equipment must describe exactly {', '.join(SPORTS)}")
    result: dict[str, SportEquipment] = {}
    for sport in SPORTS:
        entry = _object(sports[sport], sport)
        ball = _object(entry.get("ball"), f"{sport} ball")
        rule = _object(ball.get("regulation"), f"{sport} regulation")
        size = _number(rule.get("size"), f"{sport} ball size")
        regulation = BallRegulation(
            int(size),
            _range(rule.get("circumference_cm"), f"{sport} ball circumference"),
            _range(rule.get("mass_g"), f"{sport} ball mass"),
            _text(rule.get("source"), f"{sport} ball source"),
            rule.get("verified") is True,
        )
        balls = [
            _ball(_object(raw, sport), f"{sport} ball design")
            for raw in _array(ball.get("designs"), f"{sport} ball designs")
        ]
        referee = _object(entry.get("referee"), f"{sport} referee")
        kits = [
            _kit(_object(raw, sport), f"{sport} referee kit")
            for raw in _array(referee.get("kits"), f"{sport} referee kits")
        ]
        if not balls or not kits:
            raise EquipmentError(f"{sport}: needs at least one ball design and referee kit")
        for names in ([b.id for b in balls], [k.id for k in kits]):
            if len(names) != len(set(names)):
                raise EquipmentError(f"{sport}: duplicate ids")
        profiles = _profiles(entry.get("profiles", {}), sport, balls, kits)
        result[sport] = SportEquipment(sport, regulation, tuple(balls), tuple(kits), profiles)
    return result


@cache
def catalogue() -> dict[str, SportEquipment]:
    path = resources.files("just_a_sports_dataset_generator.sports") / "equipment.json"
    return parse(json.loads(path.read_text(encoding="utf-8")))


def _distance(a: RGB, b: RGB) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def _check_profile(profile: str) -> None:
    if profile not in PROFILES:
        raise ValueError(f"equipment profile must be one of: {', '.join(PROFILES)}")


def choose_ball(rng: random.Random, sport: str, profile: str = "international") -> BallDesign:
    _check_profile(profile)
    entry = catalogue()[sport]
    weights = [entry.weight(profile, "balls", d.id, d.weight) for d in entry.balls]
    return rng.choices(entry.balls, weights)[0]


def choose_referee_kit(
    rng: random.Random,
    sport: str,
    team_kits: list[RGB],
    min_distance: float = 0.25,
    profile: str = "international",
) -> RefereeKit:
    """Weighted pick among kits whose shirt differs from every team kit (linear RGB).

    The laws of all three sports require referee colours that differ from both teams.
    Kits with weight 0 in the profile are never used; if no kit qualifies, the contrast
    rule is relaxed before the profile.
    """
    _check_profile(profile)
    entry = catalogue()[sport]

    def weight(kit: RefereeKit) -> float:
        return entry.weight(profile, "referee_kits", kit.id, kit.weight)

    usable = [kit for kit in entry.referee_kits if weight(kit) > 0]
    allowed = [
        kit
        for kit in usable
        if all(_distance(linear(kit.shirt), team) >= min_distance for team in team_kits)
    ] or usable
    return rng.choices(allowed, [weight(kit) for kit in allowed])[0]
