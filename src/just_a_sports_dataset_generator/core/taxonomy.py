"""Canonical Reco Trainer sport and label compatibility contract."""

RECO_REVISION = "1fec3e34997a92e627600f68134a1b0c7eb19833"

SPORTS = (
    "basketball",
    "football",
    "futsal",
    "handball",
    "hockey",
    "rugby",
    "lacrosse",
    "american_football",
)

_LABELS_BY_SPORT: dict[str, tuple[str, ...]] = {
    "basketball": ("ball", "player", "referee", "hoop"),
    "hockey": ("puck", "player", "goalkeeper", "referee", "goal"),
    "rugby": ("ball", "player", "referee", "goalpost"),
    "american_football": ("ball", "player", "referee", "goalpost"),
    "football": ("ball", "player", "goalkeeper", "referee", "goal"),
    "futsal": ("ball", "player", "goalkeeper", "referee", "goal"),
    "handball": ("ball", "player", "goalkeeper", "referee", "goal"),
    "lacrosse": ("ball", "player", "goalkeeper", "referee", "goal"),
}


def labels_for_sport(sport: str) -> tuple[str, ...]:
    """Return Reco Trainer labels in canonical class-ID order."""
    try:
        return _LABELS_BY_SPORT[sport]
    except KeyError as exc:
        raise ValueError(f"unsupported Reco Trainer sport: {sport!r}") from exc
