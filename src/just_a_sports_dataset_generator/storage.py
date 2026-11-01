"""Local JSON, checksums, and non-overwriting publication helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def new_directory(destination: Path) -> Generator[Path, None, None]:
    """Build in a sibling staging folder; never merge into an existing dataset."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sports-staging-", dir=destination.parent))
    try:
        yield staging
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output already exists: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@contextmanager
def resumable_directory(
    destination: Path, fingerprint: dict[str, object]
) -> Generator[tuple[Path, bool], None, None]:
    """Build in ``<destination>.partial`` and keep it when interrupted, so a rerun resumes.

    Yields the work folder and whether an earlier run is being resumed. A leftover folder
    is only reused when its fingerprint (command, inputs and settings) matches exactly;
    on success it is published as ``destination``, which is never merged into.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    marker = partial / "resume.json"
    resumed = partial.exists()
    if resumed:
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if previous != fingerprint:
            raise FileExistsError(
                f"an unfinished run with other settings exists: {partial}; "
                "delete it or choose another output folder"
            )
        for stale in partial.glob(".sports-staging-*"):  # interrupted mid-scene
            shutil.rmtree(stale)
    else:
        partial.mkdir(parents=True)
        write_json(marker, fingerprint)
    yield partial, resumed
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    marker.unlink()
    partial.rename(destination)
