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
