from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def seed_for(seed: int, name: str):
    return int.from_bytes(hashlib.sha256(f"hall-v1:{seed}:{name}".encode()).digest()[:4], "big")


def asset_path(root: Path, name: str):
    from just_a_sports_dataset_generator.annotations.scene import safe_file

    return safe_file(root, name)


def validate(name: str, data):
    from jsonschema import Draft202012Validator

    schema = read(Path(__file__).resolve().parents[2] / "schemas" / name)
    Draft202012Validator(schema).validate(data)
