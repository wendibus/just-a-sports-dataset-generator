#!/usr/bin/env python3
"""Prepare an isolated, local hall run from already installed components. No downloads."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

from hall_night.common import asset_path, read, sha, write

from just_a_sports_dataset_generator.storage import new_directory


def copy_file(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def freeze(root: Path, config):
    if (root / "renders").exists() or (root / "dataset").exists():
        raise ValueError("Do not change a run after full rendering has started; choose a new root")
    repo = Path(__file__).resolve().parents[1]
    engine = root / "engine"
    if engine.exists():
        raise ValueError("An engine snapshot already exists; it is intentionally immutable")
    # Snapshot only code/contracts/licenses, never repository credentials or weights.
    with new_directory(engine) as stage:
        shutil.copytree(
            repo / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        shutil.copytree(repo / "schemas", stage / "schemas")
        shutil.copytree(repo / "licenses", stage / "licenses")
        copy_file(repo / "LICENSE", stage / "LICENSE")
        copy_file(repo / "docs/HALL_NIGHT.md", stage / "docs/HALL_NIGHT.md")
        if config.get("sport") == "futsal":
            copy_file(repo / "docs/FUTSAL_HALL.md", stage / "docs/FUTSAL_HALL.md")
        copy_file(repo / "scripts/run_hall_night.py", stage / "scripts/run_hall_night.py")
        copy_file(repo / "scripts/foreground_masks.py", stage / "scripts/foreground_masks.py")
        shutil.copytree(
            repo / "scripts/hall_night",
            stage / "scripts/hall_night",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    source = Path(config["source_crops"])
    destination = root / "source-crops"
    for crop in read(source / "crops.json")["crops"]:
        for kind in ("rgb", "mask"):
            name = crop["files"][kind]
            copy_file(asset_path(source, name), destination / name)
    copy_file(source / "crops.json", destination / "crops.json")
    copy_file(Path(config["background"]), root / "background.png")
    for key, name in (
        ("ball_reference", "ball-reference.png"),
        ("referee_reference", "referee-reference.png"),
    ):
        copy_file(Path(config[key]), root / "reference-sheets" / name)
        config[key] = str(root / "reference-sheets" / name)
    config["background"] = str(root / "background.png")
    config["source_crops"] = str(destination)
    config["generator_root"] = str(engine)
    config["background_sha256"] = sha(root / "background.png")
    if not config.get("model_hashes"):
        config["model_hashes"] = {k: sha(Path(config["model"][k])) for k in ("gguf", "lora")}
    write(root / "config.json", config)
    quote = shlex.quote
    starter = f"""#!/bin/zsh
set -euo pipefail
task_root={quote(str(root))}
task_engine={quote(str(engine))}
task_python={quote(sys.executable)}
export PYTHONPATH="$task_engine/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$task_engine"
print "Lokaler Hallen-Nachtlauf: {config["count"]} Bilder. Derselbe Befehl setzt einen Abbruch fort."
print "Fortschritt: $task_root/index.html"
print "Mit Strg+C nur diesen Generator stoppen."
/usr/bin/open "$task_root/index.html"
/usr/bin/caffeinate -i /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' "$task_python" -u "$task_engine/scripts/run_hall_night.py" "$task_root" run 2>&1 | /usr/bin/tee -a "$task_root/Nachtlauf.log"
print "Fertig: $task_root/dataset/coco"
/usr/bin/open "$task_root/index.html"
"""
    start = root / "2.000 Hallenbilder starten.command"
    start.write_text(starter)
    start.chmod(0o755)
    verification = f"""#!/bin/zsh
set -euo pipefail
export PYTHONPATH={quote(str(engine / "src"))}
cd {quote(str(engine))}
{quote(sys.executable)} {quote(str(engine / "scripts/run_hall_night.py"))} {quote(str(root))} verify
"""
    target = root / "Datensatz pruefen.command"
    target.write_text(verification)
    target.chmod(0o755)
    write(
        root / "prepared.json",
        {
            "schema_version": "1.0",
            "engine_sha256": {
                str(p.relative_to(engine)): sha(p) for p in sorted(engine.rglob("*")) if p.is_file()
            },
            "source_crop_index_sha256": sha(destination / "crops.json"),
            "downloads": False,
            "note": "Immutable code/input snapshot; model weights and Python environments remain at the existing local paths.",
        },
    )
    print("Prepared launcher:", start, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--freeze-existing",
        action="store_true",
        help="Snapshot an already prepared config before full rendering begins",
    )
    parser.add_argument("--background", type=Path)
    parser.add_argument("--crops", type=Path)
    parser.add_argument("--ball-reference", type=Path)
    parser.add_argument("--referee-reference", type=Path)
    parser.add_argument(
        "--model-request",
        type=Path,
        help="Existing local Qwen request containing base_dir, gguf and lora",
    )
    parser.add_argument("--qwen-python", type=Path)
    parser.add_argument("--matte-python", type=Path)
    parser.add_argument("--matte-model", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.freeze_existing:
        config = read(root / "config.json")
    else:
        if (root / "config.json").exists():
            parser.error("Existing configuration: use --freeze-existing or a new root")
        required = (
            "background",
            "crops",
            "ball_reference",
            "referee_reference",
            "model_request",
            "qwen_python",
            "matte_python",
            "matte_model",
        )
        if any(getattr(args, name) is None for name in required):
            parser.error("Supply all local component/input paths")
        model = read(args.model_request)
        config = {
            "schema_version": "1.0",
            "seed": 20261103,
            "count": 2000,
            "max_hours": 10,
            "resolution": 640,
            "background": str(args.background.resolve()),
            "background_sha256": sha(args.background),
            "source_crops": str(args.crops.resolve()),
            "ball_reference": str(args.ball_reference.resolve()),
            "referee_reference": str(args.referee_reference.resolve()),
            "qwen_python": str(args.qwen_python.absolute()),
            "matte_python": str(args.matte_python.absolute()),
            "matte_model": str(args.matte_model.resolve()),
            "model": {k: model[k] for k in ("base_dir", "gguf", "lora")},
            "model_request_sha256": sha(args.model_request),
            "generator_root": str(Path(__file__).resolve().parents[1]),
            "output_width": 1600,
            "gender_weights": {"men": 0.4, "women": 0.4, "mixed": 0.2},
            "ball_weights": {"single": 0.7, "multi": 0.2, "none": 0.1},
        }
    if os.name != "posix":
        print("Python runner is portable; generated .command launchers target macOS.", flush=True)
    freeze(root, config)


if __name__ == "__main__":
    main()
