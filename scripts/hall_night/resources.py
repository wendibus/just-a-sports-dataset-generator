"""Coordinate local hall runs and finish deferred background preparation."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from .common import asset_path, read, sha, validate, write


def active_runs(root):
    active = []
    for path in root.parent.glob("*/.night.lock"):
        if path.parent.resolve() == root.resolve():
            continue
        if not (path.parent / "config.json").is_file():
            continue
        with path.open("r+" if os.name == "nt" else "r") as handle:
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    active.append(path.parent.name)
            else:
                import fcntl

                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    active.append(path.parent.name)
    return active


def wait_for_other_runs(root):
    announced = None
    while running := active_runs(root):
        if running != announced:
            print(
                "Warte auf laufenden Hallen-Batch: "
                + ", ".join(running)
                + ". Strg+C beendet nur dieses Warten.",
                flush=True,
            )
            announced = running
        time.sleep(30)
    if announced:
        print("Hallen-Warteschlange frei. Starte Futsal.", flush=True)


def finish_background(root, config, invoke, dashboard):
    # Validate inputs and immutable code before launching the existing local model.
    validate("hall-night-config-v2.0.json", config)
    if (root / "run-fingerprint.json").exists():
        raise ValueError("Cannot alter a hall background after scene generation started")
    for name, digest in read(root / "prepared.json")["engine_sha256"].items():
        if sha(asset_path(Path(config["generator_root"]), name)) != digest:
            raise ValueError("Frozen engine changed: " + name)
    if sha(Path(config["background"])) != config["background_sha256"]:
        raise ValueError("Background changed before cleanup")
    for key, digest in config["model_hashes"].items():
        if sha(Path(config["model"][key])) != digest:
            raise ValueError("Model changed before cleanup")
    source = root / "background-before-futsal.png"
    if not source.exists():
        shutil.copyfile(config["background"], source)
    request = {
        "background": str(source),
        "model": config["model"],
        "output": str(root / "background-preparation"),
        "worker": str(
            Path(config["generator_root"])
            / "src/just_a_sports_dataset_generator/generators/qwen_worker.py"
        ),
    }
    write(root / "background-request.json", request)
    dashboard(
        root,
        {
            "status": "background",
            "images": 0,
            "requested": config["count"],
            "message": "Lokale Entfernung der hochgeklappten Wandtore; danach Futsal-Personen",
        },
    )
    invoke(
        [
            config["qwen_python"],
            "-I",
            "-u",
            str(Path(__file__).with_name("futsal_background.py")),
            str(root / "background-request.json"),
        ],
        root / "background-preparation.log",
    )
    config["background"] = str(root / "background-preparation/background.png")
    config["background_sha256"] = sha(Path(config["background"]))
    config["background_cleanup"] = False
    write(root / "config.json", config)
