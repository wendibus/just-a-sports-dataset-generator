#!/usr/bin/env python3
"""Offline, resumable photographic hall dataset. No automatic training or downloads."""

from __future__ import annotations

import argparse
import contextlib
import html
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from hall_night.assets import (
    asset_plan,
    build_balls,
    finalize_assets,
    prepare_references,
    prompt_for,
)
from hall_night.common import asset_path, read, seed_for, sha, write
from hall_night.export import export_and_verify, verify
from hall_night.render import make_plans, render_scene
from PIL import Image


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            handle.write("0")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another hall run is active") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another hall run is active") from exc
        yield


def invoke(command, log_path, progress=None):
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt"
        )
        try:
            while True:
                try:
                    status = process.wait(timeout=10)
                    if progress:
                        progress()
                    break
                except subprocess.TimeoutExpired:
                    if progress:
                        progress()
        except BaseException:
            if process.poll() is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    if status:
        raise RuntimeError(f"Subprocess failed ({status}); see {log_path}")


def dashboard(root, state):
    write(root / "state.json", state)
    images = sorted((root / "previews").glob("*.jpg"))[-12:]
    gallery = "".join(
        f'<a href="previews/{p.name}"><img src="previews/{p.name}"></a>' for p in images
    )
    asset_status = (
        f"<p>Personen-Bausteine: {state['assets_done']} / {state['assets_total']}</p>"
        if "assets_done" in state
        else ""
    )
    if not gallery and state.get("status") == "assets":
        photos = sorted((root / "cache/photos").glob("*.png"), key=lambda p: p.stat().st_mtime)[-6:]
        gallery = "".join(f'<img src="cache/photos/{p.name}">' for p in photos)
    complete = state["status"] == "complete"
    dataset_dir = state.get("dataset_dir", "dataset")
    links = (
        f'<p><a href="{dataset_dir}/coco/">COCO-Ordner für Reco Trainer</a> · <a href="{dataset_dir}/manifest.json">Bilanz</a> · <a href="{dataset_dir}/verification.json">Prüfung</a></p>'
        if complete
        else ""
    )
    if (root / "pilot/index.html").exists():
        links += '<p><a href="pilot/index.html">Pilotbilder ansehen</a></p>'
    refresh = "" if complete else '<meta http-equiv="refresh" content="30">'
    text = f"""<!doctype html><meta charset="utf-8">{refresh}<title>Hallen-Nachtlauf</title><style>body{{font:16px system-ui;background:#17191d;color:#eee;margin:24px}}img{{width:48%;margin:.5%}}a{{color:#9df}}pre{{white-space:pre-wrap}}</style><h1>Hallen-Nachtlauf</h1><p>Status: {state["status"]} – {html.escape(state.get("message", ""))}</p><p>{state.get("images", 0)} / {state.get("requested", 2000)} Bilder</p>{links}{asset_status}<p>Lokale 2,5D-Fotokompositionen. Vorschauen mit Labels; Trainingsbilder ohne eingeblendete Boxen.</p>{gallery}"""
    temporary = root / "index.partial.html"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(root / "index.html")


def preflight(root, config):
    from jsonschema import Draft202012Validator

    schema = read(Path(__file__).resolve().parents[1] / "schemas/hall-night-config-v1.0.json")
    Draft202012Validator(schema).validate(config)
    make_plans(config["count"], config["seed"], config["gender_weights"], config["ball_weights"])
    from just_a_sports_dataset_generator.core import labels_for_sport

    if list(labels_for_sport("basketball")) != ["ball", "player", "referee", "hoop"]:
        raise ValueError("Reco taxonomy changed")
    for key in (
        "background",
        "qwen_python",
        "matte_python",
        "matte_model",
        "source_crops",
        "ball_reference",
        "referee_reference",
    ):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"{key}: {config[key]}")
    for key, digest in config.get("model_hashes", {}).items():
        if sha(Path(config["model"][key])) != digest:
            raise ValueError("Qwen model checksum mismatch: " + key)
    for key in ("gguf", "lora", "base_dir"):
        if not Path(config["model"][key]).exists():
            raise FileNotFoundError(config["model"][key])
    if sha(Path(config["background"])) != config["background_sha256"]:
        raise ValueError("Hall background changed; start a new run directory")
    with Image.open(config["background"]) as image:
        if image.size != (3200, 900):
            raise ValueError("This calibration requires the approved 3200x900 hall")
    if shutil.disk_usage(root).free < 15 * 1024**3:
        raise ValueError("Keep at least 15 GiB free for the run, export and resume files")
    if config["resolution"] not in (512, 640, 768):
        raise ValueError("Unsupported local asset resolution")
    prepared = root / "prepared.json"
    if prepared.exists():
        for name, digest in read(prepared)["engine_sha256"].items():
            if sha(asset_path(Path(config["generator_root"]), name)) != digest:
                raise ValueError("The frozen engine was modified: " + name)
    expected = {
        "config": config,
        "base_model_file_stats": {
            str(p.relative_to(Path(config["model"]["base_dir"]))): [
                p.stat().st_size,
                p.stat().st_mtime_ns,
            ]
            for p in sorted(Path(config["model"]["base_dir"]).rglob("*"))
            if p.is_file()
        },
        "hall_schema_sha256": {
            p.name: sha(p)
            for p in sorted((Path(config["generator_root"]) / "schemas").glob("hall-*.json"))
        },
        "core_source_sha256": {
            str(p.relative_to(Path(config["generator_root"]))): sha(p)
            for p in sorted((Path(config["generator_root"]) / "src").rglob("*.py"))
        },
        "source_crops_sha256": sha(Path(config["source_crops"]) / "crops.json"),
        "ball_reference_sha256": sha(Path(config["ball_reference"])),
        "referee_reference_sha256": sha(Path(config["referee_reference"])),
        "runner_sha256": sha(Path(__file__)),
        "common_sha256": sha(Path(__file__).with_name("hall_night") / "common.py"),
        "schema_sha256": sha(
            Path(__file__).resolve().parents[1] / "schemas/hall-night-config-v1.0.json"
        ),
        "assets_sha256": sha(Path(__file__).with_name("hall_night") / "assets.py"),
        "render_sha256": sha(Path(__file__).with_name("hall_night") / "render.py"),
        "export_sha256": sha(Path(__file__).with_name("hall_night") / "export.py"),
        "matte_checks_sha256": sha(Path(__file__).with_name("hall_night") / "matte_checks.py"),
        "photo_worker_sha256": sha(Path(__file__).with_name("hall_night") / "photo_worker.py"),
    }
    fingerprint = root / "run-fingerprint.json"
    if fingerprint.exists() and read(fingerprint) != expected:
        raise ValueError(
            "Run settings/code/input changed; use a new output directory rather than mixing versions"
        )
    write(fingerprint, expected)


def generate_assets(root, config, deadline, pilot=False):
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    plan = asset_plan(config["seed"])
    prepare_references(config, cache, plan)
    for a in plan:
        a["prompt"] = prompt_for(a)
    write(cache / "asset-plan.json", {"schema_version": "1.0", "assets": plan})
    if pilot:
        wanted = {"p-woman-red-00", "p-woman-yellow-01", "p-man-red-02", "p-man-yellow-00", "r-04"}
        selected = [a for a in plan if a["id"] in wanted]
    else:
        selected = plan
    attempts = (
        read(cache / "repair-attempts.json") if (cache / "repair-attempts.json").exists() else {}
    )
    for cycle in range(3):
        if time.time() >= deadline:
            raise TimeoutError("Asset time budget exhausted; rerun to resume")
        requested = []
        for asset in selected:
            a = dict(asset)
            attempt = attempts.get(a["id"], 0)
            if attempt:
                a["seed"] = seed_for(a["seed"], f"repair-{attempt}")
                a["prompt"] += (
                    " Keep generous empty margins around both hands and shoes; one complete human only."
                )
            requested.append(a)
        request = {
            "root": str(cache),
            "worker": str(
                Path(config["generator_root"])
                / "src/just_a_sports_dataset_generator/generators/qwen_worker.py"
            ),
            "model": config["model"],
            "resolution": config["resolution"],
            "deadline": deadline,
            "assets": requested,
        }
        write(cache / "photo-request.json", request)
        print(
            f"Local photographs: {len(selected)} planned; log: {root / 'asset-generation.log'}",
            flush=True,
        )
        last_done = [-1]

        def progress(last_done=last_done):
            done = sum((cache / "photos" / (a["id"] + ".png")).exists() for a in selected)
            if done != last_done[0]:
                print(f"Personen-Bausteine: {done}/{len(selected)}", flush=True)
                last_done[0] = done
                dashboard(
                    root,
                    {
                        "status": "assets",
                        "images": 0,
                        "requested": config["count"],
                        "assets_done": done,
                        "assets_total": len(selected),
                        "message": "Qwen erzeugt lokale Fotos; danach Freistellung und Szenenaufbau",
                    },
                )

        progress()
        invoke(
            [
                config["qwen_python"],
                "-I",
                "-u",
                str(Path(__file__).with_name("hall_night") / "photo_worker.py"),
                str(cache / "photo-request.json"),
            ],
            root / "asset-generation.log",
            progress=progress,
        )
        if any(not (cache / "photos" / (a["id"] + ".png")).exists() for a in selected):
            raise TimeoutError("Photographs incomplete; resume before rendering")
        dashboard(
            root,
            {
                "status": "matting",
                "images": 0,
                "requested": config["count"],
                "assets_done": len(selected),
                "assets_total": len(selected),
                "message": "Freistellung, Randkontrolle und Prüfung auf zusätzliche Objekte",
            },
        )
        accepted, rejected = finalize_assets(config, cache, plan)
        rejected = [r for r in rejected if r["id"] in {a["id"] for a in selected}]
        if not rejected:
            break
        repair = [r for r in rejected if attempts.get(r["id"], 0) < 2]
        if not repair:
            break
        folder = cache / "rejected"
        folder.mkdir(exist_ok=True)
        for failure in repair:
            identity = failure["id"]
            attempt = attempts.get(identity, 0)
            (cache / "photos" / (identity + ".png")).rename(
                folder / f"{identity}-attempt-{attempt}.png"
            )
            attempts[identity] = attempt + 1
        write(cache / "repair-attempts.json", attempts)
    build_balls(config, cache)
    return [a for a in accepted if not pilot or a["id"] in wanted]


def render_all(root, config, plans, assets, deadline, pilot=False):
    cache = Path(config.get("cache", root / "cache"))
    balls = read(cache / "balls.json")["assets"]
    for asset in assets + balls:
        if sha(asset_path(cache, asset["file"])) != asset["sha256"]:
            raise ValueError("Cutout changed since asset QC")
    write(
        root / "scene-plan.json",
        {
            "schema_version": "1.0",
            "images": len(plans) * 2,
            "scenes": plans,
            "pilot_preview_only": pilot,
        },
    )
    snapshot = {"assets": assets, "balls": balls}
    fingerprint = root / "library-fingerprint.json"
    if fingerprint.exists() and read(fingerprint) != snapshot:
        raise ValueError(
            "The photo library changed after rendering started; use a new run directory"
        )
    write(fingerprint, snapshot)
    images = {}
    statistics = []
    for index, plan in enumerate(plans):
        stats = render_scene(config, root, plan, assets, balls, images, pilot, deadline)
        statistics.extend(stats)
        if index % 10 == 0 or index == len(plans) - 1:
            print(f"Rendered {(index + 1) * 2}/{len(plans) * 2} images", flush=True)
            dashboard(
                root,
                {
                    "status": "rendering",
                    "images": (index + 1) * 2,
                    "requested": len(plans) * 2,
                    "message": "Szenen und sichtbare Labels werden geprüft",
                },
            )
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "stage", choices=("run", "assets", "render", "pilot", "verify"), nargs="?", default="run"
    )
    parser.add_argument(
        "--hours",
        type=float,
        help="Override runtime budget for this invocation; settings and image count remain fixed",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = read(root / "config.json")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    started = time.time()
    deadline = started + 3600 * (args.hours if args.hours is not None else config["max_hours"])
    if deadline <= started:
        parser.error("Time budget must be positive")
    with lock(root / ".night.lock"):
        if args.stage == "verify":
            verify(root / "dataset")
            return
        preflight(root, config)
        dashboard(
            root,
            {
                "status": "starting",
                "images": 0,
                "requested": config["count"],
                "message": "Lokaler, fortsetzbarer Lauf; keine Downloads",
            },
        )
        try:
            if args.stage in ("run", "assets", "pilot"):
                assets = generate_assets(root, config, deadline, pilot=args.stage == "pilot")
            else:
                assets = read(root / "cache/library.json")["assets"]
            if args.stage == "assets":
                dashboard(
                    root,
                    {
                        "status": "assets-ready",
                        "images": 0,
                        "requested": config["count"],
                        "message": f"{len(assets)} Personen-Bausteine; Kontaktbögen im cache-Ordner",
                    },
                )
                return
            if args.stage == "pilot":
                destination = root / "pilot"
                destination.mkdir(exist_ok=True)
                settings = {**config, "cache": str(root / "cache")}
                plans = make_plans(12, config["seed"])
                # Guarantee all three composition types are present in the visual pilot.
                for i, plan in enumerate(plans):
                    plan["gender_mode"] = ("women", "men", "mixed")[i % 3]
                    plan["ball_mode"] = ("single", "multi", "multi", "none", "none", "single")[i]
                    plan["occlude_ball"] = plan["ball_mode"] != "none" and i < 3
                render_all(destination, settings, plans, assets, deadline, pilot=True)
                path = export_and_verify(destination, settings, plans, pilot=True)
                dashboard(
                    destination,
                    {
                        "status": "complete",
                        "images": 12,
                        "requested": 12,
                        "message": "Pilotvorschau – kleine Bibliothek, nicht als Evaluationsdatensatz verwenden",
                        "dataset_dir": "pilot-dataset",
                    },
                )
                print("Pilot complete:", path, flush=True)
                dashboard(
                    root,
                    {
                        "status": "pilot-ready",
                        "images": 12,
                        "requested": config["count"],
                        "message": "Pilot fertig; der vollständige Nachtlauf wurde nicht gestartet",
                    },
                )
                return
            plans = make_plans(
                config["count"], config["seed"], config["gender_weights"], config["ball_weights"]
            )
            render_all(root, config, plans, assets, deadline)
            dashboard(
                root,
                {
                    "status": "exporting",
                    "images": config["count"],
                    "requested": config["count"],
                    "message": "COCO-Export und unabhängige Prüfung sämtlicher Masken und Boxen",
                },
            )
            dataset = export_and_verify(root, config, plans)
            dashboard(
                root,
                {
                    "status": "complete",
                    "images": config["count"],
                    "requested": config["count"],
                    "message": "COCO vollständig exportiert und alle Masken geprüft",
                },
            )
            print(
                "COMPLETE:",
                dataset,
                "elapsed hours:",
                round((time.time() - started) / 3600, 2),
                flush=True,
            )
        except BaseException as exc:
            dashboard(
                root,
                {
                    "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                    "images": 2 * len(list((root / "renders").glob("*/done.json"))),
                    "requested": config["count"],
                    "message": str(exc) or "Unterbrochen – derselbe Befehl setzt fort",
                },
            )
            raise


if __name__ == "__main__":
    main()
