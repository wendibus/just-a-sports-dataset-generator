#!/usr/bin/env python3
"""Replace an unstarted schematic football venue with an offline Cycles CPU render."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from hall_night.common import read, sha, validate, write
from hall_night.natural_venue import assert_unstarted, prepare_layers
from prepare_hall_night import copy_file
from run_hall_night import dashboard, lock


def snapshot(repo, target):
    shutil.copytree(
        repo / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("schemas", "licenses"):
        shutil.copytree(repo / name, target / name)
    copy_file(repo / "LICENSE", target / "LICENSE")
    for name in ("HALL_NIGHT.md", "FOOTBALL.md"):
        copy_file(repo / "docs" / name, target / "docs" / name)
    for name in ("run_hall_night.py", "foreground_masks.py", "upgrade_football_venue.py"):
        copy_file(repo / "scripts" / name, target / "scripts" / name)
    shutil.copytree(
        repo / "scripts/hall_night",
        target / "scripts/hall_night",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def install(root: Path, render: Path, assets: Path):
    root = root.absolute()
    repo = Path(__file__).resolve().parents[1]
    with lock(root / ".night.lock"):
        assert_unstarted(root)
        config = read(root / "config.json")
        validate("hall-night-config-v3.0.json", config)
        if config["sport"] != "football":
            raise ValueError("Football only")
        prepared = read(root / "prepared.json")
        for name, digest in prepared["engine_sha256"].items():
            if sha(Path(config["generator_root"]) / name) != digest:
                raise ValueError("Frozen source changed unexpectedly: " + name)
        venue = root / "venue-natural"
        if venue.exists():
            raise ValueError(
                "Natural venue already installed or partially present; inspect before replacing"
            )
        if (root / "cache/goals-natural").exists():
            raise ValueError("Natural goal cache already exists")
        backup = None
        changed = False
        saved_names = (
            "config.json",
            "prepared.json",
            "camera.json",
            "preparation.json",
            "Rasenplatz-Vorlage.jpg",
            "Rasenplatz-fotorealistisch.png",
            "Referenzen.html",
            "FOOTBALL.md",
            "state.json",
            "index.html",
        )
        work = root / ".venue-staging"
        work.mkdir(exist_ok=False)
        try:
            layers = work / "layers"
            records = prepare_layers(render, layers)
            natural = work / "venue-natural"
            natural.mkdir()
            for name in (
                "with-goals.png",
                "without-goals.png",
                "goal-ids.pgm",
                "camera-check.json",
            ):
                shutil.copy2(render / name, natural / name)
            shutil.copy2(layers / "background.png", natural / "background.png")
            shutil.copy2(layers / "checks.json", natural / "checks.json")
            shutil.copytree(assets, natural / "assets")
            for a in read(assets / "sources.json")["assets"]:
                if sha(natural / "assets" / a["file"]) != a["sha256"]:
                    raise ValueError("Texture checksum mismatch")
            source = (
                repo
                / "src/just_a_sports_dataset_generator/renderers/blender_worker/football_venue.py"
            )
            write(
                natural / "provenance.json",
                {
                    "schema_version": "1.0",
                    "renderer": "Blender Cycles CPU",
                    "worker_sha256": sha(source),
                    "assets": read(assets / "sources.json"),
                    "camera": "same pinhole camera, principal point moved to y=660",
                    "photograph": False,
                    "note": "Physically shaded 3D render with CC0 photograph-based environment; no Qwen generation or real-footage evaluation",
                },
            )
            engine = work / "engine"
            snapshot(repo, engine)
            backup = root / "Sicherung-vor-natuerlichem-Platz"
            if backup.exists():
                raise ValueError("Backup already exists")
            backup.mkdir()
            for name in saved_names:
                if (root / name).exists():
                    shutil.copy2(root / name, backup / name)
            shutil.copy2(root / "cache/goals.json", backup / "goals.json")
            # Stage all data before touching the active configuration.
            changed = True
            natural.rename(venue)
            shutil.copytree(layers / "goals-natural", root / "cache/goals-natural")
            write(root / "cache/goals.json", {"schema_version": "1.0", "assets": records})
            (root / "engine").rename(backup / "engine")
            engine.rename(root / "engine")
            config["background"] = str(venue / "background.png")
            config["background_sha256"] = sha(venue / "background.png")
            config["note"] = (
                "Natural Cycles CPU venue with CC0 textures and matching goal masks. No human-photo pilot or training benefit validated."
            )
            camera = read(root / "camera.json")
            camera["principal_point_px"] = [1600.0, 660.0]
            camera["source"] = (
                "Original local Cycles CPU football venue with CC0 environment and turf textures"
            )
            write(root / "camera.json", camera)
            prepared["engine_sha256"] = {
                str(p.relative_to(root / "engine")): sha(p)
                for p in sorted((root / "engine").rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            }
            write(root / "prepared.json", prepared)
            write(root / "config.json", config)
            shutil.copy2(render / "with-goals.png", root / "Rasenplatz-fotorealistisch.png")
            from PIL import Image

            with Image.open(render / "with-goals.png") as image:
                image.convert("RGB").save(root / "Rasenplatz-Vorlage.jpg", quality=95)
            for name in ("FOOTBALL.md",):
                shutil.copy2(repo / "docs" / name, root / name)
            html = root / "Referenzen.html"
            text = html.read_text().replace(
                "Originale prozedurale CPU-Vorlage mit bekannter Kamera. Kein echtes Stadionfoto und noch kein Nachweis fotorealistischer Qualität.",
                "Lokal gerenderte räumliche Platzvorlage mit natürlicher Rasentextur, realer Lichtumgebung, Bäumen und passenden Tor-Masken. Blender Cycles auf zwei CPU-Threads; kein echtes Stadionfoto und noch kein Fußball-Personenpilot.",
            )
            text = text.replace(
                "</main>",
                '<p>Texturen: <a href="https://ambientcg.com/view?id=Grass004">ambientCG Grass004</a>, CC0; <a href="https://polyhaven.com/a/approaching_storm">Poly Haven Approaching Storm</a>, CC0. Powered by Poly Haven.</p></main>',
            )
            html.write_text(text)
            dashboard(
                root,
                {
                    "status": "prepared",
                    "images": 0,
                    "requested": config["count"],
                    "message": "Natürliche 3D-Platzvorlage fertig; Fotopilot weiterhin erst nach Ende des Reco-Trainings starten.",
                },
            )
            print(
                "Installed natural football venue:",
                root / "Rasenplatz-fotorealistisch.png",
                flush=True,
            )
        except BaseException:
            if changed and backup is not None:
                for name in saved_names:
                    if (backup / name).exists():
                        shutil.copy2(backup / name, root / name)
                    elif (root / name).is_file():
                        (root / name).unlink()
                shutil.copy2(backup / "goals.json", root / "cache/goals.json")
                if (backup / "engine").exists():
                    if (root / "engine").exists():
                        shutil.rmtree(root / "engine")
                    shutil.copytree(backup / "engine", root / "engine")
                for path in (venue, root / "cache/goals-natural"):
                    if path.exists():
                        shutil.rmtree(path)
            raise
        finally:
            if work.exists():
                shutil.rmtree(work)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument(
        "--render",
        type=Path,
        required=True,
        help="Completed local worker output containing RGB and raw goal IDs",
    )
    a = p.parse_args()
    install(a.root, a.render, a.assets)


if __name__ == "__main__":
    main()
