#!/usr/bin/env python3
"""Prepare a frozen local Futsal hall batch using already approved installed models."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from hall_night.common import read, sha, write
from hall_night.futsal import make_plans
from PIL import Image
from prepare_hall_night import copy_file, freeze
from run_hall_night import dashboard, lock


def prepare(
    root,
    base,
    referees,
    balls,
    blender,
    count=10009,
    hours=15,
    skip_cleanup=False,
    defer_background=False,
):
    if not 1 < hours <= 49:
        raise ValueError("Runtime window must exceed one hour and be at most 49 hours")
    if type(count) is not int or not 2 <= count <= 100000:
        raise ValueError("Image count must be between 2 and 100000")
    root = root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists():
        raise ValueError("Prepared run already exists; use its starter to resume")
    repo = Path(__file__).resolve().parents[1]
    config = read(base / "config.json")
    config.update(
        schema_version="2.0",
        sport="futsal",
        count=count,
        max_hours=max(0.1, hours - 1),
        seed=20261104,
        generator_root=str(repo),
        goals=str(root / "cache/goals.json"),
        ball_reference=str(balls.absolute()),
        referee_reference=str(referees.absolute()),
        note="Local Futsal photographic composites; original standing goals; estimated image calibration, no physical simulation. One hour reserved after generation for COCO export and validation.",
    )
    plans = make_plans(count, config["seed"], config["gender_weights"], config["ball_weights"])
    write(
        root / "planned-distribution.json",
        {
            "images": count,
            "scenes": len(plans),
            "gender_modes": {
                k: sum(p["frame_count"] for p in plans if p["gender_mode"] == k)
                for k in ("men", "women", "mixed")
            },
            "ball_modes": {
                k: sum(p["frame_count"] for p in plans if p["ball_mode"] == k)
                for k in ("single", "multi", "none")
            },
            "splits": {
                k: sum(p["frame_count"] for p in plans if p["split"] == k)
                for k in ("train", "val", "test")
            },
        },
    )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    # Do not compete for GPU memory with an existing basketball batch.
    if not skip_cleanup and not defer_background:
        from hall_night.resources import wait_for_other_runs

        wait_for_other_runs(root)
        with lock(base / ".night.lock"):
            request = {
                "background": config["background"],
                "model": config["model"],
                "output": str(root / "background-preparation"),
                "worker": str(
                    repo / "src/just_a_sports_dataset_generator/generators/qwen_worker.py"
                ),
            }
            write(root / "background-request.json", request)
            subprocess.run(
                [
                    config["qwen_python"],
                    "-I",
                    "-u",
                    str(repo / "scripts/hall_night/futsal_background.py"),
                    str(root / "background-request.json"),
                ],
                check=True,
            )
            config["background"] = str(root / "background-preparation/background.png")
    goals = root / "cache/goals"
    goals.mkdir(parents=True, exist_ok=True)
    if not (goals / "geometry.json").exists():
        subprocess.run(
            [
                str(blender),
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python",
                str(
                    repo
                    / "src/just_a_sports_dataset_generator/renderers/blender_worker/futsal_goal.py"
                ),
                "--",
                "--output",
                str(goals),
            ],
            check=True,
        )
    records = read(goals / "geometry.json")
    for a in records:
        a["sha256"] = sha(root / "cache" / a["file"])
    write(root / "cache/goals.json", {"schema_version": "1.0", "assets": records})
    # Validate the two supplied charts as image inputs before copying them locally.
    for path in (referees, balls):
        with Image.open(path) as img:
            img.verify()
    for key, name in (("ball_reference", "ball"), ("referee_reference", "referee")):
        normalized = root / "reference-inputs" / f"{name}.png"
        normalized.parent.mkdir(exist_ok=True)
        with Image.open(config[key]) as raw:
            raw.convert("RGB").save(normalized)
        config[key] = str(normalized)
    config["background_cleanup"] = defer_background and not skip_cleanup
    freeze(root, config)
    old = root / "2.000 Hallenbilder starten.command"
    starter = root / f"{count:,} Futsalbilder starten.command".replace(",", ".")
    old.rename(starter)
    for label, stage in [("Futsal-Pilot starten.command", "pilot")]:
        script = (
            starter.read_text()
            .replace('"$task_root" run', '"$task_root" ' + stage)
            .replace(
                'print "Fertig: $task_root/dataset/coco"',
                'print "Pilot fertig: $task_root/pilot/index.html"',
            )
        )
        path = root / label
        path.write_text(script)
        path.chmod(0o755)
    # Local illustrated documentation; supplied charts are not published without redistribution rights.
    copy_file(repo / "docs/FUTSAL_HALL.md", root / "FUTSAL_HALL.md")
    (root / "Referenzen.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Futsal – Vorlagen</title><style>body{font:18px system-ui;background:#17191d;color:#eee;margin:32px}img{width:100%;max-width:1280px}</style><h1>Futsal: Schiedsrichter und Bälle</h1><p>Vom Nutzer gelieferte illustrative Vorlagen. Farben und Formen dienen als Variation; Saison-, Marken- und Verbandsangaben sind keine verifizierte offizielle Datenbank.</p><h2>Schiedsrichter</h2><p>Kurzärmlige Shirts in Türkis, Gelb, Schwarz, Grau, Orange und Rot. Schwarze Shorts, Stutzen und Hallenschuhe. Eigenes Label referee; keine langen Basketball-Schiedsrichterhosen.</p><img src="reference-sheets/referee-reference.png"><h2>Bälle</h2><p>Zehn weiße Grundformen mit verschiedenfarbigen Mustern. Größe 4, etwa 20 cm Durchmesser; mehrere Bälle in gezielten Trainingsszenen. Teilverdeckt und unterschiedlich gedreht. Alle Varianten haben das Label ball.</p><img src="reference-sheets/ball-reference.png"><h2>Tore und Torwart</h2><img style="width:45%" src="cache/goals/goal-left.png"><img style="width:45%" src="cache/goals/goal-right.png"><p>Zwei auf dem Boden stehende 3 × 2 m Tore mit transparenten Netzmaschen. Torwarttrikots grün/blau, eigenes Label goalkeeper. Farbige Feldteams rot/gelb.</p>""",
        encoding="utf-8",
    )
    dashboard(
        root,
        {
            "status": "prepared",
            "images": 0,
            "requested": count,
            "message": "Futsal vorbereitet. Erst Pilot prüfen, dann Nachtlauf starten. Vorlagen: Referenzen.html",
        },
    )
    print("Prepared:", starter, flush=True)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--base-run", type=Path, required=True)
    p.add_argument("--referee-reference", type=Path, required=True)
    p.add_argument("--ball-reference", type=Path, required=True)
    p.add_argument(
        "--blender", type=Path, default=Path("/Applications/Blender.app/Contents/MacOS/Blender")
    )
    p.add_argument("--count", type=int, default=10009)
    p.add_argument("--hours", type=float, default=15)
    p.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Use only when the supplied hall already has no folded wall goals",
    )
    p.add_argument(
        "--defer-background",
        action="store_true",
        help="Prepare without Qwen now; finish background when the batch starts",
    )
    a = p.parse_args()
    prepare(
        a.root,
        a.base_run,
        a.referee_reference,
        a.ball_reference,
        a.blender,
        a.count,
        a.hours,
        a.skip_cleanup,
        a.defer_background,
    )


if __name__ == "__main__":
    main()
