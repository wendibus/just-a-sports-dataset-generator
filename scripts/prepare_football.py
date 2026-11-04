#!/usr/bin/env python3
"""Prepare a neutral outdoor football batch with CPU work only; never start ML."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from hall_night.common import read, sha, validate, write
from hall_night.football import asset_plan, build_balls, make_plans, prompt_for
from hall_night.football_pitch import build_pitch
from PIL import Image
from prepare_hall_night import freeze
from run_hall_night import dashboard


def prepare(root, base, referees, balls, count=10009, hours=15):
    if type(count) is not int or not 2 <= count <= 100000:
        raise ValueError("Count must be 2..100000")
    if not 1 < hours <= 49:
        raise ValueError("Time window must be >1 and <=49 hours")
    root = root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists():
        raise ValueError("Prepared root exists; use its starter or choose a new root")
    repo = Path(__file__).resolve().parents[1]
    config = read(base / "config.json")
    config.update(
        schema_version="3.0",
        sport="football",
        seed=20261104,
        count=count,
        max_hours=hours - 1,
        background=str(root / "background.png"),
        goals=str(root / "cache/goals.json"),
        background_cleanup=False,
        output_width=3200,
        generator_root=str(repo),
        note="Original procedural outdoor grass background with a known camera; experimental photographic people will be generated only on explicit start. Not a real venue photo; no training benefit validated.",
    )
    for key, source in (("referee_reference", referees), ("ball_reference", balls)):
        destination = root / "reference-inputs" / f"{key}.png"
        destination.parent.mkdir(exist_ok=True)
        with Image.open(source) as image:
            image.convert("RGB").save(destination)
        config[key] = str(destination)
    build_pitch(root, config["seed"])
    config["background_sha256"] = sha(root / "background.png")
    validate("hall-night-config-v3.0.json", config)
    plans = make_plans(count, config["seed"], config["gender_weights"], config["ball_weights"])
    write(
        root / "preparation.json",
        {
            "schema_version": "1.0",
            "sport": "football",
            "status": "prepared_cpu_only",
            "photographic_pilot_run": False,
            "images_generated": 0,
            "requested_images": count,
            "scene_groups": len(plans),
            "labels": ["ball", "player", "goalkeeper", "referee", "goal"],
            "person_assets_planned": len(asset_plan(config["seed"])),
            "gender_modes": {
                k: sum(p["frame_count"] for p in plans if p["gender_mode"] == k)
                for k in ("men", "women", "mixed")
            },
            "ball_modes": {
                k: sum(p["frame_count"] for p in plans if p["ball_mode"] == k)
                for k in ("single", "multi", "none")
            },
            "sources": "User supplied illustrative ball and referee references; original procedural pitch and goal geometry",
        },
    )
    validate("football-preparation-v1.0.json", read(root / "preparation.json"))
    freeze(root, config)
    starter = root / f"{count:,} Fussballbilder starten.command".replace(",", ".")
    (root / "2.000 Hallenbilder starten.command").rename(starter)
    text = starter.read_text().replace(
        'print "Lokaler Hallen-Nachtlauf:', 'print "Lokaler Fussball-Nachtlauf:'
    )
    text = text.replace(
        '/usr/bin/open "$task_root/index.html"',
        'print "Reco-Training vor dem Start dieser Modellarbeit beenden. Reco wird nicht automatisch erkannt."\n/usr/bin/open "$task_root/index.html"',
        1,
    )
    starter.write_text(text)
    starter.chmod(0o755)
    pilot = root / "Fussball-Pilot starten.command"
    pilot.write_text(
        text.replace('"$task_root" run', '"$task_root" pilot').replace(
            'print "Fertig: $task_root/dataset/coco"',
            'print "Pilot fertig: $task_root/pilot/index.html"',
        )
    )
    pilot.chmod(0o755)
    build_balls(config, root / "cache")
    assets = asset_plan(config["seed"])
    write(
        root / "person-profiles.json",
        {"schema_version": "1.0", "assets": [{**a, "prompt": prompt_for(a)} for a in assets]},
    )
    shutil.copy2(repo / "docs/FOOTBALL.md", root / "FOOTBALL.md")
    (root / "Referenzen.html").write_text(
        """<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Fußball – Vorbereitung und Vorlagen</title><style>body{font:17px/1.6 system-ui;background:#16191e;color:#eee;margin:32px}main{max-width:1536px;margin:auto}img{max-width:100%;height:auto}a{color:#9df}</style><main><h1>Fußball: neutraler Rasenplatz und Referenzen</h1><p>Vorbereitung ohne Qwen-Berechnung oder neues Training. Der echte Fotopilot wurde noch nicht ausgeführt. Erst nach Ende des Basketball-Trainings den Fußball-Pilot starten.</p><h2>Neutrale Spielfeldvorlage</h2><p>Originale prozedurale CPU-Vorlage mit bekannter Kamera. Kein echtes Stadionfoto und noch kein Nachweis fotorealistischer Qualität. 105 × 68 m Rasenfläche, weiße 7,32 × 2,44 m Tore mit transparenten Netzmaschen. Keine Zuschauer.</p><img src="Rasenplatz-Vorlage.jpg" alt="Leere neutrale Rasenplatz-Vorlage mit zwei großen Toren"><h2>Schiedsrichter</h2><p>Die Vorlage zeigt Blau, Neon-Gelb, Schwarz, Türkis, Rot, Grau, Orange und Gelb. Kurzarmtrikot mit Brusttaschen, schwarze Shorts, schwarze oder passende Stutzen und Fußballschuhe. Frauen und Männer, unterschiedliche Hauttöne. Einheitliches Label referee; keine erfundenen Marken- oder Verbandsklassen.</p><img src="reference-sheets/referee-reference.png" alt="Vom Nutzer gelieferte illustrative Übersicht von Fußball-Schiedsrichtertrikots"><h2>Bälle</h2><p>Zehn unterschiedliche Panel- und Farbvarianten: Weiß mit Blau/Gold, Rot/Blau/Grün, Magenta, Cyan oder Schwarz; außerdem ein gelber Ball mit dunklen Akzenten. Größe 5, ungefähr 22 cm Durchmesser. Einheitliches Label ball, auch bei teilweiser Verdeckung. Abgebildete Saison-, Marken- und Ligaangaben sind illustrative Beschreibungen und keine verifizierte offizielle Ausrüstungsliste.</p><img src="reference-sheets/ball-reference.png" alt="Vom Nutzer gelieferte Fußball-Ballübersicht"><h2>Geplante Szenen</h2><p>10.009 Bilder als voreingestelltes Ziel, 40 % Männer / 40 % Frauen / 20 % gemischt, ausschließlich Erwachsene. 70 % ein Ball / 20 % zwei bis vier / 10 % ohne sichtbaren Ball. Volle Szenen: zehn Feldspieler plus ein Torwart pro Team und ein Schiedsrichter. Der Pilot enthält weniger Personen. Ausschnitte, kleine entfernte Bälle und Verdeckungen werden variiert. Sichtbare Masken, Boxen und COCO-Kategorien passen zu Reco Trainer und RF-DETR.</p><p>336 neue Personenprofile sind geplant. Keine Hallen-Spielerfotos werden übernommen. Der 15-Stunden-Rahmen ist ein Ziel mit einer Stunde Exportreserve; keine Laufzeitgarantie. Alle Verarbeitung bleibt lokal, vorhandene installierte Modelle werden erst beim Starter geladen.</p><p>Die Nutzerbilder bleiben lokal und behalten ihre Rechte. Apache-2.0 gilt für den Projektcode, nicht automatisch für die Vorlagen. <a href="FOOTBALL.md">Technische Beschreibung</a></p></main></html>""",
        encoding="utf-8",
    )
    dashboard(
        root,
        {
            "status": "prepared",
            "images": 0,
            "requested": count,
            "message": "Fußball CPU-vorbereitet. Reco-Training zuerst beenden; danach Fotopilot starten. Vorlagen: Referenzen.html",
        },
    )
    print("Prepared without model execution:", root, flush=True)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--base-run", type=Path, required=True)
    p.add_argument("--referee-reference", type=Path, required=True)
    p.add_argument("--ball-reference", type=Path, required=True)
    p.add_argument("--count", type=int, default=10009)
    p.add_argument("--hours", type=float, default=15)
    a = p.parse_args()
    prepare(a.root, a.base_run, a.referee_reference, a.ball_reference, a.count, a.hours)


if __name__ == "__main__":
    main()
