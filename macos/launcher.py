"""Native macOS entry point for local basketball rendering (Apache-2.0)."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from just_a_sports_dataset_generator import __version__
from just_a_sports_dataset_generator.cli import main as cli_main
from just_a_sports_dataset_generator.preview import create_preview
from just_a_sports_dataset_generator.renderers.blender import find_blender
from just_a_sports_dataset_generator.storage import new_directory

TITLE = "Just a Sports Dataset Generator"
RESOURCES = Path(__file__).resolve().parent


def apple(script: str, *arguments: str) -> str | None:
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", script, *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        if "(-128)" in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip() or "Der Mac-Dialog konnte nicht geöffnet werden.")
    return result.stdout.strip()


def message(text: str) -> None:
    apple(
        "on run argv\nactivate\ndisplay dialog (item 1 of argv) with title (item 2 of argv) "
        'buttons {"OK"} default button "OK"\nend run',
        text,
        TITLE,
    )


def run_dataset(
    recipe: Path,
    destination: Path,
    demo: bool,
    *,
    reference: bool = False,
    progress: Callable[[str], None] | None = None,
    samples: int = 32,
) -> dict[str, object]:
    log = io.StringIO()
    with new_directory(destination) as staging:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            render_args = [
                "render-reference" if reference else "render-basketball",
                str(recipe),
                "--output",
                str(staging / "render"),
            ]
            if not reference:
                render_args += ["--samples", str(samples)]
            if demo:
                render_args += ["--max-scenes", "3", "--max-frames", "2" if reference else "1"]
            if cli_main(render_args, progress=progress):
                raise ValueError(log.getvalue().strip())
            if progress:
                progress("Objektmasken prüfen und COCO-Datensatz für RF-DETR erstellen …")
            if cli_main(["export", str(staging / "render"), "--output", str(staging / "dataset")]):
                raise ValueError(log.getvalue().strip())
        (staging / "protokoll.txt").write_text(log.getvalue(), encoding="utf-8")
        report = json.loads((staging / "dataset/report.json").read_text(encoding="utf-8"))
        if progress:
            progress("Farbbilder und Maskenvorschau vorbereiten …")
        create_preview(staging / "dataset", staging / "Vorschau")
    return report


def default_folder() -> Path:
    for parent in Path(sys.executable).parents:
        if parent.suffix == ".app":
            return parent.parent
    return Path.cwd()


def gui() -> int:
    choice = apple(
        "on run argv\nactivate\nset answer to display dialog "
        '("Version " & (item 1 of argv) & " · lokal · RF-DETR" & return & return & '
        '"3D-Basketball: Halle, Spieler, Schiedsrichter, Ball und Körbe mit COCO-Labels." & '
        'return & "Prozedurale Figuren; noch kein validierter Fotorealismus." & '
        'return & return & "Basketball starten: 3 Bilder. Blender 5.x wird lokal benötigt.") '
        'with title (item 2 of argv) buttons {"Beenden", "Rezept wählen", "Basketball starten"} '
        'default button "Basketball starten" cancel button "Beenden"\n'
        "return button returned of answer\nend run",
        __version__,
        TITLE,
    )
    if choice is None:
        return 0
    find_blender()
    demo = choice == "Basketball starten"
    recipe = RESOURCES / "examples/basketball-3d.json"
    if not demo:
        selected = apple(
            'activate\nset selected to choose file with prompt "Datensatz-Rezept (JSON) auswählen" '
            'of type {"public.json"}\nreturn POSIX path of selected'
        )
        if selected is None:
            return 0
        recipe = Path(selected)
    quality = apple(
        'set answer to display dialog "Bildqualität wählen. Höhere Qualität benötigt mehr Rechenzeit." '
        'with title "Basketball-Bildqualität" buttons {"Abbrechen", "Standard", "Fein"} '
        'default button "Standard" cancel button "Abbrechen"\nreturn button returned of answer'
    )
    if quality is None:
        return 0
    folder = apple(
        "on run argv\nactivate\nset selected to choose folder with prompt "
        '"Ausgabeordner wählen. Es wird ein neuer Durchlauf angelegt." '
        "default location (POSIX file (item 1 of argv))\nreturn POSIX path of selected\nend run",
        str(default_folder()),
    )
    if folder is None:
        return 0
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    destination = Path(folder) / "Ausgaben" / f"Durchlauf-{stamp}-{uuid4().hex[:6]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress_page = destination.parent / (destination.name + "-Fortschritt.html")

    def progress(message: str, *, finished: bool = False, failed: bool = False) -> None:
        url = quote(destination.name + "/Vorschau/index.html", safe="/")
        refresh = (
            f'<meta http-equiv="refresh" content="0;url={url}">'
            if finished
            else ("" if failed else '<meta http-equiv="refresh" content="5">')
        )
        page = (
            '<!doctype html><html lang="de"><meta charset="utf-8">'
            + refresh
            + "<title>Basketball wird erzeugt</title><style>body{font:18px/1.6 system-ui;"
            "background:#f3f5f8;color:#182333;max-width:800px;margin:70px auto;padding:24px}"
            "h1{font-size:32px}.box{background:white;border:1px solid #d9e0e9;border-radius:16px;padding:28px}"
            '</style><h1>Deine Basketballbilder</h1><div class="box"><p>'
            + escape(message)
            + "</p><p>Die Berechnung läuft vollständig auf deinem Mac. Je nach Qualität kann "
            "ein Bild mehrere Minuten benötigen. Die Vorschau öffnet sich nach Abschluss.</p></div></html>"
        )
        temporary = progress_page.with_suffix(".tmp")
        temporary.write_text(page, encoding="utf-8")
        temporary.replace(progress_page)

    progress("Blender wird vorbereitet …")
    subprocess.run(["/usr/bin/open", str(progress_page)], check=True)
    try:
        result = run_dataset(
            recipe, destination, demo, progress=progress, samples=64 if quality == "Fein" else 32
        )
    except Exception:
        progress("Die Berechnung wurde abgebrochen. Die App zeigt die Fehlermeldung.", failed=True)
        raise
    progress("Fertig. Die Bildvorschau wird geöffnet.", finished=True)
    completed = apple(
        "on run argv\nactivate\nset answer to display dialog (item 1 of argv) "
        'with title (item 2 of argv) buttons {"Schließen", "Trainingsordner", "Bilder ansehen"} '
        'default button "Bilder ansehen"\nreturn button returned of answer\nend run',
        f"Fertig: {result['images']} Bilder, {result['annotations']} Objektboxen.\n\n"
        f"„Bilder ansehen“ zeigt die Farbbilder und farbige Objektbereiche.\n"
        f"Für Reco Trainer / RF-DETR den Ordner „dataset/coco“ importieren.\n"
        f"Reco Trainer teilt Training und Test beim Import neu auf.\n\n{destination}",
        TITLE,
    )
    if completed == "Bilder ansehen":
        subprocess.run(["/usr/bin/open", str(destination / "Vorschau/index.html")], check=True)
    elif completed == "Trainingsordner":
        subprocess.run(["/usr/bin/open", str(destination / "dataset/coco")], check=True)
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args == ["--version"]:
        print(f"{TITLE} {__version__} (macOS arm64, local Blender basketball)")
        return 0
    if len(args) == 2 and args[0] == "--demo":
        print(
            json.dumps(run_dataset(RESOURCES / "examples/basketball-3d.json", Path(args[1]), True))
        )
        return 0
    if len(args) == 2 and args[0] == "--reference-demo":
        print(
            json.dumps(
                run_dataset(
                    RESOURCES / "examples/basketball.json", Path(args[1]), True, reference=True
                )
            )
        )
        return 0
    if args:
        return cli_main(args)
    try:
        return gui()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        message(f"Der Durchlauf konnte nicht abgeschlossen werden.\n\n{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
