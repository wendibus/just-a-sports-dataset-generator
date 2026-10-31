"""Small native macOS launcher for the reference build (Apache-2.0)."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from just_a_sports_dataset_generator import __version__
from just_a_sports_dataset_generator.cli import main as cli_main
from just_a_sports_dataset_generator.preview import create_preview
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


def run_dataset(recipe: Path, destination: Path, demo: bool) -> dict[str, object]:
    log = io.StringIO()
    with new_directory(destination) as staging:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            render_args = ["render-reference", str(recipe), "--output", str(staging / "render")]
            if demo:
                render_args += ["--max-scenes", "3", "--max-frames", "2"]
            if cli_main(render_args):
                raise ValueError(log.getvalue().strip())
            if cli_main(["export", str(staging / "render"), "--output", str(staging / "dataset")]):
                raise ValueError(log.getvalue().strip())
        (staging / "protokoll.txt").write_text(log.getvalue(), encoding="utf-8")
        report = json.loads((staging / "dataset/report.json").read_text(encoding="utf-8"))
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
        '"Geometrische Testbilder mit COCO-Labels. Noch keine fotorealistischen Sportszenen." & '
        'return & return & "Basketball-Test: 6 Bilder. Oder ein eigenes JSON-Rezept wählen.") '
        'with title (item 2 of argv) buttons {"Beenden", "Rezept wählen", "Basketball-Test"} '
        'default button "Basketball-Test" cancel button "Beenden"\n'
        "return button returned of answer\nend run",
        __version__,
        TITLE,
    )
    if choice is None:
        return 0
    demo = choice == "Basketball-Test"
    recipe = RESOURCES / "examples/basketball.json"
    if not demo:
        selected = apple(
            'activate\nset selected to choose file with prompt "Datensatz-Rezept (JSON) auswählen" '
            'of type {"public.json"}\nreturn POSIX path of selected'
        )
        if selected is None:
            return 0
        recipe = Path(selected)
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
    result = run_dataset(recipe, destination, demo)
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
        print(f"{TITLE} {__version__} (macOS arm64 reference build)")
        return 0
    if len(args) == 2 and args[0] == "--demo":
        print(json.dumps(run_dataset(RESOURCES / "examples/basketball.json", Path(args[1]), True)))
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
