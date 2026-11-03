"""Native macOS entry point for local basketball rendering (Apache-2.0)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from just_a_sports_dataset_generator import __version__
from just_a_sports_dataset_generator.cli import main as cli_main
from just_a_sports_dataset_generator.generators.qwen import ensure_installed
from just_a_sports_dataset_generator.generators.qwen_setup import InstallPlan, data_home, describe
from just_a_sports_dataset_generator.preview import create_preview
from just_a_sports_dataset_generator.renderers.blender import find_blender
from just_a_sports_dataset_generator.storage import write_json

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
        raise RuntimeError(result.stderr.strip() or "The Mac dialog could not be opened.")
    return result.stdout.strip()


def message(text: str) -> None:
    apple(
        "on run argv\nactivate\ndisplay dialog (item 1 of argv) with title (item 2 of argv) "
        'buttons {"OK"} default button "OK"\nend run',
        text,
        TITLE,
    )


def work_folder(destination: Path) -> Path:
    return destination.with_name(destination.name + ".partial")


def run_dataset(
    recipe: Path,
    destination: Path,
    demo: bool,
    *,
    reference: bool = False,
    progress: Callable[[str], None] | None = None,
    samples: int = 32,
    repaint: bool = False,
) -> dict[str, object]:
    """Render, optionally repaint, export and preview into a resumable work folder.

    Everything happens in ``<destination>.partial``. Rendering and Qwen generation resume
    where they stopped; export and preview are recreated. On success the folder is
    published as ``destination``.
    """
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    work = work_folder(destination)
    settings: dict[str, object] = {
        "recipe": str(recipe.resolve()),
        "demo": demo,
        "reference": reference,
        "samples": samples,
        "repaint": repaint,
    }
    marker = work / "launcher.json"
    if work.exists():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if previous != settings:
            raise FileExistsError(f"{work} belongs to a run with other settings")
        if progress:
            progress("Resuming an interrupted run …")
    else:
        work.mkdir(parents=True)
        write_json(marker, settings)
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        render = work / "render"
        if not render.is_dir():  # finished renders are published; render.partial resumes
            render_args = [
                "render-reference" if reference else "render-basketball",
                str(recipe),
                "--output",
                str(render),
            ]
            if not reference:
                render_args += ["--samples", str(samples)]
            if demo:
                render_args += ["--max-scenes", "3", "--max-frames", "2" if reference else "1"]
            if cli_main(render_args, progress=progress):
                raise ValueError(log.getvalue().strip())
        source = render
        if repaint:
            source = work / "photorealistic/render"
            if not source.is_dir():
                # Qwen is installed beforehand with the user's consent; --yes only skips a
                # second prompt and never triggers a download on its own here.
                repaint_args = ["repaint", str(render), "--output", str(work / "photorealistic")]
                if cli_main([*repaint_args, "--yes"], progress=progress):
                    raise ValueError(log.getvalue().strip())
        for derived in (work / "dataset", work / "Preview"):
            shutil.rmtree(derived, ignore_errors=True)  # quick to recreate from the render
        if progress:
            progress("Checking object masks and creating the COCO dataset for RF-DETR …")
        if cli_main(["export", str(source), "--output", str(work / "dataset")]):
            raise ValueError(log.getvalue().strip())
    with (work / "log.txt").open("a", encoding="utf-8") as protocol:
        protocol.write(log.getvalue())
    report = json.loads((work / "dataset/report.json").read_text(encoding="utf-8"))
    if progress:
        progress("Preparing RGB images and mask preview …")
    create_preview(work / "dataset", work / "Preview")
    marker.unlink()
    work.rename(destination)
    return report


def state_file() -> Path:
    return data_home() / "launcher-state.json"


def remember_unfinished(destination: Path) -> None:
    state_file().parent.mkdir(parents=True, exist_ok=True)
    write_json(state_file(), {"unfinished": str(destination)})


def forget_unfinished() -> None:
    state_file().unlink(missing_ok=True)


def unfinished_run() -> tuple[Path, dict[str, object]] | None:
    """The last interrupted run and its settings, if its work folder still exists."""
    try:
        destination = Path(json.loads(state_file().read_text(encoding="utf-8"))["unfinished"])
        settings = json.loads((work_folder(destination) / "launcher.json").read_text("utf-8"))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if destination.exists() or not isinstance(settings, dict):
        return None
    return destination, settings


class CancelDialog:
    """A second, non-blocking dialog with an Abbrechen button while the run is busy.

    Cancelling stops the running Blender or Qwen process (children of this app); the run
    then fails like any interrupted run and keeps every finished scene for resuming.
    """

    def __init__(self) -> None:
        self.cancelled = False
        self.closed = False
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        script = (
            'display dialog "Rendering runs locally on your Mac." & return & return & '
            '"You can cancel at any time. Finished scenes and generated players are kept '
            'and can be resumed at the next start." '
            f'with title "{TITLE}" buttons {{"Cancel"}} default button "Cancel"'
        )
        self.process = subprocess.Popen(
            ["/usr/bin/osascript", "-e", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self) -> None:
        assert self.process is not None
        output, _ = self.process.communicate()
        if not self.closed and self.process.returncode == 0 and "Cancel" in output:
            self.cancelled = True
            stop_children(exclude=self.process.pid)

    def close(self) -> None:
        self.closed = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


def stop_children(exclude: int) -> None:
    """Ask every child process of this app (Blender, Qwen worker) to stop."""
    found = subprocess.run(
        ["/usr/bin/pgrep", "-P", str(os.getpid())], capture_output=True, text=True, check=False
    ).stdout.split()
    for pid in found:
        if int(pid) != exclude:
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid), signal.SIGTERM)


def confirm_download(plan: InstallPlan) -> bool:
    """macOS consent dialog listing every file source, size, license and the destination."""
    answer = apple(
        "on run argv\nactivate\nset answer to display dialog (item 1 of argv) "
        'with title "Set up Qwen" buttons {"Cancel", "Download"} '
        'default button "Download" cancel button "Cancel"\n'
        "return button returned of answer\nend run",
        describe(plan),
    )
    return answer == "Download"


def default_folder() -> Path:
    for parent in Path(sys.executable).parents:
        if parent.suffix == ".app":
            return parent.parent
    return Path.cwd()


def gui() -> int:
    resume = unfinished_run()
    if resume is not None:
        destination, saved = resume
        answer = apple(
            "on run argv\nactivate\nset answer to display dialog (item 1 of argv) "
            'with title (item 2 of argv) buttons {"Start over", "Resume"} '
            'default button "Resume"\nreturn button returned of answer\nend run',
            "An interrupted run was found:\n"
            f"{destination.name}\n\nFinished scenes and generated players are kept; "
            "only the rest is computed.",
            TITLE,
        )
        if answer == "Resume":
            find_blender()
            return execute(
                destination,
                Path(str(saved["recipe"])),
                bool(saved["demo"]),
                int(str(saved["samples"])),
                bool(saved["repaint"]),
            )
        forget_unfinished()  # the folder stays; it can be deleted by hand
    choice = apple(
        "on run argv\nactivate\nset answer to display dialog "
        '("Version " & (item 1 of argv) & " · local · RF-DETR" & return & return & '
        '"3D basketball: hall, players, referees, ball and hoops with COCO labels." & '
        'return & "Procedural figures; photorealism not validated yet." & '
        'return & return & "Start basketball: 3 images. Requires a local Blender 5.x.") '
        'with title (item 2 of argv) buttons {"Quit", "Choose recipe", "Start basketball"} '
        'default button "Start basketball" cancel button "Quit"\n'
        "return button returned of answer\nend run",
        __version__,
        TITLE,
    )
    if choice is None:
        return 0
    find_blender()
    demo = choice == "Start basketball"
    recipe = RESOURCES / "examples/basketball-3d.json"
    if not demo:
        selected = apple(
            'activate\nset selected to choose file with prompt "Choose a dataset recipe (JSON)" '
            'of type {"public.json"}\nreturn POSIX path of selected'
        )
        if selected is None:
            return 0
        recipe = Path(selected)
    quality = apple(
        'set answer to display dialog "Choose the image quality. Higher quality takes longer." '
        'with title "Basketball image quality" buttons {"Cancel", "Standard", "Fine"} '
        'default button "Standard" cancel button "Cancel"\nreturn button returned of answer'
    )
    if quality is None:
        return 0
    photo = apple(
        'set answer to display dialog "Also repaint the players photorealistically with '
        'Qwen-Image-Edit?" & return & return & "Runs locally and takes about 1–2 minutes per '
        "person. The labels stay unchanged. On first use Qwen is set up after asking "
        '(about 31 GB)." with title "Photorealistic players" '
        'buttons {"Cancel", "No", "Yes"} default button "No" cancel button "Cancel"\n'
        "return button returned of answer"
    )
    if photo is None:
        return 0
    folder = apple(
        "on run argv\nactivate\nset selected to choose folder with prompt "
        '"Choose the output folder. A new run is created there." '
        "default location (POSIX file (item 1 of argv))\nreturn POSIX path of selected\nend run",
        str(default_folder()),
    )
    if folder is None:
        return 0
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    destination = Path(folder) / "Outputs" / f"Run-{stamp}-{uuid4().hex[:6]}"
    return execute(destination, recipe, demo, 64 if quality == "Fine" else 32, photo == "Yes")


def execute(destination: Path, recipe: Path, demo: bool, samples: int, repaint: bool) -> int:
    """Run (or resume) one run with a progress page and a Cancel dialog."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress_page = destination.parent / (destination.name + "-progress.html")

    def progress(message: str, *, finished: bool = False, failed: bool = False) -> None:
        url = quote(destination.name + "/Preview/index.html", safe="/")
        refresh = (
            f'<meta http-equiv="refresh" content="0;url={url}">'
            if finished
            else ("" if failed else '<meta http-equiv="refresh" content="5">')
        )
        page = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            + refresh
            + "<title>Generating basketball images</title><style>body{font:18px/1.6 system-ui;"
            "background:#f3f5f8;color:#182333;max-width:800px;margin:70px auto;padding:24px}"
            "h1{font-size:32px}.box{background:white;border:1px solid #d9e0e9;border-radius:16px;padding:28px}"
            '</style><h1>Your basketball images</h1><div class="box"><p>'
            + escape(message)
            + "</p><p>Everything runs on your Mac. Depending on the quality, one image can take "
            "several minutes. The preview opens when it is done.</p></div></html>"
        )
        temporary = progress_page.with_suffix(".tmp")
        temporary.write_text(page, encoding="utf-8")
        temporary.replace(progress_page)

    progress("Preparing Blender …")
    subprocess.run(["/usr/bin/open", str(progress_page)], check=True)
    remember_unfinished(destination)
    cancel = CancelDialog()
    cancel.start()
    try:
        if repaint:
            progress("Checking Qwen …")
            ensure_installed(confirm_download, progress=progress)
        result = run_dataset(
            recipe, destination, demo, progress=progress, samples=samples, repaint=repaint
        )
    except Exception:
        if cancel.cancelled:
            progress("Cancelled. You can resume at the next start.", failed=True)
            message(
                "The run was cancelled.\n\nFinished scenes are kept. "
                "At the next start the app offers \u201cResume\u201d."
            )
            return 0
        progress("The run stopped. The app shows the error message.", failed=True)
        raise
    finally:
        cancel.close()
    forget_unfinished()
    progress("Done. Opening the image preview.", finished=True)
    completed = apple(
        "on run argv\nactivate\nset answer to display dialog (item 1 of argv) "
        'with title (item 2 of argv) buttons {"Close", "Training folder", "View images"} '
        'default button "View images"\nreturn button returned of answer\nend run',
        f"Done: {result['images']} images, {result['annotations']} object boxes.\n\n"
        "\u201cView images\u201d shows the RGB images and colored object regions.\n"
        "For Reco Trainer / RF-DETR, import the folder \u201cdataset/coco\u201d.\n"
        f"Reco Trainer re-splits training and test data on import.\n\n{destination}",
        TITLE,
    )
    if completed == "View images":
        subprocess.run(["/usr/bin/open", str(destination / "Preview/index.html")], check=True)
    elif completed == "Training folder":
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
        message(f"The run could not be completed.\n\n{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
