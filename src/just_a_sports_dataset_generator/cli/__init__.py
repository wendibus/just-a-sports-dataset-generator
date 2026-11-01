from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from just_a_sports_dataset_generator import __version__
from just_a_sports_dataset_generator.composite import BLENDS, composite_crops
from just_a_sports_dataset_generator.crops import export_crops
from just_a_sports_dataset_generator.exporters import LocalDatasetExporter
from just_a_sports_dataset_generator.generators.qwen import ensure_installed, generate_crops
from just_a_sports_dataset_generator.generators.qwen_setup import (
    QUANTS,
    InstallPlan,
    SetupError,
    describe,
    installed,
)
from just_a_sports_dataset_generator.preview import create_preview
from just_a_sports_dataset_generator.renderers import ReferenceRenderer
from just_a_sports_dataset_generator.renderers.blender import BlenderBasketballRenderer
from just_a_sports_dataset_generator.renderers.contracts import RenderResult
from just_a_sports_dataset_generator.scenario import load_recipe, plan_jobs
from just_a_sports_dataset_generator.storage import (
    resumable_directory,
    sha256_file,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sports-dataset")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate", help="validate a dataset recipe")
    validate.add_argument("recipe", type=Path)
    plan = subcommands.add_parser("plan", help="write a deterministic scene-job plan")
    plan.add_argument("recipe", type=Path)
    plan.add_argument("--output", "-o", type=Path)
    render = subcommands.add_parser(
        "render-reference", help="render deterministic geometric test data"
    )
    render.add_argument("recipe", type=Path)
    render.add_argument("--output", "-o", type=Path, required=True)
    render.add_argument("--max-scenes", type=int)
    render.add_argument("--max-frames", type=int)
    basketball = subcommands.add_parser(
        "render-basketball", help="render local 3D basketball with Blender 5.x"
    )
    basketball.add_argument("recipe", type=Path)
    basketball.add_argument("--output", "-o", type=Path, required=True)
    basketball.add_argument("--max-scenes", type=int)
    basketball.add_argument("--max-frames", type=int)
    basketball.add_argument("--blender", type=Path)
    basketball.add_argument("--samples", type=int, default=32)
    basketball.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    export = subcommands.add_parser("export", help="export visible COCO annotations for RF-DETR")
    export.add_argument("source", type=Path)
    export.add_argument("--output", "-o", type=Path, required=True)
    export.add_argument("--format", choices=("coco",), default="coco")
    crops = subcommands.add_parser(
        "export-crops", help="write per-person RGB/mask/pose crops for pose-guided inpainting"
    )
    crops.add_argument("source", type=Path)
    crops.add_argument("--output", "-o", type=Path, required=True)
    crops.add_argument("--size", type=int, default=1024)
    crops.add_argument("--margin", type=float, default=0.25)
    crops.add_argument("--min-height", type=int, default=24)
    crops.add_argument("--dilate", type=int, default=9)
    composite = subcommands.add_parser(
        "composite-crops", help="paste generated person crops back into a copy of a render"
    )
    composite.add_argument("render", type=Path)
    composite.add_argument("--crops", type=Path, required=True)
    composite.add_argument("--generated", type=Path, required=True)
    composite.add_argument("--output", "-o", type=Path, required=True)
    composite.add_argument("--blend", choices=BLENDS, default="mask")
    composite.add_argument("--silhouettes", type=Path)
    composite.add_argument("--min-iou", type=float, default=0.85)
    composite.add_argument("--max-spill", type=float, default=0.15)
    composite.add_argument("--min-change", type=float, default=0.02)
    composite.add_argument("--generator", help="model/workflow name recorded in provenance")
    setup = subcommands.add_parser(
        "setup-qwen", help="install Qwen-Image-Edit-2511 locally (asks before downloading)"
    )
    setup.add_argument("--quant", choices=QUANTS, default="Q4_K_M")
    setup.add_argument("--yes", action="store_true", help="approve the shown download")
    subcommands.add_parser("qwen-status", help="show whether the local Qwen runtime is ready")
    generate = subcommands.add_parser(
        "generate-crops", help="repaint person crops with local Qwen (installs on first use)"
    )
    generate.add_argument("crops", type=Path)
    generate.add_argument("--output", "-o", type=Path, required=True)
    repaint = subcommands.add_parser(
        "repaint", help="crops, local Qwen and paste-back in one step (installs on first use)"
    )
    repaint.add_argument("render", type=Path)
    repaint.add_argument("--output", "-o", type=Path, required=True)
    repaint.add_argument("--size", type=int, default=1024)
    repaint.add_argument("--blend", choices=BLENDS, default="mask")
    for command in (generate, repaint):
        command.add_argument("--steps", type=int, default=4)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
        command.add_argument("--prompt", help="replace the built-in player/referee prompts")
        command.add_argument("--quant", choices=QUANTS, default="Q4_K_M")
        command.add_argument("--yes", action="store_true", help="approve a needed download")
    preview = subcommands.add_parser(
        "preview", help="write an offline RGB and colored-mask preview"
    )
    preview.add_argument("source", type=Path)
    preview.add_argument("--output", "-o", type=Path, required=True)
    return parser


def _confirm(approved: bool) -> Callable[[InstallPlan], bool]:
    def confirm(plan: InstallPlan) -> bool:
        print(describe(plan))
        if approved:
            return True
        if not sys.stdin.isatty():
            print("Zum Bestätigen erneut mit --yes ausführen.", file=sys.stderr)
            return False
        answer = input("Herunterladen und einrichten? [j/N] ").strip().lower()
        return answer in ("j", "ja", "y", "yes")

    return confirm


def _execute(args: argparse.Namespace, progress: Callable[[str], None] | None = None) -> int:
    report = progress or print
    if args.command == "qwen-status":
        manifest = installed()
        if manifest is None:
            print("Qwen ist nicht eingerichtet. Mit 'sports-dataset setup-qwen' installieren.")
            return 1
        print(f"Qwen bereit: {manifest['model']} {manifest['quant']} ({manifest['base_dir']})")
        return 0
    if args.command == "setup-qwen":
        manifest = ensure_installed(_confirm(args.yes), args.quant, progress=report)
        print(f"Qwen bereit: {manifest['model']} {manifest['quant']}")
        return 0
    if args.command == "generate-crops":
        made = generate_crops(
            args.crops,
            args.output,
            confirm=_confirm(args.yes),
            steps=args.steps,
            seed=args.seed,
            device=args.device,
            prompt=args.prompt,
            quant=args.quant,
            progress=report,
        )
        print(f"{made.generated} von {made.total} Personen generiert in {args.output}")
        return 0
    if args.command == "repaint":
        work = args.output
        crops_dir, generated_dir, final = work / "crops", work / "generated", work / "render"
        if final.exists():
            raise FileExistsError(f"output already exists: {final}")
        if not (crops_dir / "crops.json").is_file():
            export_crops(args.render, crops_dir, args.size)
        made = generate_crops(
            crops_dir,
            generated_dir,
            confirm=_confirm(args.yes),
            steps=args.steps,
            seed=args.seed,
            device=args.device,
            prompt=args.prompt,
            quant=args.quant,
            progress=report,
        )
        merged = composite_crops(
            args.render, crops_dir, generated_dir, final, blend=args.blend, generator=made.generator
        )
        print(
            f"{merged.applied} Personen ersetzt, {merged.rejected} verworfen, "
            f"{merged.missing} fehlen. Ergebnis: {final} (Details in composite.json)"
        )
        return 0
    if args.command == "preview":
        index = create_preview(args.source, args.output)
        print(f"preview: {index}")
        return 0
    if args.command == "composite-crops":
        merged = composite_crops(
            args.render,
            args.crops,
            args.generated,
            args.output,
            blend=args.blend,
            silhouettes=args.silhouettes,
            min_iou=args.min_iou,
            max_spill=args.max_spill,
            min_change=args.min_change,
            generator=args.generator,
        )
        print(
            f"applied {merged.applied} crops, rejected {merged.rejected}, "
            f"missing {merged.missing}; details in {args.output / 'composite.json'}"
        )
        return 0
    if args.command == "export-crops":
        result = export_crops(
            args.source, args.output, args.size, args.margin, args.min_height, args.dilate
        )
        print(
            f"wrote {result.crops} person crops to {args.output} "
            f"({result.skipped_small} smaller than {args.min_height} px skipped)"
        )
        return 0
    if args.command == "export":
        formats = (str(args.format),)
        result = LocalDatasetExporter().export(args.source, args.output, formats)
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(f"exported {result.images} images and {result.annotations} boxes to {args.output}")
        return 0

    recipe = load_recipe(args.recipe)
    if args.command == "validate":
        print(f"valid recipe: {recipe.name} ({recipe.scenes} scenes)")
        return 0

    if args.command in ("render-reference", "render-basketball"):
        if args.max_scenes is not None and args.max_scenes <= 0:
            raise ValueError("--max-scenes must be positive")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        renderer = (
            BlenderBasketballRenderer(args.blender, args.samples, args.device)
            if args.command == "render-basketball"
            else ReferenceRenderer()
        )
        jobs = plan_jobs(recipe)
        if args.max_scenes is not None:
            jobs = jobs[: args.max_scenes]
        fingerprint: dict[str, object] = {
            "command": args.command,
            "recipe_sha256": sha256_file(args.recipe),
            "renderer": renderer.metadata.plugin_id,
            "renderer_version": renderer.metadata.version,
            "assets": [dict(asset) for asset in getattr(renderer, "assets", [])],
            "samples": getattr(args, "samples", None),
            "device": getattr(args, "device", None),
            "max_scenes": args.max_scenes,
            "max_frames": args.max_frames,
        }
        with resumable_directory(args.output, fingerprint) as (staging, resumed):
            if resumed:
                message = "Abgebrochener Lauf wird fortgesetzt; fertige Szenen werden übernommen."
                print(message, flush=True)
                if progress:
                    progress(message)
            results: list[RenderResult] = []
            for index, job in enumerate(jobs):
                manifest = staging / job.job_id / "scene.json"
                if manifest.is_file():  # finished before the interruption; published atomically
                    frames = len(json.loads(manifest.read_text(encoding="utf-8"))["frames"])
                    results.append(RenderResult(job.job_id, manifest.parent, frames, manifest))
                    continue
                message = f"Szene {index + 1} von {len(jobs)} wird berechnet …"
                print(message, flush=True)
                if progress:
                    progress(message)
                results.append(renderer.render(job, staging, args.max_frames))
            write_json(
                staging / "run.json",
                {
                    "schema_version": "1.1",
                    "application_version": __version__,
                    "recipe": asdict(recipe),
                    "recipe_sha256": sha256_file(args.recipe),
                    "renderer": renderer.metadata.plugin_id,
                    "renderer_version": renderer.metadata.version,
                    "assets": renderer.assets
                    if isinstance(renderer, BlenderBasketballRenderer)
                    else [
                        {
                            "id": "builtin-geometric-fixture",
                            "license": "Apache-2.0",
                            "source": "renderers/reference.py",
                            "version": renderer.metadata.version,
                        }
                    ],
                    "jobs": [
                        {"job_id": job.job_id, "seed": job.seed, "frames": result.frames}
                        for job, result in zip(jobs, results, strict=True)
                    ],
                },
            )
        print(f"rendered {sum(result.frames for result in results)} frames to {args.output}")
        return 0

    payload = {
        "plan_version": "1.0",
        "recipe": asdict(recipe),
        "jobs": [asdict(job) for job in plan_jobs(recipe)],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {len(payload['jobs'])} jobs to {args.output}")
    else:
        print(rendered, end="")
    return 0


def main(
    argv: Sequence[str] | None = None, *, progress: Callable[[str], None] | None = None
) -> int:
    args = _parser().parse_args(argv)
    try:
        return _execute(args, progress)
    except (ValueError, OSError, SetupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
