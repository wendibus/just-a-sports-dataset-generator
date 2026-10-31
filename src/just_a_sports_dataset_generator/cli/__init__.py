from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from just_a_sports_dataset_generator import __version__
from just_a_sports_dataset_generator.exporters import LocalDatasetExporter
from just_a_sports_dataset_generator.preview import create_preview
from just_a_sports_dataset_generator.renderers import ReferenceRenderer
from just_a_sports_dataset_generator.renderers.blender import BlenderBasketballRenderer
from just_a_sports_dataset_generator.renderers.contracts import RenderResult
from just_a_sports_dataset_generator.scenario import load_recipe, plan_jobs
from just_a_sports_dataset_generator.storage import new_directory, sha256_file, write_json


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
    preview = subcommands.add_parser(
        "preview", help="write an offline RGB and colored-mask preview"
    )
    preview.add_argument("source", type=Path)
    preview.add_argument("--output", "-o", type=Path, required=True)
    return parser


def _execute(args: argparse.Namespace, progress: Callable[[str], None] | None = None) -> int:
    if args.command == "preview":
        index = create_preview(args.source, args.output)
        print(f"preview: {index}")
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
        with new_directory(args.output) as staging:
            results: list[RenderResult] = []
            for index, job in enumerate(jobs):
                message = f"Szene {index + 1} von {len(jobs)} wird berechnet …"
                print(message, flush=True)
                if progress:
                    progress(message)
                results.append(renderer.render(job, staging, args.max_frames))
            write_json(
                staging / "run.json",
                {
                    "schema_version": "1.0",
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
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
