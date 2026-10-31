from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from just_a_sports_dataset_generator import __version__
from just_a_sports_dataset_generator.exporters import LocalDatasetExporter
from just_a_sports_dataset_generator.renderers import ReferenceRenderer
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
    export = subcommands.add_parser("export", help="export visible annotations as COCO and/or YOLO")
    export.add_argument("source", type=Path)
    export.add_argument("--output", "-o", type=Path, required=True)
    export.add_argument("--format", choices=("both", "coco", "yolo"), default="both")
    return parser


def _execute(args: argparse.Namespace) -> int:
    if args.command == "export":
        formats = ("coco", "yolo") if args.format == "both" else (str(args.format),)
        result = LocalDatasetExporter().export(args.source, args.output, formats)
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(f"exported {result.images} images and {result.annotations} boxes to {args.output}")
        return 0

    recipe = load_recipe(args.recipe)
    if args.command == "validate":
        print(f"valid recipe: {recipe.name} ({recipe.scenes} scenes)")
        return 0

    if args.command == "render-reference":
        if args.max_scenes is not None and args.max_scenes <= 0:
            raise ValueError("--max-scenes must be positive")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        renderer = ReferenceRenderer()
        jobs = plan_jobs(recipe)
        if args.max_scenes is not None:
            jobs = jobs[: args.max_scenes]
        with new_directory(args.output) as staging:
            results = [renderer.render(job, staging, args.max_frames) for job in jobs]
            write_json(
                staging / "run.json",
                {
                    "schema_version": "1.0",
                    "application_version": __version__,
                    "recipe": asdict(recipe),
                    "recipe_sha256": sha256_file(args.recipe),
                    "renderer": renderer.metadata.plugin_id,
                    "renderer_version": renderer.metadata.version,
                    "assets": [
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _execute(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
