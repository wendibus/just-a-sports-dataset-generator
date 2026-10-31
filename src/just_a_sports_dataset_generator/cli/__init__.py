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
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
