from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from just_a_sports_dataset_generator.scenario import RecipeError, load_recipe, plan_jobs
from just_a_sports_dataset_generator.renderers import ReferenceRenderer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sports-dataset")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate", help="validate a dataset recipe")
    validate.add_argument("recipe", type=Path)
    plan = subcommands.add_parser("plan", help="write a deterministic scene-job plan")
    plan.add_argument("recipe", type=Path)
    plan.add_argument("--output", "-o", type=Path)
    render = subcommands.add_parser("render-reference", help="render deterministic geometric test data")
    render.add_argument("recipe", type=Path)
    render.add_argument("--output", "-o", type=Path, required=True)
    render.add_argument("--max-scenes", type=int)
    render.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        recipe = load_recipe(args.recipe)
    except RecipeError as exc:
        print(f"error: {exc}")
        return 2
    if args.command == "validate":
        print(f"valid recipe: {recipe.name} ({recipe.scenes} scenes)")
        return 0

    if args.command == "render-reference":
        if args.max_scenes is not None and args.max_scenes <= 0:
            print("error: --max-scenes must be positive")
            return 2
        if args.max_frames is not None and args.max_frames <= 0:
            print("error: --max-frames must be positive")
            return 2
        renderer = ReferenceRenderer()
        jobs = plan_jobs(recipe)
        if args.max_scenes is not None:
            jobs = jobs[: args.max_scenes]
        results = [renderer.render(job, args.output, args.max_frames) for job in jobs]
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


__all__ = ["main"]

