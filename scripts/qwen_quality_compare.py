"""Reproducible small strength comparison with independent foreground rejection.

No downloads: Qwen and the explicitly selected BiRefNet runtime must be installed.
Passing numerical checks is never automatic approval for training.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from just_a_sports_dataset_generator.annotations.scene import safe_file
from just_a_sports_dataset_generator.composite import _binary, _iou, _mean_difference
from just_a_sports_dataset_generator.foreground_quality import (
    MIN_FOREGROUND_COVERAGE,
    MIN_LOCAL_COVERAGE,
    coverage_metrics,
)
from just_a_sports_dataset_generator.generators.qwen import generate_crops
from just_a_sports_dataset_generator.storage import sha256_file, write_json


def review(crop, directory, masks, source):
    with Image.open(source / crop["files"]["mask"]) as raw:
        mask = _binary(raw)
    with Image.open(masks / (crop["id"] + ".png")) as raw:
        foreground = _binary(raw.resize(mask.size, Image.Resampling.NEAREST))
    metrics = coverage_metrics(mask, foreground)
    metrics["iou"] = round(_iou(mask, foreground), 4)
    reasons = []
    if metrics["iou"] < 0.85:
        reasons.append("silhouette differs")
    if metrics["foreground_coverage"] < MIN_FOREGROUND_COVERAGE:
        reasons.append("missing foreground")
    if metrics["min_local_coverage"] < MIN_LOCAL_COVERAGE:
        reasons.append("missing local foreground")
    with Image.open(source / crop["files"]["rgb"]) as raw:
        original = raw.convert("RGB")
    with Image.open(directory / (crop["id"] + ".png")) as raw:
        result = raw.convert("RGB").resize(mask.size)
    change = _mean_difference(result, original, mask) or 0
    metrics["change"] = round(change, 4)
    if change < 0.02:
        reasons.append("unchanged")
    return {
        "id": crop["id"],
        "status": "rejected" if reasons else "needs_visual_review",
        "reasons": reasons,
        "metrics": metrics,
        "generated_sha256": sha256_file(directory / (crop["id"] + ".png")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.65, 0.75, 0.85])
    parser.add_argument("--review-python", type=Path, required=True)
    parser.add_argument("--foreground-model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if (
        not args.strengths
        or any(not 0.1 <= s <= 1 for s in args.strengths)
        or len(set(args.strengths)) != len(args.strengths)
    ):
        parser.error("strengths must be unique numbers between 0.1 and 1")
    run = args.run.resolve()
    out = args.output.resolve()
    source = run / "crops"
    index = json.loads((source / "crops.json").read_text())
    by_id = {c["id"]: c for c in index["crops"]}
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in args.ids):
        parser.error("unsafe crop ID")
    if len(set(args.ids)) != len(args.ids) or set(args.ids) - by_id.keys():
        parser.error("unknown or duplicate crop IDs")
    chosen = [by_id[i] for i in args.ids]
    if not args.review_python.is_file():
        parser.error("foreground Python interpreter is missing")
    if not (args.foreground_model / "model.safetensors").is_file():
        parser.error("local BiRefNet weights are missing")
    if args.baseline and any(not (args.baseline / (i + ".png")).is_file() for i in args.ids):
        parser.error("baseline is missing a selected generated image")
    signature = {
        "source_sha256": sha256_file(source / "crops.json"),
        "ids": args.ids,
        "strengths": args.strengths,
        "resolution": 512,
        "seed": 0,
        "baseline": str(args.baseline.resolve()) if args.baseline else None,
    }
    plan = out / "plan.json"
    if out.exists() and (not plan.is_file() or json.loads(plan.read_text()) != signature):
        raise ValueError("Output belongs to a different comparison; choose a new folder")
    out.mkdir(parents=True, exist_ok=True)
    # Keep this handle alive for the entire comparison; the OS releases it on a crash.
    run_lock = (out / ".compare.lock").open("a+b")
    if os.name == "nt":
        import msvcrt

        run_lock.write(b"0")
        run_lock.flush()
        run_lock.seek(0)
        msvcrt.locking(run_lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(plan, signature)
    subset = out / "crops"
    subset.mkdir(exist_ok=True)
    for crop in chosen:
        for name in crop["files"].values():
            original = safe_file(source, name)
            dest = subset / name
            if not dest.resolve().is_relative_to(subset.resolve()):
                raise ValueError("unsafe crop file path")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, dest)
    write_json(subset / "crops.json", {**index, "crops": chosen})
    variants = [(f"strength-{s:g}", out / f"strength-{s:g}", s) for s in args.strengths]
    if args.baseline:
        variants.append(("baseline-0.9", args.baseline.resolve(), None))
    records = []
    for name, directory, strength in variants:
        if strength is not None:
            generate_crops(
                subset,
                directory,
                confirm=lambda _: False,
                mode="inpaint",
                strength=strength,
                resolution=512,
                seed=0,
                steps=4,
            )
        masks = out / (name + "-silhouettes")
        subprocess.run(
            [
                str(args.review_python.absolute()),
                str(Path(__file__).with_name("foreground_masks.py")),
                "--generated",
                str(directory),
                "--model-dir",
                str(args.foreground_model.resolve()),
                "--output",
                str(masks),
                "--ids",
                *args.ids,
            ],
            check=True,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        items = [review(c, directory, masks, subset) for c in chosen]
        records.append({"variant": name, "items": items})
        write_json(
            out / "quality-report.json",
            {
                "schema_version": "1.0",
                "note": "Predicted segmentation rejects suspect crops. Passing requires visual review.",
                "variants": records,
            },
        )
    rows = []
    for crop in chosen:
        with Image.open(subset / crop["files"]["rgb"]) as raw:
            original = raw.convert("RGB")
        with Image.open(subset / crop["files"]["mask"]) as raw:
            mask = _binary(raw)
        original.save(out / (crop["id"] + "-reference.png"))
        cells = [f'<td><img src="{crop["id"]}-reference.png"><p>Reference</p></td>']
        for variant in records:
            name = variant["variant"]
            directory = next(d for n, d, _ in variants if n == name)
            item = next(i for i in variant["items"] if i["id"] == crop["id"])
            with Image.open(directory / (crop["id"] + ".png")) as raw:
                generated = raw.convert("RGB").resize(original.size)
            pasted = Image.composite(generated, original, mask)
            filename = f"{crop['id']}-{name}.png"
            pasted.save(out / filename)
            cells.append(
                f'<td class="{item["status"]}"><img src="{filename}"><p>{name}: {item["status"]}</p><p>{html.escape(str(item["metrics"]))}</p></td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Player quality comparison</title>'
        "<style>body{font:14px system-ui;background:#17191d;color:#eee}td{vertical-align:top;width:280px}img{width:280px}.rejected{background:#572525}.needs_visual_review{background:#404026}</style>"
        "<h1>Player quality comparison</h1><p>Red: rejected. Yellow: visual review required. No automatic training approval.</p><p>Lightning uses a discrete step schedule: nearby strengths can produce identical pixels. Compare visible results, not just setting numbers.</p><table>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )
    run_lock.close()
    print("Comparison:", out / "index.html", flush=True)


if __name__ == "__main__":
    main()
