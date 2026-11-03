"""Review an existing batch with local BiRefNet before pasting into fresh scenes.

Never overwrites a prior render or generated batch. No model downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from just_a_sports_dataset_generator.composite import composite_crops
from just_a_sports_dataset_generator.storage import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-python", type=Path, required=True)
    parser.add_argument("--foreground-model", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    generated = args.generated.resolve()
    out = args.output.resolve()
    if (out / "render").exists():
        raise FileExistsError("Reviewed render already exists; choose a new output folder")
    crops = run / "crops"
    index = json.loads((crops / "crops.json").read_text())
    ids = [c["id"] for c in index["crops"] if (generated / (c["id"] + ".png")).is_file()]
    if not ids or any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids):
        raise ValueError("No generated crops, or unsafe crop ID")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(args.review_python.absolute()),
            str(Path(__file__).with_name("foreground_masks.py")),
            "--generated",
            str(generated),
            "--output",
            str(out / "silhouettes"),
            "--model-dir",
            str(args.foreground_model.resolve()),
            "--ids",
            *ids,
        ],
        check=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    result = composite_crops(
        run / "render",
        crops,
        generated,
        out / "render",
        silhouettes=out / "silhouettes",
        generator="Local Qwen with independent BiRefNet foreground review",
    )
    write_json(
        out / "review.json",
        {
            "schema_version": "1.0",
            "crops_sha256": sha256_file(crops / "crops.json"),
            "applied": result.applied,
            "rejected": result.rejected,
            "missing": result.missing,
            "note": "Applied means numerical checks passed; visual review is required before training.",
        },
    )
    print(
        f"Applied {result.applied}, rejected {result.rejected}, missing {result.missing}. Visual review required."
    )


if __name__ == "__main__":
    main()
