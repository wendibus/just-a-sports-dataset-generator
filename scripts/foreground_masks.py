"""Extract review silhouettes with the explicitly installed local BiRefNet.

Run in a Python environment containing torch/torchvision/timm/kornia/transformers.
No downloads; model weights and code must already exist at --model-dir.
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from torch.nn import functional
    from torchvision import transforms

    folder = args.model_dir.resolve()
    weights = folder / "model.safetensors"
    provenance = json.loads((folder / "provenance.json").read_text())
    expected = next(
        f["lfs_sha256"] for f in provenance["files"] if f["path"] == "model.safetensors"
    )
    h = hashlib.sha256()
    with weights.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError("BiRefNet weights checksum mismatch")
    package = types.ModuleType("review_birefnet")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    module = importlib.import_module("review_birefnet.birefnet")
    config = importlib.import_module("review_birefnet.BiRefNet_config")
    model = module.BiRefNet(config=config.BiRefNetConfig(bb_pretrained=False))
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.eval().to("cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for identity in args.ids:
        if Path(identity).name != identity:
            raise ValueError("Invalid crop ID")
        source = args.generated / (identity + ".png")
        with Image.open(source) as raw:
            image = raw.convert("RGB")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        output = args.output / (identity + ".png")
        stamp = output.with_suffix(".json")
        if (
            output.exists()
            and stamp.exists()
            and json.loads(stamp.read_text()).get("source_sha256") == digest
        ):
            print("Reusing silhouette:", identity, flush=True)
            continue
        print("Extracting foreground:", identity, flush=True)
        with torch.inference_mode():
            prediction = model(transform(image).unsqueeze(0))[-1].sigmoid()
            alpha = functional.interpolate(
                prediction, size=(image.height, image.width), mode="bilinear", align_corners=False
            )[0, 0]
        mask = Image.fromarray((alpha.clamp(0, 1).numpy() * 255).round().astype("uint8"))
        temporary = output.with_suffix(".partial.png")
        mask.save(temporary)
        temporary.replace(output)
        record = {
            "source_sha256": digest,
            "model_revision": provenance["revision"],
            "weights_sha256": expected,
            "semantics": "predicted foreground for rejection only; not ground truth",
        }
        stamp.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    print("Foreground review masks ready.", flush=True)


if __name__ == "__main__":
    main()
