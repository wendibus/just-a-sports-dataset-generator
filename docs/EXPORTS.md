# Local annotation and export workflow

## Run

Install Python 3.11+ and the package once. Rendering and export then work offline and do not use a hosted service.

```bash
python -m pip install -e .
sports-dataset render-reference examples/basketball.json --output outputs/render --max-frames 2
sports-dataset export outputs/render --output outputs/dataset
```

Use a fresh output directory each time. Both commands publish a completed staging folder; errors do not leave a partial dataset at the requested destination. An existing output is an error, never an instruction to overwrite. Render resumption is not implemented yet. The exporter accepts a run directory or one scene directory. The export destination must be separate from the source, outside its directory tree.

Default format is both COCO and YOLO. Select one with `--format coco` or `--format yolo`. Reference rendering currently supports only `bounding_boxes` and `instance_masks`; requested keypoints, semantic masks, or depth are rejected before publication.

## Files and annotation semantics

- `coco/{train,valid,test}/`: RGB PNGs and `_annotations.coco.json`.
- `yolo/images/{train,val,test}/`: RGB PNGs.
- `yolo/labels/{train,val,test}/`: matching detection text files, including empty files for background-only images.
- `yolo/data.yaml`: relative image folders and the canonical class names.
- `native/<scene>/`: lossless 8-bit instance-ID PNGs and `annotations.json`. RGB is referenced by image name and resides in the selected format folders.
- `manifest.json`: application/renderer versions, recipe when available, seeds, backend-supplied asset provenance, checked Reco revision, class mapping, source/output SHA-256 hashes, and runtime versions.
- `report.json`: image/annotation counts, hidden objects, scene splits, completed checks, and limitations.

Boxes use visible pixels, top-left `x,y,width,height`, and exclusive right/bottom edges. Compute them from the final instance-ID pass, never from the original object rectangle. Fully occluded/off-screen objects retain their identity in native metadata but have no visible box or export row. Area counts visible pixels; it can be smaller than bounding-box area.

COCO uses category IDs starting at **1**, visible pixel boxes, `iscrowd=0`, and uncompressed column-major RLE instance masks. YOLO uses category IDs starting at **0**, followed by normalized center-x, center-y, width, and height. Both follow Reco's exact label order. Export slightly expands normalized lengths only when necessary to avoid inward floating-point rounding; the adjustment is a few floating-point ULPs, far below one pixel. This prevents Reco's one-pixel threshold from discarding valid tiny boxes.

Image/mask dimensions, mask mode, instance identities, class names/order, frame IDs, and file paths are checked. Paths cannot escape the scene folder. Current limits are 16,777,216 pixels, 32 MiB per scene/run JSON, and 255 distinct foreground instance IDs per frame. There are no poses, track continuity guarantees, depth exports, or photorealistic images in this milestone.

## Splits and repeatability

All frames of a scene stay together. Assignment is deterministic from scene seeds and job IDs. At least one scene is reserved for each evaluation split when three or more scenes exist; larger datasets approximate 80/10/10. With two scenes, one goes to training and one to validation; with one scene, only training is populated. Small datasets produce a warning.

Repeated scene seeds and identical RGB pixels in different splits are rejected. This detects exact duplicates, not all near-duplicates or semantic leakage. Same-split duplicate images are retained; Reco may deduplicate them when importing.

The same inputs and software/platform versions produce byte-identical exports. Runtime metadata is intentionally different across environments, and PNG encoder versions may affect encoded bytes. The manifest hashes every output except itself. Asset license declarations come from backend metadata; an export is not a legal audit of third-party assets.

## Reco Trainer import

Create/open a Reco project for the same sport and select **dataset/coco/** or **dataset/yolo/** in the label-import workflow. Do not select the dataset root: native instance-mask PNGs are not RGB training images. Choose the corresponding format explicitly if prompted.

Verified against [Reco Trainer revision 4002883](https://github.com/wendibus/reco-trainer/tree/4002883eacbe07a3a7b43dd54422e9f222203e89), using its actual `cross-platform/ml_worker.py import-labels` command. Tests cover all eight taxonomy IDs and both formats, image/box counts, labels, coordinates including one-pixel widths, background frames, image checksums, and repeated import.

**Current Reco limitations:** the label importer consumes detection boxes, not COCO masks. It combines imported frames into a pseudo-video and creates its own training/evaluation assignments; exported scene boundaries and train/validation/test splits are not retained. Use the original export directly for experiments requiring these fixed splits. Successful import is not evidence of model accuracy or photorealism.

To run integration checks locally, use an existing checkout at the recorded commit:

```bash
python -m pip install -e ".[dev]"
RECO_TRAINER_PATH=/absolute/path/to/reco-trainer python -m unittest discover -s tests -v
```

PowerShell: set `$env:RECO_TRAINER_PATH` before running the Python command. Without this variable only the three Reco integration tests are skipped; ordinary annotation/export/schema tests still run. A supplied checkout at a different revision fails deliberately. Inspect upstream changes and update the recorded baseline, CI checkout, and documentation together. Tests themselves do not clone Reco or download models.

## Format versions and migration

Machine-readable JSON Schemas are in [schemas/](../schemas/).

- Recipe **1.0** is unchanged. The example now requests only supported passes.
- Renderer scene **2.0** replaces the old box semantics: `bbox_xywh` is visible or null, `visible_area` is explicit, and the reference renderer retains the original rectangle as `amodal_bbox_xywh`. Renderer version is **0.2.0**.
- Export also reads legacy scene **1.0**, discards its old box coordinates, and recomputes annotations from masks. Missing original run metadata produces a warning; nothing is invented. Source files are never rewritten.
- Render run, native annotations, export manifest, and report contracts start at **1.0**. Future breaking changes require a new version and explicit migration handling.

Reference placement has changed in renderer 0.2.0 to include image edges and support 1×1 fixtures. Its output is not byte-identical to renderer 0.1.0; keep the old version and original files when replaying old runs.

Format references: [COCO mask API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py), [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/).
