# Local COCO datasets for RF-DETR

## Run

Install Python 3.11+ and the package once. Rendering and export then work offline and do not use a hosted service.

```bash
python -m pip install -e .
sports-dataset render-reference examples/basketball.json --output outputs/render --max-frames 2
sports-dataset export outputs/render --output outputs/dataset
sports-dataset preview outputs/dataset --output outputs/Vorschau
```

Use a fresh output directory each time. Both commands publish a completed staging folder; errors do not leave a partial dataset at the requested destination. An existing output is an error, never an instruction to overwrite. Render resumption is not implemented yet. The exporter accepts a run directory or one scene directory. The export destination must be separate from the source, outside its directory tree.

COCO for RF-DETR is the only export format. `--format coco` is accepted but optional. The previous `--format yolo` and `--format both` options are rejected. Reference rendering currently supports only `bounding_boxes` and `instance_masks`; requested keypoints, semantic masks, or depth are rejected before publication.

## Files and annotation semantics

For viewing, open `outputs/Vorschau/index.html`. It shows RGB images beside colored mask thumbnails, with links to the original images. It stays outside the training dataset and displays at most 48 frames. Raw `*.instances.png` masks encode object IDs (0 for background, 1..255 for objects), so small IDs appear almost black; do not brighten or otherwise modify the raw masks. The preview renders a separate color palette and preserves all training-file checksums.

- `coco/{train,valid,test}/`: RGB PNGs and `_annotations.coco.json`.
- `native/<scene>/`: lossless 8-bit instance-ID PNGs and `annotations.json`. RGB is referenced by image name and resides in the COCO folders.
- `manifest.json`: RF-DETR training target and dataset directory, application/renderer versions, recipe when available, seeds, backend-supplied asset provenance, checked Reco revision, class mapping, source/output SHA-256 hashes, and runtime versions.
- `report.json`: image/annotation counts, hidden objects, scene splits, completed checks, and limitations.

Boxes use visible pixels, top-left `x,y,width,height`, and exclusive right/bottom edges. Compute them from the final instance-ID pass, never from the original object rectangle. Fully occluded/off-screen objects retain their identity in native metadata but have no visible box or export row. Area counts visible pixels; it can be smaller than bounding-box area.

COCO uses category IDs starting at **1**, visible pixel boxes, `iscrowd=0`, and uncompressed column-major RLE instance masks. Native `class_id` values start at **0**. Both follow Reco's exact label order; the manifest records the relationship explicitly. The exported `coco/train`, `coco/valid`, and `coco/test` folders follow the [RF-DETR dataset layout](https://rfdetr.roboflow.com/learn/train/#dataset-structure). Background-only images stay in the image list with no annotation rows.

Image/mask dimensions, mask mode, instance identities, class names/order, frame IDs, and file paths are checked. Paths cannot escape the scene folder. Current limits are 16,777,216 pixels, 32 MiB per scene/run JSON, and 255 distinct foreground instance IDs per frame. There are no poses, track continuity guarantees, depth exports, or photorealistic images in this milestone.

## Splits and repeatability

All frames of a scene stay together. Assignment is deterministic from scene seeds and job IDs. At least one scene is reserved for each evaluation split when three or more scenes exist; larger datasets approximate 80/10/10. With two scenes, one goes to training and one to validation; with one scene, only training is populated. Small datasets produce a warning.

Repeated scene seeds and identical RGB pixels in different splits are rejected. This detects exact duplicates, not all near-duplicates or semantic leakage. Same-split duplicate images are retained; Reco may deduplicate them when importing.

The same inputs and software/platform versions produce byte-identical exports. Runtime metadata is intentionally different across environments, and PNG encoder versions may affect encoded bytes. The manifest hashes every output except itself. Asset license declarations come from backend metadata; an export is not a legal audit of third-party assets.

## Reco Trainer import

Create/open a Reco project for the same sport and select **dataset/coco/** in the label-import workflow. Do not select the dataset root: native instance-mask PNGs are not RGB training images. Choose COCO if prompted. Train with RF-DETR in Reco Trainer.

Verified against [Reco Trainer revision 4002883](https://github.com/wendibus/reco-trainer/tree/4002883eacbe07a3a7b43dd54422e9f222203e89), using its actual `cross-platform/ml_worker.py import-labels --format coco` command and `build_dataset` function used by its RF-DETR training workflow. Tests cover all eight taxonomy IDs, image/box counts, class names and IDs in every training split, exact pixel boxes including one-pixel widths, background frames, image checksums, and repeated import. These tests exercise data preparation; they do not instantiate RF-DETR, download weights, or run model training.

**Current Reco limitations:** the label importer consumes detection boxes, not COCO masks. It combines imported frames into a pseudo-video and creates its own training/evaluation assignments; exported scene boundaries and train/validation/test splits are not retained. Use the original export directly for experiments requiring these fixed splits. Successful import is not evidence of model accuracy or photorealism.

To run integration checks locally, use an existing checkout at the recorded commit:

```bash
python -m pip install -e ".[dev]"
RECO_TRAINER_PATH=/absolute/path/to/reco-trainer python -m unittest discover -s tests -v
```

PowerShell: set `$env:RECO_TRAINER_PATH` before running the Python command. Without this variable only the three Reco integration tests are skipped; ordinary annotation/export/schema tests still run. A supplied checkout at a different revision fails deliberately. Inspect upstream changes and update the recorded baseline, CI checkout, and documentation together. Tests themselves do not clone Reco or download models.

## RF-DETR target and format migration

Machine-readable JSON Schemas are in [schemas/](../schemas/).

- Recipe **1.0** is unchanged. The example now requests only supported passes.
- Renderer scene **2.0** replaces the old box semantics: `bbox_xywh` is visible or null, `visible_area` is explicit, and the reference renderer retains the original rectangle as `amodal_bbox_xywh`. Renderer version is **0.2.0**.
- Export also reads legacy scene **1.0**, discards its old box coordinates, and recomputes annotations from masks. Missing original run metadata produces a warning; nothing is invented. Source files are never rewritten.
- Application/exporter **0.3.0** exports only COCO for RF-DETR. Export manifest **2.0** adds `training_target` and replaces the old `yolo_id` field with the neutral native `class_id`. The legacy manifest 1.0 schema remains available to interpret previously generated files; existing datasets are not rewritten. Their `coco/` folders remain usable.
- Render run, native annotations, and report contracts remain **1.0**. Scene and recipe formats are unchanged by the RF-DETR correction.

Reference placement has changed in renderer 0.2.0 to include image edges and support 1×1 fixtures. Its output is not byte-identical to renderer 0.1.0; keep the old version and original files when replaying old runs.

Format references: [COCO mask API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py), [RF-DETR training and dataset structure](https://rfdetr.roboflow.com/learn/train/).
