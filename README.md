# Just a Sports Dataset Generator

Cross-platform, open-source generator for photorealistic synthetic sports datasets with automatic ground-truth annotations for computer vision.

> Project status: early implementation. Recipe validation, deterministic planning, a geometric reference renderer, visibility-correct annotations, and COCO export for RF-DETR are available. The export is tested through Reco Trainer's actual importer and RF-DETR training-data preparation. Photorealistic sports scenes and the desktop interface are still planned.

## Goals

- Generate diverse, photorealistic sports scenes on Windows, macOS, and Linux.
- Run the complete workflow locally with freely available components and no paid services.
- Export images and deterministic ground truth as COCO datasets for local RF-DETR training in Reco Trainer.
- Support detection, segmentation, pose estimation, tracking, and event-recognition workflows.
- Keep render engines, sports, annotation exporters, and models replaceable through plugins.
- Make every dataset reproducible from a versioned configuration, seed, and asset manifest.

## Product scope and sport order

macOS is the primary development and validation platform. The first usable version will include a graphical interface with scene preview, generation progress, and export controls.

Sport implementations will be completed in this order:

1. Basketball (`basketball`)
2. Football / soccer (`football`)
3. Futsal (`futsal`)

All rendering, model execution, annotation, validation, and export must run on the user's own computer. Components and assets must be free to obtain and use under licenses suitable for this workflow. Hosted image services, cloud rendering, paid APIs, subscriptions, and credit-based generation are outside the project scope.

After setup, generation must work offline. Initial component downloads and the planned application/plugin updater may use the network; models and large assets still require explicit download consent. The detailed milestones and acceptance criteria are in [docs/ROADMAP.md](docs/ROADMAP.md).

## Planned capabilities

- Configurable sports, venues, athletes, equipment, uniforms, cameras, lighting, weather, and crowd density.
- Domain randomization plus scenario presets for controlled experiments.
- Automatic annotations: bounding boxes, instance/semantic masks, keypoints, depth, optical flow, object IDs, camera calibration, and scene metadata.
- COCO datasets for RF-DETR, with native instance masks; MOT export remains planned.
- Fully local generation with replaceable local render and model backends.
- Validation reports for class balance, occlusion, annotation integrity, and train/validation/test leakage.

## Architecture

The application is split into a platform-neutral core and replaceable adapters:

1. **Scenario layer** — validates dataset recipes and expands them into deterministic scene jobs.
2. **Generation layer** — creates scene graphs, assets, motion, cameras, and environmental variation.
3. **Render backends** — produce RGB frames and auxiliary passes through engine-specific plugins.
4. **Annotation pipeline** — converts scene truth into task-specific labels and validates geometry.
5. **Export layer** — writes versioned datasets, manifests, checksums, and provenance.
6. **Desktop/CLI interfaces** — share the same application services and configuration schema.
7. **Updater** — securely updates the application and plugins while treating large model downloads separately.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design and update specification.

## Reproducibility and provenance

Every run should record:

- application, plugin, renderer, model, and schema versions;
- the normalized recipe and random seeds;
- asset identifiers and licenses;
- platform and relevant hardware information;
- file checksums and annotation validation results.

## RF-DETR and Reco Trainer compatibility

**RF-DETR is the target detector and training framework.** This repository generates datasets; Reco Trainer handles RF-DETR training. Ground truth comes from scene geometry and render passes. No detector needs to predict these synthetic labels. The export uses COCO, a data format accepted by RF-DETR. YOLO models and export are outside the confirmed scope.

Sports and object-label class order follow [wendibus/reco-trainer](https://github.com/wendibus/reco-trainer). The supported sport IDs are `basketball`, `football`, `futsal`, `handball`, `hockey`, `rugby`, `lacrosse`, and `american_football`. Plans derive the exact Reco Trainer labels for the selected sport; recipes cannot invent or reorder them. Integration tests verify COCO import, RF-DETR training-data preparation, and class order against revision `4002883eacbe07a3a7b43dd54422e9f222203e89`.

Choose the exported **coco/** folder in Reco Trainer's label import, with the matching project sport. Reco currently imports detection boxes only and rebuilds train/validation/test assignments: it does **not** preserve the export's scene-separated splits. Keep the original export for evaluations that require those fixed splits. See [export instructions and limitations](docs/EXPORTS.md).

These eight IDs are recognized by the planner. Sport-specific photorealistic rendering is still planned and will follow the implementation order above.

## Update policy

- Release channel: **stable**
- Automatic application updates: **enabled**
- Automatic plugin updates: **enabled**, subject to compatibility checks
- Automatic model downloads: **disabled**
- Rollback: **enabled**

Updates are distributed through signed GitHub Releases manifests. The updater verifies signatures and SHA-256 hashes, checks operating-system, architecture, application API, plugin API, schema, and model compatibility, installs atomically, and retains the last known-good version for rollback.

## Proposed repository layout

```text
src/
  core/          # domain model and application services
  scenario/      # recipe parsing and deterministic job expansion
  annotations/   # ground-truth generation and validation
  exporters/     # COCO for RF-DETR and native annotations
  plugins/       # plugin contracts and discovery
  updater/       # signed manifests, compatibility, atomic installs
  cli/           # command-line interface
  desktop/       # desktop application shell
schemas/         # versioned JSON schemas
plugins/         # first-party sport and backend plugins
tests/           # unit, contract, integration, and golden-data tests
docs/            # user and developer documentation
```

## Development principles

- Python 3.11+ for orchestration and data tooling; performance-critical or engine-specific components may use other languages behind stable interfaces.
- Typed interfaces, versioned schemas, and dependency inversion at backend boundaries.
- No silent network access: models and large assets require an explicit user action.
- Tests must cover determinism, exporter conformance, updater security, and migration/rollback paths.
- Do not commit generated datasets, model weights, secrets, or licensed third-party assets.

## Getting started

Requires Python 3.11 or newer.

```bash
python -m pip install -e .
sports-dataset validate examples/basketball.json
sports-dataset plan examples/basketball.json --output plan.json
sports-dataset render-reference examples/basketball.json --output outputs/demo --max-frames 2
sports-dataset export outputs/demo --output outputs/dataset
```

The reference renderer writes geometric RGB and instance-ID images. Export derives boxes and COCO masks from visible pixels, converts RGB to PNG, and writes native annotations, whole-scene splits, checksums, and provenance. Existing render/export folders are never overwritten; use a new output folder for each run. The renderer supports boxes and instance masks; other requested passes fail explicitly.

These geometric images are pipeline fixtures, not suitable for demonstrating photorealistic training quality. The next milestone is local basketball scene generation, followed by the macOS interface with preview.

For development, install `python -m pip install -e ".[dev]"`, then run `python -m unittest discover -s tests -v`, `ruff check .`, `ruff format --check src tests`, and `pyright`. The [export guide](docs/EXPORTS.md) describes the optional local Reco checkout needed for importer tests. CI runs that integration check automatically against the recorded revision. Dependencies and licenses are listed in [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## Contributing

Issues and pull requests are welcome. Please read [AGENTS.md](AGENTS.md) before making structural changes. Contributions must document provenance and licensing for every bundled asset and model.

## License

Licensed under the [Apache License 2.0](LICENSE).
