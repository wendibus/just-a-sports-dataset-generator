# Just a Sports Dataset Generator

Cross-platform, open-source generator for photorealistic synthetic sports datasets with automatic ground-truth annotations for computer vision.

> Project status: early bootstrap implementation. Recipe validation, deterministic scene-job planning, plugin contracts, and a geometric reference renderer are available; photorealistic rendering and production annotation export are still planned. Interfaces and formats may change before the first stable release.

## Goals

- Generate diverse, photorealistic sports scenes on Windows, macOS, and Linux.
- Export images together with deterministic, machine-readable ground truth.
- Support detection, segmentation, pose estimation, tracking, and event-recognition workflows.
- Keep render engines, sports, annotation exporters, and models replaceable through plugins.
- Make every dataset reproducible from a versioned configuration, seed, and asset manifest.

## Planned capabilities

- Configurable sports, venues, athletes, equipment, uniforms, cameras, lighting, weather, and crowd density.
- Domain randomization plus scenario presets for controlled experiments.
- Automatic annotations: bounding boxes, instance/semantic masks, keypoints, depth, optical flow, object IDs, camera calibration, and scene metadata.
- Dataset exports for COCO, YOLO, MOT, and a native lossless scene format.
- Local-first generation with optional external render or model backends.
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

## Reco Trainer compatibility

Sports and object-label class order follow [wendibus/reco-trainer](https://github.com/wendibus/reco-trainer). The supported sport IDs are `basketball`, `football`, `futsal`, `handball`, `hockey`, `rugby`, `lacrosse`, and `american_football`. Plans derive the exact Reco Trainer labels for the selected sport; recipes cannot invent or reorder them. Compatibility tests lock this mapping so taxonomy drift is detected before release.

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
  exporters/     # COCO, YOLO, MOT, native format
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
sports-dataset render-reference examples/basketball.json --output outputs/demo --max-scenes 1 --max-frames 1
```

The CLI validates strict JSON recipes and expands each recipe into deterministic, resumable scene jobs. The built-in reference renderer exercises the full plugin and file-output path using simple geometric PPM/PGM images; it is a test backend, not a photorealistic renderer. The next milestone adds the normalized annotation model and COCO export.

## Contributing

Issues and pull requests are welcome. Please read [AGENTS.md](AGENTS.md) before making structural changes. Contributions must document provenance and licensing for every bundled asset and model.

## License

Licensed under the [Apache License 2.0](LICENSE).
