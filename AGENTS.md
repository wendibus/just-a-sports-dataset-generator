# AGENTS.md

Guidance for coding agents and contributors working in this repository.

## Mission

Build a cross-platform, open-source system that generates photorealistic synthetic sports datasets and trustworthy automatic ground-truth annotations. Prefer reproducibility, inspectability, and stable interfaces over shortcuts tied to one renderer or platform.

## Current phase

Early implementation (0.4.x). Implemented: recipes and deterministic planning, the geometric reference renderer, the local Blender basketball renderer with broadcast and Reco-panorama cameras, visible-pixel annotations, COCO export for RF-DETR, OpenPose keypoints, person crops, label-preserving paste-back, a consent-based local Qwen-Image-Edit runtime, and the Mac starter. Specified but **not implemented**: the signed updater, plugin discovery/isolation/signatures, a cross-platform desktop shell, and every sport other than basketball. Do not present planned functionality as implemented; the status table in ARCHITECTURE.md is the reference. Keep README status statements accurate as code lands.

## Confirmed product requirements

- All generation, rendering, annotation, validation, preview, and export must run locally on the user's computer and work offline after setup.
- Use components, model weights, and assets that are free to obtain and use for the intended workflow. No paid APIs, subscriptions, credits, hosted inference, or cloud rendering, including optional integrations such as Kling AI.
- Check and document each dependency's and asset's license for its intended use and distribution; free download alone is not sufficient.
- macOS is the primary development and acceptance-test platform. Keep the core cross-platform; do not assume a particular Mac chip or memory capacity.
- Deliver a graphical interface with preview in the first usable basketball version.
- Complete sport implementations in this order: basketball (`basketball`), football/soccer (`football`), then futsal (`futsal`). Other Reco Trainer sport IDs remain in the taxonomy for later work.
- Target RF-DETR training in Reco Trainer, with COCO dataset handoff. Do not add YOLO models, dependencies, or export paths without a new explicit user request. Derive synthetic ground truth from scene/render data; an RF-DETR training run is separate from dataset generation.
- Preserve Reco Trainer sport IDs, label names, and class ordering. Taxonomy availability does not imply a completed sport renderer.
- Prioritize correctness, realistic output, and verified Reco Trainer imports over turnaround speed.

See [docs/ROADMAP.md](docs/ROADMAP.md) for milestones and acceptance criteria.

## Engineering rules

- Target Python 3.11+ for orchestration and data tooling.
- Keep domain logic independent from desktop UI, render engines, storage providers, and network clients.
- Use typed public interfaces and versioned JSON schemas.
- Make generation deterministic when recipe, assets, versions, and seeds are identical.
- Treat annotations as derived ground truth with validation, never as an incidental renderer by-product.
- Put engine-, sport-, exporter-, and model-specific behavior behind plugin contracts.
- Avoid hidden downloads or telemetry. Models and large assets require explicit user action.
- Never commit credentials, generated datasets, model weights, or third-party assets without redistribution rights.

## Structure

Package `src/just_a_sports_dataset_generator/`:

- `core`: domain model and the Reco Trainer taxonomy
- `scenario`: recipe parsing, validation, deterministic job expansion
- `annotations`: visible-pixel annotations, keypoint visibility, renderer handoff validation
- `exporters`: COCO for RF-DETR, native annotations, provenance
- `renderers`: reference fixture and the Blender adapter; `renderers/blender_worker` is the
  isolated GPL-3.0-or-later worker that runs inside Blender (never import it from the host)
- `generators`: consent-based local Qwen runtime; `qwen_worker.py` runs in that runtime
- `crops.py`, `composite.py`, `preview.py`: person crops, label-preserving paste-back, previews
- `plugins`: plugin metadata contracts only (no discovery or isolation yet)
- `cli`: command-line interface; `macos/` holds the Mac starter and its PyInstaller spec

Planned: `updater` (signed manifests, atomic install, rollback) and a desktop shell.
`schemas` holds versioned contracts; `tests` holds unit, contract and integration tests.

## Change workflow

1. Read README.md and ARCHITECTURE.md before structural changes.
2. State assumptions and keep the change narrowly scoped.
3. Add or update tests with behavior changes.
4. Update schemas and migration notes when persisted formats change.
5. Run formatting, linting, typing, and relevant tests before handoff.
6. Document security, compatibility, provenance, and licensing implications.

## Testing priorities

- deterministic replay across supported platforms;
- coordinate transforms, visibility, occlusion, and annotation bounds;
- exporter conformance against small golden fixtures;
- malformed recipes and hostile plugin/manifest input;
- interrupted updates, signature/hash failures, incompatible versions, and rollback;
- migrations that preserve user projects and configuration.

Tests should use small synthetic fixtures. Do not add large binary assets to normal Git history.

## Update-system constraints

The default channel is stable. Application and compatible plugin updates may install automatically. Model downloads must remain opt-in. Verify signed manifests and SHA-256 hashes before installation, use atomic replacement, retain a last-known-good version, and refuse incompatible application/plugin/schema/model combinations.

## Pull requests

Explain what changed, why, verification performed, compatibility impact, and any follow-up work. Keep generated files out of reviews unless they are required fixtures with documented provenance.
