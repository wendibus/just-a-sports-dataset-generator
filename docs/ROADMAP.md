# Product roadmap

## Confirmed requirements

- All processing runs locally and works offline after setup.
- The application, required components, model weights, and assets are free to obtain and use for the intended workflow. Record source, version, and license before adoption or redistribution.
- No paid APIs, cloud rendering, hosted image services, subscriptions, credit requirements, or optional service integrations.
- macOS is the primary target for development and acceptance testing; retain cross-platform core interfaces.
- The first usable basketball version has a graphical interface with preview.
- Complete sport implementations in order: **Basketball -> Football/soccer -> Futsal**.
- RF-DETR is the target detector and training framework. Hand off COCO datasets to Reco Trainer; no YOLO model or export path.
- Sports and labels remain compatible with Reco Trainer. Prefer correctness and validation over delivery speed.

Downloads for installation and the planned updater may use the network. Models and large assets require explicit download consent. Rendering and dataset processing must remain available offline.

## Existing foundation

Implemented: strict JSON recipes, deterministic scene-job planning, a Reco Trainer taxonomy snapshot, basic plugin contracts, a geometric reference renderer, visible-pixel annotations, COCO export for RF-DETR, native instance masks, validation reports, checksums, and provenance.

The reference renderer is a pipeline test fixture. Photorealistic sport scenes, large-scale production validation, and the graphical interface remain to be implemented. Recognizing eight sport IDs does not mean eight complete sport implementations exist.

## Completed foundation: annotations and dataset export

- Normalized visible-pixel boxes, areas, and COCO RLE masks; fully hidden objects are omitted.
- COCO datasets for RF-DETR with Reco Trainer label names and explicit class-ID mappings.
- Image/mask/identity validation, scene-separated splits, and rejection of identical images across splits.
- Seeds, versions, source licenses, and output checksums in provenance manifests.
- Actual Reco Trainer COCO importer and RF-DETR training-data preparation tests for all eight taxonomy IDs, background frames, exact pixel boxes, and repeated imports.

Reco's importer does not preserve dataset splits and currently imports detection boxes only. See [EXPORTS.md](EXPORTS.md). This milestone does not establish photorealism or training accuracy.

Before releasing each sport milestone, compare the taxonomy with the upstream Reco Trainer source and record the checked revision. Any upstream change requires an explicit compatibility update.

## Basketball: first usable sport

- Build local court, equipment, player, and camera scenes with freely usable assets and motions.
- Generate realistic images and annotations from the same scene and visibility information.
- Verify image quality on representative basketball scenes and confirm imports in Reco Trainer.
- Provide a graphical interface for scene preview, generation settings, progress, and exports.
- Measure runtime and memory on the target Mac. Chip and memory capacity still need to be established before setting performance expectations.
- Demonstrate an end-to-end generation/export run with networking unavailable after setup.

## Football/soccer: second sport

Extend the validated pipeline for football pitch geometry, goals, teams, goalkeepers, officials, and motion. Use the canonical sport ID `football`. Apply the same graphical workflow, label checks, offline test, and Reco Trainer import acceptance criteria.

## Futsal: third sport

Adapt the pipeline to futsal court geometry, goals, ball, team setup, and indoor scenes. Use the canonical sport ID `futsal`. Share applicable components with football while validating sport-specific geometry and behavior.

## Distribution and later work

The existing stable-channel update policy remains planned: automatic application and compatible plugin updates, explicit model downloads, and rollback. Update availability must not gate local generation.

Other Reco Trainer sports follow the three priorities above; their implementation order is not yet fixed. No render engine, model, or asset pack is approved merely by inclusion in this roadmap: selections must satisfy the local operation, cost, quality, and license requirements.
