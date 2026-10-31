# Dependencies and asset provenance

The application runs locally without fees, subscriptions, hosted inference, or model downloads. Installation obtains ordinary open-source Python packages; dataset processing does not contact a service.

The export milestone was checked with these distributions on macOS, Python 3.12. License identifiers below come from the installed package metadata. Keep the relevant license notices when distributing packages; Apache-2.0 for this repository does not replace third-party terms.

| Component | Checked version | Purpose | License |
| --- | --- | --- | --- |
| Pillow | 13.0.0 | Runtime: read image passes, compute masks, write lossless PNG | MIT-CMU |
| pycocotools | 2.0.11 | Tests: independently decode COCO RLE and check boxes/areas | FreeBSD / BSD-2-Clause |
| jsonschema | 4.26.0 | Tests: validate persisted JSON contracts | MIT |
| Ruff | 0.17.3 | Development: formatting and lint | MIT |
| Pyright | 1.1.414 | Development: static typing | MIT |
| build | 1.6.1 | Development: package build verification | MIT |
| Hatchling | 1.33.0 | Build backend | MIT |

Pillow is the only direct runtime dependency. COCO validation tools and their NumPy dependency are development dependencies, not required for generation/export. No third-party imagery, athletes, model weights, motion captures, textures, or geometry assets are bundled. The reference fixture is generated entirely by repository code under Apache-2.0 and records its source/version in scene and run manifests.

RF-DETR is the target training framework in Reco Trainer. Dataset generation itself needs no detection model or weights. This repository neither installs a YOLO/Ultralytics dependency nor exports YOLO data. Full RF-DETR training is performed by Reco Trainer, outside the generator's format/import tests.

Reco Trainer is fetched only for the explicit CI compatibility test (or supplied as a local checkout). It is not bundled, modified, imported by the application, or used as a hosted service. The tests execute its existing local COCO importer and RF-DETR dataset builder against temporary synthetic data.

A future desktop bundle must inventory its exact transitive dependencies and retain their notices. Future photorealistic assets and backends require separate source, license, redistribution, and no-cost checks before adoption. No such asset pack is approved by this milestone.
