# Local photographic hall night run

`scripts/run_hall_night.py ROOT run` builds a hall-specific COCO detection/instance-mask
handoff for RF-DETR. The supplied user configuration requests exactly **2,000 images**:
1,000 independently planned 2.5D scenes with two differently cropped views each.
The source is the user's cleaned, straightened 3200 × 900 hall photograph. No paid
service, model download, cloud inference or automatic detector training is involved.

This is an experimental photographic compositor, not a 3D match simulator. It uses
separately generated people and the final alpha layers as the image-space source of
visible masks. Old Blender human boxes, joints and silhouettes are never reused.
The hall floor quadrilateral, rectified pixel horizon and estimated 2.4 m camera height
provide approximate visual scale calibration for both people and balls.
The five background board/rim/net envelopes are manually traced; they are not precise
net-thread segmentation. Supports, shadows and non-person gym equipment are background.
A different hall/image requires new calibration and a new output directory.

## Requested composition

- 800 images from men's scenes, 800 from women's scenes, 400 from mixed adult scenes.
- 1,400 images with one visible ball, 400 with two to four, 200 without a visible ball.
- 144 requested player cutouts: two genders, three requested skin-tone ranges, six poses,
  two team colours, two viewing directions. Adult stature varies from 1.58 to 2.14 m
  across the requested asset profiles. Twelve referee cutouts add grey/black,
  orange/black, striped and dark/neon kit styles. Attributes are generation requests,
  not independently verified demographic labels. All people remain class `player`
  or `referee`; demographics never become additional detector classes.
- Full court, half court, medium/action and tight crops. Tight crops can truncate
  people at the frame boundary. Crops retain the requested visible-ball bucket,
  falling back to a wider view when needed. A native-resolution retry prevents a tiny
  visible ball from disappearing during downsampling.
- Deliberate person/person and ball/person overlap. Full-scene and final-crop occlusion
  fractions are recorded separately from frame clipping. Completely hidden objects
  do not get a visible annotation. Every object that retains visible ID pixels is labelled.

The RGB and categorical instance pass are composed in the same depth order from the
same thresholded alpha masks. All crop transforms are shared; RGB uses Lanczos and IDs
nearest-neighbour. Thin antialiased photographic boundaries are therefore approximated
by categorical mask support. Soft floor contact shadows, a mild warm relight, texture
softening and joint scene exposure/colour variation help visual integration. They do
not establish physical light transport or prove training benefit.

## Local components and provenance

The host needs the existing development environment (Pillow, NumPy, jsonschema and
pycocotools). Qwen runs in its existing isolated Torch/diffusers environment; BiRefNet
runs separately in the existing matte environment (including OpenCV). They are not
loaded together. The runner refuses missing local components. Model paths/checksums,
input checksums, code fingerprints, prompts, seeds, generation timings, matte metadata,
cutout checksums and scene composition metadata remain alongside the output.

Qwen-Image-Edit-2511/Lightning: Apache-2.0; BiRefNet: MIT. No model weights are bundled
with this repository. The hall and ball/uniform reference sheets were supplied by the
user. Their redistribution rights are not asserted: they stay in the local run folder
and must not be committed or published as freely licensed project assets. Reference
sheet season/league/model claims are not verified; ball styles and generic uniform
colours are used as visual examples, not declarations of official equipment.

Matte checks reject empty/tiny or border-clipped foregrounds, retain the dominant
connected human foreground and check for conspicuous attached ball-like extras.
These heuristic checks are for rejection only, not reliable anatomy, pose, demographic
or object-content verification. Cutout contact sheets (`cache/asset-review-*.jpg`) and
the pilot must be reviewed before treating generated images as training-quality data.
Rejected candidates are retained and regenerated with deterministic alternate seeds,
at most twice; insufficient gender/kit/split pools stop the run rather than inventing
labels or substituting the wrong people.

## Start and resume

The prepared local run lives on the external volume at
`/Volumes/Extern/Just a Sports Dataset Generator/Nachtlauf-Halle`.
The prepared `engine/` and `source-crops/` snapshot isolates the night run from later
project edits. Model weights/environments remain at their existing local locations;
they are not duplicated. `scripts/prepare_hall_night.py` can prepare another run from
explicit local input/component paths or freeze an existing configuration before full
rendering starts (`--freeze-existing`). A snapshot is never silently updated.

Double-click `2.000 Hallenbilder starten.command`, or run its quoted absolute path in
Terminal. The launcher opens the live local progress page, prevents idle sleep and
denies network access to the generator. Keep the external volume mounted, the Mac
connected to power and the lid open. Avoid a concurrent Reco GPU training run.

Based on the first five 640 px photographs (about 84–86 seconds each on this machine),
allow roughly 4–7 hours for the initial full run; thermal throttling, other GPU work,
retries and storage speed can change this. The configured invocation budget is ten
hours for asset generation and rendering; export verification is allowed to finish.
This is a budget, not a promise that every run will finish overnight. The runner never
silently reduces the requested 2,000 images to meet a deadline.

Ctrl+C stops only this generator. Starting the same command resumes completed photos
and atomic scene directories after validating their checksums. Only one runner may
hold the output lock. A code, input, configuration or already-used library change
requires a new run directory, preventing mixed versions. The final export is published
atomically. An interrupted or failed run retains its progress and is clearly marked
incomplete. It never reports success before the exact count and full export checks pass.

CLI stages:

```sh
python scripts/run_hall_night.py ROOT assets
python scripts/run_hall_night.py ROOT pilot
python scripts/run_hall_night.py ROOT render
python scripts/run_hall_night.py ROOT run --hours 10
python scripts/run_hall_night.py ROOT verify
```

The small pilot intentionally reuses its five preview assets between its debug splits;
it is explicitly marked **preview only**. It is not a validation dataset. The full night
run partitions player/referee asset identities between train/validation/test and keeps
both crops of a scene in the same split: 1,600/200/200 images. Do not compare training
metrics from this single synthetic hall with real-world generalization. Keep separate
real held-out footage (including genuine women/men, occlusion and different cameras).

## Output and validation

- `index.html`, `state.json`, `Nachtlauf.log`: progress and labelled previews.
- `cache/`: local photographs, masks, cutouts, reference-ball crops and contact sheets.
- `renders/`: resumable scene groups, final RGB/instance masks and per-view metadata.
- `dataset/coco/{train,valid,test}/`: unlabelled RGB images plus `_annotations.coco.json`.
- `dataset/native/`: raw per-image instance IDs (these naturally look almost black).
- `dataset/manifest.json`: class/mode/shot counts, occlusion counts, asset splits and limits.
- `dataset/verification.json`: independent verification results.

The exporter validates scene schema 2.0 and reads final visible pixels. Independently,
pycocotools decodes **every** exported COCO RLE and compares it pixel-for-pixel with its
native final ID mask. It checks tight visible boxes, areas, category IDs/order, all visible
IDs, RGB checksums and exact image counts. Reco's classes stay `ball`, `player`, `referee`,
`hoop`, with native class IDs 0..3 and COCO category IDs 1..4.

In Reco Trainer select `dataset/coco/` and sport basketball. Its current label importer
imports detection boxes and rebuilds its own splits: it does not preserve these grouped,
identity-separated export splits. Keep the original export for any evaluation needing
those guarantees. A successful import or mask check is not a trained-model accuracy test.
