# Pose crops for photorealistic athletes

The procedural athletes are visibly synthetic. The intended upgrade path keeps Blender
as the source of truth for geometry and labels and lets a local, pose-conditioned image
model repaint only the people. The generator ships a local Qwen-Image-Edit-2511 runtime
that it installs on first use, after asking; any other local tool can produce the same
results.

## Outputs

`render-basketball` writes `poses.json` next to every `scene.json`
([schema](../schemas/poses-v1.0.json)): one OpenPose BODY_18 skeleton per player and
referee and frame, projected from the athlete geometry with the same camera as the
render (perspective or `reco_panorama`). Each keypoint is `[x, y, v]` in pixel-centre
coordinates with COCO visibility:

- `0` outside the image (coordinates zeroed),
- `1` inside but hidden — by another object (checked against the same Object Index
  pass as the labels) or, for nose, eyes and ears, by the person's own head when it is
  turned away from the camera,
- `2` visible.

```sh
sports-dataset render-basketball examples/basketball-reco.json --output outputs/reco
sports-dataset export-crops outputs/reco --output outputs/crops --size 1024
```

`export-crops` writes, for every visible person at least `--min-height` pixels tall
(default 24), a square crop around the visible box plus `--margin` (default 25 %),
resized to `--size`:

| File | Content |
| --- | --- |
| `*.rgb.png` | rendered crop; parts outside the frame are padded black |
| `*.mask.png` | exact instance mask (white = this person, visible pixels only) |
| `*.inpaint.png` | the mask dilated by `--dilate` crop pixels (default 9): the only region a generator may change |
| `*.pose.png` | OpenPose skeleton on black. Body joints with `v ≥ 1` are drawn, so a leg hidden behind another player still defines the pose; face points only with `v = 2`, as OpenPose would show them |

`crops.json` ([schema](../schemas/crops-v1.0.json)) lists every crop with its source
frame, instance ID, visible box, the crop box in source pixels and the scale (crop
pixels per source pixel), so a generated crop can be resized by `1 / scale` and pasted
back at `crop_box_xyxy`.

## One command

```sh
sports-dataset repaint outputs/reco --output outputs/reco-repaint
sports-dataset export outputs/reco-repaint/render --output outputs/dataset
```

`repaint` writes `crops/`, `generated/` and the composited `render/` into the output
folder. If Qwen is not installed yet, it first prints the download plan (files, sizes,
commits, licenses, destination, about 31 GB in total) and asks `[j/N]`; `--yes` approves
it non-interactively, `sports-dataset setup-qwen` installs ahead of time and
`sports-dataset qwen-status` reports the install. An interrupted generation resumes when
the same command is run again; finished people are skipped.

## Generating with the built-in Qwen

`generate-crops CROPS --output GENERATED` repaints each crop with Qwen-Image-Edit-2511
(GGUF `--quant`, default Q4_K_M) and the 4-step Lightning LoRA. The model receives the
RGB crop and the pose skeleton as two input images and a built-in prompt (player or
referee) asking for a photorealistic person in exactly that pose, position and size,
with the original kit colours; `--prompt` replaces it. Seeds are derived from `--seed`
and the crop ID, so a rerun reproduces a crop. `--device auto` uses Apple Metal (MPS),
then CUDA, then CPU. If the Lightning LoRA cannot be applied to the quantized model, the
run falls back to 30 steps with classifier-free guidance and says so.

`generation.json` records the model, quantization, packages, checksums and settings;
`generation.jsonl` adds seed, steps and time per person; `qwen.log` keeps the worker
output.

Expect roughly 30 GB of memory use and one to a few minutes per person on an Apple
M-series Mac with 48 GB; these figures come from public reports for the same model, not
from measurements in this project. The end-to-end run on real weights has not been
verified yet; the code path is tested with a stand-in runtime.

## Using another tool

Feed each crop to a local inpainting workflow: `rgb.png` as the image, `inpaint.png` as
the mask and `pose.png` into a pose ControlNet. Save the result as `<crop id>.png`
(ComfyUI's `<crop id>_00001_.png` counter suffix is accepted, subfolders are fine).
Results must be square; they are resized to the crop size. Record the model and
workflow with `--generator`; any model must meet the project rules: local, offline
after setup, and with a license recorded in [DEPENDENCIES.md](DEPENDENCIES.md).

## Pasting back

```sh
sports-dataset composite-crops outputs/reco --crops outputs/crops \
  --generated outputs/generated --output outputs/reco-repainted \
  --generator "Qwen-Image-Edit + pose ControlNet, workflow v1"
sports-dataset export outputs/reco-repainted --output outputs/dataset
```

The result is a copy of the render that `export` and `preview` accept unchanged.
`composite.json` ([schema](../schemas/composite-report-v1.0.json)) lists every crop as
`applied`, `rejected` (with reason and metrics) or `missing`, with the SHA-256 of each
generated file; every repainted `scene.json` gains a `pose-guided-repaint` asset entry,
which `export` carries into the dataset manifest. Crops are checked against the
untouched render, so a crop set made from a different render is refused.

**Labels are never re-detected.**

- `--blend mask` (default) writes generated pixels only where the instance-ID pass shows
  that person. Silhouettes and therefore every box and mask stay exactly as rendered;
  only the appearance changes. If the generated figure is thinner than the silhouette,
  its background fills the difference, so keep the figure close to the pose.
- `--blend inpaint` also accepts the dilated band (`dilate / scale` source pixels,
  usually under one pixel for panorama players), but never pixels of another object:
  other players, the ball and the hoops stay exactly as rendered.

Quality checks reject a crop and keep the rendered person when:

- the person barely changed (`--min-change`, mean colour difference inside the mask,
  default 0.02),
- the background band around the person changed strongly (`--max-spill`, default
  0.15), a sign of a shifted or oversized figure,
- `--silhouettes DIR` is given (for example from a SAM node in the same workflow,
  named like the results) and the silhouette's IoU with the instance mask is below
  `--min-iou` (default 0.85).

The thresholds are starting points; tune them on your first batches. The silhouette
check is a filter on generated pixels, never a source of labels.

## Limits

- Joints come from the planner's procedural pose families and motion model (see
  [scene variety](BASKETBALL.md#scene-variety)); there is no motion capture and the
  torso does not bend or twist.
- Self-occlusion is modelled only for face points; a forearm behind the torso of the
  same person still counts as visible.
- Upscaling a 40-pixel panorama player to 1024 pixels gives the generator a blurry
  input; that is intended, since it repaints the person, but the result must be
  downscaled for the dataset.
