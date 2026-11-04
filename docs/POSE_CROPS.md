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
RGB crop, with everything outside the person's inpaint mask blurred, and a built-in
prompt; `--prompt` replaces it. `--pose-image` adds the OpenPose skeleton as a second
input image (roughly twice the tokens). Seeds are derived from `--seed` and the crop ID, so a rerun reproduces a
crop. `--device auto` uses Apple Metal (MPS), then CUDA, then CPU. If the Lightning LoRA
cannot be applied to the quantized model, the run falls back to 30 steps with
classifier-free guidance and says so.

The built-in prompt (version 2, recorded in `generation.json`) was revised after the
first Mac test, in which "sleeveless jersey with shoulder straps" produced overall-style
straps with buckles and two of three crouched poses came out standing upright:

- players wear "a classic basketball tank-top jersey with wide shoulder panels sewn to
  the jersey, a deep scoop neckline, piped armholes and the number printed on the chest"
  and knee-length shorts, **in exactly the colours of image 1** (no colour is named);
  referees wear a grey striped short-sleeved shirt and black trousers;
- each crop carries its planner pose family (`pose_family` in `crops.json`), and the
  prompt names that stance explicitly, for example "in a low defensive stance, knees
  deeply bent, feet wide apart, arms spread out";
- a negative prompt (straps, buckles, overalls, backpack, square neckline, standing
  upright, deformed hands, …) applies to the 30-step fallback; the 4-step Lightning
  schedule runs without classifier-free guidance, so there the wording of the positive
  prompt does the work.

Version 3 follows the second Mac test (three crops, M4 Pro, Reco training in parallel):

| Crop | Result | Cause | Change |
| --- | --- | --- | --- |
| shoot | photorealistic, kit and pose right | — | — |
| dribble | pose right, but stiff "shoulder panels" drawn as wings and a 3D-game look | the phrase "wide shoulder panels" | phrase removed; "a real photograph, not a 3D render, not a doll" |
| defend | a second, distorted player, rock-like shoulders, romper shorts | another player inside the crop was repainted too | everything outside the person's mask is blurred before generation; "only one person" |

**Tokens.** diffusers' `QwenImageEditPlusPipeline` encodes every input image at
1024 × 1024 (4096 tokens each). With a 768 px output and two inputs, the transformer
processed about 10,500 tokens per step, and one crop took 12–15 minutes (714–878 s for 4
steps, with the training running alongside). The worker now encodes the inputs at the work
resolution and sends only the RGB crop by default: about 4,600 tokens at 768 px.

**Speed.** Qwen works at most at `--resolution` pixels per side (default 768; 1024 is
sharper and roughly 1.8× slower), then the result is scaled back to the crop size. With
Lightning, a crop takes 4 denoising steps instead of the 40 used in the first manual
test, which took about 26 minutes per image (40 × ~39 s at 640 × 960 with mflux, 8-bit).
Per-step time on MPS has not been measured for this runtime yet; `generation.jsonl`
records the seconds per crop, so the first run gives the number.

`generation.json` records the model, quantization, packages, checksums and settings;
`generation.jsonl` adds seed, steps and time per person; `qwen.log` keeps the worker
output.

On an M4 Pro (48 GB), loading the model took 195 s once per run (measured while a Reco
Trainer training ran in parallel). The manual mflux test of the same model peaked at about
39 GB, so do not run Qwen next to another GPU-heavy job: both slow down and the other job
can run out of memory.

**Memory.** The first Mac run of prompt v2 (M4 Pro, Reco training in parallel) finished two
crops and then failed with "MPS backend out of memory": PyTorch kept the freed blocks of
earlier crops in its pool. The worker now returns that cache after every crop and, on an
out-of-memory error, frees it, retries once at the same size and then at about 84 % of the
side length per try (down to 256 px). The resolution actually used is recorded per crop in
`generation.jsonl`, and the progress line shows the seconds per person.

## Inpaint mode (default since prompt version 4)

The overnight run of prompt v3 (118 isolated people, 512 px, 63 s each on an M4 Pro)
produced photorealistic people, but in `edit` mode Qwen redraws the whole crop as a new
photo: eye-level camera, a larger, centred person, a new hall. Pasted back with the instance
mask, the figures fell apart; the spill check would have rejected 117 of 118 (median
background change 0.24).

`--mode inpaint` (default) uses diffusers' `QwenImageEditInpaintPipeline` with the same
installed model: generation starts from the render, only pixels inside `inpaint.png` may
change, and `--strength` (default 0.75) sets how far the result may move away from the render.
Outline, size, camera angle and kit colours therefore come from Blender. The schedule is
stretched so that the requested 4 Lightning steps are actually run (5 scheduled steps at
strength 0.8). The prompt describes only the person and forbids lettering except the number.
`--mode edit` keeps the previous behaviour.

Prompt version 5 names the kit colours. `export-crops` adds `kit` to every crop, read
from `layout.json`: a colour word for players ("white") and the planned referee kit
("a plain grey short-sleeved shirt with black sleeves and black long trousers"). At
strength 0.9, without names, white kits had come out black or blue and grey referee
shirts had become striped.

Strength test on 12 crops of the night run (M4 Pro, 512 px, about 80 s per person): every
result stays inside the silhouette and keeps the broadcast camera angle. At 0.6 bodies and
shorts stay doll-like; at 0.75 faces, skin and fabric look real and the kit colours hold;
at 0.9 kits start to drift (a white kit became blue, white shoes appeared). Where the 3D
figure is bulkier than a real body (round shoulder caps, wide shorts), the result fills the
difference with background inside the mask, visible as a faint halo.

`scripts/qwen_inpaint_compare.py RUN` repaints 12 crops of an earlier run at strengths
0.6, 0.75 and 0.9 and writes `RUN/compare/compare.jpg` with every result pasted back with
the instance mask.

## Overnight batch

`scripts/qwen-overnight.sh [OUTPUT] [HOURS] [SCENES]` (defaults `../qwen-overnight`, 9,
60; env `STRENGTH`, default 0.75) renders broadcast scenes, exports 512 px crops with
`export-crops --isolated`, runs `generate-crops --mode inpaint --resolution 512
--time-limit` with whatever time is left and pastes the results back with
`composite-crops`. `OUTPUT/index-inpaint-<strength>.html` then shows, per person, the
render next to the repainted scene at source resolution and whether the quality checks
applied or rejected it. An existing OUTPUT reuses its render and crops. The script keeps
the Mac awake with `caffeinate`, restarts Qwen after a crash and resumes every step when
started again.

`--isolated` keeps only people whose whole crop box lies inside the frame, whose 14 body
joints are all visible (`v = 2`) and whose crop box shows at most 3 % other players or
referees. On broadcast renders that leaves about 2–4 people per scene. Bench players and
spectators are background in the ID pass and are not counted; the worker blurs them.
`generate-crops` alternates pose families, so a run stopped by `--time-limit` still covers
every pose; the limit is not a setting, so the same folder continues with a new budget.

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

## Independent foreground review and controlled strength comparison

`scripts/qwen_quality_compare.py RUN --output NEW_FOLDER --ids ID1 ID2 ID3
--strengths 0.65 0.75 0.85 --review-python /path/to/foreground-env/bin/python
--foreground-model /path/to/local/BiRefNet` compares the same source crops with
fixed seed 0 and 512 px output. Add `--baseline RUN/generated-inpaint-0.9` to include
an existing 0.9 run without regenerating it. Use a new output folder when settings
change. Matching comparisons resume completed generation and foreground extraction.
Qwen must already be installed; the comparison declines any installation request.
BiRefNet must be explicitly installed separately (MIT, local code and safetensors).
There are no automatic model downloads. The foreground helper uses CPU so it does
not keep another model in accelerator memory next to Qwen.

The comparison writes source/result previews and `quality-report.json`. Predicted
foreground masks are filters only; they never replace Blender IDs. `rejected` means
numerical checks failed. `needs_visual_review` means a human must still review skin,
uniforms, anatomy, pose, ball contact and consistency at the original frame size.

When `composite-crops --silhouettes DIR` is supplied, the compositor additionally
requires at least 90% foreground coverage and at least 55% coverage in every populated
local window of the original silhouette, allowing two pixels of boundary tolerance.
Overlapping local windows catch missing hands/feet that whole-person IoU can hide.
These are conservative starting thresholds, not calibrated anatomy detection.
Foreground estimation may reject a usable image, especially with a touching ball or
occlusion. A pass does not establish photorealism or training value.

Reports with silhouette checks use `composite-report-v1.1.json`, adding foreground
coverage, minimum local coverage, and the supplied silhouette's SHA-256. Runs without
silhouettes keep report 1.0 for compatibility. Rejected replacements leave RGB and
all original masks/annotations untouched. The original model/source provenance and
labels remain unchanged. An old `applied` status alone is not quality approval.

To audit an existing batch and produce a fresh, conservatively filtered render:

```sh
python scripts/review_generated.py /path/to/run \
  --generated /path/to/run/generated-inpaint-0.9 \
  --output /path/to/run/reviewed-batch \
  --review-python /path/to/foreground-env/bin/python \
  --foreground-model /path/to/local/BiRefNet
```

The script writes predicted masks, then invokes `composite-crops` with those masks
required. Missing or ambiguous silhouettes reject a replacement; missing generated
images remain original renders. The checked render and composite report are under
`reviewed-batch/render/`. Never treat the numerical applied count as a count of
photorealistic, anatomically correct training examples.

## Experimental photographic cutouts

`scripts/qwen_photo_candidates.py RUN --output NEW_FOLDER --ids ID1 ID2
--review-python /path/to/foreground-env/bin/python --foreground-model /path/to/BiRefNet`
tests an alternative to appearance-only inpainting. It replaces the rendered
background with neutral grey, uses the pose/kit as guidance for a 768 px photographic
edit, and extracts a new alpha mask with the explicitly installed local BiRefNet.
It declines model downloads and writes a separate comparison page and RGBA cutouts.

These are **visual candidates, not annotated scene replacements**. The original
Blender contours, projected joints and bounding boxes are not ground truth for the
edited body. Pose/camera consistency and anatomy require human review. The cutout experiment alone cannot be exported as a labelled render. Source
renders/crops remain untouched.

`scripts/assemble_basketball_example.py ROOT --approved APPROVED_CUTOUTS` assembles
one camera-specific 5-v-5 demonstration from a clean local Blender court render,
ten distinct photographic player cutouts and a referee cutout. The example script
has explicit asset IDs/placements; it is not a general scene generator. Its core
`photo_composite.py` sorts layers by ground depth and rasterizes the same thresholded
alpha (>=128) into RGB and instance IDs. Fully hidden pixels are excluded, and boxes
are recomputed from the final visible masks. Old Blender human masks/keypoints are
never reused. Contact shadows are approximate background pixels.

The example exports schema-2.0 scene metadata, COCO for RF-DETR, a labelled preview,
raw instance masks and asset hashes. It independently decodes every exported COCO
RLE with pycocotools, compares it pixel-for-pixel to final instance IDs and verifies
box extents/areas and Reco category ordering (`ball`, `player`, `referee`, `hoop`).
The example script therefore needs the developer dependencies. Foreground quality,
camera/lighting consistency and anatomy still need visual review: exact annotation
support does not imply a physically simulated match or a validated training corpus.
Qwen (Apache-2.0) and BiRefNet (MIT) remain explicitly installed local components;
this stage adds no downloads or inference service.

## Hall-specific overnight dataset

See [the local night-run workflow](HALL_NIGHT.md) for the experimental 2,000-image
photographic compositor, asset checks, visibility/crop labels, identity-separated
COCO splits, resume behaviour and the limits of single-hall synthetic data.
