# Pose crops for photorealistic athletes

The procedural athletes are visibly synthetic. The intended upgrade path keeps Blender
as the source of truth for geometry and labels and lets a local, pose-conditioned image
model (for example Qwen-Image with a pose ControlNet, run in ComfyUI) repaint only the
people. This document covers the data the generator provides for that; the generation
and paste-back steps are not implemented yet.

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

## Keeping the labels true

Labels are never re-detected. They stay valid as long as the paste-back only changes
pixels inside `inpaint.png` and the generated person fills roughly the original
silhouette. The dilation band is `dilate / scale` source pixels wide, usually under one
pixel for small panorama players. Other people, the ball and the hoops are outside the
mask and stay exactly as rendered, which also protects an occluding player in front.

Planned next steps: a paste-back command that composites only inside the inpaint mask
and rejects crops whose generated silhouette drifts (IoU against `mask.png`), and a
documented ComfyUI workflow. Any model used must meet the project rules: local,
offline after setup, and with a license recorded in [DEPENDENCIES.md](DEPENDENCIES.md).

## Limits

- Joints come from the procedural athletes (no rig), so the poses are as limited as
  their three static stances.
- Self-occlusion is modelled only for face points; a forearm behind the torso of the
  same person still counts as visible.
- Upscaling a 40-pixel panorama player to 1024 pixels gives the generator a blurry
  input; that is intended, since it repaints the person, but the result must be
  downscaled for the dataset.
