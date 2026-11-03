# Camera look

Rendered frames are cleaner than real footage: full resolution, no motion blur, no
sensor noise, no compression, perfect white balance. At Reco panorama scale, where a
player is 40–100 pixels tall, these camera artefacts shape what a detector sees far more
than skin or fabric detail. `camera-look` degrades a copy of a render like a real camera:

```sh
sports-dataset camera-look outputs/reco --output outputs/reco-look            # preset reco
sports-dataset camera-look outputs/reco --output outputs/reco-light --preset light
sports-dataset export outputs/reco-look --output outputs/dataset
```

Only the RGB images change. Instance IDs, poses, layouts and every other file are copied
byte for byte, so boxes, masks and keypoints stay the renderer's ground truth; `export`,
`preview`, `export-crops` and `composite-crops` accept the result like any render. It
works on plain renders and on repainted ones (`composite-crops` output).

## Steps

Each frame draws its own parameters from the preset ranges with a random stream of
(preset, `--seed`, scene, frame), so the same command reproduces the same images. All
parameters are written to `camera-look.json`; every `scene.json` gains a `camera-look`
asset entry.

| Step | `reco` | `light` |
| --- | --- | --- |
| Lost resolution (share of pixels per side kept, then scaled back) | 0.55–0.85 | 0.70–0.95 |
| Horizontal motion blur | 0–4 px | 0–3 px |
| White balance (warm/cool) and tint (green/magenta) | ±8 %, ±4 % | ±5 %, ±2 % |
| Contrast, saturation, brightness | 0.90–1.05, 0.80–1.05, 0.88–1.08 | 0.92–1.05, 0.90–1.05, 0.94–1.06 |
| Vignetting | 5–25 % | 0–12 % |
| Sensor noise, luma and chroma (8-bit steps) | 2–6, 0.5–2 | 1–3, 0–1 |
| In-camera sharpening (unsharp mask) | 20–60 % | 10–40 % |
| JPEG compression with 4:2:0 chroma subsampling | quality 40–75 | quality 55–85 |

Motion blur is global (camera pan), not per player. Blur and compression spread a person
by a pixel or two across its mask edge, as in real footage; the labels still mark the
person's true outline.

## Calibration

The ranges are first estimates. They should be compared with real Reco frames of the
same hall size and lighting: resolution and compression first, since they decide how
much detail a 50-pixel player still has. Comparing a detector trained with and without
the camera look on held-out real frames is the actual test.
