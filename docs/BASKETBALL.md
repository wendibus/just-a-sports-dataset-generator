# Local 3D basketball — version 0.4

The backend generates a procedural indoor court, painted markings, two baskets,
ten athletes in two kits, one or two referees, a basketball, stands with optional
spectators, lights, and a camera. A renderer-independent planner varies the game
situation, poses, ball, hall and motion per scene (see [scene variety](#scene-variety)).
It needs locally installed Blender 5.x. No models, textures, motion libraries,
accounts, or remote processing services are used. Geometry and materials are
created by the bundled worker source. Football and futsal remain future stages.

## Run and inspect

```sh
sports-dataset render-basketball examples/basketball-3d.json --output outputs/render --samples 32
sports-dataset export outputs/render --output outputs/dataset
sports-dataset preview outputs/dataset --output outputs/Vorschau
```

Use new output folders. An interrupted render resumes when the same command is run again: finished scenes in `<output>.partial` are kept, and only the missing ones are rendered. The example generates three 1280×720 images from three
independent scene seeds. `--max-scenes` and `--max-frames` limit a recipe.
`--device cpu` is the default; `--device auto` tries Metal on macOS and otherwise
uses CPU. Cold Metal compilation can be slower than CPU. The backend was checked
with Blender 5.0.1 on an Apple M4 Pro with 48 GB RAM. Rendering can take minutes
per image; timings and the selected device are recorded per scene.

The Mac starter defaults to **Basketball starten**, with Standard (32 samples)
and Fein (64 samples). A local HTML progress page reports each scene and opens
the finished preview. It does not contact a web server or need internet access.
Closing the progress page does not cancel rendering. Cancellation/resumption and
a full scene editor are not implemented. `--reference-demo` retains the old
rectangle test; `--demo` now generates the three 3D basketball images.

## Scene variety

Each scene is planned by `sports/basketball.py` (Apache-2.0, unit-tested without
Blender) from the scene seed, and the plan is saved as `layout.json` next to
`scene.json`; the Blender worker only builds geometry from it. `scene.json` records
the plan's version, situation, ball event and checksum as an asset.

- **Situations** (weights): half-court set play towards either basket (50 %),
  transition over the whole court (25 %), free throw with players on the lane (15 %),
  jump ball at the centre circle (10 %). Offense spots are sampled from typical
  positions with jitter; defenders guard their player between player and basket;
  athletes keep at least 0.6 m apart.
- **Poses**: stand, run, defend, dribble, shoot, pass, catch, rebound (with jump) and a
  referee signal, each with continuous parameters (crouch, stance width, stride, arm
  elevation, direction and elbow bend). Legs use two-bone IK with constant bone lengths
  and feet on the floor; knees always bend forward. The same joints feed `poses.json`.
- **Ball**: held or dribbled (bouncing between hand and floor), passed along an arc to
  a team-mate, shot along an arc to the rim and dropping through, at the rim, loose on
  the floor, or tossed at a jump ball. A pass or shot may already be under way at the
  first frame. The ball is labelled in every state; occluded frames simply have no box.
- **Hall type**: an arena with stands (45 %) or a multi-sport school gym (55 %), the
  typical hall of amateur leagues and therefore of many Reco recordings. School gyms
  have no stands, half of them a synthetic floor (green, blue, grey or red) instead of
  wood, court lines of other sports in other colours (handball 40 × 20 m with dashed 9 m
  lines, volleyball, three crosswise badminton courts), often a band of daylight windows
  with sun, and wooden wall bars. Arenas occasionally carry volleyball lines too. Other
  sports' lines never share the basketball line colour.
- **Hall and light**: wood tone and gloss, key, line, wall and seat colours, light
  energy, colour temperature (3800–6500 K) and number of light rows, scoreboard text.
- **Spectators**: none, sparse or dense seated figures on the stands. They render as
  background (ID 0): people outside the court are hard negatives, not labelled players.
- **Team benches**: in three of four scenes per team, 3–7 seated substitutes in the team
  kit, and often a standing coach, in front of the far stands (y ≈ 9 m). They are
  background as well: the label policy counts only people on the court as `player`. If
  your real labels also mark bench players, say so; this is a one-line policy change.
- **Motion**: frames are 0.2 s apart. Athletes move with situation-dependent velocities
  (up to about 6 m/s in transition, static at free throws and jump balls) and the ball
  follows its event, so consecutive frames differ while instance IDs stay stable.
- **Animation** (layout schema 1.1): every frame has its own joints. Runners cycle their
  legs and swing the opposite arm with a stride from their speed and face where they run;
  defenders shuffle; dribblers push the ball down in time with the bounce; passers extend
  their arms at the release; the shooter moves from the set point to the release at the
  moment the ball leaves the hands. Rebounds, jump shots and jump balls follow a
  ballistic arc from the take-off. The renderer moves each limb to the frame's joints
  (bone lengths are constant), so `poses.json` and the rendered body always agree.

## Camera presets

Recipes with `"schema_version": "1.1"` may set `"camera"`. Version 1.0 recipes keep
the broadcast camera and reject the field.

- `broadcast` (default): elevated perspective camera about 20 m behind the near
  sideline, 9–11 m high, 35 mm lens, drifting slightly between frames.
- `reco_panorama`: approximates the stitched full-court panorama that Reco Trainer
  trains on. Reco stitches two wide-angle cameras at the centre of one sideline and
  addresses the panorama by yaw/pitch; this preset renders that view directly as an
  **equirectangular** Cycles panorama from a static mount 2.0–4.5 m high and
  0.5–2.0 m behind the near sideline (±1 m along it). The horizontal sweep is
  160–180°; the vertical sweep follows the aspect ratio (square pixels), and the top
  edge is aimed 3–10° above the backboards. A roll of up to ±0.7° imitates
  stitching misalignment. All values come from the scene seed and are recorded in
  `worker-result.json`.

```sh
sports-dataset render-basketball examples/basketball-reco.json --output outputs/reco
```

Use a wide resolution such as 2560×720 (about 3.5:1); aspect ratios that would
need more than 110° vertical sweep are rejected. This is a geometric approximation,
not Reco's calibrated two-plane stitch: lens distortion residue, seams, exposure
differences between the halves and Reco's exact yaw/pitch extent are not modelled.
Matching a real calibration file is the next step. The ball-pixel check in
`worker-result.json` uses an independent equirectangular projection whose Cycles
convention (forward −Z, right +X, up +Y) was verified against rendered markers.



- Reco labels stay exactly `ball`, `player`, `referee`, `hoop`, in that order.
- All meshes belonging to one athlete share one stable instance ID. Court,
  seating, walls, supports, and shadows are background. The hoop instance covers
  backboard, rim, and net; its stanchion/base are background. This is the current
  generator annotation policy, not a claim about all human-labeled Reco datasets.
- Cycles writes RGB and **Object Index in the same render**. A 32-bit linear EXR
  carries raw IDs; Blender reads it without display transforms, checks integral
  0..255 values, flips bottom-left storage to top-left, and writes an 8-bit PGM.
  The host validates IDs and derives visible-pixel boxes and COCO RLE masks.
- RGB is antialiased and denoised; Object Index is not antialiased. Subpixel edges
  can differ by about a pixel. Motion blur and transparent backboards are disabled;
  the current opaque practice backboard avoids ambiguous transparent-object masks.
- `poses.json` holds OpenPose BODY_18 keypoints per person and frame with COCO
  visibility; see [pose crops](POSE_CROPS.md).
- Occluded/off-camera objects keep identities but produce no visible boxes.
  Exported COCO categories are one-based; native class IDs remain zero-based.
- Native/scene/export schemas remain unchanged. `worker-result.json` is a
  versioned internal diagnostic with cameras, lens, samples, device, Blender
  version, timing, limitations, and an independently projected ball center.
- Compatibility baseline: Reco Trainer `1fec3e34997a92e627600f68134a1b0c7eb19833`.
  The importer still rebuilds train/test splits. Use the untouched export for
  evaluation requiring the generator's scene-separated splits.

The local integration test can be run with `BLENDER_EXECUTABLE=/path/to/blender
python -m unittest discover -s tests -p test_blender_renderer.py -v` after installing
the project. It checks a real render, visible classes, COCO export, and mask/ball
alignment against an independent camera projection. Normal CI does not install
Blender; adapter tests and Reco integration run there.

## Realism and reproducibility limits

These are **3D prototype scenes**, not a photorealistic or validated training
corpus. The athletes have procedural anatomy, basic clothing and faces, and a simple
procedural motion model rather than motion capture; the torso does not bend or twist.
There is no cloth simulation, accurate ball-player contact dynamics or quality metric
against real basketball footage. Situations are plausible snapshots,
not simulated plays. Do not infer tracking or action recognition validity from the
exported instance identities.

Seeds determine geometry, colors, camera, and light choices. Pixel reproducibility
also requires the same worker hash, Blender version, device, sample count, and
render environment. GPU/CPU or Blender version changes can change RGB/noise and
subpixel mask boundaries. Timings/logs are diagnostic and are not deterministic.

Worker source is GPL-3.0-or-later; the host remains Apache-2.0. Blender is invoked
with factory settings and auto-execution disabled, accepts only the project's
worker script, and uses no external assets. A user-specified executable is a local
trust decision; recipes cannot provide an executable or executable code. Every
run records the worker source hash and engine version in provenance.

Official references: [Blender licensing](https://www.blender.org/about/license/),
[render passes](https://docs.blender.org/manual/en/latest/render/layers/passes.html),
[Blender 5 render-pass changes](https://developer.blender.org/docs/release_notes/5.0/rendering/),
and [FIBA's 2024 equipment dimensions](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf)
for the nominal backboard and ring geometry. The prototype is not a certified equipment simulation.
