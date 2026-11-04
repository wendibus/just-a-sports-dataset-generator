# Outdoor football preparation

This experimental photographic adapter prepares a **neutral grass pitch**, original
projected goal assets, adult player profiles and exact COCO plans for sport `football`.
Preparation is CPU-only: it does not run Qwen, install weights or start RF-DETR.
The first photographic pilot and real-footage training benefit remain unverified.

The default inherited target is **10,009 images**, with the previously chosen
40/40/20 men's/women's/mixed and 70/20/10 single/multiple/no-visible-ball ratios.
This count and a 15-hour window are configurable preparation defaults, not a runtime
guarantee. Generation gets 14 hours by default, with export/verification allowed to
finish afterwards. 336 new person assets are planned: 288 field players, 24 keepers,
24 referees. Full scenes contain ten field players and one keeper per team and one
referee; the twelve-image pilot uses a reduced roster. Related views and person
identities remain grouped by split; preserve the original export because Reco
currently rebuilds imported splits.

## Local start

```sh
PYTHONPATH=src python scripts/prepare_football.py '/path/to/Fussball-Rasen-10009' \
  --base-run '/path/to/Nachtlauf-Halle' \
  --referee-reference '/path/to/referees.png' \
  --ball-reference '/path/to/balls.png'
```

Preparation copies installed-model paths and the existing pose references from the
base run, freezes current code/schemas, and creates a separate cache. It never
changes an existing basketball/Futsal engine, dataset or Reco training job.

After stopping other GPU work, open `Fussball-Pilot starten.command`, inspect the
contact sheets and all twelve labelled pilot images, then use
`10.009 Fussballbilder starten.command` if acceptable. The starter waits for active
sibling hall-run locks, but **does not detect arbitrary Reco training processes**.
It denies network access on macOS and keeps the computer awake. Ctrl+C stops only
this runner; verified assets/scenes can be resumed with the same starter. Model
execution starts only when the user launches the pilot/batch. No background
monitor or automatic handoff from the current basketball training is installed.

## Reference descriptions and illustrations

The prepared run's `Referenzen.html` embeds both user-provided charts. They remain
outside Git: redistribution rights are not established. League/brand/season
captions are illustrative user input, not verified official equipment declarations.

The ball chart contains ten faces: white with blue/gold star motifs, red/blue/green
panels, red/blue accents, magenta/black, cyan/magenta, black/red, blue/yellow and
other patterns, plus a yellow/dark design. All use class `ball`. The extraction
excludes captions and drop shadows. These are illustrated 2D ball faces, not
licensed brand models or full physical simulations. Their approximate radius is
0.110 m, consistent with the adult ball circumference of 68–70 cm in
[IFAB Law 2](https://www.theifab.com/laws/2025-26/the-ball/).

The referee chart shows fluorescent blue/yellow, black, turquoise, red, grey,
orange and yellow shirts, chest-pocket details, black shorts, matching or black
knee socks, and football boots. Prompts request generic plain kits without text,
logos or affiliation badges. Adult women/men and different skin tones are planned;
these are prompt traits requiring visual review, not detector categories.

Field players wear short-sleeved red/yellow football jerseys, shorts, long socks,
shin guards under socks and studded outdoor football boots. Keepers have green/blue
long-sleeved contrasting kits and gloves. Poses include running, sprinting, defending,
controlling, kicking and passing; inherited 3D references only guide the local edit.
No finished basketball/Futsal cutout is silently substituted for a football player.

## Pitch and calibration

The original CPU backplate is a procedural **prototype**, not a real venue photo.
It has seeded grass colour/texture, markings and empty park surroundings. It is
intentionally neutral; its realism must be judged in the photographic pilot.
The 105 × 68 m field, elevated camera (22 m), look-at point, focal length and image
size are recorded in `camera.json`. Player ground points and projected scale use
this pinhole model instead of the hall's calibration.

The two white goal apertures measure 7.32 × 2.44 m, following
[IFAB Law 1](https://www.theifab.com/laws/latest/the-field-of-play/). Goal depth is
an original 1.8 m modelling choice. Post/net alpha is projected with the same camera;
net holes retain the background. Scene depth lets players and balls occlude goals
and be partly hidden by the net. Person pose extents and contact shadows remain
approximate 2.5D composition, not physical or metric pose ground truth.

## Labels and validation

Canonical Reco order: `ball`, `player`, `goalkeeper`, `referee`, `goal` (native IDs
0..4, COCO category IDs 1..5). Boxes, areas and RLE derive from final visible IDs
after depth composition and cropping; complete occlusion produces no detection row.
Photographic anatomy, kit compliance and cutout quality still require visual review.
Do not interpret successful mask validation or Reco import as model accuracy.

Football camera and CPU-preparation manifests have their own version-1.0 schemas.
Football configuration/export contracts are version 3.0. Person libraries retain
2.0 because their role contract already includes goalkeepers. Basketball 1.0 and
Futsal 2.0 configs/exports and their frozen snapshots are not migrated in place.
The neutral pitch/goal code is original project code, Apache-2.0. Installed Qwen
weights remain Apache-2.0 and BiRefNet MIT; user charts retain their own rights.

## Natural venue upgrade (4 November 2026)

The user's prepared `Fussball-Rasen-10009` folder now supports a more natural,
locally rendered venue instead of the schematic oval trees and flat graphics.
The optional isolated Blender worker `football_venue.py` uses two CPU threads,
Cycles, a real CC0 environment photograph and camera-projected photographic turf
detail, with a small diffuse contribution for goal shading. It does not run Qwen
or use a GPU. This is a photo-textured scene composite, not a photograph of a real
105 × 68 m football venue. Surroundings are an image-based backdrop; their metric
scale is not ground truth. Human photographic generation/pilot remains separate.

The principal point moves from y=520 to y=660 to include the natural surroundings.
The placement model, frozen football engine and `camera.json` must change together.
The worker projection is numerically compared with the compositor at known world
points. Field dimensions and goal apertures are unchanged.

Goals and their categorical Object Index masks come from the same render. The host
removes goal support from the background and restores it through goal RGB/alpha
layers; exact pixel reconstruction is verified. Physical goal shadows remain in
the background, and the compositor skips its additional ellipse for those goals.
Antialiased edge coverage is approximated by the categorical ID support, as with
the existing Blender handoff. No old human masks or boxes are reused.

Only unstarted preparations may be upgraded. Existing fingerprints, person photos,
pilots, renders or datasets block replacement. `upgrade_football_venue.py` verifies
the old engine, stages and checks new assets, retains the previous venue/engine in
`Sicherung-vor-natuerlichem-Platz`, and restores the old configuration, camera,
engine and goal manifest if publication fails. Basketball, Futsal and Reco processes
are not modified.

Small source textures are obtained explicitly for this local edit and retained with
SHA-256 checksums in `venue-natural/assets/sources.json`; they are not bundled in Git:

- [Grass004, Lennart Demes / ambientCG](https://ambientcg.com/view?id=Grass004):
  colour/normal/roughness maps, CC0-1.0 ([license](https://docs.ambientcg.com/license/)).
- [Approaching Storm, Greg Zaal / Poly Haven](https://polyhaven.com/a/approaching_storm):
  2K lighting HDRI and a sharper tonemapped camera photograph, CC0-1.0.
  Powered by Poly Haven. The live API is only used during asset preparation;
  rendering and later generation are offline.

Reproduction after those local textures are present:

```sh
blender --background --factory-startup --disable-autoexec --threads 2 \
  --python src/just_a_sports_dataset_generator/renderers/blender_worker/football_venue.py \
  -- /path/to/venue-request.json
PYTHONPATH=src python scripts/upgrade_football_venue.py /path/to/unstarted-football-run \
  --assets /path/to/texture-folder --render /path/to/completed-render
```

The request contains `schema_version: "1.0"`, `assets`, `output`, `percentage: 100`,
`samples: 32` and `environment_rotation: 0.5`. Half-resolution previews are not
accepted for installation. RGB, raw goal IDs, camera checks, source provenance and
verification results are retained locally. The existing GPL-3.0-or-later worker
boundary applies to the new Blender worker; host code remains Apache-2.0.


## Venue quality decision: rejected

The user rejected the 4 November photo-projected landscape/field result as
unconvincing. It is **not an accepted training background**, despite passing the
geometric and mask checks. Those checks establish annotation consistency, not
photorealism. The local football preparation is marked `venue-rejected.json`;
launchers and preflight refuse generation while that marker exists. Basketball,
Futsal and Reco jobs are unaffected.

Replacement candidates are actual photographs of football venues, with their
original markings and goal perspective intact. The current local visual selection
contains an elevated full-field view (Wikideas1, CC0, UW-L soccer field 02) and a
ground-level amateur view (Dejan Krsmanovic, CC BY 2.0, Subsidiary Football Yard).
The user selected B (ground-level amateur view). The source photograph is now
staged with an estimated local scale and manually reviewed detection boxes;
measured calibration, fine goal masks and the photographic person pilot remain pending. Do not remove the rejection marker merely because an image was copied.
Select the visual basis, establish perspective/scale, trace every visible target
goal and review a labelled person composite before enabling a production run.

Sources: [UW-L photograph](https://commons.wikimedia.org/wiki/File:UW-L_soccer_field_02.jpg),
[amateur photograph](https://commons.wikimedia.org/wiki/File:Subsidiary_Football_Yard_(43019524354).jpg).


## Selected real photograph B: separate staging

`prepare_football_photo.py` stages the selected **real photograph**, without Blender,
Qwen, BiRefNet, downloads or any training-process control. It verifies the approved
original SHA-256 and dimensions, retains the original, crops only the upper sky
(native crop 0,1250,7670,3407), and resizes to 3200 × 900. It preserves the photographed
turf, buildings, markings, goal geometry and natural light. This is a ground-level
training yard, not a full-pitch broadcast view. Avoid claiming an 11-a-side full-field
simulation from this image.

```sh
PYTHONPATH=src python scripts/prepare_football_photo.py /path/to/Fussball-Amateur-B-10009 \
  --photo /path/to/B-Echter-Platz-Amateur.jpg
```

The new folder is independent of the rejected venue and all basketball/Futsal snapshots.
Publication is staged; existing output/staging folders are refused. The version-1.0
`football-photo-preparation` schema records image transforms, checksums, rights,
estimated placement scale, five manual goal boxes and unfinished work. This is a
preparation contract, **not** a hall-run config or a production export. It supplies
no launch command and keeps `venue-rejected.json` in place so the existing runner
cannot accidentally treat the new photo as an accepted, calibrated venue.

The near white goal, farther white goal, blue training goal and two red mini-goal regions (one behind the near net) all
have manual detection boxes in `goal-box-review.json`. COCO category order remains
`ball`, `player`, `goalkeeper`, `referee`, `goal`; category 5 is used for each goal.
Overlapping boxes are legitimate. This review fixture has no segmentation fields:
boxes are **not** net-thread or visible-pixel masks. Do not rasterize whole goal boxes
as claimed segmentation ground truth. Fine masks and photo-aware occlusion/export
integration remain required before enabling the production runner.

Scale is a local hypothesis: the near aperture is presumed 2.44 m high and the
camera 1.65 m high. These dimensions, level and lens distortion are not measured;
the source is panoramic and its EXIF focal length is insufficient for a metric
camera. Proposed foot points lie only in the grassy foreground, below the near
net, so future people/balls are in front of it. Estimated scale grows toward the
camera. Do not reuse the 22 m elevated prototype camera, person-view prompts,
23-person roster, goal masks or its pixel boxes. The first ten pilot requests use
four field players, one keeper and one referee, ground-level views, warm diffuse
light, matched contact shadows, closer crops and controlled overlap. They retain
the agreed 40/40/20 gender scene distribution and 70/20/10 ball-count buckets.
They are requests only; no person photos have been generated or accepted yet.

Photograph: **Subsidiary Football Yard**, Dejan Krsmanovic, **CC BY 2.0**.
[Source and attribution](https://commons.wikimedia.org/wiki/File:Subsidiary_Football_Yard_(43019524354).jpg),
[license](https://creativecommons.org/licenses/by/2.0/).
The local `ATTRIBUTION.md`, source link, author, license link and change notice
must accompany future derived composites/datasets. The project's Apache-2.0 code
license does not replace the photograph's license. The source photograph and
rendered preparation outputs are not committed to Git.
