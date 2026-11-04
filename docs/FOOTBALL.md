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
