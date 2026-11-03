# Local Futsal hall batch

The experimental photographic adapter prepares **10,009** labelled images in the
approved 3200 × 900 hall. Everything runs locally with the already installed Qwen
and BiRefNet weights. No new downloads or training runs are started.

## Preparation and start

Run with the Python environment used for the basketball hall runner:

```sh
PYTHONPATH=src python scripts/prepare_futsal_hall.py '/path/to/Futsal-Halle-10009' \
  --base-run '/path/to/Nachtlauf-Halle' \
  --referee-reference '/path/to/Foto 1.jpg' \
  --ball-reference '/path/to/Foto 2.jpg' --defer-background
```

On macOS the generated `10.009 Futsalbilder starten.command` keeps the computer
awake and denies network access. It waits for active hall batches in sibling
folders before starting model work; waiting does not consume the generation
budget. The deferred background cleanup finishes on the first start. This
coordinates the existing hall runner locks, not arbitrary external training jobs. `Futsal-Pilot starten.command` generates ten
person assets and twelve preview images first. Review those before starting the
full batch. Interrupt with Ctrl+C; the same command resumes verified completed
assets and scene groups. A run lock prevents duplicate starts in one output folder.
Do not run another Qwen or RF-DETR job at the same time.

The local **Referenzen.html** description embeds both supplied referee and ball
charts. The reference images are copied into the prepared run, not Git history:
redistribution rights have not been established. Chart league/brand/season claims
are illustrative and are not treated as a verified official equipment database.

## Referee and ball references

Referees wear short-sleeved shirts in turquoise, fluorescent yellow, black, light
grey, orange and red, with black shorts, knee socks and flat indoor shoes. Prompts
request plain kits without generated text, badges or logos. They use the `referee`
class; basketball referee trousers and striped shirts are not used.

Ten illustrated ball faces are extracted from the supplied ball chart: white with
purple/yellow, blue/lime, black/lime, red/yellow, blue/red and other contrasting
panels. All have the same `ball` label. They are rotated and scaled to an approximate
20 cm diameter; the chart's typography, shadows and captions are excluded. These
are illustrated 2D ball faces, not measured physical or licensed brand models.

Field players wear short-sleeved red/yellow football shirts, shorts, long socks
and indoor shoes. Goalkeepers wear contrasting green/blue long-sleeved kits and
have their own class. Only adults are requested. Skin tone, stature, sex presentation
and pose are prompt attributes and require visual review, not demographic labels.

## Scenes, labels and counts

Reco Trainer's exact Futsal class order is:

| Native class | COCO category | Label |
| --- | --- | --- |
| 0 | 1 | ball |
| 1 | 2 | player |
| 2 | 3 | goalkeeper |
| 3 | 4 | referee |
| 4 | 5 | goal |

Full scenes contain four field players and one goalkeeper per team, plus one or
two referees. Pilot scenes use a smaller roster. Two original procedural 3 × 2 m
standing goals ([goal aperture dimensions, FA/FIFA Law 1](https://www.thefa.com/football-rules-governance/lawsandrules/laws/futsal/law-1---the-pitch)) with red/white posts and see-through net meshes are rendered locally
in Blender and placed at opposite ends. Folded wall goals are removed in local
masked background preparation. Existing basketball boards remain gym background;
they do not receive a Futsal class. Post/net pixels participate in depth ordering,
so players and balls can occlude goals and be occluded by their mesh.

The 10,009-image plan allocates individual frames before pairing compatible views:
4,004 men's, 4,003 women's, 2,002 mixed; 7,006 single-ball, 2,002 multi-ball (2–4),
and 1,001 without a visible ball. Train/validation/test sizes are 8,007/1,001/1,001.
Related views stay in one split; human asset identities are disjoint between splits.
The odd total is produced exactly, without silently rounding up or dropping a frame.

Full, half, medium and tight crops, depth-based sizing, mild hall-matched colour,
soft contact shadows, partial human overlaps and targeted partial ball occlusion
provide combinations. Boxes, areas and RLE masks derive from the **final visible
instance pixels after clipping and cropping**. Every exported RLE, box, area and
image checksum is checked. Use `dataset/coco` for RF-DETR in a Futsal Reco project.
Reco currently rebuilds imported splits; preserve this original grouped export.

## Time and quality limits

The default is 14 hours for person generation and scene composition, reserving an
additional hour of the requested 15-hour window for export and validation. Runtime
is measured on the user's machine; 10,009 images within 15 hours is a target, not
a guarantee. The run stops as incomplete at its generation deadline and can resume.
It only reports completion once the exact export has passed verification.

These are 2.5D composites in one hall, not a physical match simulation or 10,009
independently generated photographs. 192 source people are planned (144 field
players, 24 goalkeepers, 24 referees), with new outfits and football poses. Image
scale and ground plane are visually calibrated, not measured metric ground truth.
Matte/framing checks cannot prove anatomical realism or that no generated extra
object remains; inspect contact sheets and pilot images. Keep real held-out Futsal
footage to measure RF-DETR training benefit. Original code is Apache-2.0, the
isolated Blender worker is GPL-3.0-or-later, approved Qwen weights Apache-2.0 and
BiRefNet MIT. User photographs/charts remain local and retain their own rights.

## Format versions

Basketball hall config/library/export 1.0 are preserved. Futsal uses hall config,
library and export manifest 2.0, adding the sport, goalkeeper assets, goal manifest
and odd per-group frame counts. Native scenes continue to use scene schema 2.0.
There is no in-place migration of an active or frozen basketball run: prepare a
new folder with its own cache, engine snapshot and checksums.
