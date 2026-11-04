# Economical training diversity for photographic sports datasets

This is a proposed experiment, not an implemented new training preset or an
accuracy claim. Keep generation offline and free. Do not alter a completed dataset
or frozen run in place.

## What already exists

The photographic hall compositor already applies mild per-person softening
(Gaussian radius 0.2–0.45 source-image pixels), warm colour matching and contrast
reduction. Each final view receives brightness factors 0.94–1.06 and colour factors
0.95–1.04, plus varying crops, apparent scale, placements and occlusions.

RF-DETR documents default scale/crop jitter and horizontal flipping. Its optional
custom augmentation path supports additional pixel and geometric transforms.
The inspected Reco Trainer worker does not pass an explicit `aug_config` to
`model.train`; therefore do not claim that blur or custom colour jitter is enabled
there. Installed RF-DETR versions/dependencies must be checked before integration.
Sources: [RF-DETR documentation](https://rfdetr.roboflow.com/latest/learn/train/augmentations/),
[Reco worker](https://github.com/wendibus/reco-trainer/blob/main/cross-platform/ml_worker.py)
(inspected content SHA `ded838e89f7b18ef65948d04f24a6d87e5c6d505`).

## Suggested conservative experiment

These are starting hypotheses, adjusted to real target footage, not measured optima:

| Effect | Initial range / frequency | Main constraint |
| --- | --- | --- |
| Brightness and contrast | factors 0.85–1.15 on about 30% | avoid clipped highlights/shadows |
| Mild Gaussian blur | 3×3 kernel on 10–15% at around 1600 px width | retain recognisable small balls |
| JPEG degradation | quality 65–95 on 10–20% | match actual video compression |
| Reduced resolution, then restore original dimensions | occasional, mild | do not erase tiny targets |
| White-balance shift and sensor noise | infrequent, weak | preserve plausible colours |
| Short directional motion blur | a separate later experiment | match camera motion and inspect label support |

Keep most images close to natural quality. Initially choose among degradation
families rather than stacking every effect. Match severity to the final training
resolution and actual footage. Softening does not correct bad anatomy, floating
feet, inaccurate masks or a physically implausible shadow. RGB intensity changes
simulate exposure/white balance, not physically new light sources or cast shadows.

Prefer random **training-time** variants after splitting: they save storage and
provide changing samples across epochs without another Qwen generation. They do
not create new independent scenes. More stored variants also increase epoch time;
compare experiments at the same training-step budget. CPU transforms still have a
cost, so measure data loading before adding a large stack.

## Cheap changes with more scene value

Reuse existing person cutouts in new formations, overlaps, scale ranges and crops;
rerender visible IDs from the same layers. Target small/distant balls, partial
occlusion, frame edges, uncommon referee clothing and crowded interactions.
The current compositor already varies these factors; future runs can emphasize
failure cases found in real footage rather than merely making more copies.

Add genuinely empty or target-free camera views as negatives only when **all**
visible target classes are absent. A basketball hall with visible hoops is not an
empty-label negative under the current class policy. Add hard examples of false
ball detections (hands, shoes, floor markings), with all real target objects labelled.

A few additional real camera views or halls can add background diversity beyond
one hall; each needs proper scale/ground calibration. Reuse the cutouts rather than
regenerating every person. Prioritize correcting generator/matte mistakes and adding
real missed detections before spending compute on more nearly identical scenes.

## Annotation and split rules

Mild colour, noise and blur leave geometric coordinates unchanged; strong blur
that makes a tiny object unrecognisable should be rejected or reduced. Detection
labels refer to the underlying object support, not the blur halo. Such degraded
RGB variants are not exact visible-pixel segmentation ground truth for the halo.
For segmentation training, decide/document the target semantics explicitly.

Crops, flips, resizing and rotation require identical geometric transforms for
RGB, instance masks and boxes, with categorical masks resized by nearest neighbour.
Derive tight boxes/areas and drop completely hidden targets from final masks.
Adding a new occluder changes visibility and requires recomputing masks/boxes.
Do not paste random opaque rectangles over RGB while leaving visible-pixel labels
unchanged. Do not use generative whole-scene repainting with old annotations.

Assign original scenes and related identity groups to splits **before** variants.
All variants of a source stay in its split; only training receives random training
augmentation. Keep fixed clean real validation/test videos grouped by recording,
not neighbouring frames randomly split. A separate fixed degraded test can measure
robustness. Reco currently rebuilds splits during import: importing an offline
expanded folder can put sister variants into training and validation. Training-time
augmentation after Reco's split avoids this extra source of leakage. Cross-recording
and scene grouping still need explicit evaluation control.

## Measure improvement

Compare a baseline and the conservative preset using identical real held-out
recordings, seeds where practical and equal training budgets. Check per-class
precision/recall, small/occluded-ball recall and false positives, not just aggregate
synthetic validation scores. Add effects one family at a time. Keep a change only
when useful on the target footage; no universal multiplier or improvement is promised.
