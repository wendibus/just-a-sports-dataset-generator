# Balls and referee kits

The appearance catalogue
[`src/just_a_sports_dataset_generator/sports/equipment.json`](../src/just_a_sports_dataset_generator/sports/equipment.json)
([schema](../schemas/sport-equipment-v1.0.json)) describes match balls and referee kits for
basketball, football (soccer) and futsal, season 2025/26. Planners sample from it with
their own random stream; renderers only receive linear-RGB colours and a few shape flags.

| Sport | Planner and renderer use it | Status |
| --- | --- | --- |
| basketball | yes, layout schema 1.2 (`equipment` in `layout.json`) | implemented |
| football | catalogue only | the football renderer is not implemented yet |
| futsal | catalogue only | the futsal renderer is not implemented yet |

## Sources and rules for using them

The catalogue was compiled from eight datasheets supplied by the project owner (folder
`datasheets`, not part of the repository): referee uniforms for basketball, football and
futsal, and official match balls for all three sports. The sheets describe themselves as
*illustrative reconstructions, not official product photos*. Therefore:

- **Only looks are taken over**: colours, two-tone patterns, sleeves, trousers or shorts,
  socks. Brand names, league names and logos stay in `observed_in` as provenance and are
  **never rendered**: no trademarks in synthetic images, and a detector should not learn a
  particular logo as a cue for "ball" or "referee".
- **Regulation values come from the rules**, not from the sheets, where they could be
  checked. `verified: true` marks values compared with the primary rules text in this
  revision; `false` means "as in the datasheets, consistent with the rules, not checked
  against the document".
- **Weights** are the number of datasheet columns that show a look. They describe what
  the sheets show, not a measured share of real matches.

## Profiles

The material is meant to work internationally, so the default profile `international`
samples with the datasheet weights. The profile `germany` reweights the same looks for
German leagues and amateur play; select it with `"equipment_profile": "germany"` in a
recipe with schema 1.2.

| Sport | `germany` differs from `international` |
| --- | --- |
| Basketball | ball mostly orange/ivory (8 of 10.5), referee grey with black sleeves (9) or black with neon (1); no orange or striped kits |
| Football | ball mostly the pink/orange design (4); referee black (4), fluorescent yellow, fluorescent blue or red (1 each) |
| Futsal | ball mostly black/neon (4); referee black (4), grey or yellow (1 each) |

Weights of 0 switch a look off for the profile. Profiles never add looks, they only
reweight the catalogue.

## Decisions from the project owner (November 2026)

- Both profiles are kept; the material must work internationally.
- Basketball referees always wear **long black trousers**: it is the most important
  feature for recognising a referee. There is no shorts variant for basketball.
- People outside the court (bench, table officials, coaches) are excluded by the court
  boundary, in labels and in real footage. Synthetic material contains **no spectators**.

## Regulation balls

| Sport | Size | Circumference | Mass | Pressure | Bounce | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Basketball | 7 (men); 6 (women): 72.4–73.7 cm, 510–567 g | 74.9–78.0 cm | 567–650 g | — | from 1.80 m to 1.20–1.40 m | FIBA basketball equipment, via datasheets (`verified: false`) |
| Football | 5 | 68–70 cm | 410–450 g | 0.6–1.1 atm | — | [IFAB Laws of the Game, Law 2](https://www.theifab.com/laws/chapter/22/section/32) |
| Futsal | 4 | 62–64 cm | 400–440 g | 0.6–0.9 atm | from 2 m: 50–65 cm | [Futsal Laws of the Game, Law 2](https://www.thefa.com/football-rules-governance/lawsandrules/laws/futsal/law-2---the-ball) |

The basketball planner's ball radius (0.12 m, 75.4 cm circumference) lies inside the
regulation range.

## Ball designs

| Sport | Designs (weight) |
| --- | --- |
| Basketball | classic orange (3), orange with ivory panels (3), orange/black two-tone (3), dark brown (3) |
| Football | white base with colour graphics in ten palettes (blue/gold stars, red/green/blue, pink/orange, purple/magenta, cyan/blue/pink, pink/black, blue/yellow/green, red/blue/black, yellow/black) and one yellow high-visibility ball, 1 each |
| Futsal | white base with neon or saturated graphics: blue/neon (2), black/neon (2), red/blue (2), purple/yellow, red/orange, purple/neon, blue (1 each) |

Basketball two-tone balls are rendered with the panel colour on alternate octants between
the three seam circles: a simplification of the 12-panel layout that keeps colour shares and
contrast.

## Referee kits

**Basketball** (rules: grey shirt, long black trousers, black socks and shoes; the shirt
must differ from both team colours):

| Kit | Weight | Seen in |
| --- | --- | --- |
| grey shirt, black sleeves | 7 | NBA, FIBA, DBB, ACB, FIP, NBB, B.League |
| orange shirt, black sleeves | 1 | EuroLeague |
| black shirt with neon accents | 1 | FFBB |
| black/white vertical stripes | 1 | NCAA |

All basketball referees now wear **long black trousers**; until layout 1.1 they wore
shorts with bare legs. Long trousers change the referee silhouette and therefore the
`referee` masks: retrain, or at least re-evaluate, models trained on older renders.

**Football**: fluorescent blue, fluorescent yellow, black (2), turquoise, red, grey,
orange, yellow (2); short sleeves with two chest pockets, black shorts, socks in the shirt
colour or black.

**Futsal**: black (9), yellow (6), fluorescent blue (3), red (3), grey (3), turquoise,
orange; short sleeves with breast pockets, black shorts, black or matching socks.

**Contrast rule.** All three rule books require referee colours that differ from both
teams. `choose_referee_kit` only picks kits whose shirt is at least 0.25 (linear-RGB
distance) away from every team kit and falls back to all kits if none qualifies.

## Corrections to the datasheets

Checked while building the catalogue:

- **Futsal ball pressure**: the futsal ball sheets give 0.4–0.6 bar; the Futsal Laws of
  the Game specify 0.6–0.9 atm (600–900 g/cm²). The catalogue uses the rule.
- **Contradictory basketball balls**: one sheet shows the EuroLeague and French LNB balls
  as dark brown, the other as orange/black two-tone. Both looks are in the catalogue;
  neither is assigned to a league.
- **NCAA ball**: one sheet names a Nike ball, the other a Wilson ball. Irrelevant for
  rendering, since brands are not used.
- The outdoor ball labelled "Wilson DRI-FIT" mixes brands; it is not used.

## Adding a sport or a look

Add designs or kits in `equipment.json` (sRGB hex colours, a positive `weight`, an
identifier `id`); `parse` validates the file and the tests check it against the schema.
New entries change which equipment existing seeds get (the rest of the layout stays the
same, since equipment uses its own random stream); mention that in the change notes.
Changing the field set needs a new schema version.
