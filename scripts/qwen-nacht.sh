#!/bin/bash
# Overnight Qwen batch: render broadcast scenes, keep only whole, isolated people and repaint
# as many as fit into the time budget. Safe to run again: every step resumes.
#
#   scripts/qwen-nacht.sh [OUTPUT] [HOURS] [SCENES]
#
# Defaults: ../qwen-nacht, 9 hours, 60 scenes, 512 px, inpaint strength 0.75 (env STRENGTH).
# An existing OUTPUT reuses its render and crops. Results are pasted back into a copy of the
# render with composite-crops; OUTPUT/index-inpaint-<strength>.html shows, per person, the
# render next to the repainted scene at source resolution. Runs under caffeinate so the Mac stays awake.
set -euo pipefail

OUT="${1:-../qwen-nacht}"
HOURS="${2:-9}"
SCENES="${3:-60}"
RESOLUTION="${RESOLUTION:-512}"
STRENGTH="${STRENGTH:-0.75}"
GENERATED="generated-inpaint-$STRENGTH"
REPAINTED="repainted-inpaint-$STRENGTH"

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
if [ -z "${QWEN_NACHT_AWAKE:-}" ] && command -v caffeinate >/dev/null; then
  QWEN_NACHT_AWAKE=1 exec caffeinate -ims "$SELF" "$OUT" "$HOURS" "$SCENES"
fi
cd "$(dirname "$SELF")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
exec > >(tee -a "$OUT/nacht.log") 2>&1

START=$(date +%s)
BUDGET=$(python -c "print(int(float('$HOURS') * 3600))")
echo "== $(date '+%F %T') Start: $SCENES Szenen, $HOURS h, ${RESOLUTION} px -> $OUT"

if [ ! -f "$OUT/recipe.json" ]; then
  cat > "$OUT/recipe.json" <<EOF
{"schema_version": "1.1", "name": "qwen-nacht", "sport": "basketball", "seed": 2026,
 "scenes": $SCENES, "frames_per_scene": 1, "resolution": {"width": 1920, "height": 1080},
 "camera": "broadcast", "annotations": ["bounding_boxes", "instance_masks"]}
EOF
fi

if [ ! -f "$OUT/render/run.json" ]; then
  echo "== $(date '+%T') Szenen rendern (Blender) …"
  sports-dataset render-basketball "$OUT/recipe.json" --output "$OUT/render"
fi

if [ ! -f "$OUT/crops/crops.json" ]; then
  echo "== $(date '+%T') Ganze, frei stehende Personen ausschneiden …"
  sports-dataset export-crops "$OUT/render" --output "$OUT/crops" --size "$RESOLUTION" \
    --min-height 100 --isolated
fi

LEFT=$(( (BUDGET - ($(date +%s) - START)) / 60 ))
if [ "$LEFT" -lt 5 ]; then
  echo "== Keine Zeit mehr für Qwen übrig."
else
  echo "== $(date '+%T') Qwen: höchstens $LEFT Minuten …"
  # A crash (e.g. memory) must not stop the night: retry while time is left.
  for attempt in 1 2 3; do
    LEFT=$(( (BUDGET - ($(date +%s) - START)) / 60 ))
    [ "$LEFT" -lt 5 ] && break
    if sports-dataset generate-crops "$OUT/crops" --output "$OUT/$GENERATED" \
      --mode inpaint --strength "$STRENGTH" --resolution "$RESOLUTION" --time-limit "$LEFT"; then
      break
    fi
    echo "== Qwen abgebrochen (Versuch $attempt), neuer Start …"
  done
fi

echo "== $(date '+%T') Ergebnisse in die Szenen einsetzen …"
rm -rf "$OUT/$REPAINTED"
sports-dataset composite-crops "$OUT/render" --crops "$OUT/crops" \
  --generated "$OUT/$GENERATED" --output "$OUT/$REPAINTED" \
  --generator "Qwen-Image-Edit-2511 inpaint, strength $STRENGTH"

python - "$OUT" "$REPAINTED" <<'PY'
import html, json, sys
from pathlib import Path
from PIL import Image

out, repainted = Path(sys.argv[1]), sys.argv[2]
report = json.loads((out / repainted / "composite.json").read_text())
status = {item["id"]: item for item in report["items"]}
crops = json.loads((out / "crops/crops.json").read_text())["crops"]
view = out / f"{repainted}-ansicht"
view.mkdir(exist_ok=True)
rows, counts = [], {}
for crop in crops:
    item = status.get(crop["id"], {})
    state = item.get("status", "missing")
    counts[state] = counts.get(state, 0) + 1
    if state == "missing":
        continue
    box = tuple(crop["crop_box_xyxy"])
    pair = Image.new("RGB", (512, 256))
    for column, root in enumerate((out / "render", out / repainted)):
        with Image.open(root / crop["source_rgb"]) as frame:
            pair.paste(frame.crop(box).resize((256, 256), Image.Resampling.LANCZOS), (column * 256, 0))
    pair.save(view / f"{crop['id']}.jpg", quality=90)
    reason = html.escape(str(item.get("reason") or ""))
    rows.append(
        f"<figure class='{state}'><img src='{view.name}/{crop['id']}.jpg'>"
        f"<figcaption>{html.escape(crop['id'])} · {html.escape(crop.get('pose_family') or '?')}"
        f" · {state} {reason}</figcaption></figure>"
    )
summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
page = out / f"index-{repainted.removeprefix('repainted-')}.html"
page.write_text(
    "<!doctype html><meta charset='utf-8'><title>Qwen Inpaint</title><style>"
    "body{font:14px system-ui;background:#111;color:#ddd;margin:16px}"
    "figure{display:inline-block;margin:8px}img{width:512px;height:256px}"
    "figcaption{font-size:12px}.rejected img{outline:3px solid #c33}</style>"
    f"<h1>Render | eingesetzt ({summary})</h1>{''.join(rows)}"
)
print(f"Übersicht: {page} ({summary})")
PY
echo "== $(date '+%F %T') Fertig."
