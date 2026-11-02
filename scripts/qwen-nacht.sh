#!/bin/bash
# Overnight Qwen batch: render broadcast scenes, keep only whole, isolated people and repaint
# as many as fit into the time budget. Safe to run again: every step resumes.
#
#   scripts/qwen-nacht.sh [OUTPUT] [HOURS] [SCENES]
#
# Defaults: ../qwen-nacht, 9 hours, 60 scenes, 512 px. At the end OUTPUT/index.html shows
# every rendered person next to its Qwen result. Runs under caffeinate so the Mac stays awake.
set -euo pipefail

OUT="${1:-../qwen-nacht}"
HOURS="${2:-9}"
SCENES="${3:-60}"
RESOLUTION="${RESOLUTION:-512}"

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
    if sports-dataset generate-crops "$OUT/crops" --output "$OUT/generated" \
      --resolution "$RESOLUTION" --time-limit "$LEFT"; then
      break
    fi
    echo "== Qwen abgebrochen (Versuch $attempt), neuer Start …"
  done
fi

python - "$OUT" <<'PY'
import html, json, sys
from pathlib import Path

out = Path(sys.argv[1])
crops = json.loads((out / "crops/crops.json").read_text())["crops"]
seconds = {}
records = out / "generated/generation.jsonl"
if records.is_file():
    for line in records.read_text().splitlines():
        entry = json.loads(line)
        seconds[entry["id"]] = entry["seconds"]
done = [c for c in crops if (out / "generated" / f"{c['id']}.png").is_file()]
rows = "".join(
    f"<figure><img src='crops/{c['files']['rgb']}'><img src='generated/{c['id']}.png'>"
    f"<figcaption>{html.escape(c['id'])} · {html.escape(c.get('pose_family') or '?')} · "
    f"{seconds.get(c['id'], '?')} s</figcaption></figure>"
    for c in done
)
mean = sum(seconds.values()) / len(seconds) if seconds else 0
(out / "index.html").write_text(
    "<!doctype html><meta charset='utf-8'><title>Qwen Nacht</title><style>"
    "body{font:14px system-ui;background:#111;color:#ddd;margin:16px}"
    "figure{display:inline-block;margin:8px}img{width:256px;height:256px}"
    "figcaption{font-size:12px}</style>"
    f"<h1>Qwen Nacht: {len(done)} von {len(crops)} Personen, Ø {mean:.0f} s</h1>{rows}"
)
print(f"Übersicht: {out / 'index.html'} ({len(done)} von {len(crops)}, Ø {mean:.0f} s)")
PY
echo "== $(date '+%F %T') Fertig."
