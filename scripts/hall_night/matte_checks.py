"""Run in the existing matte environment: remove stray foreground, reject ball-like extras."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--ids", nargs="+", required=True)
a = p.parse_args()
records = {}
for identity in a.ids:
    path = a.root / "photos" / (identity + ".png")
    mask_path = a.root / "masks" / (identity + ".png")
    rgb = cv2.imread(str(path))
    mask = cv2.imread(str(mask_path), 0)
    binary = np.uint8(mask >= 128)
    # Connect sub-pixel matte gaps, preserving original support after component selection.
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
    clean = np.uint8((labels == main) & (binary > 0)) * 255 if main else np.zeros_like(mask)
    roi = cv2.boundingRect(clean)
    x, y, w, h = roi
    suspicious = []
    if h:
        gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), (5, 5), 1)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1.2, 30, param1=90, param2=31, minRadius=12, maxRadius=65
        )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        if circles is not None:
            yy, xx = np.indices(mask.shape)
            for cx, cy, r in circles[0]:
                if cy < y + 0.25 * h:
                    continue  # Human heads are not extra objects.
                if not 0.05 * h <= r <= 0.10 * h:
                    continue
                distance = (xx - cx) ** 2 + (yy - cy) ** 2
                ring = (distance > (1.08 * r) ** 2) & (distance < (1.23 * r) ** 2)
                background_fraction = float(((clean == 0) & ring).sum() / max(1, int(ring.sum())))
                if background_fraction < 0.65:
                    continue
                region = (distance < (0.8 * r) ** 2) & (clean > 0)
                area = max(1, int(region.sum()))
                brown_orange = (
                    (hsv[:, :, 0] > 3)
                    & (hsv[:, :, 0] < 26)
                    & (hsv[:, :, 1] > 125)
                    & (hsv[:, :, 2] > 65)
                )
                fraction = float((brown_orange & region).sum() / area)
                if fraction > 0.68 and area > 0.9 * np.pi * (0.8 * r) ** 2:
                    suspicious.append(
                        [round(float(cx), 1), round(float(cy), 1), round(float(r), 1)]
                    )
    target = a.root / "masks" / (identity + ".clean.png")
    cv2.imwrite(str(target), clean)
    records[identity] = {
        "raw_mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "clean_mask_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "removed_stray_pixels": int(((binary > 0) & (clean == 0)).sum()),
        "possible_extra_balls": suspicious,
        "semantics": "Conservative quality rejection only; not automatic anatomy or demographic verification.",
    }
(a.root / "matte-checks.json").write_text(json.dumps(records, indent=2) + "\n")
print("Additional matte checks:", len(records), "photos", flush=True)
