#!/usr/bin/env python3
"""Stage selected real football photo B offline, without starting any model."""

from __future__ import annotations

import argparse
from pathlib import Path

from hall_night.football_photo import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--photo", required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.photo, args.output) / "index.html")


if __name__ == "__main__":
    main()
