#!/usr/bin/env python3
"""Fail when a patient identifier appears in more than one dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rooted4d.data import find_split_leakage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("splits", type=Path)
    args = parser.parse_args()
    split_mapping = json.loads(args.splits.read_text(encoding="utf-8"))
    leakage = find_split_leakage(split_mapping)
    if leakage:
        print(json.dumps({"passed": False, "leakage": leakage}, indent=2))
        return 1
    print(json.dumps({"passed": True, "splits": sorted(split_mapping)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

