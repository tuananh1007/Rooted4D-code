"""Validate a Rooted4D global identity split lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rooted4d.data import validate_split_lock_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    report = validate_split_lock_file(args.path)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
