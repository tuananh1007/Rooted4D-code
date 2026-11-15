#!/usr/bin/env python3
"""Clone audited public baselines and record exact revisions.

This script never vendors third-party source into the Rooted4D package. Clones
belong under an ignored ``externals`` directory and retain provider licenses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "baselines" / "catalog.yaml"


def load_catalog() -> dict[str, dict[str, str]]:
    raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    return raw["baselines"]


def print_catalog(catalog: dict[str, dict[str, str]]) -> None:
    for key, entry in catalog.items():
        print(
            f"{key:12}  {entry['status']:44}  {entry['license']:40}  {entry['url']}"
        )


def run(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def clone_baseline(key: str, destination: Path, catalog: dict[str, dict[str, str]]) -> None:
    if key not in catalog:
        raise KeyError(f"Unknown baseline '{key}'")
    entry = catalog[key]
    if entry["status"] != "available":
        raise RuntimeError(f"{key} is not runnable: {entry['status']}")
    target = destination / key
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", entry["url"], str(target)])
    revision = run(["git", "rev-parse", "HEAD"], cwd=target)

    lock_path = destination / "LOCKED_REVISIONS.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    lock[key] = {
        "title": entry["title"],
        "url": entry["url"],
        "revision": revision,
        "license_at_audit": entry["license"],
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Cloned {key} at {revision}")
    print(f"Review provider terms before execution or modification: {entry['license']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--clone", metavar="BASELINE")
    parser.add_argument("--dest", type=Path, default=ROOT / "externals")
    args = parser.parse_args(argv)
    catalog = load_catalog()
    if args.list:
        print_catalog(catalog)
        return 0
    try:
        clone_baseline(args.clone, args.dest, catalog)
    except Exception as exc:
        print(f"Baseline bootstrap failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

