"""Command-line interface for the semi-synthetic proof-of-concept dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rooted4d.data.semisynthetic import (
    audit_semisynthetic_dataset,
    compare_semisynthetic_datasets,
    dataset_fingerprint,
)

from .config import load_config
from .dataset_builder import build_semisynthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rooted4d-semisynthetic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="build and verify a new dataset")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="audit manifests, arrays, and split isolation")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--skip-checksums", action="store_true")
    audit.add_argument("--skip-arrays", action="store_true")

    fingerprint = subparsers.add_parser("fingerprint", help="print a deterministic tree hash")
    fingerprint.add_argument("--root", type=Path, required=True)

    compare = subparsers.add_parser("compare", help="compare two independently generated trees")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        config = load_config(args.config)
        root = build_semisynthetic_dataset(config, args.output)
        report = audit_semisynthetic_dataset(root, verify_checksums=True, verify_arrays=True)
        summary = {
            "root": str(root),
            "fingerprint": dataset_fingerprint(root),
            "audit": report.as_dict(),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report.passed else 1
    if args.command == "audit":
        report = audit_semisynthetic_dataset(
            args.root,
            verify_checksums=not args.skip_checksums,
            verify_arrays=not args.skip_arrays,
        )
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0 if report.passed else 1
    if args.command == "fingerprint":
        print(json.dumps({"fingerprint": dataset_fingerprint(args.root)}, indent=2))
        return 0
    same = compare_semisynthetic_datasets(args.left, args.right)
    print(json.dumps({"identical": same}, indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
