"""Command-line entry point for provider-aware data acquisition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .catalog import DatasetEntry, get_dataset, load_catalog
from .validation import validate_dataset


def _print_table(entries: dict[str, DatasetEntry]) -> None:
    rows = [(key, value.status, value.access, value.title) for key, value in entries.items()]
    widths = [max(len(str(row[i])) for row in (["KEY", "STATUS", "ACCESS", "TITLE"], *rows)) for i in range(4)]
    header = "  ".join(name.ljust(widths[i]) for i, name in enumerate(("KEY", "STATUS", "ACCESS", "TITLE")))
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _manual_message(entry: DatasetEntry) -> str:
    url = entry.access_url or entry.landing_url or "No public URL"
    return (
        f"{entry.title} requires the provider's manual access route.\n"
        f"Access page: {url}\n"
        f"Access class: {entry.access}\n"
        f"Terms/license: {entry.license}\n"
        "Rooted4D will not automate provider registration or acceptance."
    )


def _command(entry: DatasetEntry, destination: Path) -> str:
    if entry.command:
        return entry.command.format(dest=str(destination))
    return _manual_message(entry)


def _fetch_huggingface(entry: DatasetEntry, destination: Path) -> None:
    if not entry.repo_id:
        raise RuntimeError(f"{entry.key} has no Hugging Face repository id")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the downloader with: python -m pip install -e '.[download]'"
        ) from exc
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=entry.repo_id,
        repo_type="dataset",
        local_dir=str(destination),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rooted4d-data",
        description="Audit, acquire, and validate Rooted4D data without bypassing provider terms.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("list", help="List cataloged datasets and current access status")

    info = sub.add_parser("info", help="Show one complete catalog record")
    info.add_argument("dataset")

    command = sub.add_parser("command", help="Print an official download command or access URL")
    command.add_argument("dataset")
    command.add_argument("--dest", type=Path, default=Path("data/raw"))

    fetch = sub.add_parser("fetch", help="Execute a supported provider download after explicit acknowledgement")
    fetch.add_argument("dataset")
    fetch.add_argument("--dest", type=Path, required=True)
    fetch.add_argument("--acknowledge-provider-terms", action="store_true")

    validate = sub.add_parser("validate", help="Check a local dataset's expected file layout")
    validate.add_argument("dataset")
    validate.add_argument("--root", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "list":
        _print_table(load_catalog())
        return 0

    try:
        entry = get_dataset(args.dataset)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.action == "info":
        print(yaml.safe_dump(entry.as_dict(), sort_keys=False, allow_unicode=True))
        return 0
    if args.action == "command":
        print(_command(entry, args.dest))
        return 0
    if args.action == "validate":
        report = validate_dataset(entry, args.root)
        print(json.dumps(report.as_dict(), indent=2))
        return 0 if report.passed else 1
    if args.action == "fetch":
        if not args.acknowledge_provider_terms:
            print(
                "Refusing to download without --acknowledge-provider-terms. "
                "Review and accept the provider conditions yourself first.",
                file=sys.stderr,
            )
            return 2
        if entry.fetch_kind != "huggingface_dataset":
            print(_manual_message(entry), file=sys.stderr)
            return 2
        try:
            _fetch_huggingface(entry, args.dest)
        except Exception as exc:  # provider/authentication errors need user action
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1
        print(f"Downloaded {entry.key} to {args.dest.resolve()}")
        return 0

    raise AssertionError(f"Unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())

