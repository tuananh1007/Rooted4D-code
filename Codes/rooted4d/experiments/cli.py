"""Command line entry point for the Rooted4D experiment campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .campaign import EXPERIMENTS, run_all


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalars and arrays, which appear in verdict payloads."""

    def default(self, o):  # noqa: D102 - json.JSONEncoder hook
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _summarize(payload: dict) -> str:
    lines = ["", "=" * 78, "ROOTED4D EXPERIMENT CAMPAIGN SUMMARY", "=" * 78]
    environment = payload["environment"]
    lines.append(
        f"rooted4d {environment['rooted4d_version']} | python {environment['python']} | "
        f"numpy {environment['numpy']} | {environment['machine']}"
    )
    for name, result in payload["results"].items():
        lines.append("")
        lines.append(f"[{name}] {result['experiment']}  ({result['runtime_seconds']}s)")
        lines.append(f"  claim: {result['claim']}")
        for key, value in result["verdict"].items():
            lines.append(f"    {key}: {json.dumps(value, cls=_NumpyEncoder)}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Rooted4D experiment campaign")
    parser.add_argument(
        "--output", required=True, help="directory for results.json and summary.txt"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(EXPERIMENTS),
        help="run a subset of experiments (default: all)",
    )
    arguments = parser.parse_args(argv)

    def progress(name: str, index: int, total: int, elapsed: float) -> None:
        print(f"  [{index}/{total}] {name} finished in {elapsed:.1f}s", flush=True)

    print("running experiments:", flush=True)
    payload = run_all(
        tuple(arguments.only) if arguments.only else None, progress=progress
    )
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, cls=_NumpyEncoder) + "\n", encoding="utf-8"
    )
    summary = _summarize(payload)
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
