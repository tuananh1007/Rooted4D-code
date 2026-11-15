#!/usr/bin/env python3
"""Estimate segmentation-error correlation length from paired delineations.

This is the measurement the ActivePlaneTwin direction depends on. Point it at
pairs of delineations of the same slice, two observers or an expert against an
automated segmentation, and it reports how many independent measurements a
contour actually carries.

Input is a directory of ``.npz`` files, each containing two 2D boolean arrays
``a`` and ``b`` for the same slice, plus optional ``spacing_mm``. Keeping the
interface at the array level avoids a hard dependency on a NIfTI or DICOM reader
and works with whatever converter the data happens to need.

    python scripts/measure_contour_correlation.py pairs/ --label "ACDC LV cavity"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rooted4d.active.contour_noise import (  # noqa: E402
    contour_sample_spacing_mm,
    estimate_correlation_length,
    mask_boundary_contour,
    residuals_from_mask_pairs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory of .npz files with masks 'a' and 'b'")
    parser.add_argument("--samples", type=int, default=128, help="contour resampling count")
    parser.add_argument("--label", default="unnamed", help="what these delineations are")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    arguments = parser.parse_args(argv)

    files = sorted(Path(arguments.directory).glob("*.npz"))
    if not files:
        print(f"no .npz pairs found in {arguments.directory}", file=sys.stderr)
        return 1

    pairs = []
    spacings = []
    for path in files:
        payload = np.load(path)
        if "a" not in payload or "b" not in payload:
            print(f"skipping {path.name}: needs arrays 'a' and 'b'", file=sys.stderr)
            continue
        pairs.append((payload["a"], payload["b"]))
        spacings.append(
            tuple(payload["spacing_mm"]) if "spacing_mm" in payload else (1.0, 1.0)
        )
    if not pairs:
        print("no usable pairs", file=sys.stderr)
        return 1

    spacing = spacings[0]
    residuals = residuals_from_mask_pairs(
        pairs, spacing_mm=spacing, n_samples=arguments.samples
    )
    reference = mask_boundary_contour(pairs[0][0], spacing_mm=spacing)
    sample_spacing = contour_sample_spacing_mm(reference, arguments.samples)
    estimate = estimate_correlation_length(
        residuals, sample_spacing_mm=sample_spacing
    )

    report = {"label": arguments.label, "n_pairs": len(pairs), **estimate.as_dict()}
    print(json.dumps(report, indent=2))
    print(
        f"\nA contour of {estimate.n_samples} samples carries about "
        f"{estimate.effective_independent_samples:.1f} independent measurements.",
        file=sys.stderr,
    )
    print(
        "Below roughly 10, several planes are genuinely needed and per-subject "
        "selection has room to help. Above roughly 50, one plane saturates and "
        "an offline-fixed configuration will match it.",
        file=sys.stderr,
    )
    if arguments.output:
        Path(arguments.output).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
