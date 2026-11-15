#!/usr/bin/env python3
"""Fit the real donor pool and write it in the shared topology.

The v1 pool was built ad hoc in a session and left no committed entry point, so
it could not be reproduced from the repository. This is that entry point.

Chambers come from ACDC left-ventricle cavity segmentations and valves from
MVSeg2023 mitral leaflet labels. Person pseudonyms are assigned per source case
and are disjoint across the two pools, because a chamber donor and a valve donor
are different people and the split lock must never merge them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rooted4d.active.acdc import load_acdc_fits
from rooted4d.donors.export import export_donors
from rooted4d.donors.valve import load_valve_fits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acdc-root", default="data/raw/acdc")
    parser.add_argument("--mvseg-root", default="data/raw/mvseg2023")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-annulus", type=int, default=24)
    parser.add_argument("--n-rings", type=int, default=8)
    parser.add_argument("--n-angles", type=int, default=32)
    parser.add_argument("--valve-annulus", type=int, default=32)
    parser.add_argument("--valve-depth", type=int, default=6)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-valves", type=int, default=None)
    parser.add_argument("--chamber-levels", type=int, default=1,
                        help="chamber rings between the annulus and the apex (W2.6)")
    parser.add_argument("--leaflet-rows", type=int, default=1,
                        help="leaflet rows beyond the root, the last the free edge (W2.6)")
    arguments = parser.parse_args(argv)

    output = Path(arguments.output)
    if output.exists():
        raise SystemExit(
            f"{output} already exists; refusing to overwrite a donor pool that "
            "committed results may depend on. Write to a new directory."
        )

    print(f"fitting ACDC chambers from {arguments.acdc_root}", flush=True)
    chambers = load_acdc_fits(
        arguments.acdc_root,
        n_rings=arguments.n_rings,
        n_angles=arguments.n_angles,
        max_patients=arguments.max_patients,
    )
    # One donor per patient, not per frame: two frames of the same heart are the
    # same person, and treating them as two donors would put a patient on both
    # sides of a split.
    by_patient: dict[str, object] = {}
    for fit in chambers:
        by_patient.setdefault(fit.patient, fit)
    chambers = [by_patient[key] for key in sorted(by_patient)]
    print(f"  {len(chambers)} chamber donors", flush=True)

    print(f"fitting MVSeg2023 valves from {arguments.mvseg_root}", flush=True)
    valves = load_valve_fits(
        arguments.mvseg_root,
        n_annulus=arguments.valve_annulus,
        n_depth=arguments.valve_depth,
        limit=arguments.max_valves,
    )
    print(f"  {len(valves)} valve donors", flush=True)

    manifest = export_donors(
        chambers, valves, output, n_annulus=arguments.n_annulus,
        chamber_levels=arguments.chamber_levels, leaflet_rows=arguments.leaflet_rows,
    )
    record = json.loads(Path(manifest).read_text())
    print(f"wrote {manifest} with {len(record['donors'])} donors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
