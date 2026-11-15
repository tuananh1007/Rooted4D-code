"""E120: the supplement's table of each arm's medians before and after the session-gauge correction.

    python e120_table.py [results/benchmark/E120/E120_comparison.json] > table.tex

Rows use the supplement's names for the arms. Each study's own report has the
paired intervals; this table lists the stored and rerun medians of rotation,
translation and phase-map error, with an empty cell where a study reports none.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GROUPS = [
    ("Small session, 60 samples", [
        ("Zero start, image", "E70_v3n96o", "image_cold"),
        (r"\quad + 2 neighboring", "E74_v3n96o", "landmarks_2_cold"),
        (r"\quad + 2 commissures", "E74_v3n96o", "commissures_cold"),
        (r"\quad + 4 landmarks", "E74_v3n96o", "landmarks_4_cold"),
        (r"\quad + 8 landmarks", "E74_v3n96o", "landmarks_8_cold"),
        ("Warm start, image", "E70_v3n96o", "image_warm"),
        (r"\quad + 2 neighboring", "E70_v3n96o", "landmarks_2"),
        (r"\quad + 2 commissures", "E74_v3n96o", "commissures_warm"),
        (r"\quad + 4 landmarks", "E70_v3n96o", "landmarks_4"),
        (r"\quad + 8 landmarks", "E70_v3n96o", "landmarks_8"),
        ("Warm start, CMR image, TEE points", "E71_v3n96o", "cmr_image_tee_point"),
        ("Warm start, CMR points, TEE image", "E71_v3n96o", "cmr_point_tee_image"),
        ("Warm start, both image", "E71_v3n96o", "both_image"),
        ("Search, image", "E91_v3n96o", "image_search"),
        (r"\quad + 2 commissures", "E91_v3n96o", "commissures_search"),
        (r"\qquad \SI{1}{\milli\meter} RMS", "E96_v3n96o", "noise_1mm"),
        (r"\qquad \SI{2}{\milli\meter} RMS", "E96_v3n96o", "noise_2mm"),
        (r"\qquad \SI{4}{\milli\meter} RMS", "E96_v3n96o", "noise_4mm"),
        (r"\qquad \SI{8}{\milli\meter} RMS", "E96_v3n96o", "noise_8mm"),
        (r"\qquad one only", "E96_v3n96o", "one_only"),
        (r"\qquad \ang{15} along the ring", "E96_v3n96o", "slide_one"),
        (r"\qquad swapped", "E96_v3n96o", "swap"),
    ]),
    ("Realistic session, 60 samples", [
        ("Zero start, image", "E91_v4n96o", "image_uninformed"),
        (r"\quad + 2 commissures", "E91_v4n96o", "commissures_uninformed"),
        ("Search, image", "E91_v4n96o", "image_search"),
        (r"\quad + 2 commissures", "E91_v4n96o", "commissures_search"),
    ]),
    ("Preliminary configuration", [
        ("Zero start, image (60)", "E30", "image_lbfgs"),
        ("Warm start, image (60)", "E30", "image_from_point"),
        ("Warm start, image (30)", "E32", "0"),
        (r"\quad + 2 landmarks", "E32", "2"),
        (r"\quad + 4 landmarks", "E32", "4"),
        (r"\quad + 8 landmarks", "E32", "8"),
    ]),
]


def cell(record: dict | None, digits: int) -> str:
    if not record or record.get("stored_median") is None or record.get("rerun_median") is None:
        return "-- & --"
    return f"{record['stored_median']:.{digits}f} & {record['rerun_median']:.{digits}f}"


def main(path: str) -> int:
    studies = json.loads(Path(path).read_text())["studies"]
    lines = []
    for title, rows in GROUPS:
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}}\\")
        for label, study, arm in rows:
            record = studies.get(study, {}).get("arms", {}).get(arm)
            if record is None:
                raise SystemExit(f"no rerun for {study}/{arm}")
            lines.append(f"{label} & {cell(record.get('rotation_deg'), 2)} & {cell(record.get('translation_mm'), 2)}"
                         f" & {cell(record.get('phase_map_cycles'), 3)}\\\\")
        lines.append(r"\addlinespace[2pt]")
    print("\n".join(lines[:-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark/E120/E120_comparison.json"))
