"""Table 1 of the paper, recomputed from the stored per-sample test rows.

    python Codes/scripts/experiments/main_table.py              # Table 1, realistic-session benchmark
    python Codes/scripts/experiments/main_table.py v3n96o       # the same rows on the small-session benchmark

Runs in seconds and needs no data. Medians are recomputed from the per-sample
rows in ``results/benchmark/``:
- late fusion and Rooted4D's joint fit: ``E100/rows/<benchmark>_test/``;
- Rooted4D, the joint fit with its residual field: ``E116/reference_rows/``,
  the rows that ``reproduce_rooted4d_main.py compare`` checks a rerun against.
Each recomputed median is checked against the study's report
(``E116/E116_test.json``), and on the realistic-session benchmark against the
value printed in Table 1. The two comparators built from recent methods (late
fusion + BCPD, GPMM fusion) are read from that report, since their per-sample
rows are not stored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path("results/benchmark")
REPORT = RESULTS / "E116/E116_test.json"
COLUMNS = (("heldout_cmr_mm", "CMR (mm)", "%.2f"), ("heldout_tee_mm", "TEE (mm)", "%.2f"),
           ("attachment_residual_mm", "Gap (mm)", "%.2f"), ("phase_map_cycles", "Phase (cyc.)", "%.3f"),
           ("rotation_deg", "Rot. (deg)", "%.2f"))
# (label, arm in the report, where its per-sample rows are)
ROWS = (("Late fusion", "S/late_fusion_trajectory", "e100"),
        ("Late fusion + BCPD", "LF+BCPD", None),
        ("GPMM fusion", "GPMM-G", None),
        ("Rooted4D, joint fit", "S/joint", "e100"),
        ("Rooted4D", "R4D+GP", "e116"))
# Table 1 as printed in the paper (realistic-session benchmark).
PRINTED = {"Late fusion": ("1.11", "1.68", "2.08", "0.013", "2.39"),
           "Late fusion + BCPD": ("1.11", "1.83", "1.87", "0.013", "2.39"),
           "GPMM fusion": ("0.95", "1.64", "0", "0.053", "3.71"),
           "Rooted4D, joint fit": ("1.11", "1.66", "0", "0.008", "1.77"),
           "Rooted4D": ("0.73", "1.09", "0", "0.008", "1.77")}


def _rows(source: str, arm: str, benchmark: str) -> list[dict]:
    if source == "e100":
        shards = sorted((RESULTS / "E100/rows" / f"{benchmark}_test").glob("shard_*.jsonl"))
        records = [json.loads(line) for shard in shards for line in shard.read_text().splitlines() if line.strip()]
        return [record["variants"][arm] for record in records]
    path = RESULTS / "E116/reference_rows" / f"{benchmark}_full.jsonl"
    return [json.loads(line)[arm] for line in path.read_text().splitlines() if line.strip()]


def _cell(value: float, key: str, fmt: str) -> str:
    return "0" if key == "attachment_residual_mm" and value == 0.0 else fmt % value


def main(argv: list[str]) -> int:
    benchmark = argv[0] if argv else "v4n96o"
    report = json.loads(REPORT.read_text())["benchmarks"][f"{benchmark}/full"]["medians"]
    title = {"v4n96o": "Table 1, realistic-session benchmark", "v3n96o": "small-session benchmark (Table S1)"}
    print(f"{title.get(benchmark, benchmark)}: medians over the test samples\n")
    print("%-22s" % "Method" + "".join("%14s" % label for _, label, _ in COLUMNS) + "   source")
    failed = False
    for label, arm, source in ROWS:
        reported = report[arm]
        if source is None:
            medians, note = reported, "report (rows not stored)"
        else:
            rows = _rows(source, arm, benchmark)
            medians = {key: float(np.median([row[key] for row in rows])) for key, _, _ in COLUMNS}
            agree = all(np.isclose(medians[key], reported[key], rtol=0, atol=1e-9) for key, _, _ in COLUMNS)
            failed |= not agree
            note = f"{len(rows)} stored rows, {'equal to' if agree else 'DIFFER FROM'} the report"
        cells = [_cell(medians[key], key, fmt) for key, _, fmt in COLUMNS]
        if benchmark == "v4n96o":
            same = tuple(cells) == PRINTED[label]
            failed |= not same
            note += ", as printed" if same else f", PRINTED {PRINTED[label]}"
        print("%-22s" % label + "".join("%14s" % c for c in cells) + "   " + note)
    print("\npassed" if not failed else "\nFAILED")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
