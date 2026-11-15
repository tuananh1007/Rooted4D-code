"""E110: is the joint fit non-inferior to late fusion in held-out accuracy?

Pre-registered in ``Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md``.
A reading of the stored test rows of every instance against a margin fixed
there: delta = 0.35 mm, the generator's CMR localization noise per coordinate.
Rooted4D is non-inferior when the upper end of the 95% identity-bootstrap
interval of Rooted4D minus late fusion (E62's statistic) is below delta, and
equivalent when both ends lie inside (-delta, +delta). No fit is made.

    python e110_noninferiority.py <results root>   -> results/benchmark/E110/E110_noninferiority.json

``<results root>`` holds ``e100/<key>/TEST_DONE`` and ``e94/v4oriented/TEST_DONE``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md"
MARGIN_MM = 0.35
JOINT, LATE = "S/joint", "S/late_fusion_trajectory"
#: (results subdirectory, key, label), the paper's instances first.
INSTANCES = [
    ("e100", "v4n96o", "Realistic session, 96 nodes (primary)"),
    ("e100", "v3n96o", "Small session, 96 nodes"),
    ("e100", "v4n48o", "Realistic session, 48 nodes"),
    ("e94", "v4oriented", "Realistic session, 24 nodes"),
    ("e100", "v3n48o", "Small session, 48 nodes"),
    ("e100", "v3o", "Small session, 24 nodes"),
] + [("e100", f"{base}s{k}", f"{label}, split {k}")
     for base, label in (("v4n96o", "Realistic session, 96 nodes"), ("v3n96o", "Small session, 96 nodes"))
     for k in range(1, 6)]
OUT = Path("results/benchmark/E110/E110_noninferiority.json")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reading(entry: dict) -> dict:
    return {"non_inferior": bool(entry["upper"] < MARGIN_MM),
            "equivalent": bool(entry["lower"] > -MARGIN_MM and entry["upper"] < MARGIN_MM),
            "one_sided_upper_bound_mm": entry["upper"]}


def main(root: str) -> int:
    stage = _load("e62_stage_b")
    results = Path(root)
    report = {"experiment": "E110", "preregistration": PREREGISTRATION, "margin_mm": MARGIN_MM,
              "arms": {"rooted4d": JOINT, "late_fusion": LATE},
              "statistic": "E62: Rooted4D minus late fusion, identity-averaged, sum of modality medians, "
                           "%d identity resamples, seed %d" % (stage.RESAMPLES, stage.SEED),
              "instances": {}}
    for sub, key, label in INSTANCES:
        done = results / sub / key / "TEST_DONE"
        if not done.exists():
            report["instances"][key] = {"label": label, "missing": str(done)}
            print(f"{key}: no test rows at {done}")
            continue
        directory = results / sub / key / done.read_text().strip()
        rows = stage.load_rows([directory])
        paired = stage.paired(rows, JOINT, LATE)
        entry = {"label": label, "rows": str(directory), "samples": len(rows),
                 "identities": paired["descriptive"]["identities"],
                 "failures": paired["descriptive"]["failures"]}
        for part in ("sum", "cmr", "tee"):
            entry[part] = dict(paired[part], **reading(paired[part]))
        report["instances"][key] = entry
        print("%-40s sum %+.3f [%+.3f, %+.3f]  cmr %+.3f [%+.3f, %+.3f]  tee %+.3f [%+.3f, %+.3f]  %s" % (
            label, paired["sum"]["observed"], paired["sum"]["lower"], paired["sum"]["upper"],
            paired["cmr"]["observed"], paired["cmr"]["lower"], paired["cmr"]["upper"],
            paired["tee"]["observed"], paired["tee"]["lower"], paired["tee"]["upper"],
            "non-inferior" if entry["sum"]["non_inferior"] else "NOT non-inferior"))
    present = [e for e in report["instances"].values() if "sum" in e]
    report["summary"] = {
        "instances_read": len(present),
        "non_inferior_sum": sum(e["sum"]["non_inferior"] for e in present),
        "non_inferior_all_three": sum(all(e[p]["non_inferior"] for p in ("sum", "cmr", "tee")) for e in present),
        "equivalent_sum": sum(e["sum"]["equivalent"] for e in present),
        "largest_upper_bound_sum_mm": max(e["sum"]["upper"] for e in present) if present else None,
        "largest_upper_bound_any_mm": max(e[p]["upper"] for e in present for p in ("sum", "cmr", "tee"))
        if present else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=1))
    print("wrote", OUT)
    return 0


def latex(path: Path = OUT) -> str:
    """Rows for a table: instance, the three intervals, and the reading."""

    tables = _load("e64_manuscript_tables")
    report = json.loads(path.read_text())
    lines = []
    for key, entry in report["instances"].items():
        if "sum" not in entry:
            continue
        verdict = "yes" if all(entry[p]["non_inferior"] for p in ("sum", "cmr", "tee")) else "no"
        lines.append("%s & %s & %s & %s & %s\\\\" % (entry["label"], tables.interval(entry["sum"]),
                                                     tables.interval(entry["cmr"]), tables.interval(entry["tee"]),
                                                     verdict))
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1:2] == ["latex"]:
        print(latex())
    else:
        sys.exit(main(*sys.argv[1:]))
