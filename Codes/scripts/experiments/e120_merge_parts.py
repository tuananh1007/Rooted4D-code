"""E120: merge the per-arm partial rows of E91 or E96 into one row per sample.

    python e120_merge_parts.py <component directory> <part directory> [<part directory> ...]

Each part directory is what ``uniformity_runner`` wrote for one group of arms
run under ``R4D_ARMS`` (``run.json`` and ``shard_*.jsonl``). The merged row of a
sample has the study's own fields, its variants in the study's own order, the
CMR stage time of the first part that ran the stage, and the digest of the CMR
start, which every part that ran the stage must agree on. Nothing is written
unless every sample of the split has every arm exactly once, the parts agree on
each sample's identity, commissures and failure, and the component directory
holds no rows yet. The merge writes ``shard_000_of_001.jsonl``, the parts'
common ``run.json``, and ``parts.json`` (which part ran which arms).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SHARED = ("sample", "identity", "commissures", "failure_reason")


def study_arms(experiment: str) -> tuple:
    if experiment == "E91":
        import e91_image_search as module
    elif experiment == "E96":
        import e96_commissure_precision as module
    else:
        raise SystemExit(f"not a split study: {experiment}")
    return module.ARMS


def expected_arms(experiment: str, row: dict, arms: tuple) -> set:
    """The arms the study's own code runs on this sample (both commissures are needed for the landmark arms)."""
    if experiment == "E91":
        return set(arms) if len(row["commissures"]) == 2 else {"image_search", "image_uninformed"}
    return set() if "failure_reason" in row else set(arms)


def read_part(directory: Path) -> dict:
    rows = {}
    for shard in sorted(directory.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["sample"] in rows:
                raise SystemExit(f"{row['sample']} twice in {directory}")
            rows[row["sample"]] = row
    return rows


def merge(component: Path, parts: list[Path]) -> dict:
    runs = [json.loads((part / "run.json").read_text()) for part in parts]
    if any(run != runs[0] for run in runs):
        raise SystemExit("the parts' run.json differ")
    experiment, benchmark = runs[0]["experiment"], runs[0]["benchmark"]
    arms = study_arms(experiment)
    samples = sorted(p.name for p in (Path(benchmark) / "samples" / "test").glob("*.npz"))
    if not samples:
        raise SystemExit(f"no test samples under {benchmark}")
    if component.exists() and any(component.glob("shard_*.jsonl")):
        raise SystemExit(f"{component} already holds rows")
    by_part = [read_part(part) for part in parts]
    merged, ran = [], {}
    for sample in samples:
        pieces = [(part, rows[sample]) for part, rows in zip(parts, by_part) if sample in rows]
        if len(pieces) != len(parts):
            raise SystemExit(f"{sample} is missing from {len(parts) - len(pieces)} part(s)")
        first = pieces[0][1]
        for part, piece in pieces[1:]:
            for key in SHARED:
                if piece.get(key) != first.get(key):
                    raise SystemExit(f"{sample}: {part} disagrees on {key}")
        digests = {piece["cmr_start_sha256"] for _, piece in pieces if "cmr_start_sha256" in piece}
        if len(digests) > 1:
            raise SystemExit(f"{sample}: the parts started from different CMR fits")
        variants = {}
        for part, piece in pieces:
            for name, outcome in piece["variants"].items():
                if name in variants:
                    raise SystemExit(f"{sample}: arm {name} in two parts")
                variants[name] = outcome
                ran.setdefault(str(part), set()).add(name)
        if set(variants) != expected_arms(experiment, first, arms):
            raise SystemExit(f"{sample}: arms {sorted(variants)} differ from the study's")
        row = {key: first[key] for key in ("sample", "identity") if key in first}
        row["variants"] = {name: variants[name] for name in arms if name in variants}
        row["commissures"] = first["commissures"]
        if "failure_reason" in first:
            row["failure_reason"] = first["failure_reason"]
        timed = [piece["cmr_stage_seconds"] for _, piece in pieces if "cmr_stage_seconds" in piece]
        if timed:
            row["cmr_stage_seconds"] = timed[0]
        if digests:
            row["cmr_start_sha256"] = digests.pop()
        merged.append(row)
    component.mkdir(parents=True, exist_ok=True)
    (component / "shard_000_of_001.jsonl").write_text("".join(json.dumps(row) + "\n" for row in merged))
    (component / "run.json").write_text(json.dumps(runs[0], indent=2, sort_keys=True) + "\n")
    report = {"experiment": experiment, "benchmark": benchmark, "samples": len(merged),
              "parts": {part: sorted(names, key=arms.index) for part, names in ran.items()}}
    (component / "parts.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    print(json.dumps(merge(Path(argv[1]), [Path(a) for a in argv[2:]]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
