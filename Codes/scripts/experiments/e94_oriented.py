"""E94: the main protocol on correctly oriented composites.

Pre-registered in ``Docs/E94_ORIENTED_COMPOSITES_PREREGISTRATION_2026-09-28.md``.
Every benchmark so far carries its valves upside down relative to the chamber
(``Docs/COMPOSITE_ORIENTATION_FINDING_2026-09-28.md``). This builds the
realistic-session benchmark again with ``donors.orient_valves: true`` and
nothing else changed (the same split seed, donors, pairing and sample seeds),
and takes it through E88's unattended chain, keeping the fitted parameters of
Rooted4D and trajectory late fusion for E90's measurements.

    python e94_oriented.py config                  -> Codes/configs/semisynthetic_real_donors_v4_oriented.yaml
    python e94_oriented.py chain                   -> the protocol on it (on Delta; reads R4D_E88_OUT)
    python e94_oriented.py analyze <test dir> v4=<original dirs,...>
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E94_ORIENTED_COMPOSITES_PREREGISTRATION_2026-09-28.md"
SOURCE = Path("Codes/configs/semisynthetic_real_donors_v4.yaml")
CONFIG = Path("Codes/configs/semisynthetic_real_donors_v4_oriented.yaml")
BENCHMARK = "data/derived/semisynthetic-real-v4-oriented"
KEY = "v4oriented"
REPORT = Path("results/benchmark/E94/E94_oriented.json")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_config() -> None:
    text = SOURCE.read_text()
    anchor = "  split_lock: null\n"
    assert text.count(anchor) == 1
    body = text.replace(anchor, anchor + (
        "  # E94: each valve is re-indexed to its chamber's ring handedness before it is\n"
        "  # carried onto the ring, so its free edge lands on the ventricular side.\n"
        "  orient_valves: true\n"))
    header = (f"# E94 ({PREREGISTRATION}): the realistic-session benchmark on correctly\n"
              f"# oriented composites. Identical to {SOURCE.name} except donors.orient_valves.\n"
              "# Written by e94_oriented.py config; do not edit by hand.\n")
    CONFIG.write_text(header + body)
    print("wrote", CONFIG)


def cmd_chain() -> None:
    e88 = _load("e88_replication")
    chain = e88.Chain("v4", 0, key=KEY, benchmark=BENCHMARK, configs=Path("Codes/configs/e94"),
                      experiment="E94", preregistration=PREREGISTRATION, store_parameters=True)
    chain.selection_rounds()
    arms = chain.session_weight()
    directory = chain.test(arms)
    (chain.root / "TEST_DONE").write_text(str(Path(directory).relative_to(chain.root)) + "\n")
    print(f"[{KEY}] test rows in {directory}", flush=True)


def _samples(directories) -> set:
    return {(json.loads(line)["identity"], json.loads(line)["sample"])
            for directory in directories for shard in sorted(Path(directory).glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()}


def cmd_analyze(test_dir: str, original: str) -> None:
    """E64's T1-T4 and E87's comparisons on E94's test, beside the original benchmark's."""

    e88 = _load("e88_replication")
    stage = e88._load("e62_stage_b")
    bench, directories = original.split("=", 1)
    assert bench == "v4"
    spec = json.loads((Path(test_dir) / "spec.json").read_text())
    assert spec["benchmark"] == BENCHMARK and spec["split"] == "test", spec["benchmark"]
    oriented = stage.load_rows([test_dir])
    before = stage.load_rows(directories.split(","))
    assert len(oriented) == len(before) == 60
    # Same identities and samples: the composites differ only in valve orientation.
    # (load_rows keys rows by sample and drops the name, so it is read from the shards.)
    assert _samples([test_dir]) == _samples(directories.split(","))
    report = {"experiment": "E94", "preregistration": PREREGISTRATION,
              "oriented": e88.test_report(stage, oriented),
              "original": e88.test_report(stage, before)}
    report["side_by_side"] = {
        name: {"original": e88._entry(report["original"], path),
               "oriented": e88._entry(report["oriented"], path)}
        for name, path in e88.KEY_DIFFERENCES}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    for name, pair in report["side_by_side"].items():
        print("%-32s original %+.3f [%+.3f, %+.3f]   oriented %+.3f [%+.3f, %+.3f]" % (
            name, *(pair["original"][k] for k in ("observed", "lower", "upper")),
            *(pair["oriented"][k] for k in ("observed", "lower", "upper"))))
    print("wrote", REPORT)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"config": cmd_config, "chain": cmd_chain, "analyze": cmd_analyze}[command](*arguments)
