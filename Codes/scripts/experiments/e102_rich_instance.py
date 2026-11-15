"""E102: the method ordering on a richer procedural instance (plan v2, W2.6).

Pre-registered in ``Docs/E102_RICHER_INSTANCE_PREREGISTRATION_2026-10-03.md``.
The reduced instance carries one chamber ring between the annulus and the apex
and one free-edge row per leaflet. This instance carries six chamber rings and
three leaflet rows (``make_toy_complex(chamber_levels=6, leaflet_rows=3)``), at
24 ring nodes, on E100's small-session construction otherwise unchanged
(``v3o``: oriented valves, the same donors, pairing, seeds and split), and is
taken through the same protocol.

    python e102_rich_instance.py configs        -> Codes/configs/e102/v3or.yaml
    python e102_rich_instance.py check          the build checks (audit, free-edge side, v3o's sample ids)
    python e102_rich_instance.py chain          E88's chain: selection rounds, session weight, one test
    python e102_rich_instance.py gate           the representation's donor-grouped cross-validation
    python e102_rich_instance.py analyze        -> results/benchmark/E102/E102_analysis.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E102_RICHER_INSTANCE_PREREGISTRATION_2026-10-03.md"
KEY = "v3or"
CONFIGS = Path("Codes/configs/e102")
SOURCE = Path("Codes/configs/e100/v3o.yaml")
BENCHMARK = Path("data/derived/e102-v3or")
REFERENCE = Path("data/derived/e100-v3o")
REFERENCE_KEY = "v3o"
DONORS = "data/derived/donors_v3_rich24"
CHAMBER_LEVELS, LEAFLET_ROWS = 6, 3
REPORT = Path("results/benchmark/E102/E102_analysis.json")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def cmd_configs() -> None:
    text = SOURCE.read_text()
    text = text[text.index("schema_version:"):]
    text = _replace_once(text, "manifest: ../../../data/derived/donors_v3/donors.json",
                         f"manifest: ../../../{DONORS}/donors.json")
    text = _replace_once(text, "  n_annulus: 24\n",
                         "  n_annulus: 24\n"
                         "  # E102 (W2.6): six chamber rings between the annulus and the apex, and\n"
                         "  # three leaflet rows beyond the root, the last the free edge.\n"
                         f"  chamber_levels: {CHAMBER_LEVELS}\n"
                         f"  leaflet_rows: {LEAFLET_ROWS}\n")
    header = (f"# E102 ({PREREGISTRATION}): {KEY}.\n"
              f"# {SOURCE} on the richer procedural instance, from the donor pool {DONORS}.\n"
              "# Written by e102_rich_instance.py configs; do not edit by hand.\n")
    CONFIGS.mkdir(parents=True, exist_ok=True)
    out = CONFIGS / f"{KEY}.yaml"
    out.write_text(header + text)
    print("wrote", out)


def cmd_check() -> None:
    """The audit, the free edge on the ventricular side in every sample, and v3o's sample ids."""

    from rooted4d.geometry import make_toy_complex

    audit = json.loads((BENCHMARK / "audit.json").read_text())
    assert audit.get("passed", True) and not audit.get("errors"), "audit failed"
    failed = False
    for split in ("train", "val", "test"):
        names = sorted(p.name for p in (BENCHMARK / "samples" / split).glob("*.npz"))
        heights = []
        for name in names:
            vertices = np.load(BENCHMARK / "samples" / split / name)["template_vertices"]
            groups = make_toy_complex((len(vertices) - 1) // (1 + CHAMBER_LEVELS + LEAFLET_ROWS),
                                      chamber_levels=CHAMBER_LEVELS, leaflet_rows=LEAFLET_ROWS).groups
            ring = vertices[groups["annulus"]].mean(axis=0)
            apex = vertices[groups["chamber_unique"][-1]]
            axis = (apex - ring) / np.linalg.norm(apex - ring)
            heights.append(float(((vertices[groups["leaflet_free"]] - ring) @ axis).mean()))
        heights = np.array(heights)
        other = sorted(p.name for p in (REFERENCE / "samples" / split).glob("*.npz"))
        same = names == other
        line = (f"{split}: {len(names)} samples, free-edge height {heights.mean():+.1f} mm "
                f"[{heights.min():+.1f}, {heights.max():+.1f}] | same sample ids as {REFERENCE}: {same}")
        if (heights <= 0).any() or not same:
            failed = True
            line += "  FAILED"
        print(line)
    if failed:
        raise SystemExit("a build check failed")
    print("checks passed")


def _chain():
    e88 = _load("e88_replication")
    return e88.Chain("v3", 0, key=KEY, benchmark=str(BENCHMARK), configs=CONFIGS, experiment="E102",
                     preregistration=PREREGISTRATION)


def cmd_chain() -> None:
    chain = _chain()
    chain.selection_rounds()
    arms = chain.session_weight()
    directory = chain.test(arms)
    (chain.root / "TEST_DONE").write_text(str(Path(directory).relative_to(chain.root)) + "\n")
    print(f"[{KEY}] test rows in {directory}", flush=True)


def cmd_gate() -> None:
    components = _load("e100_components")
    print("gate rows in", components.study_gate(_chain()), flush=True)


def _test_rows(stage, root: Path):
    marker = root / "TEST_DONE"
    rows = stage.load_rows([str(root / marker.read_text().strip())])
    assert len(rows) == 60, (root, len(rows))
    return rows


def reading(rich: dict, reference: dict, key_differences) -> dict:
    """The pre-stated reading: (a), (b) and (c) decide; the sign rule against v3o is reported."""

    representation = rich["T1_representation"]["comparison"]["sum"]
    trajectory = rich["rooted4d_minus_trajectory"]
    conditions = {
        "a_representation_confirmed": bool(representation["upper"] < 0),
        "b_no_fusion_cost": all(rich["T3_fusion_cost"][m] != "cost remains" for m in ("cmr", "tee")),
        "c_gap_and_phase": bool(trajectory["attachment_residual_mm"]["upper"] < 0
                                and trajectory["phase_map_cycles"]["upper"] < 0),
    }
    signs = {}
    for name, path in key_differences:
        mine, theirs = rich, reference
        for step in path:
            mine, theirs = mine[step], theirs[step]
        excludes = lambda v: v["upper"] < 0 or v["lower"] > 0  # noqa: E731
        signs[name] = {
            "reference_excludes_zero": bool(excludes(theirs)),
            "rich_excludes_zero": bool(excludes(mine)),
            "opposite_signs": bool(excludes(theirs) and excludes(mine)
                                   and np.sign(mine["observed"]) != np.sign(theirs["observed"])),
        }
    return {"conditions": conditions, "ordering_holds": all(conditions.values()),
            "sign_rule_against_reference": {"holds": not any(s["opposite_signs"] for s in signs.values()),
                                            "differences": signs}}


def cmd_analyze() -> None:
    e88 = _load("e88_replication")
    stage = e88._load("e62_stage_b")
    out_root = Path(os.environ["R4D_E88_OUT"])
    rich_root, reference_root = out_root / KEY, Path(os.environ["R4D_E100_OUT"]) / REFERENCE_KEY
    rich = e88.test_report(stage, _test_rows(stage, rich_root))
    reference = e88.test_report(stage, _test_rows(stage, reference_root))
    selection = json.loads((rich_root / "session_selection.json").read_text())
    rich["session_weights_changed"] = {arm: a["selected"] for arm, a in selection["arms"].items() if a["changed"]}
    report = {"experiment": "E102", "preregistration": PREREGISTRATION,
              "instance": {"key": KEY, "benchmark": str(BENCHMARK), "chamber_levels": CHAMBER_LEVELS,
                           "leaflet_rows": LEAFLET_ROWS},
              "reference": {"key": REFERENCE_KEY, "benchmark": str(REFERENCE)},
              "statistics": "E62's: identity-averaged, sum of modality medians, differences of medians, "
                            "%d identity resamples, seed %d" % (stage.RESAMPLES, stage.SEED),
              "rich": rich, "reference_test": reference,
              "reading": reading(rich, reference, e88.KEY_DIFFERENCES)}
    gate_dirs = [p for p in (rich_root / "gate").glob("*") if p.is_dir()] if (rich_root / "gate").exists() else []
    if gate_dirs:
        rows = stage.load_rows([str(gate_dirs[0])])
        report["cross_validation"] = {
            "samples": len(rows), "identities": len({r["identity"] for r in rows}),
            "representation": stage.paired(rows, "S/joint", "G/joint"),
            "fusion_cost": {m: stage.paired(rows, f"{m}/joint", f"{m}/independent") for m in stage.MODELS},
        }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["reading"]["conditions"], indent=1), "ordering holds:", report["reading"]["ordering_holds"],
          "| sign rule:", report["reading"]["sign_rule_against_reference"]["holds"])
    print("wrote", REPORT)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"configs": cmd_configs, "check": cmd_check, "chain": cmd_chain, "gate": cmd_gate,
     "analyze": cmd_analyze}[command](*arguments)
