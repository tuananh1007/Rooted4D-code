"""E103: late fusion with cross-modal transfer on MITEA (weak point 4).

Pre-registered in ``Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md``.
E99's held-out-frame variants on the same-scan pairing: late fusion is re-run
with its fits stored (the code check: its held-out errors must reproduce E99's
rows), and the held-out frame is also predicted by transfer through late
fusion's own registration, ``G`` (TEE-fit frame to CMR frame) and ``chi``
(TEE-fit phase to CMR phase):

* echo end diastole held out, at acquisition time t: ``P_T G^-1 x_C(chi(phi_T(t)))``;
* CMR end diastole held out, at phase 0.5:           ``G x_T(chi^-1(0.5))``.

Rooted4D's rows are E99's, read from ``results/benchmark/E99/E99_fits_test.json``.

    R4D_MODE=fit R4D_ALLOW_TEST=1 python e103_mitea_transfer.py   -> results/benchmark/E103/E103_rows.json
    R4D_MODE=analyze python e103_mitea_transfer.py                -> results/benchmark/E103/E103_analysis.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md"
E99_ROWS = Path("results/benchmark/E99/E99_fits_test.json")
OUT = Path("results/benchmark/E103")
VARIANTS = ("echo_ED", "cmr_ED")
#: The code check: late fusion's own held-out error must match E99's to this (mm).
REPRODUCTION_TOLERANCE_MM = 0.005


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transfer_errors(outcome_extras: dict, problem, base, late_options: dict) -> dict:
    """Held-out error of the other modality's frame, carried through late fusion's registration."""

    from rooted4d.geometry.rigid import RigidTransform
    from rooted4d.inference.baselines import parameters_from_record
    from rooted4d.inference.trajectory_registration import inverse_phase, session_free_trajectory
    from rooted4d.motion import MonotoneCircularPhaseWarp

    model, observations = problem.model, problem.observations
    overrides = late_options.get("modality_overrides") or {}
    # The configs ``_late_fusion_fits`` fitted with; only the motion and session
    # settings enter the trajectories, and the overrides do not change them.
    cmr_config = replace(base, use_tee=False, estimate_pose=False, estimate_phase_warp=False,
                         use_session=False, **overrides.get("cmr", {}))
    tee_config = replace(base, use_cmr=False, use_session=False, **overrides.get("tee", {}))
    cmr = parameters_from_record(outcome_extras["cmr"])
    tee = parameters_from_record(outcome_extras["tee"])
    registration = outcome_extras["registration"]
    G = RigidTransform(np.asarray(registration["rotation"]), np.asarray(registration["translation"]))
    chi = MonotoneCircularPhaseWarp(np.asarray(registration["logits"]), offset=float(registration["offset"]))
    out = {}
    if len(observations.heldout_tee_times):
        times = np.asarray(observations.heldout_tee_times, dtype=np.float64)
        vertices = np.asarray(observations.heldout_tee_vertices)
        cmr_phase = chi.evaluate(tee.warp().evaluate(times))
        chamber = session_free_trajectory(cmr, model, cmr_config, cmr_phase, vertices)
        placed = tee.transform().apply(G.inverse().apply(chamber.reshape(-1, 3))).reshape(chamber.shape)
        distances = np.linalg.norm(placed - np.asarray(observations.heldout_tee_points), axis=-1)
        out["transfer_tee_mm"] = float(distances.mean())
    if len(observations.heldout_cmr_times):
        phases = np.asarray(observations.heldout_cmr_times, dtype=np.float64)
        vertices = np.asarray(observations.heldout_cmr_vertices)
        source = inverse_phase(chi, phases)
        valve = session_free_trajectory(tee, model, tee_config, source, vertices)
        placed = G.apply(valve.reshape(-1, 3)).reshape(valve.shape)
        distances = np.linalg.norm(placed - np.asarray(observations.heldout_cmr_points), axis=-1)
        out["transfer_cmr_mm"] = float(distances.mean())
    return out


def cmd_fit() -> None:
    if os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing test subjects without R4D_ALLOW_TEST=1")
    from rooted4d.inference import SolverConfig
    from rooted4d.inference.baselines import run_arm

    e99 = _load("e99_mitea_fusion", HERE / "e99_mitea_fusion.py")
    stored = {(r["subject"], r["pairing"]): r for r in json.loads(E99_ROWS.read_text())["rows"]}
    model, spec, _ = e99.build_model()
    base = replace(SolverConfig(**spec["base_config"]), estimate_phase_warp=False)   # as E99
    by_name = {v["name"]: v for v in spec["variants"]}
    joint_config = replace(base, **by_name["S/joint"]["overrides"])
    late_options = dict(by_name["S/independent"].get("options") or {}, store_parameters=True)
    chambers = e99._chambers("test")
    rows = []
    for subject in sorted({k[0] for k in chambers}):
        values = {"cmr": {p: chambers.get((subject, "scan1", p, "label")) for p in e99.PHASES},
                  "echo": {p: chambers.get((subject, "scan1", p, "prediction")) for p in e99.PHASES}}
        reference = stored.get((subject, "same_scan"))
        if reference is None or reference.get("front_end_failure") or any(
                v is None for side in values.values() for v in side.values()):
            rows.append({"subject": subject, "front_end_failure": True})
            continue
        perturbation = reference["perturbation"]          # E99's own draw, read back
        row = {"subject": subject, "variants": {}}
        for held_out in VARIANTS:
            problem = e99.make_problem(values["cmr"], values["echo"], model, joint_config,
                                       rotation=np.asarray(perturbation["axis_angle"]),
                                       translation=np.asarray(perturbation["translation"]),
                                       offset=float(perturbation["offset"]), held_out=held_out)
            started = time.perf_counter()
            alignment = e99.two_frame_alignment(problem.observations)
            outcome = run_arm("late_fusion_trajectory", problem, base, alignment=alignment,
                              options=late_options).as_dict()
            record = {k: outcome.get(k) for k in ("succeeded", "heldout_cmr_mm", "heldout_tee_mm",
                                                  "rotation_deg", "translation_mm", "phase_error_cycles")}
            if outcome.get("succeeded"):
                record.update(transfer_errors(outcome["extras"], problem, base, late_options))
            e99_late = reference["variants"].get(f"late_fusion_trajectory/{held_out}", {})
            e99_r4d = reference["variants"].get(f"rooted4d_full/{held_out}", {})
            key = "heldout_tee_mm" if held_out == "echo_ED" else "heldout_cmr_mm"
            record["e99_late_fusion_mm"] = e99_late.get(key)
            record["e99_rooted4d_mm"] = e99_r4d.get(key) if e99_r4d.get("succeeded") else None
            record["seconds"] = round(time.perf_counter() - started, 3)
            row["variants"][held_out] = record
        rows.append(row)
        print(subject, {h: (round(v.get("heldout_tee_mm") or v.get("heldout_cmr_mm") or float("nan"), 3),
                            v.get("e99_late_fusion_mm") and round(v["e99_late_fusion_mm"], 3),
                            round(v.get("transfer_tee_mm") or v.get("transfer_cmr_mm") or float("nan"), 3),
                            v.get("e99_rooted4d_mm") and round(v["e99_rooted4d_mm"], 3))
                        for h, v in row["variants"].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E103_rows.json").write_text(json.dumps(
        {"experiment": "E103", "preregistration": PREREGISTRATION, "rows": rows}, indent=2, default=float) + "\n")
    print("wrote", OUT / "E103_rows.json")


def cmd_analyze() -> None:
    e99 = _load("e99_mitea_fusion", HERE / "e99_mitea_fusion.py")
    rows = [r for r in json.loads((OUT / "E103_rows.json").read_text())["rows"] if not r.get("front_end_failure")]
    report = {"experiment": "E103", "preregistration": PREREGISTRATION, "subjects": len(rows),
              "statistic": "E99's: per-subject values, difference of medians, 20000 resamples, seed 20260910",
              "code_check": {}, "medians": {}, "rooted4d_minus": {}, "reading": {}}
    for held_out, own, transfer in (("echo_ED", "heldout_tee_mm", "transfer_tee_mm"),
                                    ("cmr_ED", "heldout_cmr_mm", "transfer_cmr_mm")):
        records = {r["subject"]: r["variants"][held_out] for r in rows if held_out in r["variants"]}
        deviations = [abs(v[own] - v["e99_late_fusion_mm"]) for v in records.values()
                      if v.get(own) is not None and v.get("e99_late_fusion_mm") is not None]
        report["code_check"][held_out] = {"max_deviation_mm": float(max(deviations)),
                                          "reproduces": bool(max(deviations) <= REPRODUCTION_TOLERANCE_MM),
                                          "subjects": len(deviations)}
        ours = {s: v["e99_rooted4d_mm"] for s, v in records.items() if v.get("e99_rooted4d_mm") is not None}
        late = {s: v[own] for s, v in records.items() if v.get(own) is not None}
        moved = {s: v[transfer] for s, v in records.items() if v.get(transfer) is not None}
        report["medians"][held_out] = {"rooted4d": float(np.median(list(ours.values()))),
                                       "late_fusion": float(np.median(list(late.values()))),
                                       "late_fusion_transfer": float(np.median(list(moved.values())))}
        against_transfer = e99._paired(ours, moved)
        report["rooted4d_minus"][held_out] = {"late_fusion": e99._paired(ours, late),
                                              "late_fusion_transfer": against_transfer,
                                              "transfer_minus_own": e99._paired(moved, late)}
        report["reading"][held_out] = ("advantage survives transfer" if against_transfer["upper"] < 0
                                       else "transfer matches or beats joint fitting")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E103_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    mode = os.environ.get("R4D_MODE")
    if mode == "fit":
        cmd_fit()
    elif mode == "analyze":
        cmd_analyze()
    else:
        sys.exit(f"unknown R4D_MODE {mode!r}")
