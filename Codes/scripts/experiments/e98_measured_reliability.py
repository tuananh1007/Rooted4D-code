"""E98: reliability flags under measured front-end error (plan v2, W2.4 on W1.3's data).

Pre-registered in ``Docs/E98_MEASURED_RELIABILITY_PREREGISTRATION_2026-09-29.md``.
E95's signals and statistics, unchanged, on E89's measured-error run of Rooted4D:
alpha and the final objective from the scored rows, the Laplace posterior spread
at each stored fit on the measured-error observations (no refit), against
rotation, translation, phase-map and anatomy error. Held-out errors are not
targets: under measured error they measure agreement with the front end.

    python e98_measured_reliability.py <e59 rows root>   -> results/benchmark/E98/E98_measured_reliability.json

Runs after E89's analysis (``results/benchmark/E89/E89_analysis.json``), whose
per-sample anatomy errors it reads. With ``R4D_E89_INSTANCE=e100`` it reads and
writes E100's copies (E89's switch; E100's amendment).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E98_MEASURED_RELIABILITY_PREREGISTRATION_2026-09-29.md"
TARGETS = ("rotation_deg", "translation_mm", "phase_map_cycles", "anatomy_rms_mm")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def laplace_spreads(rows: dict, spec: dict, e95) -> dict:
    """E95's posterior spreads, at the measured-error run's stored fits."""

    driver = _load("e59_driver", HERE / "e59_unknown_alignment.py")
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.geometry.rigid import axis_angle_to_matrix
    from rooted4d.inference import SolverConfig
    from rooted4d.inference.baselines import parameters_from_record
    from rooted4d.inference.joint import SceneFit, _pack
    from rooted4d.inference.posterior import fit_laplace_posterior, propagate_measurements
    from rooted4d.motion import circular_distance

    joint = next(v for v in spec["variants"] if v["name"] == "S/joint")
    config = replace(SolverConfig(**spec["base_config"]), **joint["overrides"])
    model, *_ = driver.build_model(Path(spec["model_benchmark"]), spec["models"]["S"])
    root = Path(spec["benchmark"])
    out = {}
    for name, row in rows.items():
        outcome = row["variants"]["S/joint"]
        if not outcome.get("succeeded"):
            continue
        parameters = parameters_from_record(outcome["extras"]["parameters"])
        problem = benchmark_problem(root / "samples" / "test" / name, model, config)
        fit = SceneFit(parameters=parameters, cost=float(outcome["cost"]), iterations=0, stage_costs=(),
                       attachment_residual=0.0, packed=_pack(parameters, model, config), model=model,
                       config=config, observations=problem.observations)
        posterior = fit_laplace_posterior(fit)
        rotation = axis_angle_to_matrix(np.asarray(parameters.pose_axis_angle))
        measurements = {
            "rotation": lambda p: float(np.degrees(np.arccos(np.clip(
                (np.trace(axis_angle_to_matrix(np.asarray(p.pose_axis_angle)) @ rotation.T) - 1.0) / 2.0,
                -1.0, 1.0)))),
            "translation": lambda p: float(np.linalg.norm(np.asarray(p.pose_translation)
                                                          - np.asarray(parameters.pose_translation))),
            "phase": lambda p: float(circular_distance(np.array([p.phase_offset]),
                                                       np.array([parameters.phase_offset]))[0]),
        }
        draws = propagate_measurements(posterior, fit, measurements, n_samples=e95.POSTERIOR_SAMPLES,
                                       seed=e95.SEED)
        out[name] = {"laplace_rotation_deg": float(np.sqrt((draws["rotation"] ** 2).mean())),
                     "laplace_translation_mm": float(np.sqrt((draws["translation"] ** 2).mean())),
                     "laplace_phase_cycles": float(np.sqrt((draws["phase"] ** 2).mean()))}
    return out


def main(rows_root: str) -> int:
    e95 = _load("e95_reliability", HERE / "e95_reliability.py")
    e89 = _load("e89_measured_error", HERE / "e89_measured_error.py")
    spec, rows = e89._spec_rows(Path(rows_root), "measured")
    analysis = json.loads(e89.REPORT.read_text())
    anatomy = {r["sample"]: r.get("anatomy_rms_mm") for r in analysis["records"]["measured"]
               if r["method"] == "S/joint"}
    spreads = laplace_spreads(rows, spec, e95)
    names = sorted(rows)
    joint = [rows[n]["variants"]["S/joint"] for n in names]
    identities = [rows[n]["identity"] for n in names]
    signals = {"alpha": np.array([np.nan if o.get("alpha") is None else o["alpha"] for o in joint]),
               "cost": np.array([o.get("cost", np.nan) for o in joint], dtype=float)}
    for key in ("laplace_rotation_deg", "laplace_translation_mm", "laplace_phase_cycles"):
        signals[key] = np.array([spreads.get(n, {}).get(key, np.nan) for n in names])
    errors = {t: np.array([np.nan if o.get(t) is None else o[t] for o in joint], dtype=float)
              for t in TARGETS if t != "anatomy_rms_mm"}
    errors["anatomy_rms_mm"] = np.array([np.nan if anatomy.get(n) is None else anatomy[n] for n in names])
    report = {"experiment": "E98", "preregistration": PREREGISTRATION, "samples": len(names), "signals": {}}
    for name, signal in signals.items():
        targets = TARGETS if name not in e95.MATCHING else (e95.MATCHING[name],)
        report["signals"][name] = {
            target: {"spearman": e95.correlation(identities, signal, errors[target]),
                     "flag": e95.flag_rule(signal, errors[target], e95.SIGNS[name])}
            for target in targets}
        for target, result in report["signals"][name].items():
            rho, expected = result["spearman"], e95.SIGNS[name]
            result["useful_flag"] = bool((rho["upper"] < 0 if expected < 0 else rho["lower"] > 0)
                                         and result["flag"]["precision"] >= 0.3)
    e100 = os.environ.get("R4D_E89_INSTANCE") == "e100"
    out = Path(f"results/benchmark/{'E100/' if e100 else ''}E98/E98_measured_reliability.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for name, targets in report["signals"].items():
        for target, result in targets.items():
            s = result["spearman"]
            print("%-24s %-18s rho %+.2f [%+.2f, %+.2f]  flag %d/%d  useful %s" % (
                name, target, s["rho"], s["lower"], s["upper"], result["flag"]["hits"],
                result["flag"]["flagged"], result["useful_flag"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
