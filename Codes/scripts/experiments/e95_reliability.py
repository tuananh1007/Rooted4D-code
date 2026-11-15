"""E95: do the reliability signals flag the bad cases? (plan v2, W2.4)

Pre-registered in ``Docs/E95_RELIABILITY_FLAGS_PREREGISTRATION_2026-09-28.md``.
Signals per test sample of Rooted4D (main configuration): the identifiability
diagnostic alpha and the final objective, from the scored rows, and the Laplace
posterior spread of rotation, translation and phase offset, evaluated at the
fits E87 stored (no refit). Targets: the sample's errors.

    python e95_reliability.py        -> results/benchmark/E95/E95_reliability.json
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E95_RELIABILITY_FLAGS_PREREGISTRATION_2026-09-28.md"
ROWS = {"v4": Path("results/benchmark/E75/v4_test"), "v3": Path("results/benchmark/E64/v3_test")}
STORED = Path("results/benchmark/E87/v4_test")  # S/joint_rerun, E75's configuration, with parameters
STORED_ARM = "S/joint_rerun"
REPORT = Path("results/benchmark/E95/E95_reliability.json")
#: E100 points ROWS, STORED, STORED_ARM and REPORT at the rebuilt primaries
#: (``e100_components.py``); the defaults are E95's.
TARGETS = ("rotation_deg", "translation_mm", "phase_map_cycles", "heldout_cmr_mm", "heldout_tee_mm")
#: Expected sign of the rank correlation between a signal and the error.
SIGNS = {"alpha": -1, "cost": +1, "laplace_rotation_deg": +1, "laplace_translation_mm": +1,
         "laplace_phase_cycles": +1}
#: Which spread flags which target (the matching quantity).
MATCHING = {"laplace_rotation_deg": "rotation_deg", "laplace_translation_mm": "translation_mm",
            "laplace_phase_cycles": "phase_map_cycles"}
RESAMPLES, SEED, POSTERIOR_SAMPLES = 20000, 20260910, 2000


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(directory: Path) -> list:
    return [json.loads(line) for shard in sorted(directory.glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()]


def laplace_spreads() -> dict:
    """Posterior spread of pose and phase at each stored Rooted4D fit."""

    driver = _load("e59_driver", HERE / "e59_unknown_alignment.py")
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.geometry.rigid import axis_angle_to_matrix
    from rooted4d.inference import SolverConfig
    from rooted4d.inference.baselines import parameters_from_record
    from rooted4d.inference.joint import SceneFit, _pack
    from rooted4d.inference.posterior import fit_laplace_posterior, propagate_measurements
    from rooted4d.motion import circular_distance

    spec = json.loads((STORED / "spec.json").read_text())
    joint = next(v for v in spec["variants"] if v["name"] == STORED_ARM)
    config = replace(SolverConfig(**spec["base_config"]), **joint["overrides"])
    root = Path(spec["benchmark"])
    model, *_ = driver.build_model(root, spec["models"]["S"])
    out = {}
    for row in _rows(STORED):
        outcome = row["variants"][STORED_ARM]
        if not outcome.get("succeeded"):
            continue
        parameters = parameters_from_record(outcome["extras"]["parameters"])
        problem = benchmark_problem(root / "samples" / "test" / row["sample"], model, config)
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
        draws = propagate_measurements(posterior, fit, measurements, n_samples=POSTERIOR_SAMPLES,
                                       seed=SEED)
        # RMS deviation from the MAP value: the spread of each quantity.
        out[row["sample"]] = {"laplace_rotation_deg": float(np.sqrt((draws["rotation"] ** 2).mean())),
                              "laplace_translation_mm": float(np.sqrt((draws["translation"] ** 2).mean())),
                              "laplace_phase_cycles": float(np.sqrt((draws["phase"] ** 2).mean()))}
    return out


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties (SciPy's rankdata where available; the same result)."""
    try:
        from scipy.stats import rankdata

        return rankdata(values)
    except ImportError:
        pass
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def correlation(identities, signal, error) -> dict:
    """Spearman over samples, with an identity-level bootstrap interval."""

    keep = np.isfinite(signal) & np.isfinite(error)
    identities, signal, error = np.asarray(identities)[keep], signal[keep], error[keep]
    groups = {i: np.flatnonzero(identities == i) for i in np.unique(identities)}
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        chosen = rng.integers(0, len(keys), size=len(keys))
        index = np.concatenate([groups[keys[c]] for c in chosen])
        draws[r] = spearman(signal[index], error[index])
    return {"rho": spearman(signal, error), "lower": float(np.nanpercentile(draws, 2.5)),
            "upper": float(np.nanpercentile(draws, 97.5)), "samples": int(keep.sum())}


def flag_rule(signal, error, sign) -> dict:
    """Flag the worst decile by signal; how many of the error's worst decile does it catch?"""

    keep = np.isfinite(signal) & np.isfinite(error)
    signal, error = signal[keep], error[keep]
    n = len(signal)
    k = max(1, int(round(0.1 * n)))
    flagged = set(np.argsort(sign * signal)[::-1][:k])        # worst by the signal's expected sign
    bad = set(np.argsort(error)[::-1][:k])                     # largest errors
    hits = len(flagged & bad)
    tail = sum(math.comb(k, j) * math.comb(n - k, k - j) for j in range(hits, k + 1)) / math.comb(n, k)
    return {"flagged": k, "hits": hits, "precision": hits / k, "recall": hits / k,
            "chance": k / n, "chance_tail_probability": tail}


def main() -> int:
    spreads = laplace_spreads()
    report = {"experiment": "E95", "preregistration": PREREGISTRATION, "benchmarks": {}}
    for bench, directory in ROWS.items():
        rows = sorted(_rows(directory), key=lambda r: r["sample"])
        joint = [r["variants"]["S/joint"] for r in rows]
        identities = [r["identity"] for r in rows]
        signals = {"alpha": np.array([np.nan if o.get("alpha") is None else o["alpha"] for o in joint]),
                   "cost": np.array([o.get("cost", np.nan) for o in joint], dtype=float)}
        if bench == "v4":
            for key in ("laplace_rotation_deg", "laplace_translation_mm", "laplace_phase_cycles"):
                signals[key] = np.array([spreads.get(r["sample"], {}).get(key, np.nan) for r in rows])
        errors = {t: np.array([np.nan if o.get(t) is None else o[t] for o in joint], dtype=float)
                  for t in TARGETS}
        entry = {"samples": len(rows), "signals": {}}
        for name, signal in signals.items():
            targets = TARGETS if name not in MATCHING else (MATCHING[name],)
            entry["signals"][name] = {
                target: {"spearman": correlation(identities, signal, errors[target]),
                         "flag": flag_rule(signal, errors[target], SIGNS[name])}
                for target in targets}
            for target, result in entry["signals"][name].items():
                rho = result["spearman"]
                expected = SIGNS[name]
                result["useful_flag"] = bool(
                    (rho["upper"] < 0 if expected < 0 else rho["lower"] > 0)
                    and result["flag"]["precision"] >= 0.3)
        report["benchmarks"][bench] = entry
    out = REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for bench, entry in report["benchmarks"].items():
        for name, targets in entry["signals"].items():
            for target, result in targets.items():
                s = result["spearman"]
                print("%s %-24s %-18s rho %+.2f [%+.2f, %+.2f]  flag %d/%d  useful %s" % (
                    bench, name, target, s["rho"], s["lower"], s["upper"], result["flag"]["hits"],
                    result["flag"]["flagged"], result["useful_flag"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
