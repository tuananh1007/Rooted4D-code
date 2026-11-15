"""E29: does the evidence-aware prior earn its place where a prior can matter?

Section 18 measured the evidence-aware prior against a uniform one at 0.008 mm
on held-out CMR and 0.027 on translation, and section 8 downgraded it to barely
measurable. Both were run in the data-rich regime the benchmark supplies: some
two thousand point residuals against a handful of prior terms. A prior cannot
matter there, and measuring it there says nothing about the prior.

This is the same mistake the registration ablations had to be rescued from. The
directed annular arc term looked worthless from a good initialization and was
worth 6.7 degrees once tested under the failure it exists for. A prior exists for
scarcity, so it is tested under scarcity: evidence is thinned by dropping
observed CMR frames and TEE vertices, and by raising observation noise, and the
two priors are compared across that range.

Thinning is applied to the *visible* evidence only. The held-out mask is
untouched, so every arm at every scarcity level is scored on identical hidden
data and the numbers stay comparable down the column.
"""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rooted4d.experiments.benchmark_eval import (
    benchmark_problem,
    fit_benchmark_model,
    sample_paths,
)
from rooted4d.inference.joint import Observations, SolverConfig, fit_joint_scene
from rooted4d.inference.problem import JointProblem, pose_errors

ROOT = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v4"))
N_MODES = int(os.environ.get("R4D_MODES", 32))
N_SAMPLES = int(os.environ.get("R4D_SAMPLES", 20))
BASE = SolverConfig(
    fourier_order=2, session_order=1, motion_prior_weight=1.0, max_iterations=40
)
# Evidence-aware weights are the configured ones; the uniform arm sets every
# prior to the same value, which is what "uniform instead of evidence-aware"
# means in the baseline lock.
ARMS = {
    "evidence_aware": BASE,
    "uniform_prior": replace(
        BASE, identity_prior_weight=1.0, motion_prior_weight=1.0,
        session_prior_weight=1.0,
    ),
}
# fraction of visible frames/vertices retained, and a noise multiplier
LEVELS = (
    ("full", 1.00, 0.0),
    ("half", 0.50, 0.0),
    ("quarter", 0.25, 0.0),
    ("eighth", 0.125, 0.0),
    ("eighth_noisy", 0.125, 1.5),
)

print("fitting shape space...", flush=True)
model, _ = fit_benchmark_model(ROOT, n_modes=N_MODES, n_motion=4, n_session=2)


def thin(problem, keep, noise_mm, seed):
    """Retain a fraction of visible CMR frames and TEE vertices, add noise."""

    rng = np.random.default_rng(seed)
    obs = problem.observations
    n_frames = max(2, int(round(keep * len(obs.cmr_times))))
    frames = np.sort(rng.choice(len(obs.cmr_times), n_frames, replace=False))
    n_vertices = max(6, int(round(keep * len(obs.tee_vertices))))
    columns = np.sort(rng.choice(len(obs.tee_vertices), n_vertices, replace=False))

    cmr_points = obs.cmr_points[frames]
    tee_points = obs.tee_points[:, columns, :]
    if noise_mm > 0.0:
        cmr_points = cmr_points + rng.normal(0.0, noise_mm, cmr_points.shape)
        tee_points = tee_points + rng.normal(0.0, noise_mm, tee_points.shape)
    thinned = Observations(
        cmr_times=obs.cmr_times[frames],
        cmr_vertices=obs.cmr_vertices,
        cmr_points=cmr_points,
        tee_times=obs.tee_times,
        tee_vertices=obs.tee_vertices[columns],
        tee_points=tee_points,
        # The held-out partition is untouched, so scores stay comparable.
        heldout_cmr_times=obs.heldout_cmr_times,
        heldout_cmr_vertices=obs.heldout_cmr_vertices,
        heldout_cmr_points=obs.heldout_cmr_points,
        heldout_tee_times=obs.heldout_tee_times,
        heldout_tee_vertices=obs.heldout_tee_vertices,
        heldout_tee_points=obs.heldout_tee_points,
    )
    return JointProblem(
        observations=thinned, truth=problem.truth, model=problem.model,
        config=problem.config, cmr_noise_mm=noise_mm, tee_noise_mm=noise_mm,
    )


rows = []
paths = sample_paths(ROOT, "test")[:N_SAMPLES]
t0 = time.time()
for index, path in enumerate(paths, 1):
    for label, keep, noise in LEVELS:
        for name, cfg in ARMS.items():
            problem = thin(benchmark_problem(path, model, cfg), keep, noise, index)
            fit = fit_joint_scene(problem.observations, model, cfg)
            errors = fit.heldout_errors()
            pose = pose_errors(fit.parameters, problem.truth)
            rows.append({
                "arm": name, "level": label, "keep": keep, "noise_mm": noise,
                "sample": path.stem,
                "heldout_cmr_mm": errors["heldout_cmr_mm"],
                "heldout_tee_mm": errors["heldout_tee_mm"],
                "translation_mm": pose["translation_mm"],
                "rotation_deg": pose["rotation_deg"],
                "visible_cmr_frames": int(len(problem.observations.cmr_times)),
                "visible_tee_vertices": int(len(problem.observations.tee_vertices)),
            })
    print("  %d/%d  %.0fs" % (index, len(paths), time.time() - t0), flush=True)

KEYS = ("heldout_cmr_mm", "heldout_tee_mm", "translation_mm", "rotation_deg")


def median_of(arm, level, key):
    values = [r[key] for r in rows
              if r["arm"] == arm and r["level"] == level and np.isfinite(r[key])]
    return float(np.median(values)) if values else float("nan")


def paired(level, key):
    subset = {r["arm"]: {} for r in rows}
    for r in rows:
        if r["level"] == level:
            subset.setdefault(r["arm"], {})[r["sample"]] = r[key]
    samples = sorted(subset["evidence_aware"])
    x = np.array([subset["evidence_aware"][s] for s in samples], dtype=float)
    y = np.array([subset["uniform_prior"][s] for s in samples], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    diff = (x - y)[mask]
    rng = np.random.default_rng(0)
    obs = float(diff.mean())
    null = (rng.choice([-1.0, 1.0], size=(20000, diff.size)) * diff).mean(axis=1)
    return {"difference": obs, "p": float((np.abs(null) >= abs(obs)).mean()),
            "n": int(diff.size)}


table = {
    label: {
        "visible_cmr_frames": int(np.median(
            [r["visible_cmr_frames"] for r in rows if r["level"] == label])),
        "visible_tee_vertices": int(np.median(
            [r["visible_tee_vertices"] for r in rows if r["level"] == label])),
        "arms": {arm: {k: median_of(arm, label, k) for k in KEYS} for arm in ARMS},
        "paired_evidence_minus_uniform": {k: paired(label, k) for k in KEYS},
    }
    for label, _, _ in LEVELS
}
json.dump({
    "experiment": "E29_prior_under_scarcity",
    "benchmark": str(ROOT), "split": "test", "n_modes": N_MODES,
    "n_samples": len(paths), "levels": [list(v) for v in LEVELS],
    "by_level": table, "rows": rows,
}, open(os.environ.get("R4D_OUT", "results/benchmark") + "/E29_prior_scarcity.json", "w"), indent=2)

print("\n%-14s %7s %7s %11s %11s %13s"
      % ("level", "frames", "tee vx", "evid cmr", "unif cmr", "diff (p)"), flush=True)
for label, _, _ in LEVELS:
    entry = table[label]
    d = entry["paired_evidence_minus_uniform"]["heldout_cmr_mm"]
    print("%-14s %7d %7d %11.4f %11.4f %8.4f (%.3f)%s"
          % (label, entry["visible_cmr_frames"], entry["visible_tee_vertices"],
             entry["arms"]["evidence_aware"]["heldout_cmr_mm"],
             entry["arms"]["uniform_prior"]["heldout_cmr_mm"],
             d["difference"], d["p"], " *" if d["p"] < 0.05 else ""), flush=True)
print("\ntranslation, evidence-aware minus uniform:", flush=True)
for label, _, _ in LEVELS:
    d = table[label]["paired_evidence_minus_uniform"]["translation_mm"]
    print("  %-14s %9.4f mm  p=%.4f%s"
          % (label, d["difference"], d["p"], " *" if d["p"] < 0.05 else ""), flush=True)
print("done", flush=True)
