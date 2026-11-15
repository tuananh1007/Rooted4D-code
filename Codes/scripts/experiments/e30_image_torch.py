"""E30: image-domain fitting once the Jacobian stops being the bottleneck.

E23 and E27 fitted image evidence with the NumPy Levenberg-Marquardt solver and
central-difference Jacobians. That cost 486 s per sample, forced twelve
iterations and six samples, and produced 33.7 degrees and 39.6 mm -- far worse
than point evidence. Two things were tangled in that result: whether image
evidence is worse, and whether twelve Gauss-Newton steps on a narrow-banded
non-convex objective are enough to find out.

The differentiable port separates them. A central-difference Jacobian needs 102
residual evaluations, forward-mode autodiff needs 27, and a gradient needs less
than one. A first-order method therefore takes hundreds of steps for the price
of one differenced Gauss-Newton step, and that is what is run here.

Arms, all from the same cold start, all scored on the same held-out mask:

* ``point_lm``     - point evidence, the NumPy solver, the existing baseline;
* ``image_lbfgs``  - image evidence only, torch gradients, L-BFGS;
* ``image_from_point`` - the same, warm-started from the point fit, which is what
  a pipeline with a segmentation front end would actually do.
"""

import glob
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rooted4d.differentiable import (
    TorchScene,
    evidence_tensors,
    fit_image_lbfgs,
    prior_weights,
)
from rooted4d.experiments.benchmark_eval import (
    benchmark_problem,
    fit_benchmark_model,
    sample_paths,
)
from rooted4d.inference.image_evidence import evidence_from_sample, noise_weights
from rooted4d.inference.joint import (
    SolverConfig,
    _pack,
    _unpack,
    fit_joint_scene,
    initial_parameters,
)
from rooted4d.inference.problem import pose_errors
from rooted4d.motion import circular_distance

ROOT = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3"))
N_MODES = int(os.environ.get("R4D_MODES", 12))
N_SAMPLES = int(os.environ.get("R4D_SAMPLES", 20))
LBFGS_ITERS = int(os.environ.get("R4D_ITERS", 250))
SHADOW_TOLERANCE = 0.1
W_CMR, W_TEE = noise_weights(0.015, 0.12)

print("fitting shape space...", flush=True)
model, _ = fit_benchmark_model(ROOT, n_modes=N_MODES, n_motion=4, n_session=2)
POINT = SolverConfig(fourier_order=2, session_order=1, motion_prior_weight=1.0,
                     max_iterations=12)
IMAGE = SolverConfig(fourier_order=2, session_order=1, motion_prior_weight=1.0,
                     use_cmr=False, use_tee=False,
                     cmr_image_weight=W_CMR, tee_image_weight=W_TEE, max_iterations=12)
scene = TorchScene.from_model(model, IMAGE)
weights = prior_weights(IMAGE)

rows = []
paths = sample_paths(ROOT, "test")[:N_SAMPLES]
# E120: R4D_SHARD=k/n fits every n-th sample from the k-th and writes only its
# rows; R4D_MERGE=<glob> fits nothing and summarizes the rows those shards wrote,
# in sample order. Unset, nothing changes.
SHARD, MERGE = os.environ.get("R4D_SHARD"), os.environ.get("R4D_MERGE")
fitted = list(paths)
if SHARD:
    k, n = (int(v) for v in SHARD.split("/"))
    fitted = fitted[k::n]
if MERGE:
    order = {p.stem: i for i, p in enumerate(paths)}
    rows = sorted((row for name in sorted(glob.glob(MERGE)) for row in json.load(open(name))["rows"]),
                  key=lambda row: order[row["sample"]])
    per_sample = Counter(row["sample"] for row in rows)
    if set(per_sample) != set(order) or set(per_sample.values()) != {3}:
        raise SystemExit("the shards do not hold every sample's rows exactly once")
    fitted = []
t0 = time.time()
for index, path in enumerate(fitted, 1):
    ev = evidence_from_sample(path, model, budget=200, n_cmr_frames=4, n_tee_frames=4)
    tensors = evidence_tensors(ev, W_CMR, W_TEE)
    # One-sided TEE likelihood. Acoustic shadowing removes signal and never adds
    # it, so predicting more echo than was observed is weaker evidence of bad
    # geometry than predicting less. Symmetric weighting let the unmodelled
    # shadow drag the surface away from the anatomy: warm-started from a 4.7
    # degree pose the fit ended at 149 degrees. This needs no knowledge of where
    # the shadow falls, which reading it from the generator seed would.
    tensors["shadow_tolerance"] = SHADOW_TOLERANCE
    problem = benchmark_problem(path, model, POINT)
    with np.load(path, allow_pickle=True) as blob:
        true_phases = np.asarray(blob["tee_true_phases"], dtype=float)
        all_times = np.asarray(blob["tee_times"], dtype=float)
    frames = np.searchsorted(all_times, ev.tee_times)

    started = time.time()
    point_fit = fit_joint_scene(problem.observations, model, POINT,
                                phase_offset_candidates=6)
    point_seconds = time.time() - started

    cold = torch.as_tensor(_pack(initial_parameters(model, IMAGE), model, IMAGE))
    warm = torch.as_tensor(_pack(point_fit.parameters, model, IMAGE))

    results = {"point_lm": (point_fit.parameters, point_seconds)}
    for name, start in (("image_lbfgs", cold), ("image_from_point", warm)):
        started = time.time()
        vector, history = fit_image_lbfgs(start, scene, tensors, weights,
                                          iterations=LBFGS_ITERS)
        results[name] = (
            _unpack(vector.numpy(), problem.truth, model, IMAGE),
            time.time() - started,
        )

    for name, (params, seconds) in results.items():
        errors = pose_errors(params, problem.truth)
        mapped = params.warp().evaluate(ev.tee_times)
        rows.append({
            "arm": name, "sample": path.stem, "seconds": seconds,
            "rotation_deg": errors["rotation_deg"],
            "translation_mm": errors["translation_mm"],
            "phase_map_cycles": float(np.median(
                circular_distance(np.asarray(mapped, dtype=float), true_phases[frames])
            )),
        })
    print("  %d/%d  %.0fs" % (index, len(fitted), time.time() - t0), flush=True)

if SHARD:
    json.dump({"shard": SHARD, "rows": rows}, open(os.environ.get("R4D_OUT", "results/benchmark") + "/E30_image_torch_shard_%03d_of_%03d.json" % (k, n), "w"), indent=2)
    raise SystemExit(0)


KEYS = ("rotation_deg", "translation_mm", "phase_map_cycles", "seconds")
ARMS = ("point_lm", "image_lbfgs", "image_from_point")


def summarize(arm, key):
    values = np.array([r[key] for r in rows if r["arm"] == arm and np.isfinite(r[key])])
    return {"n": int(values.size), "median": float(np.median(values)),
            "mean": float(values.mean()), "p95": float(np.percentile(values, 95))}


per_arm = {arm: {k: summarize(arm, k) for k in KEYS} for arm in ARMS}
by = {}
for row in rows:
    by.setdefault(row["arm"], {})[row["sample"]] = row
samples = sorted(by["point_lm"])
paired = {}
for arm in ARMS[1:]:
    paired[arm] = {}
    for key in KEYS[:3]:
        x = np.array([by[arm][s][key] for s in samples], dtype=float)
        y = np.array([by["point_lm"][s][key] for s in samples], dtype=float)
        diff = x - y
        rng = np.random.default_rng(0)
        obs = float(diff.mean())
        null = (rng.choice([-1.0, 1.0], size=(20000, diff.size)) * diff).mean(axis=1)
        paired[arm][key] = {"median_minus_point": float(np.median(diff)),
                            "mean_minus_point": obs,
                            "p": float((np.abs(null) >= abs(obs)).mean())}

json.dump({
    "experiment": "E30_image_domain_with_exact_gradients",
    "benchmark": str(ROOT), "split": "test",
    "n_modes": N_MODES, "n_samples": len(paths), "lbfgs_iterations": LBFGS_ITERS, "shadow_tolerance": SHADOW_TOLERANCE,
    "device": "cpu", "supersedes": "E23 and E27, which were optimizer-limited",
    "per_arm": per_arm, "paired_vs_point": paired, "rows": rows,
}, open(os.environ.get("R4D_OUT", "results/benchmark") + "/E30_image_torch.json", "w"), indent=2)

print("\n%-20s %13s %15s %14s %10s"
      % ("arm", "rotation deg", "translation mm", "phase cycles", "seconds"), flush=True)
for arm in ARMS:
    v = per_arm[arm]
    print("%-20s %13.3f %15.3f %14.4f %10.1f"
          % (arm, v["rotation_deg"]["median"], v["translation_mm"]["median"],
             v["phase_map_cycles"]["median"], v["seconds"]["median"]), flush=True)
print("\nmedian difference from point_lm (negative favours the image arm)", flush=True)
for arm, table in paired.items():
    for key, entry in table.items():
        print("  %-20s %-18s %9.4f  p=%.4f%s"
              % (arm, key, entry["median_minus_point"], entry["p"],
                 " *" if entry["p"] < 0.05 else ""), flush=True)
print("done", flush=True)
