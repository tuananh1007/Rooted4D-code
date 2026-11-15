"""E32: does a handful of correspondences rescue image-domain fitting?

Section 19 measured the reason image evidence cannot recover pose. A surface
likelihood is correspondence-free, so rotating a near-axisymmetric annulus about
its own axis reproduces very nearly the same surface: the image objective changes
1 percent under a 180-degree rotation where point evidence changes by a factor of
189. Section 21 sharpened it at full scale -- an arm warm-started from a 2.2
degree pose ends at 67.6 degrees while its translation moves by 0.29 mm, so the
damage sits in one unconstrained degree of freedom and nowhere else.

That diagnosis makes a prediction that can be wrong: supplying a small number of
points whose *identity* is known should collapse the rotation error, because
identity is exactly the information a surface likelihood lacks. Two landmarks are
the minimum that breaks a rotational symmetry, so if the diagnosis is right, two
should already do most of the work and adding more should help little.

The landmarks are annular vertices, evenly spaced so they span the ring. This is
realistic rather than a convenience: the anterolateral and posteromedial
commissures are standard annotations on the mitral annulus and the production
template specification already lists them among the frozen landmarks.

Every arm is warm-started from the point fit, which is the configuration section
21 showed failing, so the comparison isolates the landmark term.
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
from rooted4d.geometry import make_toy_complex
from rooted4d.inference.image_evidence import evidence_from_sample, noise_weights
from rooted4d.inference.joint import SolverConfig, _pack, _unpack, fit_joint_scene
from rooted4d.inference.problem import pose_errors
from rooted4d.motion import circular_distance

ROOT = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3"))
OUT = os.environ.get("R4D_OUT", "results/benchmark")
N_MODES = int(os.environ.get("R4D_MODES", 32))
N_SAMPLES = int(os.environ.get("R4D_SAMPLES", 30))
LBFGS_ITERS = int(os.environ.get("R4D_ITERS", 400))
SHADOW_TOLERANCE = 0.1
LANDMARK_COUNTS = (0, 2, 4, 8)
W_CMR, W_TEE = noise_weights(0.015, 0.12)
# Landmarks are millimetres with the same noise as any TEE point observation, so
# they are weighted like one rather than by a number chosen to make them work.
LANDMARK_WEIGHT = 1.0 / 0.45**2

print("fitting shape space...", flush=True)
model, _ = fit_benchmark_model(ROOT, n_modes=N_MODES, n_motion=4, n_session=2)
POINT = SolverConfig(fourier_order=2, session_order=1, motion_prior_weight=1.0,
                     max_iterations=12)
IMAGE = SolverConfig(fourier_order=2, session_order=1, motion_prior_weight=1.0,
                     use_cmr=False, use_tee=False,
                     cmr_image_weight=W_CMR, tee_image_weight=W_TEE, max_iterations=12)
scene = TorchScene.from_model(model, IMAGE)
weights = prior_weights(IMAGE)
annulus = np.sort(make_toy_complex(n_annulus=24).groups["annulus"])

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
    if set(per_sample) != set(order) or set(per_sample.values()) != {len(LANDMARK_COUNTS)}:
        raise SystemExit("the shards do not hold every sample's rows exactly once")
    fitted = []
t0 = time.time()
for index, path in enumerate(fitted, 1):
    ev = evidence_from_sample(path, model, budget=200, n_cmr_frames=4, n_tee_frames=4)
    base = evidence_tensors(ev, W_CMR, W_TEE)
    base["shadow_tolerance"] = SHADOW_TOLERANCE
    problem = benchmark_problem(path, model, POINT)
    with np.load(path, allow_pickle=True) as blob:
        true_phases = np.asarray(blob["tee_true_phases"], dtype=float)
        all_times = np.asarray(blob["tee_times"], dtype=float)
        tee_index = np.asarray(blob["tee_indices"], dtype=np.int64)
        tee_points = np.asarray(blob["tee_points"], dtype=float)
    frames = np.searchsorted(all_times, ev.tee_times)

    point_fit = fit_joint_scene(problem.observations, model, POINT,
                                phase_offset_candidates=6)
    warm = torch.as_tensor(_pack(point_fit.parameters, model, IMAGE))
    reference = pose_errors(point_fit.parameters, problem.truth)

    # Annular vertices that survived dropout, so a landmark is something the
    # acquisition actually saw.
    available = np.intersect1d(annulus, tee_index)
    for count in LANDMARK_COUNTS:
        evidence = dict(base)
        if count > 0 and len(available) >= count:
            chosen = available[np.linspace(0, len(available) - 1, count).round().astype(int)]
            chosen = np.unique(chosen)
            columns = np.searchsorted(tee_index, chosen)
            evidence["landmark_vertices"] = torch.as_tensor(chosen, dtype=torch.int64)
            evidence["landmark_points"] = torch.as_tensor(
                tee_points[np.ix_(frames, columns)], dtype=torch.float64
            )
            evidence["landmark_times"] = torch.as_tensor(ev.tee_times, dtype=torch.float64)
            evidence["landmark_weight"] = LANDMARK_WEIGHT
            used = int(len(chosen))
        else:
            used = 0
        started = time.time()
        vector, _ = fit_image_lbfgs(warm, scene, evidence, weights,
                                    iterations=LBFGS_ITERS)
        params = _unpack(vector.numpy(), problem.truth, model, IMAGE)
        errors = pose_errors(params, problem.truth)
        mapped = params.warp().evaluate(ev.tee_times)
        rows.append({
            "landmarks": used, "sample": path.stem,
            "rotation_deg": errors["rotation_deg"],
            "translation_mm": errors["translation_mm"],
            "phase_map_cycles": float(np.median(
                circular_distance(np.asarray(mapped, dtype=float), true_phases[frames])
            )),
            "point_rotation_deg": reference["rotation_deg"],
            "point_translation_mm": reference["translation_mm"],
            "seconds": time.time() - started,
        })
    print("  %d/%d  %.0fs" % (index, len(fitted), time.time() - t0), flush=True)

if SHARD:
    json.dump({"shard": SHARD, "rows": rows}, open(OUT + "/E32_image_landmarks_shard_%03d_of_%03d.json" % (k, n), "w"), indent=2)
    raise SystemExit(0)


KEYS = ("rotation_deg", "translation_mm", "phase_map_cycles")


def median_of(count, key):
    values = [r[key] for r in rows if r["landmarks"] == count and np.isfinite(r[key])]
    return float(np.median(values)) if values else float("nan")


table = {
    str(count): {k: median_of(count, k) for k in KEYS} for count in LANDMARK_COUNTS
}
reference = {
    "point_rotation_deg": float(np.median([r["point_rotation_deg"] for r in rows])),
    "point_translation_mm": float(np.median([r["point_translation_mm"] for r in rows])),
}
paired = {}
for count in LANDMARK_COUNTS[1:]:
    paired[str(count)] = {}
    for key in KEYS:
        by = {}
        for r in rows:
            by.setdefault(r["landmarks"], {})[r["sample"]] = r[key]
        samples = sorted(by[0])
        x = np.array([by[count][s] for s in samples], dtype=float)
        y = np.array([by[0][s] for s in samples], dtype=float)
        diff = x - y
        rng = np.random.default_rng(0)
        obs = float(diff.mean())
        null = (rng.choice([-1.0, 1.0], size=(20000, diff.size)) * diff).mean(axis=1)
        paired[str(count)][key] = {
            "median_minus_none": float(np.median(diff)),
            "p": float((np.abs(null) >= abs(obs)).mean()),
        }

json.dump({
    "experiment": "E32_image_with_landmarks",
    "benchmark": str(ROOT), "split": "test", "n_modes": N_MODES,
    "n_samples": len(paths), "lbfgs_iterations": LBFGS_ITERS,
    "shadow_tolerance": SHADOW_TOLERANCE, "landmark_weight": LANDMARK_WEIGHT,
    "start": "warm, from the point-evidence fit",
    "by_landmark_count": table, "point_reference": reference,
    "paired_vs_no_landmarks": paired, "rows": rows,
}, open(OUT + "/E32_image_landmarks.json", "w"), indent=2)

print("\nwarm-started image fit, by number of landmarks with known identity", flush=True)
print("%12s %14s %16s %14s" % ("landmarks", "rotation deg", "translation mm", "phase cycles"),
      flush=True)
for count in LANDMARK_COUNTS:
    e = table[str(count)]
    print("%12d %14.3f %16.3f %14.4f"
          % (count, e["rotation_deg"], e["translation_mm"], e["phase_map_cycles"]), flush=True)
print("%12s %14.3f %16.3f %14s"
      % ("point ref", reference["point_rotation_deg"],
         reference["point_translation_mm"], "-"), flush=True)
print("\nmedian difference from no landmarks:", flush=True)
for count, table_ in paired.items():
    for key, entry in table_.items():
        print("  %2s landmarks %-18s %+10.4f  p=%.4f%s"
              % (count, key, entry["median_minus_none"], entry["p"],
                 " *" if entry["p"] < 0.05 else ""), flush=True)
print("done", flush=True)
