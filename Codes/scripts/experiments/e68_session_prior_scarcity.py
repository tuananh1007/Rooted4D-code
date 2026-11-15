"""E68: the session-prior weight under scarce evidence, in the frozen structured configuration.

Pre-registered in ``Docs/E65_E72_UNIFORMITY_PREREGISTRATION_2026-09-24.md``. On
the v4 test split, the frozen S configuration (session prior weight 12) is
compared with the same configuration at session prior weight 1, while the
visible evidence is thinned. That is the one difference E29 actually measured
under the name "evidence-aware prior"; here it is named for what it is.

Thinning retains a fraction of the visible CMR frames and TEE vertices, drawn
with the sample's own seed and the level, and the last level adds 1.5 mm of
noise. The held-out mask is untouched, so every level is scored on the same
hidden evidence. The input-derived start is computed from what remains visible.

    R4D_ALLOW_TEST=1 python e68_session_prior_scarcity.py

E83 repeats it after E75 changed the frozen weight: ``R4D_EXPERIMENT=E83
R4D_FROZEN_SESSION_PRIOR=0.015625``. E100 adds ``R4D_BENCH=<key>`` and the
instance's selected weight. The defaults are E68's.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import e59_unknown_alignment as e59  # noqa: E402
from uniformity_runner import frozen_s, run  # noqa: E402
from rooted4d.experiments.benchmark_eval import benchmark_problem  # noqa: E402
from rooted4d.inference import SolverConfig  # noqa: E402
from rooted4d.inference.baselines import phase_map_error  # noqa: E402
from rooted4d.inference.initialization import input_derived_alignment, start_from  # noqa: E402
from rooted4d.inference.joint import Observations, fit_joint_scene  # noqa: E402
from rooted4d.inference.problem import pose_errors  # noqa: E402

BENCH = os.environ.get("R4D_BENCH", "v4")  # E100 passes a rebuilt instance
LEVELS = (("full", 1.00, 0.0), ("half", 0.50, 0.0), ("quarter", 0.25, 0.0),
          ("eighth", 0.125, 0.0), ("eighth_noisy", 0.125, 1.5))
EXPERIMENT = os.environ.get("R4D_EXPERIMENT", "E68")
FROZEN_SESSION_PRIOR = float(os.environ.get("R4D_FROZEN_SESSION_PRIOR", 12.0))
#: The comparison arm is beta_s = 1 (E68, E83). If the frozen weight is itself 1,
#: it is the first evaluation's fixed 12 instead (E100's amendment).
COMPARISON_SESSION_PRIOR = 12.0 if FROZEN_SESSION_PRIOR == 1.0 else 1.0
ARMS = {"frozen": FROZEN_SESSION_PRIOR, f"session_prior_{COMPARISON_SESSION_PRIOR:g}": COMPARISON_SESSION_PRIOR}
STATE: dict = {}


def initialise(recipe, base, weight, root):
    STATE["model"] = e59.build_model(Path(root), recipe)[0]
    STATE["config"] = SolverConfig(**{**base, "identity_prior_weight": weight})
    if os.environ.get("R4D_SMOKE") == "1":  # code-path check only, never a result
        STATE["config"] = replace(STATE["config"], max_iterations=2)


def thin(observations: Observations, keep: float, noise_mm: float, seed: int) -> Observations:
    """E29's thinning, with the draw seeded by the sample and level rather than run order."""
    rng = np.random.default_rng(seed)
    n_frames = max(2, int(round(keep * len(observations.cmr_times))))
    frames = np.sort(rng.choice(len(observations.cmr_times), n_frames, replace=False))
    n_vertices = max(6, int(round(keep * len(observations.tee_vertices))))
    columns = np.sort(rng.choice(len(observations.tee_vertices), n_vertices, replace=False))
    cmr_points = observations.cmr_points[frames]
    tee_points = observations.tee_points[:, columns, :]
    if noise_mm > 0.0:
        cmr_points = cmr_points + rng.normal(0.0, noise_mm, cmr_points.shape)
        tee_points = tee_points + rng.normal(0.0, noise_mm, tee_points.shape)
    return replace(observations, cmr_times=observations.cmr_times[frames], cmr_points=cmr_points,
                   tee_vertices=observations.tee_vertices[columns], tee_points=tee_points)


def fit_sample(path_string: str) -> dict:
    model, base = STATE["model"], STATE["config"]
    path = Path(path_string)
    with np.load(path, allow_pickle=True) as blob:
        identity = str(blob["identity_id"])
        sample_seed = int(blob["seed__sample"]) % (2**31)
    problem = benchmark_problem(path, model, base)
    row = {"sample": path.name, "identity": identity, "variants": {}}
    for index, (level, keep, noise) in enumerate(LEVELS):
        observations = (problem.observations if keep == 1.0 and noise == 0.0
                        else thin(problem.observations, keep, noise, sample_seed * 16 + index))
        thinned = replace(problem, observations=observations)
        # Heavy thinning can leave fewer than four vertices both modalities
        # observe, where the input-derived start is undefined. Both arms then
        # start from the zero start, so they still share one start (amendment in
        # the pre-registration, made before any test run).
        try:
            alignment = input_derived_alignment(observations)
        except ValueError:
            alignment = None
        for arm, session_weight in ARMS.items():
            config = replace(base, session_prior_weight=session_weight)
            start = None if alignment is None else start_from(alignment, model, config)
            fit = fit_joint_scene(observations, model, config, start=start)
            outcome = dict(fit.heldout_errors())
            outcome.update(pose_errors(fit.parameters, problem.truth))
            outcome["phase_map_cycles"] = phase_map_error(fit.parameters, thinned)
            outcome["succeeded"] = bool(np.isfinite(fit.cost))
            outcome["visible_cmr_frames"] = int(len(observations.cmr_times))
            outcome["visible_tee_vertices"] = int(len(observations.tee_vertices))
            outcome["start"] = "zero" if alignment is None else "input_derived"
            row["variants"][f"{level}/{arm}"] = outcome
    return row


def main() -> int:
    recipe, base, weight, root = frozen_s(BENCH)
    split = "val" if os.environ.get("R4D_SMOKE") == "1" else "test"  # smoke never reads test
    paths = e59.sample_paths(Path(root), split)
    return run(f"{EXPERIMENT}_{BENCH}_{split}", split, paths, fit_sample, initialise, (recipe, base, weight, root),
               extra={"experiment": EXPERIMENT, "benchmark": root, "levels": LEVELS, "arms": ARMS,
                      "recipe": recipe, "base_config": base, "identity_prior_weight": weight})


if __name__ == "__main__":
    sys.exit(main())
