"""E69: the frozen structured fits on v3 validation and test, kept for calibration and warm starts.

Pre-registered in ``Docs/E65_E72_UNIFORMITY_PREREGISTRATION_2026-09-24.md``. Each
sample is fitted in the frozen S configuration from the input-derived start,
exactly as E64 fitted it, and three things are recorded:

* the errors E64 recorded, for the reproduction check against E64's rows;
* the packed parameter vector, the warm start of E70 and E71;
* the four calibration scalars, predicted and true: phase offset, translation
  norm, rotation angle (from the rotation matrix, so it is in the principal
  branch), and the annulus diameter of the static identity geometry.

    R4D_SPLIT=val|test [R4D_ALLOW_TEST=1] python e69_frozen_fits.py
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
from rooted4d.geometry import RigidTransform  # noqa: E402
from rooted4d.inference import SolverConfig  # noqa: E402
from rooted4d.inference.baselines import phase_map_error  # noqa: E402
from rooted4d.inference.initialization import input_derived_alignment, start_from  # noqa: E402
from rooted4d.inference.joint import _pack, fit_joint_scene  # noqa: E402
from rooted4d.inference.problem import pose_errors  # noqa: E402
from rooted4d.motion import circular_distance  # noqa: E402

BENCH = os.environ.get("R4D_BENCH", "v3")  # E100 passes a rebuilt instance
STATE: dict = {}


def initialise(recipe, base, weight, root):
    STATE["model"] = e59.build_model(Path(root), recipe)[0]
    STATE["config"] = SolverConfig(**{**base, "identity_prior_weight": weight})
    if os.environ.get("R4D_SMOKE") == "1":  # code-path check only, never a result
        STATE["config"] = replace(STATE["config"], max_iterations=2)


def diameter(points: np.ndarray) -> float:
    """Twice the mean distance of the ring vertices from their centroid."""
    centre = points.mean(axis=0)
    return float(2.0 * np.linalg.norm(points - centre, axis=1).mean())


def angle_deg(axis_angle: np.ndarray) -> float:
    rotation = RigidTransform.from_axis_angle(np.asarray(axis_angle), np.zeros(3)).rotation
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def fit_sample(path_string: str) -> dict:
    model, config = STATE["model"], STATE["config"]
    path = Path(path_string)
    with np.load(path, allow_pickle=True) as blob:
        identity = str(blob["identity_id"])
        true_template = np.asarray(blob["template_vertices"], dtype=np.float64)
    problem = benchmark_problem(path, model, config)
    alignment = input_derived_alignment(problem.observations)
    fit = fit_joint_scene(problem.observations, model, config,
                          start=start_from(alignment, model, config))
    truth, estimate = problem.truth, fit.parameters
    ring = np.sort(model.groups["annulus"])
    static = model.physical_coordinates(estimate.identity)
    errors = dict(fit.heldout_errors())
    errors.update(pose_errors(estimate, truth))
    errors["phase_error_cycles"] = float(circular_distance(
        np.array([estimate.phase_offset]), np.array([truth.phase_offset]))[0])
    errors["phase_map_cycles"] = phase_map_error(estimate, problem)
    errors["attachment_residual_mm"] = float(fit.attachment_residual)
    return {
        "sample": path.name, "identity": identity, "errors": errors,
        "scalars": {
            "phase_offset_cycles": [float(estimate.phase_offset % 1.0), float(truth.phase_offset % 1.0)],
            "translation_norm_mm": [float(np.linalg.norm(estimate.pose_translation)),
                                    float(np.linalg.norm(truth.pose_translation))],
            "rotation_angle_deg": [angle_deg(estimate.pose_axis_angle), angle_deg(truth.pose_axis_angle)],
            "annulus_diameter_mm": [diameter(static[ring]), diameter(true_template[ring])],
        },
        "packed": _pack(estimate, model, config),
        "cost": float(fit.cost), "iterations": int(fit.iterations),
    }


def main() -> int:
    split = os.environ.get("R4D_SPLIT", "val")
    recipe, base, weight, root = frozen_s(BENCH)
    paths = e59.sample_paths(Path(root), split)
    return run(f"E69_{BENCH}_{split}", split, paths, fit_sample, initialise,
               (recipe, base, weight, root),
               extra={"experiment": "E69", "benchmark": root, "split": split, "recipe": recipe,
                      "base_config": base, "identity_prior_weight": weight})


if __name__ == "__main__":
    sys.exit(main())
