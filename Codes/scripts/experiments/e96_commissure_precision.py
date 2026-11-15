"""E96: how precise must the commissure annotations be? (plan v2, W2.3)

Pre-registered in ``Docs/E96_COMMISSURE_PRECISION_PREREGISTRATION_2026-09-29.md``.
E91's image-domain search with the two commissures, from the same CMR-image
stage, with the annotations degraded in the ways a clinician's can be:

* positional error: each annotated point displaced by isotropic Gaussian error
  whose RMS length is s in {0, 1, 2, 4, 8} mm, independently per frame;
* a swap: the two commissures' labels exchanged;
* a slide: one commissure labelled as the ring vertex 15 degrees away, so the
  point is right and its identity off by n/24 vertices (one at 24 nodes, four
  at E100's 96, per its amendment);
* one only: a single commissure supplied.

    R4D_ALLOW_TEST=1 python e96_commissure_precision.py        (small-session test)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import e59_unknown_alignment as e59  # noqa: E402
import e91_image_search as e91  # noqa: E402
from e70_image_evidence import commissure_landmarks  # noqa: E402
from uniformity_runner import run  # noqa: E402
from rooted4d.experiments.benchmark_eval import benchmark_problem  # noqa: E402
from rooted4d.inference.image_evidence import evidence_from_sample  # noqa: E402

PREREGISTRATION = "Docs/E96_COMMISSURE_PRECISION_PREREGISTRATION_2026-09-29.md"
BENCH = os.environ.get("R4D_BENCH", "v3")  # E100 passes a rebuilt instance
NOISE_MM = (0.0, 1.0, 2.0, 4.0, 8.0)
ARMS = tuple(f"noise_{noise:g}mm" for noise in NOISE_MM) + ("swap", "slide_one", "one_only")


def _seed(sample: str, arm: str) -> int:
    return int.from_bytes(hashlib.sha256(f"E96/{sample}/{arm}".encode()).digest()[:8], "little")


def fit_sample(path_string: str) -> dict:
    import torch
    from rooted4d.differentiable import evidence_tensors
    from rooted4d.differentiable.scene import image_residuals, landmark_residuals, prior_residuals

    state = e91.STATE
    model, point, scene, weights = (state[k] for k in ("model", "point", "scene", "weights"))
    path = Path(path_string)
    with np.load(path, allow_pickle=True) as blob:
        identity = str(blob["identity_id"])
        true_phases = np.asarray(blob["tee_true_phases"], dtype=float)
        all_times = np.asarray(blob["tee_times"], dtype=float)
        tee_index = np.asarray(blob["tee_indices"], dtype=np.int64)
        tee_points = np.asarray(blob["tee_points"], dtype=float)
    problem = benchmark_problem(path, model, point)
    record = evidence_from_sample(path, model, budget=200, n_cmr_frames=4, n_tee_frames=4)
    frames = np.searchsorted(all_times, record.tee_times)
    phases = true_phases[frames]
    evidence = evidence_tensors(record, e91.W_CMR, e91.W_TEE)
    evidence["shadow_tolerance"] = 0.1
    n = len(state["annulus"])
    available = np.intersect1d(state["annulus"], tee_index)
    chosen = commissure_landmarks(available, n)
    row = {"sample": path.name, "identity": identity, "variants": {}, "commissures": [int(v) for v in chosen]}
    if len(chosen) != 2:
        row["failure_reason"] = "fewer than two commissure vertices observed"
        return row
    points = tee_points[np.ix_(frames, np.searchsorted(tee_index, chosen))]  # (frames, 2, 3)

    def objective(ev):
        return lambda v: (torch.cat([image_residuals(v, scene, ev), landmark_residuals(v, scene, ev),
                                     prior_residuals(v, scene, weights)]) ** 2).sum()

    def with_landmarks(vertices, landmark_points):
        ev = dict(evidence)
        ev.update(landmark_vertices=torch.as_tensor(np.asarray(vertices), dtype=torch.int64),
                  landmark_points=torch.as_tensor(landmark_points, dtype=torch.float64),
                  landmark_times=torch.as_tensor(record.tee_times, dtype=torch.float64),
                  landmark_weight=e91.LANDMARK_WEIGHT)
        return ev

    cmr_only = dict(evidence)
    cmr_only["tee_values"] = None
    started = time.perf_counter()
    cmr_start = e91._lbfgs(torch.as_tensor(state["packed"].copy()), objective(cmr_only), iterations=e91.FINAL)
    row["cmr_stage_seconds"] = round(time.perf_counter() - started, 3)
    wanted = e91.selected_arms(ARMS)  # E120: R4D_ARMS runs only these arms (see e91.selected_arms)
    if wanted is not None:
        row["cmr_start_sha256"] = e91.digest(cmr_start)

    def arm(name, vertices, landmark_points):
        if wanted is not None and name not in wanted:
            return
        # Contiguous copies: a reversed view (the swap) cannot become a tensor.
        vertices = np.array(vertices, dtype=np.int64)
        landmark_points = np.ascontiguousarray(landmark_points)
        started = time.perf_counter()
        try:
            ev = with_landmarks(vertices, landmark_points)
            outcome = e91.search_fit(objective(ev), {"vertices": np.asarray(vertices), "points": landmark_points},
                                     problem, record.tee_times, phases, cmr_start)
            outcome["within_5deg"] = bool(outcome["rotation_deg"] <= 5.0)
        except Exception as error:  # a failed arm is recorded, never fatal
            outcome = {"succeeded": False, "failure_reason": f"{type(error).__name__}: {error}"}
        outcome["seconds"] = round(time.perf_counter() - started, 3)
        row["variants"][name] = outcome

    for noise in NOISE_MM:
        rng = np.random.default_rng(_seed(path.name, f"noise{noise:g}"))
        # Isotropic, with RMS length ``noise``: each axis has SD noise / sqrt(3).
        perturbed = points + rng.normal(0.0, noise / np.sqrt(3.0), size=points.shape)
        arm(f"noise_{noise:g}mm", chosen, perturbed)
    arm("swap", chosen[::-1], points)
    # One commissure's identity slid 15 degrees along the ring (n/24 vertices), away from the other.
    slid = np.array([(chosen[0] - max(1, n // 24)) % n, chosen[1]])
    arm("slide_one", slid, points)
    arm("one_only", chosen[:1], points[:, :1, :])
    return row


def main() -> int:
    spec = json.loads(Path(e91.SPECS[BENCH]).read_text())
    joint = next(v for v in spec["variants"] if v["name"] == "S/joint")
    smoke = os.environ.get("R4D_SMOKE") == "1"  # code-path check only: validation, one sample
    split = "val" if smoke else "test"
    paths = e59.sample_paths(Path(spec["benchmark"]), split)
    if smoke:
        paths = paths[:1]
    return run(f"E96_{BENCH}_{split}" + ("_smoke" if smoke else ""), split, paths, fit_sample, e91.initialise,
               (spec["models"]["S"], spec["base_config"], joint["overrides"], spec["benchmark"]),
               extra={"experiment": "E96", "preregistration": PREREGISTRATION, "benchmark": spec["benchmark"],
                      "noise_mm": NOISE_MM, "search": "E91 (amended)"})


if __name__ == "__main__":
    sys.exit(main())
