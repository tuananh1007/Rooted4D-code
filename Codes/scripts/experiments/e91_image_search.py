"""E91: the image-domain cold start by search (plan v2, W1.4).

Pre-registered in ``Docs/E91_IMAGE_SEARCH_PREREGISTRATION_2026-09-28.md``.
E74 fitted image evidence from the uninformed start by optimization alone and
landed in the wrong rotation basin (median 113 degrees); commissures fixed the
rotation but not translation or phase. Here the start is found by search: the
start pose rotated about the model's annular axis (36 values) crossed with 24
phase offsets, scored by forward evaluation; the five best are refined with
translation and warp shape held, and the best is then released entirely.

    R4D_BENCH=v3|v4 R4D_ALLOW_TEST=1 python e91_image_search.py
    R4D_BENCH=v3 R4D_SMOKE=1 python e91_image_search.py      (validation, 2 iterations: code path only)
    R4D_ARMS=image_search,...                               (E120: only these arms; see selected_arms)

Runs through ``uniformity_runner.run`` (resumable, sharded, spawn pool), as E70.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import e59_unknown_alignment as e59  # noqa: E402
from e70_image_evidence import commissure_landmarks  # noqa: E402
from uniformity_runner import run  # noqa: E402
from rooted4d.experiments.benchmark_eval import benchmark_problem  # noqa: E402
from rooted4d.geometry.rigid import axis_angle_to_matrix  # noqa: E402
from rooted4d.inference import SolverConfig  # noqa: E402
from rooted4d.inference.image_evidence import evidence_from_sample, noise_weights  # noqa: E402
from rooted4d.inference.joint import _pack, _unpack, initial_parameters  # noqa: E402
from rooted4d.inference.problem import pose_errors  # noqa: E402
from rooted4d.motion import circular_distance  # noqa: E402

PREREGISTRATION = "Docs/E91_IMAGE_SEARCH_PREREGISTRATION_2026-09-28.md"
#: Each benchmark's frozen test spec: its models, base settings and S/joint weights.
SPECS = {"v3": "Codes/configs/e62/v3_test.json", "v4": "Codes/configs/e75/v4_e75_test.json",
         # E100: the rebuilt primaries, at their chains' final configurations.
         "v3n96o": "Codes/configs/e100/v3n96o/v3n96o_e88_test.json",
         "v4n96o": "Codes/configs/e100/v4n96o/v4n96o_e88_test.json"}
ROTATIONS, OFFSETS, TOP, REFINE, FINAL = 36, 24, 5, 100, 400
W_CMR, W_TEE = noise_weights(0.015, 0.12)
LANDMARK_WEIGHT = 1.0 / 0.45**2
STATE: dict = {}
ARMS = ("image_search", "commissures_search", "image_uninformed", "commissures_uninformed")
SEARCH_ARMS = {"image_search", "commissures_search"}


def selected_arms(names) -> set | None:
    """The arms ``R4D_ARMS`` (comma-separated) selects, or None for all of them.

    E120 runs a sample's arms in separate processes, each writing a partial row,
    and merges the rows (``e120_merge_parts.py``), so that no process outlasts a
    one-hour job. The arms share nothing but the CMR stage, which each process
    recomputes; its digest goes into each partial row, and the merge checks that
    every process started from the same point. Unset, nothing changes.
    """
    value = os.environ.get("R4D_ARMS")
    if not value:
        return None
    chosen = set(value.split(","))
    if chosen - set(names):
        raise SystemExit(f"unknown arms in R4D_ARMS: {sorted(chosen - set(names))}")
    return chosen


def digest(tensor) -> str:
    """SHA-256 of a tensor's float64 bytes, to show that two processes computed the same start."""
    return hashlib.sha256(tensor.detach().numpy().astype("<f8").tobytes()).hexdigest()


def initialise(recipe, base, overrides, root):
    import torch

    torch.set_num_threads(1)
    from rooted4d.differentiable import TorchScene
    from rooted4d.differentiable.scene import prior_weights

    global REFINE, FINAL
    if os.environ.get("R4D_SMOKE") == "1":  # code-path check only, never a result
        REFINE, FINAL = 2, 2
    model = e59.build_model(Path(root), recipe)[0]
    point = replace(SolverConfig(**base), **overrides)
    image = replace(point, use_cmr=False, use_tee=False, cmr_image_weight=W_CMR, tee_image_weight=W_TEE)
    scene = TorchScene.from_model(model, image)
    annulus = np.sort(model.groups["annulus"])
    assert np.array_equal(annulus, np.arange(len(annulus)))
    ring = model.template_physical[model.observation_rows(annulus)]
    centre = ring.mean(axis=0)
    axis = np.linalg.svd(ring - centre)[2][2]
    # Vector layout of the packed parameters (the NumPy solver's, which TorchScene reads).
    zero = initial_parameters(model, image)
    packed = _pack(zero, model, image)
    n_before_pose = len(packed) - (3 + 3 + 1 + model.n_phase_nodes)
    layout = {"axis_angle": slice(n_before_pose, n_before_pose + 3),
              "translation": slice(n_before_pose + 3, n_before_pose + 6),
              "offset": n_before_pose + 6,
              "logits": slice(n_before_pose + 7, len(packed))}
    STATE.update(model=model, point=point, image=image, scene=scene, zero=zero, packed=packed,
                 weights=prior_weights(image, model), annulus=annulus, ring_centre=centre,
                 ring_axis=axis / np.linalg.norm(axis), layout=layout)


def _lbfgs(start, objective, free=None, iterations=FINAL):
    """E70's L-BFGS on the free entries of the vector; the rest are held."""
    import torch

    base = start.clone().detach()
    mask = torch.ones_like(base, dtype=torch.bool) if free is None else free
    variable = base[mask].clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([variable], lr=0.4, max_iter=iterations, history_size=25,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                                  tolerance_change=1e-14)

    def closure():
        optimizer.zero_grad()
        full = base.clone()
        full[mask] = variable
        cost = objective(full)
        cost.backward()
        return cost

    optimizer.step(closure)
    out = base.clone()
    out[mask] = variable.detach()
    return out


def candidates(landmarks: dict | None, base):
    """``base`` with its pose rotated about the annular axis, crossed with phase offsets.

    ``base`` carries shape and motion from the CMR-image fit: with the uninformed
    start's zero motion the anatomy is static and every phase offset scores the
    same (found in the validation code check), so the phase search needs motion.
    """
    import torch

    model, layout = STATE["model"], STATE["layout"]
    # The base's shape over the cycle: its motion has zero cycle mean, so the
    # mean anatomy is the identity's shape. The ring's axis and centre, and the
    # commissure midpoint, are taken from it rather than from the template.
    shape = model.physical_coordinates(base[: model.n_modes].numpy())
    ring = shape[model.observation_rows(STATE["annulus"])]
    centre = ring.mean(axis=0)
    axis = np.linalg.svd(ring - centre)[2][2]
    axis = axis / np.linalg.norm(axis)
    model_midpoint = None
    if landmarks is not None:
        model_midpoint = shape[model.observation_rows(landmarks["vertices"])].mean(axis=0)
        observed_midpoint = landmarks["points"].reshape(-1, 3).mean(axis=0)
    for i in range(ROTATIONS):
        angle = 2.0 * np.pi * i / ROTATIONS
        wrapped = angle - 2.0 * np.pi if angle > np.pi else angle
        rotation = axis_angle_to_matrix(axis * wrapped)
        if model_midpoint is None:
            translation = centre - rotation @ centre            # rotate about the ring's own centre
        else:
            translation = observed_midpoint - rotation @ model_midpoint
        for j in range(OFFSETS):
            vector = base.clone()
            vector[layout["axis_angle"]] = torch.as_tensor(axis * wrapped)
            vector[layout["translation"]] = torch.as_tensor(translation)
            vector[layout["offset"]] = j / OFFSETS
            yield (float(np.degrees(angle)), j / OFFSETS), vector


def _score(vector, problem, evidence_times, true_phases):
    model, image = STATE["model"], STATE["image"]
    parameters = _unpack(np.asarray(vector, dtype=np.float64), problem.truth, model, image)
    errors = pose_errors(parameters, problem.truth)
    mapped = parameters.warp().evaluate(evidence_times)
    return {"succeeded": bool(np.all(np.isfinite(vector))),
            "rotation_deg": errors["rotation_deg"], "translation_mm": errors["translation_mm"],
            "phase_map_cycles": float(np.median(circular_distance(np.asarray(mapped, float), true_phases)))}


def search_fit(cost, landmarks, problem, evidence_times, phases, base) -> dict:
    """The search start: grid, five refinements with translation and warp held, then all released."""
    import torch

    layout = STATE["layout"]
    held = torch.ones(len(STATE["packed"]), dtype=torch.bool)
    held[layout["translation"]] = False
    held[layout["logits"]] = False
    scored = []
    with torch.no_grad():
        for key, vector in candidates(landmarks, base):
            scored.append((float(cost(vector)), key, vector))
    scored.sort(key=lambda item: item[0])
    # The five best rotations, each at its best phase offset: the five best
    # candidates overall were one rotation at neighbouring offsets in the code
    # check, which would refine a single basin five times.
    best_per_rotation = {}
    for value, key, vector in scored:
        best_per_rotation.setdefault(key[0], (value, key, vector))
    distinct = sorted(best_per_rotation.values(), key=lambda item: item[0])
    refined = []
    for value, key, vector in distinct[:TOP]:
        result = _lbfgs(vector, cost, free=held, iterations=REFINE)
        refined.append((float(cost(result).detach()), key, result))
    refined.sort(key=lambda item: item[0])
    final = _lbfgs(refined[0][2], cost, iterations=FINAL)
    outcome = _score(final.numpy(), problem, evidence_times, phases)
    outcome.update(grid_best=[{"rotation_deg": k[0], "offset": k[1], "objective": v} for v, k, _ in distinct[:TOP]],
                   refined=[{"rotation_deg": k[0], "offset": k[1], "objective": v} for v, k, _ in refined],
                   final_objective=float(cost(final).detach()))
    return outcome


def uninformed_fit(cost, problem, evidence_times, phases) -> dict:
    """E74's uninformed start: 400 iterations from the zero start."""
    import torch

    final = _lbfgs(torch.as_tensor(STATE["packed"].copy()), cost, iterations=FINAL)
    outcome = _score(final.numpy(), problem, evidence_times, phases)
    outcome["final_objective"] = float(cost(final).detach())
    return outcome


def fit_sample(path_string: str) -> dict:
    import torch
    from rooted4d.differentiable import evidence_tensors
    from rooted4d.differentiable.scene import image_residuals, landmark_residuals, prior_residuals

    model, point, scene, weights, layout = (STATE[k] for k in ("model", "point", "scene", "weights", "layout"))
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
    evidence = evidence_tensors(record, W_CMR, W_TEE)
    evidence["shadow_tolerance"] = 0.1
    available = np.intersect1d(STATE["annulus"], tee_index)
    chosen = commissure_landmarks(available, len(STATE["annulus"]))
    columns = np.searchsorted(tee_index, chosen)
    landmark_points = tee_points[np.ix_(frames, columns)]
    with_landmarks = dict(evidence)
    with_landmarks.update(landmark_vertices=torch.as_tensor(chosen, dtype=torch.int64),
                          landmark_points=torch.as_tensor(landmark_points, dtype=torch.float64),
                          landmark_times=torch.as_tensor(record.tee_times, dtype=torch.float64),
                          landmark_weight=LANDMARK_WEIGHT)

    def objective(ev):
        return lambda v: (torch.cat([image_residuals(v, scene, ev), landmark_residuals(v, scene, ev),
                                     prior_residuals(v, scene, weights)]) ** 2).sum()

    row = {"sample": path.name, "identity": identity, "variants": {},
           "commissures": [int(v) for v in chosen]}
    wanted = selected_arms(ARMS)

    def arm(name, function):
        if wanted is not None and name not in wanted:
            return
        started = time.perf_counter()
        try:
            outcome = function()
        except Exception as error:  # a failed arm is recorded, never fatal
            outcome = {"succeeded": False, "failure_reason": f"{type(error).__name__}: {error}"}
        outcome["seconds"] = round(time.perf_counter() - started, 3)
        row["variants"][name] = outcome

    # Shape and motion from the CMR images alone (TEE evidence off), shared by
    # both search arms: the CMR frames are at known phases, so this is the
    # motion the TEE phase search is scored against. The uninformed arms do not
    # use it, so a process running only those (R4D_ARMS) skips it.
    cmr_start = None
    if wanted is None or wanted & SEARCH_ARMS:
        cmr_only = dict(evidence)
        cmr_only["tee_values"] = None
        started = time.perf_counter()
        cmr_start = _lbfgs(torch.as_tensor(STATE["packed"].copy()), objective(cmr_only), iterations=FINAL)
        row["cmr_stage_seconds"] = round(time.perf_counter() - started, 3)
        if wanted is not None:
            row["cmr_start_sha256"] = digest(cmr_start)

    def search(ev, landmarks):
        return search_fit(objective(ev), landmarks, problem, record.tee_times, phases, cmr_start)

    def uninformed(ev):
        return uninformed_fit(objective(ev), problem, record.tee_times, phases)

    enough = len(chosen) == 2
    arm("image_search", lambda: search(evidence, None))
    if enough:
        arm("commissures_search", lambda: search(with_landmarks, {"vertices": chosen, "points": landmark_points}))
    arm("image_uninformed", lambda: uninformed(evidence))
    if enough:
        arm("commissures_uninformed", lambda: uninformed(with_landmarks))
    return row


def main() -> int:
    bench = os.environ.get("R4D_BENCH", "v3")
    spec = json.loads(Path(SPECS[bench]).read_text())
    joint = next(v for v in spec["variants"] if v["name"] == "S/joint")
    smoke = os.environ.get("R4D_SMOKE") == "1"
    split = "val" if smoke else "test"  # smoke never reads test
    paths = e59.sample_paths(Path(spec["benchmark"]), split)
    if smoke:
        paths = paths[:2]
    return run(f"E91_{bench}_{split}" + ("_smoke" if smoke else ""), split, paths, fit_sample, initialise,
               (spec["models"]["S"], spec["base_config"], joint["overrides"], spec["benchmark"]),
               extra={"experiment": "E91", "preregistration": PREREGISTRATION, "benchmark": spec["benchmark"],
                      "frozen_from": SPECS[bench], "rotations": ROTATIONS, "offsets": OFFSETS, "top": TOP,
                      "refine_iterations": REFINE, "final_iterations": FINAL, "smoke": smoke})


if __name__ == "__main__":
    sys.exit(main())
