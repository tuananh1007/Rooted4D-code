"""E70 and E71: image evidence in the frozen structured configuration.

Pre-registered in ``Docs/E65_E72_UNIFORMITY_PREREGISTRATION_2026-09-24.md``.
Both use the PyTorch objective of the method section with the covariance-aware
identity prior carried across (``prior_weights(config, model)``), L-BFGS for 400
iterations, image weights 1/0.015^2 and 1/0.12^2, and E32's evidence
construction (200 query points over 4 CMR and 4 TEE frames). Warm starts are
E69's frozen point fits, loaded rather than refitted.

* **E70** (``R4D_STUDY=e70``): image-only fitting from the zero start and from
  the point fit, and image evidence plus 2, 4 and 8 identified annular
  landmarks, warm-started. One-sided TEE residual (tolerance 0.1), as E30/E32.
* **E71** (``R4D_STUDY=e71``): the image channels one at a time -- CMR image
  with TEE points, CMR points with TEE image, both image -- warm-started, with
  the symmetric TEE residual, so the unmodeled shadow is what is measured.
* **E74** (``R4D_STUDY=e74``, pre-registered in
  ``Docs/E74_LANDMARKS_COLD_START_PREREGISTRATION_2026-09-25.md``): E70's
  image evidence from the zero start with 2, 4 and 8 landmarks chosen by E70's
  rule, and with two landmarks at the commissure positions of the benchmark
  ring from both starts; E70's image-only zero-start arm is re-run as the
  same-run reference.

    R4D_STUDY=e70|e71|e74 R4D_E69=<dir of E69 test shards> R4D_ALLOW_TEST=1 python e70_image_evidence.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import e59_unknown_alignment as e59  # noqa: E402
from uniformity_runner import frozen_s, run  # noqa: E402
from rooted4d.experiments.benchmark_eval import benchmark_problem  # noqa: E402
from rooted4d.inference import SolverConfig  # noqa: E402
from rooted4d.inference.image_evidence import evidence_from_sample, noise_weights  # noqa: E402
from rooted4d.inference.joint import _pack, _unpack, initial_parameters  # noqa: E402
from rooted4d.inference.problem import pose_errors  # noqa: E402
from rooted4d.motion import circular_distance  # noqa: E402

BENCH = os.environ.get("R4D_BENCH", "v3")  # E100 passes a rebuilt instance
ITERATIONS = 400
LANDMARK_COUNTS = (2, 4, 8)
W_CMR, W_TEE = noise_weights(0.015, 0.12)
LANDMARK_WEIGHT = 1.0 / 0.45**2
STATE: dict = {}


def initialise(recipe, base, weight, root, study, warm_starts):
    import torch

    torch.set_num_threads(1)
    from rooted4d.differentiable import TorchScene
    from rooted4d.differentiable.scene import prior_weights

    model = e59.build_model(Path(root), recipe)[0]
    point = SolverConfig(**{**base, "identity_prior_weight": weight})
    if os.environ.get("R4D_SMOKE") == "1":  # code-path check only, never a result
        global ITERATIONS
        ITERATIONS = 2
    image = replace(point, use_cmr=False, use_tee=False,
                    cmr_image_weight=W_CMR, tee_image_weight=W_TEE)
    STATE.update(model=model, point=point, image=image, study=study,
                 scene=TorchScene.from_model(model, image),
                 weights=prior_weights(image, model), warm=warm_starts,
                 annulus=np.sort(model.groups["annulus"]))
    # The commissure rule reads ring order from vertex indices.
    assert np.array_equal(STATE["annulus"], np.arange(len(STATE["annulus"])))


def _lbfgs(start, residual_function):
    """E30's optimizer settings, on an arbitrary residual stack."""
    import torch

    vector = start.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([vector], lr=0.4, max_iter=ITERATIONS, history_size=25,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                                  tolerance_change=1e-14)

    def closure():
        optimizer.zero_grad()
        cost = (residual_function(vector) ** 2).sum()
        cost.backward()
        return cost

    optimizer.step(closure)
    return vector.detach()


def _score(vector, problem, evidence_times, true_phases):
    model, image = STATE["model"], STATE["image"]
    parameters = _unpack(np.asarray(vector, dtype=np.float64), problem.truth, model, image)
    errors = pose_errors(parameters, problem.truth)
    mapped = parameters.warp().evaluate(evidence_times)
    return {"succeeded": bool(np.all(np.isfinite(vector))),
            "rotation_deg": errors["rotation_deg"], "translation_mm": errors["translation_mm"],
            "phase_map_cycles": float(np.median(circular_distance(np.asarray(mapped, float), true_phases)))}


def e70_landmarks(available: np.ndarray, count: int) -> np.ndarray:
    """E32's and E70's rule: evenly spaced by index among the observed annular vertices.

    The ring wraps, so for two landmarks this takes the first and last observed
    vertices by index, which are usually neighbours on the ring (0 and 23), not
    points that span it.
    """
    return np.unique(available[np.linspace(0, len(available) - 1, count).round().astype(int)])


def commissure_landmarks(available: np.ndarray, n_annulus: int) -> np.ndarray:
    """The observed annular vertices nearest the two commissures of the benchmark ring.

    The procedural complex gives the anterior leaflet the root arc 0..n/2-1 and
    the posterior leaflet n/2..n-1 (``make_toy_complex``), so the commissures
    lie at the junctions between n/2-1 and n/2 and between n-1 and 0. For each
    junction the observed vertex nearest it by circular index distance is taken,
    ties going to the anterior arc; with every vertex observed that is 0 and n/2-1.
    """
    half = n_annulus // 2
    chosen = []
    for junction in (half - 0.5, n_annulus - 0.5):
        def key(vertex, junction=junction):
            distance = abs((vertex - junction + n_annulus / 2) % n_annulus - n_annulus / 2)
            return (distance, 0 if vertex < half else 1)
        chosen.append(min((int(v) for v in available), key=key))
    return np.unique(np.asarray(chosen, dtype=np.int64))


def fit_sample(path_string: str) -> dict:
    import torch
    from rooted4d.differentiable import evidence_tensors
    from rooted4d.differentiable.scene import (
        image_residuals, landmark_residuals, point_residuals, prior_residuals,
    )

    model, point, image, scene, weights = (STATE[k] for k in ("model", "point", "image", "scene", "weights"))
    path = Path(path_string)
    with np.load(path, allow_pickle=True) as blob:
        identity = str(blob["identity_id"])
        true_phases = np.asarray(blob["tee_true_phases"], dtype=float)
        all_times = np.asarray(blob["tee_times"], dtype=float)
        tee_index = np.asarray(blob["tee_indices"], dtype=np.int64)
        tee_points = np.asarray(blob["tee_points"], dtype=float)
    problem = benchmark_problem(path, model, point)
    evidence_record = evidence_from_sample(path, model, budget=200, n_cmr_frames=4, n_tee_frames=4)
    frames = np.searchsorted(all_times, evidence_record.tee_times)
    phases = true_phases[frames]
    warm = torch.as_tensor(np.asarray(STATE["warm"][path.name], dtype=np.float64))
    cold = torch.as_tensor(_pack(initial_parameters(model, image), model, image))
    row = {"sample": path.name, "identity": identity, "variants": {}}
    row["variants"]["point_fit"] = _score(warm.numpy(), problem, evidence_record.tee_times, phases)

    def run_arm(name, start, residual_function):
        started = time.perf_counter()
        try:
            vector = _lbfgs(start, residual_function)
            outcome = _score(vector.numpy(), problem, evidence_record.tee_times, phases)
            if os.environ.get("R4D_KEEP_VECTORS") == "1":  # for figures (E81); off in E70 and E74
                outcome["vector"] = vector.numpy().tolist()
        except Exception as error:  # a failed arm is recorded, never fatal
            outcome = {"succeeded": False, "failure_reason": f"{type(error).__name__}: {error}"}
        outcome["seconds"] = round(time.perf_counter() - started, 3)
        row["variants"][name] = outcome

    if STATE["study"] == "e70":
        evidence = evidence_tensors(evidence_record, W_CMR, W_TEE)
        evidence["shadow_tolerance"] = 0.1

        def image_only(ev):
            return lambda v: torch.cat([image_residuals(v, scene, ev), landmark_residuals(v, scene, ev),
                                        prior_residuals(v, scene, weights)])

        run_arm("image_cold", cold, image_only(evidence))
        run_arm("image_warm", warm, image_only(evidence))
        available = np.intersect1d(STATE["annulus"], tee_index)
        for count in LANDMARK_COUNTS:
            if len(available) < count:
                row["variants"][f"landmarks_{count}"] = {"succeeded": False,
                                                         "failure_reason": "too few annular vertices observed"}
                continue
            chosen = e70_landmarks(available, count)
            columns = np.searchsorted(tee_index, chosen)
            with_landmarks = dict(evidence)
            with_landmarks["landmark_vertices"] = torch.as_tensor(chosen, dtype=torch.int64)
            with_landmarks["landmark_points"] = torch.as_tensor(tee_points[np.ix_(frames, columns)],
                                                                dtype=torch.float64)
            with_landmarks["landmark_times"] = torch.as_tensor(evidence_record.tee_times, dtype=torch.float64)
            with_landmarks["landmark_weight"] = LANDMARK_WEIGHT
            run_arm(f"landmarks_{count}", warm, image_only(with_landmarks))
            row["variants"][f"landmarks_{count}"]["landmarks_used"] = int(len(chosen))
    elif STATE["study"] == "e74":
        evidence = evidence_tensors(evidence_record, W_CMR, W_TEE)
        evidence["shadow_tolerance"] = 0.1

        def image_only(ev):
            return lambda v: torch.cat([image_residuals(v, scene, ev), landmark_residuals(v, scene, ev),
                                        prior_residuals(v, scene, weights)])

        def with_landmarks(chosen):
            columns = np.searchsorted(tee_index, chosen)
            ev = dict(evidence)
            ev["landmark_vertices"] = torch.as_tensor(chosen, dtype=torch.int64)
            ev["landmark_points"] = torch.as_tensor(tee_points[np.ix_(frames, columns)], dtype=torch.float64)
            ev["landmark_times"] = torch.as_tensor(evidence_record.tee_times, dtype=torch.float64)
            ev["landmark_weight"] = LANDMARK_WEIGHT
            return ev

        run_arm("image_cold", cold, image_only(evidence))
        available = np.intersect1d(STATE["annulus"], tee_index)
        arms = [(f"landmarks_{count}_cold", cold, e70_landmarks(available, count), count)
                for count in LANDMARK_COUNTS]
        commissures = commissure_landmarks(available, len(STATE["annulus"]))
        arms += [("commissures_cold", cold, commissures, 2), ("commissures_warm", warm, commissures, 2)]
        for name, start, chosen, count in arms:
            if len(available) < count or len(chosen) < min(count, 2):
                row["variants"][name] = {"succeeded": False, "failure_reason": "too few annular vertices observed"}
                continue
            run_arm(name, start, image_only(with_landmarks(chosen)))
            row["variants"][name]["landmarks_used"] = [int(v) for v in chosen]
    else:
        observations = problem.observations
        points = {"cmr_points": torch.as_tensor(observations.cmr_points),
                  "cmr_times": torch.as_tensor(observations.cmr_times),
                  "cmr_vertices": torch.as_tensor(observations.cmr_vertices, dtype=torch.int64),
                  "tee_points": torch.as_tensor(observations.tee_points),
                  "tee_times": torch.as_tensor(observations.tee_times),
                  "tee_vertices": torch.as_tensor(observations.tee_vertices, dtype=torch.int64)}
        full = evidence_tensors(evidence_record, W_CMR, W_TEE)  # symmetric: no shadow tolerance

        def mixed(cmr_image: bool, tee_image: bool):
            image_evidence = dict(full)
            point_evidence = dict(points)
            if not cmr_image:
                image_evidence["cmr_values"] = None
            else:
                point_evidence["cmr_points"] = None
            if not tee_image:
                image_evidence["tee_values"] = None
            else:
                point_evidence["tee_points"] = None
            return lambda v: torch.cat([image_residuals(v, scene, image_evidence),
                                        point_residuals(v, scene, point_evidence),
                                        prior_residuals(v, scene, weights)])

        run_arm("cmr_image_tee_point", warm, mixed(True, False))
        run_arm("cmr_point_tee_image", warm, mixed(False, True))
        run_arm("both_image", warm, mixed(True, True))
    return row


def load_warm_starts(directory: str) -> dict:
    starts = {}
    for shard in sorted(Path(directory).glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                starts[record["sample"]] = record["packed"]
    return starts


def main() -> int:
    study = os.environ.get("R4D_STUDY", "e70")
    recipe, base, weight, root = frozen_s(BENCH)
    warm = load_warm_starts(os.environ["R4D_E69"])
    split = "val" if os.environ.get("R4D_SMOKE") == "1" else "test"  # smoke never reads test
    paths = [p for p in e59.sample_paths(Path(root), split) if p.name in warm]
    return run(f"{study.upper()}_{BENCH}_{split}", split, paths, fit_sample, initialise,
               (recipe, base, weight, root, study, warm),
               extra={"experiment": study.upper(), "benchmark": root, "iterations": ITERATIONS,
                      "warm_starts": os.environ["R4D_E69"], "samples_with_warm_start": len(paths)})


if __name__ == "__main__":
    sys.exit(main())
