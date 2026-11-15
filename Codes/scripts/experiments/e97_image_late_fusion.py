"""E97: late fusion under image evidence, the like-for-like comparison (plan v2, W1.4 item 3).

Pre-registered in ``Docs/E97_IMAGE_LATE_FUSION_PREREGISTRATION_2026-09-29.md``.
E74 and E91 fit Rooted4D to image evidence; the point-evidence comparison with
late fusion does not transfer, because point evidence gives both methods vertex
correspondence and image evidence gives neither. Here late fusion is built from
image-only fits of each modality, from the same zero start and with the same
evidence as E91's uninformed arms, and registered as E87 registers its point
fits:

* the CMR fit: CMR image evidence only, pose, phase offset and warp held (the
  CMR frame is the reference and its frames are at known phases);
* the TEE fit: TEE image evidence only, with its own pose, offset and warp,
  images alone (``lf_image``) or with the two commissure landmarks
  (``lf_commissures``);
* neither fit has a session term, as in E87 (its coefficients are held at
  zero);
* E87's trajectory registration of the TEE fit onto the CMR fit, and E87's
  scoring of the placed pose, the composed phase map and the interface gap.

    R4D_BENCH=v3|v4 R4D_ALLOW_TEST=1 python e97_image_late_fusion.py
    R4D_BENCH=v3 R4D_SMOKE=1 python e97_image_late_fusion.py     (validation, 2 iterations: code path only)
    python e97_image_late_fusion.py analyze                        -> results/benchmark/E97/E97_analysis.json
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
import e91_image_search as e91  # noqa: E402
from e70_image_evidence import commissure_landmarks  # noqa: E402
from uniformity_runner import run  # noqa: E402
from rooted4d.experiments.benchmark_eval import benchmark_problem  # noqa: E402
from rooted4d.inference.image_evidence import evidence_from_sample  # noqa: E402
from rooted4d.inference.problem import pose_errors  # noqa: E402
from rooted4d.motion import circular_distance  # noqa: E402

PREREGISTRATION = "Docs/E97_IMAGE_LATE_FUSION_PREREGISTRATION_2026-09-29.md"
STATE: dict = {}
KEYS = ("rotation_deg", "translation_mm", "phase_map_cycles")
RESAMPLES, SEED = 20000, 20260910


def initialise(recipe, base, overrides, root):
    """E91's initialisation, plus the masks of late fusion's two fits."""

    import torch

    e91.initialise(recipe, base, overrides, root)
    scene, layout, packed = e91.STATE["scene"], e91.STATE["layout"], e91.STATE["packed"]
    session_start = scene.n_modes + scene.motion_rows * scene.n_motion
    session_stop = session_start + scene.session_rows * scene.n_session * len(scene.sessions)
    assert session_stop == layout["axis_angle"].start, (session_stop, layout["axis_angle"].start)
    tee_free = torch.ones(len(packed), dtype=torch.bool)
    tee_free[session_start:session_stop] = False            # no session term, as in E87's late fusion
    cmr_free = tee_free.clone()
    cmr_free[layout["axis_angle"]] = False                  # the CMR frame is the reference
    cmr_free[layout["translation"]] = False
    cmr_free[layout["offset"]] = False
    cmr_free[layout["logits"]] = False                      # CMR frames are at known phases
    point = e91.STATE["point"]
    STATE.update(cmr_free=cmr_free, tee_free=tee_free,
                 registration_config=replace(point, use_tee=False, estimate_pose=False,
                                             estimate_phase_warp=False, use_session=False))


def _register_and_score(cmr_vector, tee_vector, problem, evidence_times, true_phases) -> dict:
    """E87's registration and scoring (``baselines.run_late_fusion_trajectory``), on image fits."""

    from rooted4d.inference.initialization import _matrix_to_axis_angle
    from rooted4d.inference.joint import _unpack
    from rooted4d.inference.trajectory_registration import register_trajectories, registered_interface_gap

    model, image = e91.STATE["model"], e91.STATE["image"]
    config = STATE["registration_config"]
    cmr = _unpack(np.asarray(cmr_vector, dtype=np.float64), problem.truth, model, image)
    tee = _unpack(np.asarray(tee_vector, dtype=np.float64), problem.truth, model, image)
    observations = problem.observations
    vertices = np.union1d(np.intersect1d(observations.cmr_vertices, observations.tee_vertices),
                          np.sort(model.groups["annulus"]))
    registration = register_trajectories(cmr, tee, model, config, vertices)
    pose = registration.place(tee.transform())
    tee_warp = tee.warp()
    offset = float(registration.warp.evaluate(tee_warp.evaluate(np.array([0.0])))[0])
    estimated = replace(tee, pose_axis_angle=_matrix_to_axis_angle(pose.rotation),
                        pose_translation=np.asarray(pose.translation, dtype=np.float64), phase_offset=offset)
    errors = pose_errors(estimated, problem.truth)
    mapped = registration.warp.evaluate(tee_warp.evaluate(np.asarray(evidence_times, dtype=np.float64)))
    finite = bool(np.all(np.isfinite(cmr_vector)) and np.all(np.isfinite(tee_vector))
                  and np.isfinite(registration.cost))
    return {"succeeded": finite,
            "rotation_deg": errors["rotation_deg"], "translation_mm": errors["translation_mm"],
            "phase_map_cycles": float(np.median(circular_distance(np.asarray(mapped, float), true_phases))),
            "interface_gap_mm": registered_interface_gap(cmr, tee, registration, model, config),
            "registration_rms_mm": float(registration.rms_mm)}


def fit_sample(path_string: str) -> dict:
    import torch
    from rooted4d.differentiable import evidence_tensors
    from rooted4d.differentiable.scene import image_residuals, landmark_residuals, prior_residuals

    scene, weights, annulus = (e91.STATE[k] for k in ("scene", "weights", "annulus"))
    model, point = e91.STATE["model"], e91.STATE["point"]
    path = Path(path_string)
    with np.load(path, allow_pickle=True) as blob:
        identity = str(blob["identity_id"])
        true_phases = np.asarray(blob["tee_true_phases"], dtype=float)
        all_times = np.asarray(blob["tee_times"], dtype=float)
        tee_index = np.asarray(blob["tee_indices"], dtype=np.int64)
        tee_points = np.asarray(blob["tee_points"], dtype=float)
    problem = benchmark_problem(path, model, point)
    # E91's evidence, unchanged.
    record = evidence_from_sample(path, model, budget=200, n_cmr_frames=4, n_tee_frames=4)
    frames = np.searchsorted(all_times, record.tee_times)
    phases = true_phases[frames]
    evidence = evidence_tensors(record, e91.W_CMR, e91.W_TEE)
    evidence["shadow_tolerance"] = 0.1
    available = np.intersect1d(annulus, tee_index)
    chosen = commissure_landmarks(available, len(annulus))
    columns = np.searchsorted(tee_index, chosen)
    landmark_points = tee_points[np.ix_(frames, columns)]
    cmr_only = dict(evidence)
    cmr_only["tee_values"] = None
    tee_only = dict(evidence)
    tee_only["cmr_values"] = None
    tee_landmarks = dict(tee_only)
    tee_landmarks.update(landmark_vertices=torch.as_tensor(chosen, dtype=torch.int64),
                         landmark_points=torch.as_tensor(landmark_points, dtype=torch.float64),
                         landmark_times=torch.as_tensor(record.tee_times, dtype=torch.float64),
                         landmark_weight=e91.LANDMARK_WEIGHT)

    def objective(ev):
        return lambda v: (torch.cat([image_residuals(v, scene, ev), landmark_residuals(v, scene, ev),
                                     prior_residuals(v, scene, weights)]) ** 2).sum()

    zero = torch.as_tensor(e91.STATE["packed"].copy())
    row = {"sample": path.name, "identity": identity, "variants": {},
           "commissures": [int(v) for v in chosen]}
    started = time.perf_counter()
    cmr_cost = objective(cmr_only)
    cmr_vector = e91._lbfgs(zero, cmr_cost, free=STATE["cmr_free"], iterations=e91.FINAL)
    row["cmr_stage_seconds"] = round(time.perf_counter() - started, 3)
    row["cmr_final_objective"] = float(cmr_cost(cmr_vector).detach())

    def arm(name, ev):
        started = time.perf_counter()
        try:
            cost = objective(ev)
            tee_vector = e91._lbfgs(zero, cost, free=STATE["tee_free"], iterations=e91.FINAL)
            outcome = _register_and_score(cmr_vector.numpy(), tee_vector.numpy(), problem,
                                          record.tee_times, phases)
            outcome["tee_final_objective"] = float(cost(tee_vector).detach())
        except Exception as error:  # a failed arm is recorded, never fatal
            outcome = {"succeeded": False, "failure_reason": f"{type(error).__name__}: {error}"}
        outcome["seconds"] = round(time.perf_counter() - started, 3)
        row["variants"][name] = outcome

    arm("lf_image", tee_only)
    if len(chosen) == 2:
        arm("lf_commissures", tee_landmarks)
    return row


def main() -> int:
    bench = os.environ.get("R4D_BENCH", "v3")
    spec = json.loads(Path(e91.SPECS[bench]).read_text())
    joint = next(v for v in spec["variants"] if v["name"] == "S/joint")
    smoke = os.environ.get("R4D_SMOKE") == "1"
    split = "val" if smoke else "test"  # smoke never reads test
    paths = e59.sample_paths(Path(spec["benchmark"]), split)
    if smoke:
        paths = paths[:2]
    return run(f"E97_{bench}_{split}" + ("_smoke" if smoke else ""), split, paths, fit_sample, initialise,
               (spec["models"]["S"], spec["base_config"], joint["overrides"], spec["benchmark"]),
               extra={"experiment": "E97", "preregistration": PREREGISTRATION, "benchmark": spec["benchmark"],
                      "frozen_from": e91.SPECS[bench], "iterations": e91.FINAL, "smoke": smoke})


# --------------------------------------------------------------- analysis ----

def _rows(directory: Path) -> dict:
    return {r["sample"]: r for r in (json.loads(line) for shard in sorted(directory.glob("shard_*.jsonl"))
                                     for line in shard.read_text().splitlines() if line.strip())}


def _summary(values: list[dict]) -> dict:
    ok = [v for v in values if v.get("succeeded")]
    out = {k: float(np.median([v[k] for v in ok])) for k in KEYS}
    rotation = np.array([v["rotation_deg"] for v in ok])
    out.update(samples=len(values), failures=len(values) - len(ok),
               within_5deg=float(np.mean(rotation <= 5.0)), beyond_90deg=float(np.mean(rotation > 90.0)))
    if ok and "interface_gap_mm" in ok[0]:
        gap = np.array([v["interface_gap_mm"] for v in ok])
        out.update(interface_gap_mm=float(np.median(gap)), gap_above_2mm=float(np.mean(gap > 2.0)))
    return out


def _paired(pairs: list[tuple[str, dict, dict]], key: str) -> dict:
    """E62's statistic: identity-averaged, difference of medians (first - second)."""

    identities = sorted({i for i, _, _ in pairs})
    a = np.array([np.nanmean([x[key] if x.get("succeeded") else np.nan for i, x, _ in pairs if i == identity])
                  for identity in identities])
    b = np.array([np.nanmean([y[key] if y.get("succeeded") else np.nan for i, _, y in pairs if i == identity])
                  for identity in identities])
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def cmd_analyze(directories: dict | None = None, out: str = "results/benchmark/E97/E97_analysis.json") -> None:
    """``directories`` maps a benchmark to its (E97 rows, E91 rows); the default is E97's own."""

    directories = directories or {b: (f"results/benchmark/E97/E97_{b}_test", f"results/benchmark/E91/E91_{b}_test")
                                  for b in ("v3", "v4")}
    report = {"experiment": "E97", "preregistration": PREREGISTRATION, "benchmarks": {}}
    for bench, (late_dir, joint_dir) in directories.items():
        late = _rows(Path(late_dir))
        joint = _rows(Path(joint_dir))
        assert sorted(late) == sorted(joint) and len(late) == 60, (bench, len(late), len(joint))
        entry = {"samples": len(late), "arms": {}, "rooted4d_minus_late_fusion": {}}
        for evidence, ours, theirs in (("image", "image_uninformed", "lf_image"),
                                       ("commissures", "commissures_uninformed", "lf_commissures")):
            entry["arms"][theirs] = _summary([late[s]["variants"][theirs] for s in sorted(late)])
            entry["arms"][ours] = _summary([joint[s]["variants"][ours] for s in sorted(joint)])
            pairs = [(late[s]["identity"], joint[s]["variants"][ours], late[s]["variants"][theirs])
                     for s in sorted(late)]
            entry["rooted4d_minus_late_fusion"][evidence] = {k: _paired(pairs, k) for k in KEYS}
        for search in ("image_search", "commissures_search"):           # descriptive, beside
            entry["arms"][search] = _summary([joint[s]["variants"][search] for s in sorted(joint)])
        report["benchmarks"][bench] = entry
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1)[:4000])
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1:] == ["analyze"]:
        cmd_analyze()
    else:
        sys.exit(main())
