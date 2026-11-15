"""E118: E109's qualitative fusion samples, drawn for the methods added after it.

Not an experiment: nothing here is selected or scored for the paper. E109 drew
the realistic-session test samples at the 25th, 50th and 75th percentile of
Rooted4D's held-out CMR + TEE error, and the worst, for the fusion methods of
E100. The fusion table has since gained late fusion + BCPD, late fusion + NDP
and GPMM fusion (E115), and Rooted4D's residual field (E116). This draws them on
the same samples, at their selected settings, over E109's eight phases and in
its frame (the CMR frame, session-free):

* late fusion + BCPD and + NDP: late fusion's CMR chamber, and its TEE anatomy
  moved by the attachment. The eight drawn phases are every fourth of the 32
  phases at which the arm attaches, so every vertex is appended as a query of
  the arm's own registrations (both warp each query on its own), and the arm's
  numbers are recomputed from the same calls.
* GPMM fusion: the single model's posterior mean, interpolated over phase.
* Rooted4D: the joint fit's anatomy plus the shared field (the main
  configuration), and plus the per-modality variant's CMR field.

Each drawn fit must reproduce its stored row to REFIT_TOLERANCE_MM: E115's rows
for the comparators, E116's for the fields, E100's for the joint fit.

    R4D_ALLOW_TEST=1 R4D_RESULTS=<results> R4D_OUT=<dir> python e118_qualitative_recent.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

KEY = "v4n96o"
PHASES = np.linspace(0.0, 1.0, 8, endpoint=False)
#: E109's fusion samples (results/qualitative96/fusion_index.json).
SAMPLES = (("q25", "sample-beca31cd32c236b3.npz"), ("q50", "sample-925a033bfbf6e571.npz"),
           ("q75", "sample-66fc7a9456bbb0f4.npz"), ("worst", "sample-b0a5362d9ad0a167.npz"))
JOINT, LATE = "S/joint", "S/late_fusion_trajectory"
FIELD = "R4D+GP"
REFIT_TOLERANCE_MM = 5e-3
SELECTION = Path("results/benchmark/E115/E115_val.json")
MAIN = Path("Codes/configs/rooted4d_main.json")
ENDPOINTS = ("heldout_cmr_mm", "heldout_tee_mm", "attachment_residual_mm")


def _results() -> Path:
    return Path(os.environ["R4D_RESULTS"])


def _rows(directory: Path) -> dict:
    return {row["sample"]: row["variants"]
            for shard in sorted(directory.glob("shard_*.jsonl"))
            for row in map(json.loads, filter(str.strip, shard.read_text().splitlines()))}


def _stage(experiment: str) -> Path:
    index = json.loads((_results() / experiment / "test" / "index.json").read_text())
    return _results() / experiment / "test" / index[f"{KEY}/full"]


def _e100() -> Path:
    root = _results() / "e100" / KEY
    return root / (root / "TEST_DONE").read_text().strip()


def _differences(numbers: dict, row: dict) -> dict:
    return {k: float(numbers[k] - row[k]) for k in ENDPOINTS if k in numbers and row.get(k) is not None}


def _attached(problem, config, alignment, overrides, methods: dict, n: int):
    """Late fusion, then each attachment at every phase the arm fits.

    Follows ``run_registration_attachment`` step by step, with every physical
    vertex appended to its queries. Returns late fusion's drawn chamber, its
    unattached TEE anatomy, and per method the attached anatomy and the arm's
    held-out and attachment numbers.
    """

    from rooted4d.inference.baselines import _late_fusion_fits
    from rooted4d.inference.published_fusion import _mean_and_p95
    from rooted4d.inference.recent_fusion import GAP_PHASES, ring_order
    from rooted4d.inference.trajectory_registration import (
        inverse_phase, register_trajectories, session_free_trajectory,
    )

    observations, model = problem.observations, problem.model
    cmr_fit, tee_fit, cmr_config, tee_config = _late_fusion_fits(problem, config, alignment, overrides)
    vertices = np.union1d(np.intersect1d(observations.cmr_vertices, observations.tee_vertices),
                          np.sort(model.groups["annulus"]))
    registration = register_trajectories(cmr_fit.parameters, tee_fit.parameters, model, cmr_config, vertices)
    G, chi = registration.transform, registration.warp
    cmr, tee = cmr_fit.parameters, tee_fit.parameters

    template = model.template_physical
    ring = np.sort(model.groups["annulus"])
    ring = ring[ring_order(template[model.observation_rows(ring)])]
    leaflet = np.setdiff1d(np.asarray(observations.tee_vertices), ring)
    nearest = leaflet[np.argmin(np.linalg.norm(
        template[model.observation_rows(ring)][:, None] - template[model.observation_rows(leaflet)][None],
        axis=-1), axis=1)]
    heldout_vertices = np.asarray(observations.heldout_tee_vertices)
    n_ring, n_hinge, n_arm = len(ring), len(nearest), len(ring) + len(nearest) + len(heldout_vertices)
    queries = np.concatenate([ring, nearest, heldout_vertices, np.arange(n)])

    heldout_times = np.asarray(observations.heldout_tee_times, dtype=np.float64)
    heldout_tee = np.asarray(tee.warp().evaluate(heldout_times), dtype=np.float64)
    tee_phases = np.concatenate([inverse_phase(chi, GAP_PHASES), heldout_tee])
    cmr_phases = np.concatenate([GAP_PHASES, np.asarray(chi.evaluate(heldout_tee), dtype=np.float64)])
    moving = G.apply(session_free_trajectory(tee, model, tee_config, tee_phases, queries))
    fixed = session_free_trajectory(cmr, model, cmr_config, cmr_phases, ring)

    gap_count, step = len(GAP_PHASES), len(GAP_PHASES) // len(PHASES)
    if not np.array_equal(GAP_PHASES[::step], PHASES):
        raise RuntimeError("the drawn phases are not among the arm's")
    inverse, tee_pose = G.inverse(), tee.transform()
    observed = np.asarray(observations.heldout_tee_points, dtype=np.float64)
    out = {}
    for name, (method, primary) in methods.items():
        moved = np.stack([method(moving[i, :n_ring], fixed[i], moving[i])[primary][1]
                          for i in range(len(tee_phases))])
        valve = moved[gap_count:, n_ring + n_hinge:n_arm]
        out[name] = {
            "valve": moved[:gap_count:step, n_arm:],
            "numbers": {"heldout_cmr_mm": cmr_fit.heldout_errors()["heldout_cmr_mm"],
                        "heldout_tee_mm": _mean_and_p95(tee_pose.apply(inverse.apply(valve)), observed)[0],
                        "attachment_residual_mm": float(np.abs(moved[:gap_count, :n_ring] - fixed[:gap_count]).max())},
        }
    chamber = session_free_trajectory(cmr, model, cmr_config, PHASES, np.arange(n))
    return chamber, moving[:gap_count:step, n_arm:], out


def run_sample(task) -> dict:
    from dataclasses import replace

    import e59_unknown_alignment as e59
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.experiments.fitted_geometry import fitted_trajectory
    from rooted4d.inference import SolverConfig
    from rooted4d.inference.baselines import _start
    from rooted4d.inference.initialization import input_derived_alignment
    from rooted4d.inference.joint import fit_joint_scene
    from rooted4d.inference.published_fusion import _mean_and_p95
    from rooted4d.inference.recent_fusion import (
        bcpd_attachment, circular_interpolate, fit_gpmm_fusion, ndp_attachment, score_gpmm_fusion,
    )
    from rooted4d.inference.residual_field import FieldSettings, config_name, fit_residual_field

    label, name = task
    e100 = _e100()
    spec = json.loads((e100 / "spec.json").read_text())
    variants = {v["name"]: v for v in spec["variants"]}
    root = Path(spec["benchmark"])
    models = {key: e59.build_model(root, recipe)[0] for key, recipe in spec["models"].items()}
    base = SolverConfig(**spec.get("base_config", {}))
    S, G = models["S"], models["G"]
    n = S.n_physical
    if G.n_physical != n:
        raise RuntimeError("the two models do not share the physical vertices")
    stored = {"e100": _rows(e100)[name], "e115": _rows(_stage("e115"))[name], "e116": _rows(_stage("e116"))[name]}
    chosen = json.loads(SELECTION.read_text())["selection"][KEY]
    main = json.loads(MAIN.read_text())["benchmarks"][KEY]

    path = root / "samples" / "test" / name
    problems = {key: benchmark_problem(path, model, base) for key, model in models.items()}
    # The E59 driver takes the shared start from the first model's problem.
    alignment = input_derived_alignment(next(iter(problems.values())).observations)
    geometry, numbers, checks = {}, {}, {}

    # Late fusion + BCPD and + NDP (E115's arms at their selected settings).
    beta, lam = (float(v) for v in chosen["LF+BCPD"])
    levels = int(chosen["LF+NDP"])
    methods = {"lf_bcpd": (bcpd_attachment([(beta, lam)], gamma=0.1), f"{beta:g},{lam:g}"),
               "lf_ndp": (ndp_attachment(levels, seed=0), f"{levels}")}
    overrides = variants[LATE]["options"]["modality_overrides"]
    chamber, unattached, attached = _attached(problems["S"], base, alignment, overrides, methods, n)
    geometry["late_fusion_refit_chamber"], geometry["late_fusion_refit_valve"] = chamber, unattached
    for key, variant in (("lf_bcpd", "LF+BCPD"), ("lf_ndp", "LF+NDP")):
        geometry[f"{key}_chamber"], geometry[f"{key}_valve"] = chamber, attached[key]["valve"]
        numbers[key] = attached[key]["numbers"]
        checks[key] = _differences(numbers[key], stored["e115"][variant])

    # GPMM fusion (E115's GPMM-G).
    scale, width, noise = (float(v) for v in chosen["GPMM-G"])
    fit = fit_gpmm_fusion(G, problems["G"].observations, alignment, gaussian_scale=scale, gaussian_width=width,
                          noise=noise, iterations=25, scale_floor=base.identity_prior_scale_floor)
    numbers["gpmm"] = score_gpmm_fusion(fit, G, problems["G"].observations, problems["G"].truth, base)
    checks["gpmm"] = _differences(numbers["gpmm"], stored["e115"]["GPMM-G"])
    positions = G.template_physical[None] + circular_interpolate(fit["phases"], fit["deformations"], PHASES)
    geometry["gpmm"] = positions[:, G.observation_rows(np.arange(n)), :]

    # Rooted4D: the joint fit, refitted, then each field (E116's arm).
    joint_config = replace(base, **(variants[JOINT].get("overrides") or {}))
    observations = problems["S"].observations
    fit = fit_joint_scene(observations, S, joint_config, start=_start(alignment, S, joint_config))
    numbers["rooted4d_joint"] = dict(fit.heldout_errors(), attachment_residual_mm=0.0)
    checks["rooted4d_joint"] = _differences(numbers["rooted4d_joint"], stored["e100"][JOINT])
    joint = fitted_trajectory(fit.parameters, S, joint_config, PHASES, n)
    geometry["rooted4d_joint_refit"] = joint
    rows = S.observation_rows(np.arange(n))
    cmr_times = np.asarray(observations.heldout_cmr_times, dtype=np.float64)
    tee_phases = np.asarray(fit.parameters.warp().evaluate(np.asarray(observations.heldout_tee_times)),
                            dtype=np.float64)
    cmr_rows = S.observation_rows(np.asarray(observations.heldout_cmr_vertices))
    tee_rows = S.observation_rows(np.asarray(observations.heldout_tee_vertices))
    for key, setting in (("rooted4d_field", main["field"]), ("rooted4d_field_session", main["reported_variant"])):
        settings = FieldSettings(setting["sharing"], float(setting["static"]), float(setting["dynamic"]),
                                 float(setting["width"]))
        field = fit_residual_field(fit, settings)
        numbers[key] = {
            "heldout_cmr_mm": _mean_and_p95(field.geometry(fit, cmr_times, "cmr")[:, cmr_rows],
                                            observations.heldout_cmr_points)[0],
            "heldout_tee_mm": _mean_and_p95(fit.parameters.transform().apply(
                field.geometry(fit, tee_phases, "tee")[:, tee_rows]), observations.heldout_tee_points)[0],
            "attachment_residual_mm": 0.0}
        reference = stored["e116"][FIELD]["extras"]["by_config"][config_name(
            settings.sharing, settings.static, settings.dynamic, settings.width)]
        checks[key] = _differences(numbers[key], reference)
        # The session-free anatomy plus the field of the CMR session (the shared field is both sessions').
        geometry[key] = joint + field.displacement(PHASES, "cmr")[:, rows, :]

    worst = max(abs(v) for arm in checks.values() for v in arm.values())
    stem = Path(name).stem
    destination = Path(os.environ["R4D_OUT"]) / "fusion" / f"{label}_{stem}"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "recent.npz", phases=PHASES,
                        **{f"geometry_{k}": v for k, v in geometry.items()})
    entry = {"label": label, "sample": name, "numbers": numbers, "checks": checks,
             "largest_check_mm": worst, "passed": bool(worst <= REFIT_TOLERANCE_MM),
             "settings": {"LF+BCPD": [beta, lam], "LF+NDP": levels, "GPMM-G": [scale, width, noise],
                          "field": main["field"], "per_modality_field": main["reported_variant"]}}
    print(json.dumps(entry), flush=True)
    return entry


def main() -> int:
    import multiprocessing

    out = Path(os.environ["R4D_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    labels = os.environ.get("R4D_LABELS")
    tasks = [s for s in SAMPLES if not labels or s[0] in labels.split(",")]
    with multiprocessing.get_context("fork").Pool(min(len(tasks), int(os.environ.get("R4D_WORKERS", "4")))) as pool:
        report = pool.map(run_sample, tasks)
    passed = all(entry["passed"] for entry in report)
    (out / "recent_index.json").write_text(json.dumps(
        {"tolerance_mm": REFIT_TOLERANCE_MM, "passed": passed, "samples": report}, indent=2) + "\n")
    print("wrote", out / "recent_index.json", "| checks passed:", passed, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
