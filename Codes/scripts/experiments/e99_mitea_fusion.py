"""E99, part 2: the decomposition on real two-modality chamber data (plan v2, W3.1).

Pre-registered in ``Docs/E99_MITEA_FUSION_PREREGISTRATION_2026-09-29.md``.
MITEA's CMR-derived labels (one side) and part 1's echo predictions (the other),
fitted as reduced chambers in one frame per subject and placed in the model's
frame; the echo side is moved by a known rigid perturbation and phase offset,
and Rooted4D and trajectory late fusion are fitted through ``run_arm`` with the
main protocol's start and configuration. The model is the realistic-session
benchmark's, with its session basis re-learned from part 1's training subjects'
scan-rescan differences.

    R4D_MODE=geometry R4D_SPLIT=train|val|test python e99_mitea_fusion.py
    R4D_MODE=fit      R4D_SPLIT=val|test       python e99_mitea_fusion.py   (test needs R4D_ALLOW_TEST=1)
    R4D_MODE=analyze                           python e99_mitea_fusion.py

``geometry`` writes every reduced chamber of the split's subjects to
``results/benchmark/E99/geometry_<split>.npz``. It reads labels only for the
training subjects, and labels and predictions for the others; with
``R4D_LABELS_AS_PREDICTIONS=1`` (the code check before the network exists) the
labels stand in for the predictions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E99_MITEA_FUSION_PREREGISTRATION_2026-09-29.md"
SOURCE = Path(os.environ.get("R4D_MITEA", "data/raw/mitea"))
SPLIT = Path("Codes/configs/e99/mitea_split.json")
MODEL_SPEC = Path("Codes/configs/e75/v4_e75_test.json")
OUT = Path("results/benchmark/E99")                 # numbers only
DERIVED = Path("data/derived/mitea-e99")             # derived MITEA geometry: never committed
N_ANNULUS = 24
PHASES = {"ES": 0.0, "ED": 0.5}   # the benchmark cycle: minimum volume at 0, maximum at 0.5
SCANS = ("scan1", "scan2")
POSE_ROTATION_SD, POSE_TRANSLATION_SD, OFFSET_RANGE = 0.14, 9.0, 0.2
RESAMPLES, SEED = 20000, 20260910


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:15], 16)


def _prediction_root(split: str) -> Path:
    return Path(os.environ.get("R4D_MITEA_PRED", f"data/nnunet/predictions_mitea_{split}"))


# -------------------------------------------------------------- geometry ----

def _groups():
    from rooted4d.experiments.fitted_geometry import ring_groups

    return ring_groups(N_ANNULUS)


def reduce_chamber(fit) -> np.ndarray | None:
    """The reduced chamber of a ring fit: annulus, lower ring, apex (49 vertices).

    ``chamber_donor_vertices``' mapping. ``fit_cavity_in_frame`` stacks its rings
    along the frame's base-to-apex axis, so ring 0 is the base.
    """

    from rooted4d.donors.export import _resample_ring

    if fit is None:
        return None
    rings = fit.vertices[: fit.n_rings * fit.n_angles].reshape(fit.n_rings, fit.n_angles, 3)
    out = np.empty((2 * N_ANNULUS + 1, 3))
    out[:N_ANNULUS] = _resample_ring(rings[0], N_ANNULUS)
    out[N_ANNULUS:2 * N_ANNULUS] = _resample_ring(rings[len(rings) // 2], N_ANNULUS)
    out[2 * N_ANNULUS] = rings[-1].mean(axis=0)
    return out


def _handedness(chamber: np.ndarray) -> float:
    ring = chamber[:N_ANNULUS]
    area = np.cross(ring, np.roll(ring, -1, axis=0)).sum(axis=0)
    return float(np.sign(area @ (chamber[2 * N_ANNULUS] - ring.mean(axis=0))))


def _reverse(chamber: np.ndarray) -> np.ndarray:
    """Reverse the ring direction, keeping vertex 0: k -> -k (mod n), both rings."""

    order = (-np.arange(N_ANNULUS)) % N_ANNULUS
    out = chamber.copy()
    out[:N_ANNULUS] = chamber[order]
    out[N_ANNULUS:2 * N_ANNULUS] = chamber[N_ANNULUS + order]
    return out


def _procrustes(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proper rotation and translation taking ``source`` onto ``target`` (least squares)."""

    s, t = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - s).T @ (target - t))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, t - rotation @ s


def subject_geometry(subject: str, predictions: Path | None, template: np.ndarray) -> dict:
    """Every reduced chamber of one subject, in the model's frame.

    Keys ``(scan, phase, source)``; ``source`` is ``label`` or ``prediction``. A
    failed cavity fit is None.
    """

    from rooted4d.donors.mitea import cavity_world_points, fit_cavity_in_frame, lv_frame, rigid_align

    labels = {(scan, phase): cavity_world_points(SOURCE / "labels" / f"{subject}_{scan}_{phase}.nii.gz")
              for scan in SCANS for phase in PHASES}
    frame = lv_frame(labels[("scan1", "ES")])
    sources = {"label": lambda scan, phase: labels[(scan, phase)]}
    if predictions is not None:
        sources["prediction"] = lambda scan, phase: cavity_world_points(
            predictions / f"{subject}_{scan}_{phase}.nii.gz")
    chambers = {}
    for phase in PHASES:
        # Scan 2 is placed on scan 1 by ICP of its label cavity; the same move is applied to its prediction.
        _, rotation, translation = rigid_align(labels[("scan2", phase)], labels[("scan1", phase)])
        for scan in SCANS:
            for source, read in sources.items():
                try:
                    points = read(scan, phase)
                except Exception:  # an empty prediction is a front-end failure
                    chambers[(scan, phase, source)] = None
                    continue
                if scan == "scan2":
                    points = points @ rotation.T + translation
                chambers[(scan, phase, source)] = reduce_chamber(fit_cavity_in_frame(
                    points, frame, n_rings=8, n_angles=32, subject=subject, scan=scan, phase=phase))
    reference = [chambers[("scan1", p, "label")] for p in PHASES]
    if any(c is None for c in reference):
        return {"subject": subject, "chambers": chambers, "failed": True}
    chamber_template = template[: 2 * N_ANNULUS + 1]
    reversed_ring = _handedness(0.5 * (reference[0] + reference[1])) != _handedness(chamber_template)
    if reversed_ring:
        chambers = {k: (None if v is None else _reverse(v)) for k, v in chambers.items()}
        reference = [_reverse(c) for c in reference]
    rotation, translation = _procrustes(0.5 * (reference[0] + reference[1]), chamber_template)
    placed = {k: (None if v is None else v @ rotation.T + translation) for k, v in chambers.items()}
    return {"subject": subject, "chambers": placed, "failed": False, "reversed_ring": bool(reversed_ring),
            "procrustes_rms_mm": float(np.sqrt(((0.5 * (placed[("scan1", "ES", "label")]
                                                        + placed[("scan1", "ED", "label")])
                                                 - chamber_template) ** 2).sum(axis=1).mean()))}


def _model_template() -> np.ndarray:
    driver = _load("e59_driver", HERE / "e59_unknown_alignment.py")
    spec = json.loads(MODEL_SPEC.read_text())
    model, *_ = driver.build_model(Path(spec["benchmark"]), spec["models"]["S"])
    return model.template_physical[model.observation_rows(np.arange(3 * N_ANNULUS + 1))]


def cmd_geometry(split: str) -> None:
    if split == "test" and os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing test subjects without R4D_ALLOW_TEST=1")
    subjects = json.loads(SPLIT.read_text())[split]
    limit = int(os.environ.get("R4D_LIMIT", "0"))   # code checks only
    if limit:
        subjects = subjects[:limit]
    template = _model_template()
    labels_only = split == "train"
    stand_in = os.environ.get("R4D_LABELS_AS_PREDICTIONS") == "1"
    predictions = None if labels_only or stand_in else _prediction_root(split)
    arrays, meta = {}, {}
    for subject in subjects:
        result = subject_geometry(subject, predictions, template)
        meta[subject] = {k: result[k] for k in ("failed", "reversed_ring", "procrustes_rms_mm") if k in result}
        for (scan, phase, source), chamber in result["chambers"].items():
            if chamber is not None:
                arrays[f"{subject}|{scan}|{phase}|{source}"] = chamber
                if stand_in and source == "label":
                    arrays[f"{subject}|{scan}|{phase}|prediction"] = chamber
        print(subject, meta[subject], flush=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_standin" if stand_in else ""
    np.savez_compressed(DERIVED / f"geometry_{split}{suffix}.npz", **arrays)
    (OUT / f"geometry_{split}{suffix}.json").write_text(json.dumps(
        {"experiment": "E99", "preregistration": PREREGISTRATION, "split": split,
         "labels_as_predictions": stand_in, "subjects": meta}, indent=2) + "\n")
    print("wrote", DERIVED / f"geometry_{split}{suffix}.npz")


def _chambers(split: str, stand_in: bool = False) -> dict:
    suffix = "_standin" if stand_in else ""
    with np.load(DERIVED / f"geometry_{split}{suffix}.npz") as blob:
        out = {}
        for key in blob.files:
            subject, scan, phase, source = key.split("|")
            out[(subject, scan, phase, source)] = blob[key]
    return out


# ----------------------------------------------------------------- model ----

def _train_root() -> Path:
    """Part 1's training subjects as benchmark samples: shape, ES-ED motion, scan-rescan session.

    One sample per subject, in the model frame. The shared anatomy is the cycle
    mean of the scan-1 label chambers, and its motion the first cosine harmonic
    that reproduces ES at phase 0 and ED at 0.5; the session field is the
    scan-2 minus scan-1 difference, split the same way into a constant and a
    first-harmonic part, as the benchmark generator composes its own. The free
    edge is never observed and is the ACDC-derived template's, unmoved.
    """

    template = _model_template()
    chambers = _chambers("train")
    n_vertices, n_chamber = 3 * N_ANNULUS + 1, 2 * N_ANNULUS + 1
    root = DERIVED / "train_root"
    out = root / "samples" / "train"
    out.mkdir(parents=True, exist_ok=True)
    for subject in sorted({k[0] for k in chambers}):
        get = {(scan, p): chambers.get((subject, scan, p, "label")) for scan in SCANS for p in PHASES}
        if any(v is None for v in get.values()):
            continue

        def full(chamber):
            vertices = template.copy()
            vertices[:n_chamber] = chamber
            return vertices

        es, ed = full(get[("scan1", "ES")]), full(get[("scan1", "ED")])
        base = 0.5 * (es + ed)
        motion = np.zeros((6, n_vertices, 3))
        motion[0] = 0.5 * (es - ed)                      # cos(2 pi tau): +1 at ES (0), -1 at ED (0.5)
        d_es = np.zeros((n_vertices, 3))
        d_ed = np.zeros((n_vertices, 3))
        d_es[:n_chamber] = get[("scan2", "ES")] - get[("scan1", "ES")]
        d_ed[:n_chamber] = get[("scan2", "ED")] - get[("scan1", "ED")]
        indices = np.arange(n_chamber)
        np.savez_compressed(
            out / f"{subject}.npz", template_vertices=base, motion_coefficients=motion,
            session_static=0.5 * (d_es + d_ed), session_dynamic=0.5 * (d_es - d_ed),
            identity_id=subject, cmr_times=np.array([0.0, 0.5]), cmr_indices=indices,
            cmr_points=np.stack([es[:n_chamber], ed[:n_chamber]]), tee_times=np.array([0.0, 0.5]),
            tee_indices=indices, tee_points=np.stack([es[:n_chamber], ed[:n_chamber]]),
            pose_rotation=np.eye(3), pose_translation=np.zeros(3), phase_offset=0.0,
            phase_logits=np.zeros(8))
    return root


def build_model():
    """The model, learned from part 1's training subjects with E75's recipe."""

    driver = _load("e59_driver", HERE / "e59_unknown_alignment.py")
    spec = json.loads(MODEL_SPEC.read_text())
    root = _train_root()
    model, *rest = driver.build_model(root, spec["models"]["S"])
    return model, spec, {"training_subjects": len(list((root / "samples" / "train").glob("*.npz"))),
                         "recipe": spec["models"]["S"]}


# -------------------------------------------------------------- problems ----

def make_problem(cmr: dict, echo: dict, model, config, *, rotation, translation, offset, held_out=None):
    from rooted4d.geometry.rigid import axis_angle_to_matrix
    from rooted4d.inference.initialization import _matrix_to_axis_angle
    from rooted4d.inference.joint import Observations
    from rooted4d.inference.problem import JointProblem
    from rooted4d.inference.scene import SceneParameters

    vertices = np.arange(2 * N_ANNULUS + 1)
    matrix = axis_angle_to_matrix(np.asarray(rotation))
    cmr_times = np.array([PHASES[p] for p in PHASES])
    cmr_points = np.stack([cmr[p] for p in PHASES])
    tee_times = np.array([(PHASES[p] - offset) % 1.0 for p in PHASES])
    tee_points = np.stack([echo[p] @ matrix.T + translation for p in PHASES])
    order = np.argsort(tee_times)
    tee_times, tee_points = tee_times[order], tee_points[order]
    names = [list(PHASES)[i] for i in order]
    empty_times, empty_points = np.zeros(0), np.zeros((0, len(vertices), 3))
    visible_cmr, hidden_cmr = np.arange(len(PHASES)), np.zeros(0, dtype=int)
    visible_tee, hidden_tee = np.arange(len(PHASES)), np.zeros(0, dtype=int)
    if held_out == "cmr_ED":
        visible_cmr, hidden_cmr = np.array([0]), np.array([1])
    elif held_out == "echo_ED":
        ed = names.index("ED")
        visible_tee, hidden_tee = np.array([1 - ed]), np.array([ed])
    observations = Observations(
        cmr_times=cmr_times[visible_cmr], cmr_vertices=vertices, cmr_points=cmr_points[visible_cmr],
        tee_times=tee_times[visible_tee], tee_vertices=vertices, tee_points=tee_points[visible_tee],
        heldout_cmr_times=cmr_times[hidden_cmr] if len(hidden_cmr) else empty_times,
        heldout_cmr_vertices=vertices if len(hidden_cmr) else np.zeros(0, dtype=int),
        heldout_cmr_points=cmr_points[hidden_cmr] if len(hidden_cmr) else np.zeros((0, 0, 3)),
        heldout_tee_times=tee_times[hidden_tee] if len(hidden_tee) else empty_times,
        heldout_tee_vertices=vertices if len(hidden_tee) else np.zeros(0, dtype=int),
        heldout_tee_points=tee_points[hidden_tee] if len(hidden_tee) else np.zeros((0, 0, 3)))
    rows = 2 * config.fourier_order
    session_rows = 2 * config.session_order + (1 if config.session_includes_offset else 0)
    truth = SceneParameters(
        pose_axis_angle=_matrix_to_axis_angle(matrix), pose_translation=np.asarray(translation, dtype=float),
        phase_offset=float(offset % 1.0), phase_logits=np.zeros(model.n_phase_nodes),
        identity=np.zeros(model.n_modes), motion=np.zeros((rows, model.bases.n_motion)),
        session={name: np.zeros((session_rows, model.bases.n_session)) for name in model.sessions})
    return JointProblem(observations=observations, truth=truth, model=model, config=config,
                        cmr_noise_mm=float("nan"), tee_noise_mm=float("nan"))


def two_frame_alignment(observations):
    """The input-derived start's pose part, with the phase left to the coarse search.

    ``input_derived_alignment`` estimates phase by dynamic time warping, which
    needs at least three frames, and MITEA has two per modality. Its pose part,
    the Kabsch fit between the time-averaged positions of the shared vertices,
    needs none, so it is kept; the phase offset starts at zero and the solver's
    24-offset coarse search, which runs for every fit, chooses it.
    """

    from rooted4d.inference.initialization import StartingAlignment, _columns, _matrix_to_axis_angle
    from rooted4d.inference.phase_pose import kabsch

    shared = np.intersect1d(observations.cmr_vertices, observations.tee_vertices)
    cmr = np.asarray(observations.cmr_points)[:, _columns(np.asarray(observations.cmr_vertices), shared), :]
    tee = np.asarray(observations.tee_points)[:, _columns(np.asarray(observations.tee_vertices), shared), :]
    transform = kabsch(cmr.mean(axis=0), tee.mean(axis=0))
    return StartingAlignment(pose_axis_angle=_matrix_to_axis_angle(transform.rotation),
                             pose_translation=np.asarray(transform.translation, dtype=np.float64),
                             phase_offset=0.0, shared_vertices=int(len(shared)), temporal_cost=float("nan"))


def session_recovery(parameters, model, config, measured: dict, total: dict) -> dict:
    """Fitted cross-session difference (echo minus CMR session) against the two differences."""

    from rooted4d.inference.joint import latent_trajectory

    phases = np.array([PHASES[p] for p in PHASES])
    rows = model.observation_rows(np.arange(2 * N_ANNULUS + 1))
    cmr = model.physical_coordinates(latent_trajectory(parameters, phases, "cmr", model, config))[:, rows]
    tee = model.physical_coordinates(latent_trajectory(parameters, phases, "tee", model, config))[:, rows]
    estimate = tee - cmr
    out = {}
    for name, difference in (("scan_rescan", measured), ("total", total)):
        target = np.stack([difference[p] for p in PHASES])
        out[f"recovered_{name}"] = float(1.0 - ((estimate - target) ** 2).sum() / (target ** 2).sum())
        out[f"{name}_rms_mm"] = float(np.sqrt((target ** 2).sum(axis=-1).mean()))
    out["estimate_rms_mm"] = float(np.sqrt((estimate ** 2).sum(axis=-1).mean()))
    return out


def cmd_fit(split: str) -> None:
    from rooted4d.inference import SolverConfig
    from rooted4d.inference.baselines import parameters_from_record, run_arm
    if split == "test" and os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing test subjects without R4D_ALLOW_TEST=1")
    stand_in = os.environ.get("R4D_LABELS_AS_PREDICTIONS") == "1"
    model, spec, model_record = build_model()
    # Two frames per modality cannot identify an eight-knot warp: its shape is held
    # (uniform) and only the offset is estimated, for both methods.
    base = replace(SolverConfig(**spec["base_config"]), estimate_phase_warp=False)
    by_name = {v["name"]: v for v in spec["variants"]}
    joint_config = replace(base, **by_name["S/joint"]["overrides"])
    late_options = dict(by_name["S/independent"].get("options") or {}, store_parameters=True)
    chambers = _chambers(split, stand_in)
    subjects = sorted({k[0] for k in chambers})
    rows = []
    for subject in subjects:
        def get(scan, source):
            values = {p: chambers.get((subject, scan, p, source)) for p in PHASES}
            return None if any(v is None for v in values.values()) else values

        pairings = {"same_scan": (get("scan1", "label"), get("scan1", "prediction")),
                    "cross_scan": (get("scan1", "label"), get("scan2", "prediction"))}
        for pairing, (cmr, echo) in pairings.items():
            row = {"subject": subject, "pairing": pairing, "variants": {}}
            if cmr is None or echo is None:
                row["front_end_failure"] = True
                rows.append(row)
                continue
            rng = np.random.default_rng(_seed(subject, pairing))
            rotation = rng.normal(0.0, POSE_ROTATION_SD, size=3)
            translation = rng.normal(0.0, POSE_TRANSLATION_SD, size=3)
            offset = float(rng.uniform(-OFFSET_RANGE, OFFSET_RANGE))
            row["perturbation"] = {"axis_angle": rotation.tolist(), "translation": translation.tolist(),
                                   "offset": offset}
            variants = (None, "echo_ED", "cmr_ED") if pairing == "same_scan" else (None,)
            for held_out in variants:
                problem = make_problem(cmr, echo, model, joint_config, rotation=rotation,
                                       translation=translation, offset=offset, held_out=held_out)
                try:
                    alignment = two_frame_alignment(problem.observations)
                except Exception as error:
                    alignment = None
                    row.setdefault("start_failures", []).append(f"{held_out}: {type(error).__name__}: {error}")
                for arm, config, options in (("rooted4d_full", joint_config, {"store_parameters": True}),
                                             ("late_fusion_trajectory", base, late_options)):
                    name = f"{arm}{'' if held_out is None else '/' + held_out}"
                    started = time.perf_counter()
                    try:
                        if alignment is None:
                            raise RuntimeError("no input-derived start")
                        outcome = run_arm(arm, problem, config, alignment=alignment, options=options).as_dict()
                    except Exception as error:
                        outcome = {"succeeded": False, "failure_reason": f"{type(error).__name__}: {error}"}
                    if (arm == "rooted4d_full" and held_out is None and pairing == "cross_scan"
                            and outcome.get("succeeded")):
                        measured = {p: chambers[(subject, "scan2", p, "label")] - cmr[p] for p in PHASES}
                        total = {p: echo[p] - cmr[p] for p in PHASES}
                        outcome["session"] = session_recovery(
                            parameters_from_record(outcome["extras"]["parameters"]), model, joint_config,
                            measured, total)
                    outcome.pop("extras", None)
                    outcome["seconds"] = round(time.perf_counter() - started, 3)
                    row["variants"][name] = outcome
            rows.append(row)
            print(subject, pairing, {k: (round(v.get("rotation_deg") or float("nan"), 2),
                                         round(v.get("translation_mm") or float("nan"), 2))
                                     for k, v in row["variants"].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_standin" if stand_in else ""
    (OUT / f"E99_fits_{split}{suffix}.json").write_text(json.dumps(
        {"experiment": "E99", "preregistration": PREREGISTRATION, "split": split,
         "labels_as_predictions": stand_in, "model": model_record, "rows": rows}, indent=2, default=float) + "\n")
    print("wrote", OUT / f"E99_fits_{split}{suffix}.json")


# -------------------------------------------------------------- analysis ----

def _paired(first: dict, second: dict) -> dict:
    subjects = sorted(set(first) & set(second))
    a = np.array([first[s] for s in subjects], dtype=float)
    b = np.array([second[s] for s in subjects], dtype=float)
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)), "subjects": len(subjects),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def cmd_analyze() -> None:
    stand_in = os.environ.get("R4D_LABELS_AS_PREDICTIONS") == "1"
    split = os.environ.get("R4D_SPLIT", "test")
    suffix = "_standin" if stand_in else ""
    rows = json.loads((OUT / f"E99_fits_{split}{suffix}.json").read_text())["rows"]
    keys = ("rotation_deg", "translation_mm", "phase_error_cycles", "attachment_residual_mm", "alpha",
            "heldout_cmr_mm", "heldout_tee_mm")
    report = {"experiment": "E99", "preregistration": PREREGISTRATION, "split": split,
              "front_end_failures": sorted({(r["subject"], r["pairing"]) for r in rows if r.get("front_end_failure")}),
              "medians": {}, "rooted4d_minus_late_fusion": {}}
    for pairing in ("same_scan", "cross_scan"):
        subset = [r for r in rows if r["pairing"] == pairing and not r.get("front_end_failure")]
        names = sorted({n for r in subset for n in r["variants"]})
        report["medians"][pairing] = {
            name: {k: float(np.nanmedian([r["variants"][name].get(k) if r["variants"][name].get("succeeded")
                                          and r["variants"][name].get(k) is not None else np.nan
                                          for r in subset]))
                   for k in keys}
            for name in names}
        for suffix_name in ("", "/echo_ED", "/cmr_ED"):
            ours, theirs = f"rooted4d_full{suffix_name}", f"late_fusion_trajectory{suffix_name}"
            if ours not in names:
                continue
            entry = {}
            for k in ("rotation_deg", "translation_mm", "phase_error_cycles", "heldout_cmr_mm", "heldout_tee_mm"):
                first = {r["subject"]: r["variants"][ours].get(k) for r in subset
                         if r["variants"][ours].get("succeeded") and r["variants"][ours].get(k) is not None}
                second = {r["subject"]: r["variants"][theirs].get(k) for r in subset
                          if r["variants"][theirs].get("succeeded") and r["variants"][theirs].get(k) is not None}
                if first and second:
                    entry[k] = _paired(first, second)
            report["rooted4d_minus_late_fusion"][f"{pairing}{suffix_name}"] = entry
    sessions = [r["variants"]["rooted4d_full"]["session"] for r in rows
                if r["pairing"] == "cross_scan" and r["variants"].get("rooted4d_full", {}).get("session")]
    report["session_recovery"] = {k: float(np.median([s[k] for s in sessions])) for k in sessions[0]} if sessions else None
    # With frames at phases 0 and 0.5 only, alpha is zero by structure (the
    # record); its values are reported, not correlated.
    alphas = [r["variants"]["rooted4d_full"].get("alpha") for r in rows
              if r["variants"].get("rooted4d_full", {}).get("succeeded")]
    report["alpha"] = {"values": len(alphas), "nonzero": int(sum(1 for a in alphas if a and a > 1e-8)),
                       "max": float(max(alphas)) if alphas else None}
    (OUT / f"E99_analysis_{split}{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1)[:5000])


if __name__ == "__main__":
    mode = os.environ.get("R4D_MODE")
    split_name = os.environ.get("R4D_SPLIT", "val")
    if mode == "geometry":
        cmd_geometry(split_name)
    elif mode == "fit":
        cmd_fit(split_name)
    elif mode == "analyze":
        cmd_analyze()
    else:
        sys.exit(f"unknown R4D_MODE {mode!r}")
