"""E109: qualitative samples at 96 ring nodes, and the published methods on the same hearts.

Not an experiment: nothing here is selected or scored for the paper. E78 and E81
drew the qualitative samples on the 24-node benchmarks. The paper's tables are
now at 96 ring nodes (E100), so this draws them again there, and adds the two
published methods the comparison table could run, TetHeart and GHDHeart
(Supplementary panel A), on the chamber donors of the same fusion samples.

Samples are chosen by rule, as in E81: the realistic-session test samples at the
25th, 50th and 75th percentile of Rooted4D's held-out CMR + TEE error in E100,
and the worst. Each fusion sample's chamber is an ACDC test patient's, so the
published methods are drawn on that patient's labelled frames.

Parts (``R4D_PART``), each writing under ``$R4D_OUT/qualitative96``:

* ``picks``: the fusion samples and their chamber donors' ACDC frames
  (``picks.json``). Each donor is refitted and must reproduce its stored donor
  vertices exactly, which proves the donor-to-patient map (as E108 found).
* ``fusion`` (v4n96o test, E100's main configuration): inputs, the session-free
  anatomy over eight phases of Rooted4D and of late fusion (both from the
  parameters E100 stored, so they are the table's own fits), and refits of exact
  repair, soft attachment and Rooted4D with the global basis, each checked
  against E100's rows. Rooted4D is refitted as well, as a code check.
* ``image`` (v3n96o test, E100's E70 and E74): image evidence from the zero
  start, with two commissures or eight landmarks, and from the warm start.
* ``modes`` (the v4n96o shape spaces): E81's sweeps at 96 nodes.
* ``chamber`` (ACDC test): Rooted4D's chamber on the donors' labelled frames,
  from labels (E36's fit) and behind nnU-Net (E84's) and the compact U-Net.
* ``tetheart`` (TetHeart's environment): its stored test predictions for the
  same frames (E49g), in millimetres, as predicted and clipped to the label's
  stack extent, scored as E49g scored them.
* ``ghdheart`` (GHDHeart's environment, GPU): GHDHeart refitted on the same
  frames at its validated setting (E37: 64 basis functions, 2000 iterations).
  E37 did not seed its global registration, so this is a re-run, not a
  reproduction; the index records both scores.
* ``render`` (locally, ``R4D_IN=<qualitative96 dir>``): every panel from the
  saved arrays, and the checks of the chamber parts against the committed rows.

    R4D_ALLOW_TEST=1 R4D_PART=<part> python e109_qualitative_96.py
    R4D_PART=render R4D_IN=results/qualitative96 python e109_qualitative_96.py
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

QUANTILES = (0.25, 0.50, 0.75)
PHASES = np.linspace(0.0, 1.0, 8, endpoint=False)
OUT = Path(os.environ.get("R4D_OUT", "results")) / "qualitative96"
JOINT = "S/joint"
LATE = "S/late_fusion_trajectory"
#: A refit must land within this of E100's row (re-runs are not bitwise; the
#: tables report 0.01 mm). Anything larger means a different configuration.
REFIT_TOLERANCE_MM = 5e-3
CAVITY = 3
CHAMBER_RINGS, CHAMBER_ANGLES = 8, 32          # panel A's chamber surface (E36, E84)
DONOR_ANGLES = 96                              # the 96-node donor pool (E108)
GHD_BASIS, GHD_SEED = 64, 20261006


# ------------------------------------------------------------------ shared ----

def _rows(directory) -> list:
    rows = []
    for shard in sorted(Path(directory).glob("shard_*.jsonl")):
        rows += [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    return rows


def _pick(scores: dict, worst: bool) -> dict:
    ordered = sorted(scores, key=lambda k: (scores[k], k))
    picks = {f"q{int(p * 100):02d}": ordered[int(round(p * (len(ordered) - 1)))] for p in QUANTILES}
    if worst:
        picks["worst"] = ordered[-1]
    return picks


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_index(name: str, report) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_index.json").write_text(json.dumps(_clean(report), indent=2) + "\n")
    print("wrote", OUT / f"{name}_index.json", flush=True)


def _picks() -> dict:
    return json.loads(Path(os.environ.get("R4D_PICKS", str(OUT / "picks.json"))).read_text())


def _acdc() -> Path:
    return Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))


def _frame_number(path: Path) -> int:
    return int(path.name.split("_frame")[-1].split("_")[0])


def _labels(path: Path):
    import nibabel as nib

    handle = nib.load(str(path))
    return np.asarray(handle.dataobj), tuple(float(z) for z in handle.header.get_zooms()[:3])


def _frames():
    """(pick label, patient, frame, label path) for every frame drawn in the chamber parts."""

    for entry in _picks()["samples"]:
        for frame in entry["frames"]:
            yield entry["label"], entry["patient"], frame["frame"], _acdc() / frame["label_path"]


def _external_destination(label: str, patient: str) -> Path:
    return OUT / "external" / f"{label}_{patient}"


# ------------------------------------------------------------------- picks ----

def run_picks() -> dict:
    from rooted4d.active.acdc import fit_lv_surface
    from rooted4d.donors.export import _canonicalize, chamber_donor_vertices
    from rooted4d.geometry.toy import make_toy_complex

    directory = Path(os.environ["R4D_E100_ROWS"])
    spec = json.loads((directory / "spec.json").read_text())
    rows = {r["sample"]: r for r in _rows(directory)}
    scores = {s: r["variants"][JOINT]["heldout_cmr_mm"] + r["variants"][JOINT]["heldout_tee_mm"]
              for s, r in rows.items() if r["variants"][JOINT].get("succeeded")}
    root = Path(spec["benchmark"])
    acdc = _acdc()
    # build_donor_pool numbers the chamber donors in sorted patient order.
    folders = sorted({p.parent for p in acdc.rglob("*_gt.nii.gz")}, key=lambda p: p.name)
    donors = Path(os.environ.get("R4D_DONORS", "data/derived/donors_v3_n96"))
    report = {"benchmark": str(root), "rows": str(directory), "rule": "q25, q50, q75 and worst of Rooted4D's "
              "held-out CMR + TEE error (S/joint) on the test split", "samples": []}
    for label, name in _pick(scores, worst=True).items():
        with np.load(root / "samples" / "test" / name, allow_pickle=True) as blob:
            heart = str(blob["whole_heart_donor_id"])
            valve = str(blob["valve_donor_id"])
            identity = str(blob["identity_id"])
            n_annulus = (len(blob["template_vertices"]) - 1) // 3
        index = int(heart.rsplit("-", 1)[1])
        folder = folders[index]
        paths = sorted(folder.glob("*_gt.nii.gz"))
        # The pool kept the first frame that fits, in sorted path order (E108).
        donor_frame, fit = None, None
        for path in paths:
            labels, zooms = _labels(path)
            fit = fit_lv_surface(labels, zooms, n_rings=CHAMBER_RINGS, n_angles=DONOR_ANGLES,
                                 patient=folder.name, frame=_frame_number(path))
            if fit is not None:
                donor_frame = _frame_number(path)
                break
        chamber = make_toy_complex(n_annulus=n_annulus).groups["cmr_visible"]
        stored = np.load(donors / "fixed_topology" / f"heart-{index:04d}.npz")["vertices"]
        reproduces = float(np.abs(_canonicalize(chamber_donor_vertices(fit, n_annulus=n_annulus))[chamber]
                                  - stored[chamber]).max())
        if reproduces > 1e-9:
            raise SystemExit(f"{heart} does not reproduce from {folder.name} ({reproduces} mm)")
        entry = {"label": label, "sample": name, "identity": identity, "chamber_donor": heart,
                 "valve_donor": valve, "patient": folder.name, "donor_frame": donor_frame,
                 "donor_reproduces_mm": reproduces, "heldout_sum_mm": scores[name],
                 "frames": [{"frame": _frame_number(p), "label_path": str(p.relative_to(acdc)),
                             "donor": _frame_number(p) == donor_frame} for p in paths]}
        report["samples"].append(entry)
        print(json.dumps(entry), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "picks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


# ------------------------------------------------------------------ fusion ----

def _detached_sides(parameters, detached, parent, config, phases):
    """A soft-attachment fit's chamber side, valve side and ring copy, each over every physical vertex.

    The detached model carries each component's own copy of the shared ring: the
    ring component's (the attachment map's ``annulus``), the chamber's and the
    leaflets'. The penalty ties every copy to the ring component's, and point
    observations of the ring land on that copy (``observation_index`` takes the
    first occurrence), so it is there that the fit's interface gap opens. The
    chamber side reads its ring from the chamber block and the valve side from the
    leaflet blocks; the third array is the ring component's copy.
    """

    from dataclasses import replace

    from rooted4d.inference.joint import latent_trajectory

    latent = latent_trajectory(parameters, np.mod(phases, 1.0), "cmr", detached, replace(config, use_session=False))
    local = detached.physical_coordinates(latent)
    blocks, start = {}, 0
    for name, indices in parent.attachment_map.local_to_global.items():
        indices = np.asarray(indices)
        blocks[name] = (indices, local[:, start:start + len(indices)])
        start += len(indices)
    first = local[:, detached.observation_rows(np.arange(parent.n_physical)), :]
    chamber, valve = first.copy(), first.copy()
    for name, (indices, block) in blocks.items():
        if name == "annulus":
            continue
        (chamber if name == "chamber" else valve)[:, indices] = block
    return chamber, valve, first


def _heldout_differences(errors: dict, row: dict) -> dict:
    return {k: float(errors[k] - row[k]) for k in ("heldout_cmr_mm", "heldout_tee_mm")}


def run_fusion() -> list:
    from dataclasses import replace

    import e59_unknown_alignment as e59
    import e78_qualitative_samples as q
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.experiments.fitted_geometry import fitted_trajectory, truth_trajectory
    from rooted4d.geometry.rigid import RigidTransform
    from rooted4d.geometry.toy import make_toy_complex
    from rooted4d.inference import SolverConfig
    from rooted4d.inference.baselines import _start, parameters_from_record
    from rooted4d.inference.initialization import input_derived_alignment, start_from
    from rooted4d.inference.joint import fit_joint_scene
    from rooted4d.inference.repair import exact_repair
    from rooted4d.inference.trajectory_registration import (
        TrajectoryRegistration, inverse_phase, registered_interface_gap,
    )
    from rooted4d.motion import MonotoneCircularPhaseWarp, circular_distance

    directory = Path(os.environ["R4D_E100_ROWS"])
    spec = json.loads((directory / "spec.json").read_text())
    variants = {v["name"]: v for v in spec["variants"]}
    rows = {r["sample"]: r["variants"] for r in _rows(directory)}
    root = Path(spec["benchmark"])
    models = {key: e59.build_model(root, recipe)[0] for key, recipe in spec["models"].items()}
    base = SolverConfig(**spec.get("base_config", {}))

    def config(name):
        return replace(base, **(variants[name].get("overrides") or {}))

    def overrides(name):
        return (variants[name].get("options") or {}).get("modality_overrides") or {}

    S, G = models["S"], models["G"]
    n = S.n_physical
    annulus = np.sort(S.groups["annulus"])
    report = []
    for entry in _picks()["samples"]:
        name, label = entry["sample"], entry["label"]
        path = root / "samples" / "test" / name
        row = rows[name]
        problems = {key: benchmark_problem(path, model, base) for key, model in models.items()}
        # The E59 driver takes the shared start from the first model's problem.
        alignment = input_derived_alignment(next(iter(problems.values())).observations)
        observations = problems["S"].observations
        times = np.asarray(observations.tee_times, dtype=np.float64)
        true_map = np.asarray(problems["S"].truth.warp().evaluate(times), dtype=np.float64)
        geometry, maps, checks = {}, {"truth": true_map}, {}

        def phase_check(key, estimated, variant):
            maps[key] = np.asarray(estimated, dtype=np.float64)
            value = float(np.median(circular_distance(maps[key], true_map)))
            checks.setdefault(variant, {})["phase_map_cycles"] = value - row[variant]["phase_map_cycles"]

        # Rooted4D: the stored fit is the table's; the refit is a code check.
        joint_config = config(JOINT)
        stored = parameters_from_record(row[JOINT]["extras"]["parameters"])
        geometry["rooted4d"] = fitted_trajectory(stored, S, joint_config, PHASES, n)
        phase_check("rooted4d", stored.warp().evaluate(times), JOINT)
        refit = fit_joint_scene(observations, S, joint_config, start=_start(alignment, S, joint_config))
        checks[JOINT].update(_heldout_differences(refit.heldout_errors(), row[JOINT]))
        checks[JOINT]["refit_geometry_vs_stored_mm"] = float(np.abs(
            fitted_trajectory(refit.parameters, S, joint_config, PHASES, n) - geometry["rooted4d"]).max())

        # Late fusion: both reconstructions and the trajectory registration, as stored.
        extras = row[LATE]["extras"]
        cmr, tee = parameters_from_record(extras["cmr"]), parameters_from_record(extras["tee"])
        stored_registration = extras["registration"]
        registration = TrajectoryRegistration(
            transform=RigidTransform(np.asarray(stored_registration["rotation"], dtype=np.float64),
                                     np.asarray(stored_registration["translation"], dtype=np.float64)),
            warp=MonotoneCircularPhaseWarp(np.asarray(stored_registration["logits"], dtype=np.float64),
                                           offset=float(stored_registration["offset"])),
            rms_mm=float("nan"), cost=float("nan"), iterations=0)
        late_cmr_config = replace(base, use_tee=False, estimate_pose=False, estimate_phase_warp=False,
                                  use_session=False, **overrides(LATE).get("cmr", {}))
        late_tee_config = replace(base, use_cmr=False, use_session=False, **overrides(LATE).get("tee", {}))
        geometry["late_fusion_chamber"] = fitted_trajectory(cmr, S, late_cmr_config, PHASES, n)
        geometry["late_fusion_valve"] = registration.transform.apply(fitted_trajectory(
            tee, S, late_tee_config, inverse_phase(registration.warp, PHASES), n))
        phase_check("late_fusion", registration.warp.evaluate(tee.warp().evaluate(times)), LATE)
        checks[LATE]["interface_gap_mm"] = (registered_interface_gap(cmr, tee, registration, S, late_cmr_config)
                                            - row[LATE]["attachment_residual_mm"])

        # Exact repair, refitted.
        repaired = exact_repair(observations, S, config("S/exact_repair"), alignment=alignment,
                                modality_overrides=overrides("S/exact_repair"))
        geometry["exact_repair_chamber"] = fitted_trajectory(repaired["cmr"].parameters, S, repaired["cmr"].config,
                                                             PHASES, n)
        geometry["exact_repair_valve"] = fitted_trajectory(repaired["tee"].parameters, S, repaired["tee"].config,
                                                           PHASES, n)
        phase_check("exact_repair", repaired["tee"].parameters.warp().evaluate(times), "S/exact_repair")
        checks["S/exact_repair"].update({
            "heldout_cmr_mm": repaired["cmr"].heldout_errors()["heldout_cmr_mm"]
            - row["S/exact_repair"]["heldout_cmr_mm"],
            "heldout_tee_mm": repaired["tee"].heldout_errors()["heldout_tee_mm"]
            - row["S/exact_repair"]["heldout_tee_mm"],
            "ring_disagreement_drawn_mm": float(np.abs(geometry["exact_repair_chamber"][:, annulus]
                                                       - geometry["exact_repair_valve"][:, annulus]).max())})

        # Soft attachment at its selected penalty, refitted (run_arm's construction).
        soft = variants["S/soft_matched"]
        detached = S.detached(matched=True)
        soft_config = replace(config("S/soft_matched"), soft_attachment_weight=soft["soft_attachment_weight"])
        fit = fit_joint_scene(observations, detached, soft_config, start=_start(alignment, detached, soft_config))
        geometry["soft_chamber"], geometry["soft_valve"], geometry["soft_ring"] = _detached_sides(
            fit.parameters, detached, S, soft_config, PHASES)
        phase_check("soft", fit.parameters.warp().evaluate(times), "S/soft_matched")
        checks["S/soft_matched"].update(_heldout_differences(fit.heldout_errors(), row["S/soft_matched"]))
        checks["S/soft_matched"]["ring_gap_drawn_mm"] = float(np.abs(
            geometry["soft_chamber"][:, annulus] - geometry["soft_valve"][:, annulus]).max())
        # Over the eight drawn phases, without sessions; the row's residual is the fit's own.
        checks["S/soft_matched"]["ring_copy_gap_drawn_mm"] = float(np.abs(
            geometry["soft_ring"][:, annulus] - geometry["soft_chamber"][:, annulus]).max())

        # Rooted4D with the global basis, refitted.
        global_config = config("G/joint")
        fit = fit_joint_scene(problems["G"].observations, G, global_config,
                              start=_start(alignment, G, global_config))
        geometry["rooted4d_global"] = fitted_trajectory(fit.parameters, G, global_config, PHASES, G.n_physical)
        phase_check("rooted4d_global", fit.parameters.warp().evaluate(times), "G/joint")
        checks["G/joint"].update(_heldout_differences(fit.heldout_errors(), row["G/joint"]))

        worst = max(abs(v) for arm in checks.values() for k, v in arm.items()
                    if k.startswith("heldout") or k in ("phase_map_cycles", "interface_gap_mm"))
        if worst > REFIT_TOLERANCE_MM:
            raise SystemExit(f"{name}: a drawn fit is {worst} from E100's rows: {checks}")

        with np.load(path, allow_pickle=True) as blob:
            truth = truth_trajectory(blob)(PHASES)
            cmr_images = np.asarray(blob["cmr_surface_likelihood"], dtype=np.float32)
            tee_images = np.asarray(blob["tee_surface_likelihood"], dtype=np.float32)
        start = start_from(alignment, S, joint_config).transform()
        tee_in_cmr = q._inverse(start, observations.tee_points.reshape(-1, 3)).reshape(observations.tee_points.shape)
        faces = make_toy_complex(n_annulus=len(annulus)).faces
        destination = OUT / "fusion" / f"{label}_{Path(name).stem}"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "data.npz", phases=PHASES, truth=truth, annulus=annulus,
            chamber_faces=faces["chamber"], anterior_faces=faces["anterior_leaflet"],
            posterior_faces=faces["posterior_leaflet"],
            anterior_vertices=np.asarray(S.attachment_map.local_to_global["anterior_leaflet"]),
            posterior_vertices=np.asarray(S.attachment_map.local_to_global["posterior_leaflet"]),
            cmr_points=observations.cmr_points, cmr_times=observations.cmr_times,
            cmr_vertices=observations.cmr_vertices, tee_points=observations.tee_points, tee_times=times,
            tee_vertices=observations.tee_vertices, tee_points_in_cmr_frame_at_start=tee_in_cmr,
            cmr_slice_likelihood=cmr_images, tee_sector_likelihood=tee_images,
            **{f"geometry_{k}": v for k, v in geometry.items()}, **{f"phase_map_{k}": v for k, v in maps.items()})
        published = {key: {m: row[variant][m] for m in ("heldout_cmr_mm", "heldout_tee_mm", "phase_map_cycles",
                                                       "attachment_residual_mm", "rotation_deg", "translation_mm")}
                     for key, variant in (("rooted4d", JOINT), ("late_fusion", LATE),
                                          ("exact_repair", "S/exact_repair"), ("soft", "S/soft_matched"),
                                          ("rooted4d_global", "G/joint"))}
        report_entry = {"label": label, "sample": name, "patient": entry["patient"],
                        "rows": published, "checks": checks, "largest_check_mm_or_cycles": worst}
        report.append(report_entry)
        print(json.dumps(_clean(report_entry)), flush=True)
    return report


# ------------------------------------------------------------------- image ----

def _image_task(task):
    import e70_image_evidence as d

    path, study = task
    d.STATE["study"] = study
    return study, d.fit_sample(path)


def _image_initialise(recipe, base, weight, root, warm):
    import e70_image_evidence as d

    # Only the eight-landmark arm is drawn; the two- and four-landmark arms are
    # independent fits, so leaving them out changes no other arm and keeps the
    # part inside an hour. The image fits from the zero start sit in a narrow
    # basin and do not reproduce across CPU types (a Zen 4 workstation landed
    # 110 degrees away), so this part runs on the nodes E100 ran on.
    d.LANDMARK_COUNTS = (8,)
    d.initialise(recipe, base, weight, root, "e74", warm)


def run_image() -> list:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    os.environ["R4D_KEEP_VECTORS"] = "1"
    os.environ["R4D_BENCH"] = "v3n96o"
    import e59_unknown_alignment as e59
    import e70_image_evidence as d
    from uniformity_runner import frozen_s
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.geometry.toy import make_toy_complex
    from rooted4d.inference.image_evidence import evidence_from_sample
    from rooted4d.inference.joint import _unpack, predict
    from rooted4d.motion.fourier import FourierMotion
    from rooted4d.synthetic.benchmark import session_residual

    recipe, base, weight, root = frozen_s("v3n96o")
    warm = d.load_warm_starts(os.environ["R4D_E69"])
    published74 = {r["sample"]: r["variants"] for r in _rows(os.environ["R4D_E74_ROWS"])}
    published70 = {r["sample"]: r["variants"] for r in _rows(os.environ["R4D_E70_ROWS"])}
    scores = {s: v["commissures_cold"]["rotation_deg"] for s, v in published74.items()
              if v["commissures_cold"].get("succeeded")}
    picks = _pick(scores, worst=False)
    paths = {p.name: p for p in e59.sample_paths(Path(root), "test")}
    tasks = [(str(paths[name]), study) for name in picks.values() for study in ("e74", "e70")]
    fitted = {}
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=mp.get_context("spawn"),
                             initializer=_image_initialise, initargs=(recipe, base, weight, root, warm)) as pool:
        for study, row in pool.map(_image_task, tasks):
            fitted[(row["sample"], study)] = row
    d.initialise(recipe, base, weight, root, "e74", warm)
    model, image = d.STATE["model"], d.STATE["image"]
    faces = make_toy_complex(n_annulus=len(d.STATE["annulus"])).faces
    report = []
    for label, name in picks.items():
        path = paths[name]
        row74, row70 = fitted[(name, "e74")], fitted[(name, "e70")]
        arms = {"point fit": ("point_fit", np.asarray(warm[name])),
                "image, zero start": ("image_cold", row74["variants"]["image_cold"]["vector"]),
                "+ 2 commissures, zero start": ("commissures_cold", row74["variants"]["commissures_cold"]["vector"]),
                "+ 8 landmarks, zero start": ("landmarks_8_cold", row74["variants"]["landmarks_8_cold"]["vector"]),
                "image, warm start": ("image_warm", row70["variants"]["image_warm"]["vector"])}
        checks = {}
        for key, source, published in (("image_cold", row74, published74), ("commissures_cold", row74, published74),
                                       ("landmarks_8_cold", row74, published74), ("image_warm", row70, published70)):
            checks[key] = source["variants"][key]["rotation_deg"] - published[name][key]["rotation_deg"]
        if max(abs(v) for v in checks.values()) > 1e-3:
            raise SystemExit(f"{name}: image fits differ from E100's rows by {checks}")
        problem = benchmark_problem(path, model, d.STATE["point"])
        record = evidence_from_sample(path, model, budget=200, n_cmr_frames=4, n_tee_frames=4)
        when = np.array([float(record.tee_times[0])])
        everything = np.arange(model.n_physical)
        geometry, rotations = {}, {}
        for key, vector in arms.values():
            parameters = _unpack(np.asarray(vector, dtype=np.float64), problem.truth, model, image)
            _, posed = predict(parameters, problem.observations, model, image, cmr_times=np.empty(0),
                               tee_times=when, tee_vertices=everything)
            geometry[key] = posed[0]
            rotations[key] = (row70 if key in ("image_warm", "point_fit") else row74)["variants"][key]["rotation_deg"]
        with np.load(path, allow_pickle=True) as blob:
            tau = float(np.asarray(problem.truth.warp().evaluate(when))[0])
            motion = FourierMotion(blob["template_vertices"], blob["motion_coefficients"])
            shape = motion.evaluate(np.array([tau]))[0] + 0.5 * session_residual(
                np.array([tau]), blob["session_static"], blob["session_dynamic"])[0]
            truth = shape @ np.asarray(blob["pose_rotation"]).T + np.asarray(blob["pose_translation"])
        destination = OUT / "image" / f"{label}_{Path(name).stem}"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "data.npz", truth=truth, annulus=d.STATE["annulus"],
                            anterior_faces=faces["anterior_leaflet"], posterior_faces=faces["posterior_leaflet"],
                            commissures=np.asarray(row74["variants"]["commissures_cold"]["landmarks_used"]),
                            landmarks_8=np.asarray(row74["variants"]["landmarks_8_cold"]["landmarks_used"]),
                            arm_titles=np.asarray(list(arms)), arm_keys=np.asarray([k for k, _ in arms.values()]),
                            rotations=np.asarray([rotations[k] for k, _ in arms.values()]),
                            **{f"geometry_{k}": v for k, v in geometry.items()})
        entry = {"label": label, "sample": name, "rotation_deg": rotations,
                 "difference_from_E100_rows_deg": checks}
        report.append(entry)
        print(json.dumps(_clean(entry)), flush=True)
    return report


# ------------------------------------------------------------------- modes ----

def run_modes() -> list:
    import e59_unknown_alignment as e59
    from rooted4d.geometry.toy import make_toy_complex

    spec = json.loads((Path(os.environ["R4D_E100_ROWS"]) / "spec.json").read_text())
    root = Path(spec["benchmark"])
    report, arrays = [], {}
    for key, recipe in spec["models"].items():
        model = e59.build_model(root, recipe)[0]
        annulus = np.sort(model.groups["annulus"])
        ring_rows = (3 * annulus[:, None] + np.arange(3)).ravel()
        ring_map = model.shape_basis[ring_rows]
        singular = np.linalg.svd(ring_map, compute_uv=False)
        rank = int(np.count_nonzero(singular > 1e-8 * singular[0]))
        null = np.linalg.svd(ring_map)[2][rank:].T
        scales = np.asarray(model.shape_scales)
        sweeps = {"leading": (np.eye(len(scales))[0], float(scales[0]))}
        if null.shape[1]:
            values, vectors = np.linalg.eigh(null.T @ np.diag(scales**2) @ null)
            sweeps["annulus_preserving"] = (null @ vectors[:, -1], float(np.sqrt(values[-1])))
        for name, (direction, std) in sweeps.items():
            shapes = np.stack([model.physical_coordinates(c * std * direction) for c in (-2.0, 0.0, 2.0)])
            arrays[f"{key}_{name}"] = shapes
            moved = float(np.abs(shapes[[0, 2]][:, annulus] - shapes[1][annulus]).max())
            report.append({"space": key, "sweep": name, "std": std, "annulus_moves_mm": moved,
                           "rank": int(model.shape_basis.shape[1]), "ring_map_rank": rank,
                           "annulus_preserving_directions": int(null.shape[1])})
            print(json.dumps(report[-1]), flush=True)
    faces = make_toy_complex(n_annulus=len(annulus)).faces
    destination = OUT / "modes"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "data.npz", annulus=annulus, chamber_faces=faces["chamber"],
                        anterior_faces=faces["anterior_leaflet"], posterior_faces=faces["posterior_leaflet"],
                        summary=np.asarray(json.dumps(report)), **arrays)
    return report


# ----------------------------------------------------------------- chamber ----

def run_chamber() -> list:
    import nibabel as nib

    import e76_chamber_shape_fit as stage_c
    from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
    from rooted4d.experiments.component_eval import (
        cavity_boundary_points, point_to_mesh_mm, symmetric_surface_distance,
    )

    faces = lv_topology_faces(CHAMBER_RINGS, CHAMBER_ANGLES)
    segmenter = stage_c.Segmenter(Path(os.environ["R4D_SEGMENTER"]))
    predictions = Path(os.environ["R4D_NNUNET_ACDC"])
    report = []
    for label, patient, frame, path in _frames():
        labels, zooms = _labels(path)
        image_path = Path(str(path).replace("_gt.nii.gz", ".nii.gz"))
        image = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
        case = path.name.replace("_gt.nii.gz", "")
        nnunet = np.asarray(nib.load(str(predictions / f"{case}.nii.gz")).dataobj)
        compact = segmenter(image_path, labels.shape, zooms[:2])
        points = cavity_boundary_points(labels, zooms, cavity_label=CAVITY)
        arrays = {"image": image, "zooms": np.asarray(zooms), "labels_reference": labels,
                  "labels_nnunet": nnunet.astype(np.int16), "labels_compact": compact.astype(np.int16),
                  "reference_points": points, "faces": faces}
        entry = {"label": label, "patient": patient, "frame": frame, "case": case}
        # E36's label-input fit (panel A's label row) and E84's two image-input fits.
        for key, source, base_first in (("rooted4d_label", labels, None), ("rooted4d_nnunet", nnunet, True),
                                        ("rooted4d_compact", compact, True)):
            fit = fit_lv_surface(source, zooms, n_rings=CHAMBER_RINGS, n_angles=CHAMBER_ANGLES,
                                 patient=patient, frame=frame, base_first=base_first)
            if fit is None:
                entry[key] = None
                continue
            score = symmetric_surface_distance(points, fit.vertices, faces)
            entry[key] = {"symmetric_mm": score["symmetric_mean_mm"], "hausdorff_mm": score["hausdorff_mm"],
                          "volume_ml": stage_c.score(points, fit.vertices, faces)["volume_ml"]}
            arrays[f"{key}_vertices"] = fit.vertices
            arrays[f"{key}_points_distance_mm"] = point_to_mesh_mm(points, fit.vertices, faces)
        destination = _external_destination(label, patient)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / f"frame{frame:02d}_rooted4d.npz", **arrays)
        report.append(entry)
        print(json.dumps(_clean(entry)), flush=True)
    return report


# ---------------------------------------------------------------- tetheart ----

def _tetheart_frame(task):
    """One frame of TetHeart's stored predictions, in millimetres, raw and clipped, scored as E49g."""

    import e49d_tetheart_rescore as t
    from rooted4d.experiments.component_eval import point_to_mesh_mm, symmetric_surface_distance

    label, patient, frame, path, transform = task
    labels, zooms = _labels(Path(path))
    name = f"ACDC_{patient}_frame{frame:02d}"
    mesh = t.load_mesh(t.PRED_DIR / f"{name}_{t.EXPECTED_CAVITY_CLASS}_pred.obj")
    vertices = t.to_millimetres(mesh.vertices, transform)
    faces = np.asarray(mesh.faces)
    reference = t.boundary_points(labels, zooms, CAVITY, False)       # with end caps, as E49g
    clipped_vertices, clipped_faces, how = t.clip_to_z(vertices, faces, reference[:, 2].min(), reference[:, 2].max())
    entry = {"label": label, "patient": patient, "frame": frame, "clip_method": how,
             "stack_extent_mm": {"prediction": float(np.ptp(vertices[:, 2])),
                                 "label": float(np.ptp(reference[:, 2]))}}
    for key, (v, f) in (("raw", (vertices, faces)), ("clipped", (clipped_vertices, clipped_faces))):
        score = symmetric_surface_distance(reference, v, f)           # E49g: R4D_MATCH_E36=1
        entry[key] = {"symmetric_mm": score["symmetric_mean_mm"], "hausdorff_mm": score["hausdorff_mm"]}
    destination = _external_destination(label, patient)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / f"frame{frame:02d}_tetheart.npz", raw_vertices=vertices, raw_faces=faces,
                        clipped_vertices=clipped_vertices, clipped_faces=clipped_faces, reference_points=reference,
                        clipped_points_distance_mm=point_to_mesh_mm(reference, clipped_vertices, clipped_faces))
    return entry


def run_tetheart() -> list:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    transforms = {(row["case"], row["frame"]): row
                  for row in json.loads(Path(os.environ["R4D_TET_FRAME_REPORT"]).read_text())["rows"]
                  if row.get("status") == "written"}
    tasks = [(label, patient, frame, str(path), transforms[(patient, frame)])
             for label, patient, frame, path in _frames()]
    workers = int(os.environ.get("R4D_WORKERS", "4"))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        report = list(pool.map(_tetheart_frame, tasks))
    for entry in report:
        print(json.dumps(_clean(entry)), flush=True)
    return report


# ---------------------------------------------------------------- ghdheart ----

def run_ghdheart() -> list:
    os.environ["R4D_MODE"] = "test"
    os.environ.setdefault("R4D_ITER", "2000")
    os.environ.setdefault("R4D_LR", "0.01")
    import torch

    import e37_ghdheart_chamber as g
    from rooted4d.experiments.component_eval import (
        cavity_boundary_points, point_to_mesh_mm, symmetric_surface_distance,
    )

    report = []
    for label, patient, frame, path in _frames():
        labels, zooms = _labels(path)
        np.random.seed(GHD_SEED)
        torch.manual_seed(GHD_SEED)
        fitted = g.fit_one(labels, zooms, GHD_BASIS)
        entry = {"label": label, "patient": patient, "frame": frame, "basis": GHD_BASIS,
                 "iterations": g.ITERATIONS, "lr": g.LR, "seed": GHD_SEED, "device": g.DEVICE}
        if fitted is None:
            entry["failed"] = True
            report.append(entry)
            continue
        vertices, faces = fitted
        points = cavity_boundary_points(labels, zooms, cavity_label=CAVITY)
        score = symmetric_surface_distance(points, vertices, faces)
        entry.update(symmetric_mm=score["symmetric_mean_mm"], hausdorff_mm=score["hausdorff_mm"])
        destination = _external_destination(label, patient)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / f"frame{frame:02d}_ghdheart.npz", vertices=vertices,
                            faces=np.asarray(faces), reference_points=points,
                            points_distance_mm=point_to_mesh_mm(points, vertices, np.asarray(faces)))
        report.append(entry)
        print(json.dumps(_clean(entry)), flush=True)
    return report


# ------------------------------------------------------------------ render ----

def render(root: Path) -> int:
    import e109_render as r

    return r.render_all(root)


PARTS = {"picks": run_picks, "fusion": run_fusion, "image": run_image, "modes": run_modes,
         "chamber": run_chamber, "tetheart": run_tetheart, "ghdheart": run_ghdheart}


def main() -> int:
    part = os.environ.get("R4D_PART", "")
    if part == "render":
        return render(Path(os.environ["R4D_IN"]))
    if part not in PARTS:
        raise SystemExit(f"R4D_PART must be one of {sorted(PARTS) + ['render']}")
    if os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing to read test samples without R4D_ALLOW_TEST=1")
    report = PARTS[part]()
    if part != "picks":
        _write_index(part, {"part": part, "samples": report})
    return 0


if __name__ == "__main__":
    sys.exit(main())
