"""E81: more qualitative samples for the figures.

Not an experiment: nothing here is selected or scored for the paper. Samples are
chosen by rule, at the 25th, 50th and 75th percentiles of an error and, where a
failure is informative, the worst case. Every figure's arrays are saved
(``data.npz``) on the cluster and the panels are drawn from them afterwards
(``R4D_PART=render``), because the cluster environment has no plotting library.
Each part checks what it draws against the rows the paper reports.

* ``arms`` (v4 test, E75's configuration): late fusion, exact repair, Rooted4D
  with the global basis and with the structured one, against the truth, with
  error maps, the annulus from above, and each arm's estimated phase map.
* ``image`` (v3 test, E70 and E74): image evidence from the zero start, with
  two commissures or eight landmarks, and from the warm start, seen from above
  against the truth, so the rotation error is visible as a turned leaflet.
* ``modes`` (the v4 shape spaces): the structured space's highest-variance
  direction that leaves the annulus fixed, against each space's leading mode.
  The global basis has no annulus-preserving direction.
* ``valve`` (MVSeg2023 test): the TEE volume around the valve with reference
  and nnU-Net labels, the network's leaflet probabilities, and the fitted
  leaflet sheets with distances.
* ``chamber`` (ACDC test): end-diastolic and end-systolic pairs chosen by the
  ejection-fraction error of the image-input pipeline, and its worst frame.

    R4D_ALLOW_TEST=1 R4D_PART=<part> python e81_qualitative_more.py
    R4D_PART=render R4D_IN=<qualitative dir> python e81_qualitative_more.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import e78_qualitative_samples as q  # noqa: E402

OUT = Path(os.environ.get("R4D_OUT", "results")) / "qualitative"
ANTERIOR_C, POSTERIOR_C, CHAMBER_C = q.ANTERIOR_C, q.POSTERIOR_C, q.CHAMBER_C


def _rows(directory):
    rows = []
    for shard in sorted(Path(directory).glob("shard_*.jsonl")):
        rows += [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    return rows


def _pick(scores: dict, worst: bool) -> dict:
    ordered = sorted(scores, key=lambda k: (scores[k], k))
    picks = {f"q{int(p * 100):02d}": ordered[int(round(p * (len(ordered) - 1)))] for p in q.QUANTILES}
    if worst:
        picks["worst"] = ordered[-1]
    return picks


# -------------------------------------------------------------------- arms ----

def run_arms() -> list:
    import e59_unknown_alignment as e59
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.inference import SolverConfig
    from rooted4d.inference.initialization import input_derived_alignment, start_from
    from rooted4d.inference.joint import fit_joint_scene, predict
    from rooted4d.inference.repair import exact_repair

    spec = json.loads(Path("Codes/configs/e75/v4_e75_test.json").read_text())
    variants = {v["name"]: v for v in spec["variants"]}
    root = Path(spec["benchmark"])
    models = {key: e59.build_model(root, recipe)[0] for key, recipe in spec["models"].items()}
    base = spec["base_config"]

    def config(name):
        return SolverConfig(**{**base, **(variants[name].get("overrides") or {})})

    rows = {r["sample"]: r["variants"] for r in _rows(os.environ["R4D_E75_ROWS"])}
    scores = {s: v["S/joint"]["heldout_cmr_mm"] + v["S/joint"]["heldout_tee_mm"]
              for s, v in rows.items() if v["S/joint"].get("succeeded")}
    paths = {p.name: p for p in e59.sample_paths(root, "test")}
    report = []
    for label, name in _pick(scores, worst=True).items():
        path = paths[name]
        problem = {k: benchmark_problem(path, m, SolverConfig(**base)) for k, m in models.items()}
        # The E59 driver takes the shared start from the first model's problem.
        alignment = input_derived_alignment(problem[next(iter(models))].observations)
        observations = problem["S"].observations
        everything = np.arange(models["S"].n_physical)
        geometry, maps, checks = {}, {}, {}
        times = observations.tee_times
        for key in ("S", "G"):
            cfg = config(f"{key}/joint")
            fit = fit_joint_scene(problem[key].observations, models[key], cfg,
                                  start=start_from(alignment, models[key], cfg))
            geometry[f"rooted4d_{key}"], _ = predict(fit.parameters, problem[key].observations, models[key], cfg,
                                                     cmr_times=q.PHASES, cmr_vertices=everything,
                                                     tee_times=np.empty(0))
            maps[f"rooted4d_{key}"] = np.asarray(fit.parameters.warp().evaluate(times))
            held = fit.heldout_errors()
            checks[f"{key}/joint"] = {k: held[k] - rows[name][f"{key}/joint"][k]
                                      for k in ("heldout_cmr_mm", "heldout_tee_mm")}
        independent = config("S/independent")
        overrides = (variants["S/independent"].get("options") or {}).get("modality_overrides", {})
        cmr_fit, cmr_config, tee_fit, unposed, offset, spatial, temporal = q._late_fusion_internals(
            problem["S"], independent, alignment, overrides)
        geometry["late_fusion_chamber"], _ = predict(cmr_fit.parameters, observations, models["S"], cmr_config,
                                                     cmr_times=q.PHASES, cmr_vertices=everything,
                                                     tee_times=np.empty(0))
        own, _ = predict(unposed, observations, models["S"], cmr_config, cmr_times=(q.PHASES + offset) % 1.0,
                         cmr_vertices=everything, tee_times=np.empty(0))
        geometry["late_fusion_valve"] = q._inverse(spatial, tee_fit.parameters.transform().apply(
            own.reshape(-1, 3))).reshape(own.shape)
        maps["late_fusion"] = np.asarray(temporal.target_to_source_phase)
        checks["S/independent"] = {"heldout_cmr_mm": cmr_fit.heldout_errors()["heldout_cmr_mm"]
                                   - rows[name]["S/independent"]["heldout_cmr_mm"],
                                   "heldout_tee_mm": tee_fit.heldout_errors()["heldout_tee_mm"]
                                   - rows[name]["S/independent"]["heldout_tee_mm"]}
        repair_options = (variants["S/exact_repair"].get("options") or {}).get("modality_overrides")
        repaired = exact_repair(observations, models["S"], config("S/exact_repair"), alignment=alignment,
                                modality_overrides=repair_options)
        from dataclasses import replace as _replace

        for side, fit in (("chamber", repaired["cmr"]), ("valve", repaired["tee"])):
            parameters = fit.parameters if side == "chamber" else _replace(
                fit.parameters, pose_axis_angle=np.zeros(3), pose_translation=np.zeros(3))
            geometry[f"exact_repair_{side}"], _ = predict(parameters, observations, models["S"], fit.config,
                                                          cmr_times=q.PHASES, cmr_vertices=everything,
                                                          tee_times=np.empty(0))
        maps["exact_repair"] = np.asarray(repaired["tee"].parameters.warp().evaluate(times))
        checks["S/exact_repair"] = {"heldout_cmr_mm": repaired["cmr"].heldout_errors()["heldout_cmr_mm"]
                                    - rows[name]["S/exact_repair"]["heldout_cmr_mm"],
                                    "heldout_tee_mm": repaired["tee"].heldout_errors()["heldout_tee_mm"]
                                    - rows[name]["S/exact_repair"]["heldout_tee_mm"]}
        worst_gap = max(abs(d) for arm in checks.values() for d in arm.values())
        assert worst_gap < 1e-3, checks
        with np.load(path, allow_pickle=True) as blob:
            truth = q._truth_cmr_frame(blob, q.PHASES)
        maps["truth"] = np.asarray(problem["S"].truth.warp().evaluate(times))
        destination = OUT / "arms" / f"{label}_{Path(name).stem}"
        destination.mkdir(parents=True, exist_ok=True)
        from rooted4d.geometry.toy import make_toy_complex

        faces = make_toy_complex(24).faces
        np.savez_compressed(destination / "data.npz", phases=q.PHASES, truth=truth, tee_times=times,
                            annulus=np.sort(models["S"].groups["annulus"]),
                            chamber_faces=faces["chamber"], anterior_faces=faces["anterior_leaflet"],
                            posterior_faces=faces["posterior_leaflet"],
                            **{f"geometry_{k}": v for k, v in geometry.items()},
                            **{f"phase_map_{k}": v for k, v in maps.items()})
        entry = {"label": label, "sample": name, "rooted4d_S_heldout_sum_mm": scores[name],
                 "largest_difference_from_E75_rows_mm": worst_gap}
        report.append(entry)
        print(json.dumps(entry), flush=True)
    return report


def render_arms(folder: Path) -> None:
    plt = q._plotting()
    with np.load(folder / "data.npz") as z:
        data = {k: z[k] for k in z.files}
    faces = {"chamber": data["chamber_faces"], "anterior_leaflet": data["anterior_faces"],
             "posterior_leaflet": data["posterior_faces"]}
    truth, annulus = data["truth"], data["annulus"]
    arms = [("late fusion", data["geometry_late_fusion_chamber"], data["geometry_late_fusion_valve"]),
            ("exact repair", data["geometry_exact_repair_chamber"], data["geometry_exact_repair_valve"]),
            ("Rooted4D, global basis", data["geometry_rooted4d_G"], data["geometry_rooted4d_G"]),
            ("Rooted4D", data["geometry_rooted4d_S"], data["geometry_rooted4d_S"])]
    everything = np.concatenate([truth.reshape(-1, 3), data["geometry_rooted4d_S"].reshape(-1, 3)])
    centre, half = everything.mean(axis=0), 0.55 * np.max(np.ptp(everything, axis=0))
    limits = [(c - half, c + half) for c in centre]
    phase = 0
    fig = plt.figure(figsize=(6.6, 2.7))
    for index, (title, chamber, valve) in enumerate(arms):
        ax = q._axes3d(fig, int(f"14{index + 1}"), title, limits)
        ax.set_box_aspect((1, 1, 1), zoom=1.25)
        error_chamber = np.linalg.norm(chamber[phase] - truth[phase], axis=1)
        error_valve = np.linalg.norm(valve[phase] - truth[phase], axis=1)
        collection = q._surface(ax, chamber[phase], faces["chamber"], None, 0.9, error_chamber, vmax=5.0)
        for key in ("anterior_leaflet", "posterior_leaflet"):
            q._surface(ax, valve[phase], faces[key], None, 0.9, error_valve, vmax=5.0)
    fig.colorbar(collection, ax=fig.axes, shrink=0.6, pad=0.02, label="error (mm)")
    fig.savefig(folder / "arms_error_maps.png")
    plt.close(fig)
    ring = truth[phase][annulus]
    centre_ring = ring.mean(axis=0)
    plane = np.linalg.svd(ring - centre_ring)[2][:2]

    def flat(points):
        return (np.asarray(points) - centre_ring) @ plane.T

    loop = np.append(annulus, annulus[0])
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.9), sharex=True, sharey=True)
    for ax, (title, chamber, valve) in zip(axes, arms):
        ax.plot(*flat(truth[phase][loop]).T, color="0.6", lw=2.4)
        ax.plot(*flat(chamber[phase][loop]).T, color=CHAMBER_C, lw=1.1)
        if valve is not chamber:
            ax.plot(*flat(valve[phase][loop]).T, "--", color=ANTERIOR_C, lw=1.1)
        ax.set_title(title, fontsize=6.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("grey: true annulus; blue: the chamber side's annulus; red dashed: the valve side's, where the "
                 "two are separate reconstructions", fontsize=6, y=0.02)
    fig.savefig(folder / "arms_annulus_top_view.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    times = data["tee_times"]
    truth_map = data["phase_map_truth"]
    for key, label, colour, style in (("late_fusion", "late fusion", "0.5", "--"),
                                      ("exact_repair", "exact repair", POSTERIOR_C, ":"),
                                      ("rooted4d_G", "Rooted4D, global basis", ANTERIOR_C, "-."),
                                      ("rooted4d_S", "Rooted4D", "k", "-")):
        estimate = data[f"phase_map_{key}"]
        # Unwrap each map relative to the truth so a circular error reads as a gap, not a jump.
        error = (estimate - truth_map + 0.5) % 1.0 - 0.5
        # The legend carries the arm's reported statistic for this sample: the median absolute error.
        ax.plot(times, error, style, color=colour, lw=1.1, label="%s (%.3f)" % (label, np.median(np.abs(error))))
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel("TEE acquisition time (normalized)")
    ax.set_ylabel("phase error (cycles)")
    ax.legend(fontsize=6, frameon=False, title="median absolute error", title_fontsize=6, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(folder / "phase_map_errors.png")
    plt.close(fig)


# ------------------------------------------------------------------- image ----

def run_image() -> list:
    import e59_unknown_alignment as e59
    import e70_image_evidence as d
    from uniformity_runner import frozen_s
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.inference.image_evidence import evidence_from_sample
    from rooted4d.inference.joint import _unpack, predict
    from rooted4d.motion.fourier import FourierMotion
    from rooted4d.synthetic.benchmark import session_residual

    os.environ["R4D_KEEP_VECTORS"] = "1"
    recipe, base, weight, root = frozen_s("v3")
    warm = d.load_warm_starts(os.environ["R4D_E69"])
    published = {r["sample"]: r["variants"] for r in _rows(os.environ["R4D_E74_ROWS"])}
    published70 = {r["sample"]: r["variants"] for r in _rows(os.environ["R4D_E70_ROWS"])}
    scores = {s: v["commissures_cold"]["rotation_deg"] for s, v in published.items()}
    paths = {p.name: p for p in e59.sample_paths(Path(root), "test")}
    d.initialise(recipe, base, weight, root, "e74", warm)
    model, image = d.STATE["model"], d.STATE["image"]
    report = []
    for label, name in _pick(scores, worst=False).items():
        path = paths[name]
        d.STATE["study"] = "e74"
        row74 = d.fit_sample(str(path))
        d.STATE["study"] = "e70"
        row70 = d.fit_sample(str(path))
        arms = {"point fit": ("point_fit", np.asarray(warm[name])),
                "image, zero start": ("image_cold", row74["variants"]["image_cold"]["vector"]),
                "+ 2 commissures, zero start": ("commissures_cold", row74["variants"]["commissures_cold"]["vector"]),
                "+ 8 landmarks, zero start": ("landmarks_8_cold", row74["variants"]["landmarks_8_cold"]["vector"]),
                "image, warm start": ("image_warm", row70["variants"]["image_warm"]["vector"])}
        for key, source in (("image_cold", row74), ("commissures_cold", row74), ("landmarks_8_cold", row74),
                            ("image_warm", row70)):
            reference = (published if source is row74 else published70)[name][key]["rotation_deg"]
            assert abs(source["variants"][key]["rotation_deg"] - reference) < 1e-6, (key, reference)
        problem = benchmark_problem(path, model, d.STATE["point"])
        record = evidence_from_sample(path, model, budget=200, n_cmr_frames=4, n_tee_frames=4)
        when = np.array([float(record.tee_times[0])])
        everything = np.arange(model.n_physical)
        geometry, rotations = {}, {}
        for title, (key, vector) in arms.items():
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
        from rooted4d.geometry.toy import make_toy_complex

        faces = make_toy_complex(24).faces
        destination = OUT / "image" / f"{label}_{Path(name).stem}"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "data.npz", truth=truth, annulus=np.sort(model.groups["annulus"]),
                            anterior_faces=faces["anterior_leaflet"], posterior_faces=faces["posterior_leaflet"],
                            commissures=np.asarray(row74["variants"]["commissures_cold"]["landmarks_used"]),
                            landmarks_8=np.asarray(row74["variants"]["landmarks_8_cold"]["landmarks_used"]),
                            arm_titles=np.asarray(list(arms)), arm_keys=np.asarray([k for k, _ in arms.values()]),
                            rotations=np.asarray([rotations[k] for k, _ in arms.values()]),
                            **{f"geometry_{k}": v for k, v in geometry.items()})
        entry = {"label": label, "sample": name, "rotation_deg": rotations}
        report.append(entry)
        print(json.dumps(entry), flush=True)
    return report


def render_image(folder: Path) -> None:
    plt = q._plotting()
    with np.load(folder / "data.npz") as z:
        data = {k: z[k] for k in z.files}
    truth, annulus = data["truth"], data["annulus"]
    plane = np.linalg.svd(truth[annulus] - truth[annulus].mean(axis=0))[2][:2]

    def flat(points):
        # Down the true annular axis, with the shape's own annulus centroid at the origin: a fit
        # that lands in the wrong place is shown for its rotation alone, which is what the panel is for.
        points = np.asarray(points)
        return (points - points[annulus].mean(axis=0)) @ plane.T

    loop = np.append(annulus, annulus[0])
    keys, rotations = [str(k) for k in data["arm_keys"]], data["rotations"]
    names = {"point_fit": "point fit\n(the warm start)", "image_cold": "image\nfrom the zero start",
             "commissures_cold": "image + 2 commissures\nfrom the zero start",
             "landmarks_8_cold": "image + 8 landmarks\nfrom the zero start",
             "image_warm": "image\nfrom the warm start"}
    reference = flat(truth)
    shapes = {key: flat(data[f"geometry_{key}"]) for key in keys}
    extent = 1.08 * np.abs(np.concatenate([reference] + list(shapes.values()))).max()
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6, 1.75))
    for ax, key, rotation in zip(axes, keys, rotations):
        shape = shapes[key]
        for faces, colour in ((data["anterior_faces"], "0.75"), (data["posterior_faces"], "0.88")):
            for face in faces:
                ax.fill(*reference[face].T, color=colour, lw=0)
        ax.plot(*reference[loop].T, color="0.45", lw=1.6)
        for faces, colour in ((data["anterior_faces"], ANTERIOR_C), (data["posterior_faces"], POSTERIOR_C)):
            for face in faces:
                ax.fill(*shape[face].T, color=colour, alpha=0.35, lw=0)
        ax.plot(*shape[loop].T, color="k", lw=1.0)
        ax.plot(*shape[annulus[:1]].T, "o", color="k", ms=2.5)
        ax.plot(*reference[annulus[:1]].T, "o", color="0.45", ms=2.5)
        if "commissures" in key:
            ax.plot(*reference[data["commissures"]].T, "^", color="#2ca02c", ms=3.5)
        if "landmarks_8" in key:
            ax.plot(*reference[data["landmarks_8"]].T, "^", color="#2ca02c", ms=3.0)
        ax.set_title("%s\n%.1f\u00b0" % (names.get(key, key), rotation), fontsize=5.8)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Seen down the true annular axis, each fit moved onto the true annulus centre, so only rotation "
                 "shows.\nGrey: truth; red and orange: the fitted anterior and posterior leaflets; dots: vertex 0; "
                 "triangles: landmarks.", fontsize=5.5, y=0.1)
    fig.savefig(folder / "image_evidence_top_view.png")
    plt.close(fig)


# ------------------------------------------------------------------- modes ----

def run_modes() -> list:
    import e59_unknown_alignment as e59

    spec = json.loads(Path("Codes/configs/e62/v4_gate.json").read_text())
    root = Path(spec["benchmark"])
    from rooted4d.geometry.toy import make_toy_complex

    faces = make_toy_complex(24).faces
    report, arrays = [], {}
    for key, recipe in spec["models"].items():
        model = e59.build_model(root, recipe)[0]
        annulus = np.sort(model.groups["annulus"])
        ring_rows = (3 * annulus[:, None] + np.arange(3)).ravel()
        ring_map = model.shape_basis[ring_rows]
        singular = np.linalg.svd(ring_map, compute_uv=False)
        rank = int(np.count_nonzero(singular > 1e-8 * singular[0]))
        right = np.linalg.svd(ring_map)[2]
        null = right[rank:].T
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
    destination = OUT / "modes"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "data.npz", annulus=annulus, chamber_faces=faces["chamber"],
                        anterior_faces=faces["anterior_leaflet"], posterior_faces=faces["posterior_leaflet"],
                        summary=np.asarray(json.dumps(report)), **arrays)
    return report


def render_modes(folder: Path) -> None:
    plt = q._plotting()
    with np.load(folder / "data.npz") as z:
        data = {k: z[k] for k in z.files}
    summary = {(r["space"], r["sweep"]): r for r in json.loads(str(data["summary"]))}
    faces = {"chamber": data["chamber_faces"], "anterior_leaflet": data["anterior_faces"],
             "posterior_leaflet": data["posterior_faces"]}
    annulus = data["annulus"]
    rows = [("S", "annulus_preserving", "structured, annulus fixed: highest-variance direction"),
            ("S", "leading", "structured: leading mode"),
            ("G", "leading", "global: leading mode (none fixes the annulus)")]
    everything = np.concatenate([data[f"{s}_{n}"].reshape(-1, 3) for s, n, _ in rows])
    centre, half = everything.mean(axis=0), 0.55 * np.max(np.ptp(everything, axis=0))
    limits = [(c - half, c + half) for c in centre]
    fig = plt.figure(figsize=(6.6, 5.8))
    fig.subplots_adjust(hspace=0.38, wspace=0.0)
    for row, (space, sweep, title) in enumerate(rows):
        shapes = data[f"{space}_{sweep}"]
        first = None
        for column, label in enumerate(("-2 sd", "mean", "+2 sd")):
            ax = q._axes3d(fig, int(f"33{row * 3 + column + 1}"), label, limits)
            ax.set_box_aspect((1, 1, 1), zoom=1.25)  # the shapes are tall and narrow; use the panel
            first = first or ax
            q._surface(ax, shapes[column], faces["chamber"], CHAMBER_C, 0.15)
            q._surface(ax, shapes[column], faces["anterior_leaflet"], ANTERIOR_C, 0.6)
            q._surface(ax, shapes[column], faces["posterior_leaflet"], POSTERIOR_C, 0.6)
            q._ring(ax, shapes[1], annulus, "0.6", width=2.0)
            q._ring(ax, shapes[column], annulus, "k", width=1.0)
        entry = summary[(space, sweep)]
        fig.text(0.5, first.get_position().y1 + 0.035,
                 "%s (annulus moves %.2f mm)" % (title, entry["annulus_moves_mm"]),
                 ha="center", va="bottom", fontsize=7, weight="bold")
    fig.savefig(folder / "shape_space_sweeps.png")
    plt.close(fig)


# ------------------------------------------------------------------- valve ----

def run_valve() -> list:
    import nibabel as nib

    from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL, fit_valve
    from rooted4d.experiments.component_eval import leaflet_faces, point_to_mesh_mm

    # label:case pairs joined by "+", because sbatch --export splits its value on commas.
    cases = dict(item.split(":") for item in os.environ["R4D_CASES"].split("+"))
    root = Path(os.environ.get("R4D_MVSEG", "data/raw/mvseg2023")) / "test"
    predictions = Path(os.environ["R4D_NNUNET_TEST"])
    probabilities = Path(os.environ["R4D_PROBABILITIES"])
    report = []
    for label, case in cases.items():
        us = nib.load(str(root / f"{case}-US.nii.gz"))
        reference = np.asarray(nib.load(str(root / f"{case}-label.nii.gz")).dataobj)
        predicted = np.asarray(nib.load(str(predictions / f"{case}.nii.gz")).dataobj)
        zooms = tuple(float(z) for z in us.header.get_zooms()[:3])
        volume = np.asarray(us.dataobj, dtype=np.float32)
        with np.load(probabilities / f"{case}.npz") as blob:
            probability = blob["probabilities"]
        # nnU-Net reads NIfTI through SimpleITK, so its arrays are (z, y, x); nibabel's are (x, y, z).
        # Always reorder: a shape test cannot tell the two apart when x and z are equally long (test_002).
        probability = probability.transpose(0, 3, 2, 1)
        assert probability.shape[1:] == predicted.shape, (probability.shape, predicted.shape)
        agreement = float((probability.argmax(axis=0) == predicted).mean())
        assert agreement > 0.99, agreement
        occupied = np.argwhere((reference > 0) | (predicted > 0))
        low = np.maximum(occupied.min(axis=0) - 12, 0)
        high = np.minimum(occupied.max(axis=0) + 13, reference.shape)
        box = tuple(slice(a, b) for a, b in zip(low, high))
        sheets, distances = {}, {}
        for source, labels in (("predicted", predicted), ("reference", reference)):
            fit = fit_valve(labels, zooms, n_annulus=32, n_depth=6, case=case)
            for leaflet, sheet, bins, code in (("anterior", fit.anterior, fit.anterior_root_indices, ANTERIOR_LABEL),
                                               ("posterior", fit.posterior, fit.posterior_root_indices,
                                                POSTERIOR_LABEL)):
                faces = leaflet_faces(bins, 6, 32)
                sheets[f"{source}_{leaflet}_vertices"] = sheet.reshape(-1, 3)
                sheets[f"{source}_{leaflet}_faces"] = faces
                if source == "predicted":
                    points = np.argwhere(reference == code) * np.asarray(zooms)
                    distances[f"{leaflet}_points"] = points
                    distances[f"{leaflet}_distance_mm"] = point_to_mesh_mm(points, sheet.reshape(-1, 3), faces)
        destination = OUT / "valve" / f"{label}_{case}"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "data.npz", image=volume[box], reference=reference[box].astype(np.uint8),
                            predicted=predicted[box].astype(np.uint8),
                            probability=probability[(slice(None),) + box].astype(np.float16),
                            box_origin_voxels=low, zooms=np.asarray(zooms), **sheets, **distances)
        entry = {"label": label, "case": case, "argmax_agreement": agreement,
                 "median_distance_mm": {k: float(np.median(distances[f"{k}_distance_mm"]))
                                        for k in ("anterior", "posterior")}}
        report.append(entry)
        print(json.dumps(entry), flush=True)
    return report


def render_valve(folder: Path) -> None:
    plt = q._plotting()
    with np.load(folder / "data.npz") as z:
        data = {k: z[k] for k in z.files}
    image, reference, predicted, probability = data["image"], data["reference"], data["predicted"], data["probability"]
    centre = np.round(np.argwhere(reference > 0).mean(axis=0)).astype(int)
    cuts = [("axial", (slice(None), slice(None), centre[2])), ("coronal", (slice(None), centre[1], slice(None))),
            ("sagittal", (centre[0], slice(None), slice(None)))]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.3))
    for column, (name, cut) in enumerate(cuts):
        ax = axes[0, column]
        ax.imshow(image[cut].T, cmap="gray", origin="lower")
        for code, colour in ((2, "#2ca02c"), (1, "#17becf")):
            ax.contour((reference[cut] == code).T, levels=[0.5], colors=[colour], linewidths=0.8, origin="lower")
            ax.contour((predicted[cut] == code).T, levels=[0.5], colors=["#d62728" if code == 2 else "#ff7f0e"],
                       linewidths=0.8, linestyles="--", origin="lower")
        ax.set_title(f"{name}: labels", fontsize=6.5)
        ax.set_axis_off()
        ax = axes[1, column]
        rgb = np.stack([probability[2][cut].astype(float), probability[1][cut].astype(float),
                        np.zeros(probability[1][cut].shape)], axis=-1)
        ax.imshow(image[cut].T, cmap="gray", origin="lower")
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", alpha=0.55)
        ax.set_title(f"{name}: nnU-Net probability", fontsize=6.5)
        ax.set_axis_off()
    fig.suptitle("Labels: reference solid (anterior green, posterior cyan), nnU-Net dashed (anterior red, "
                 "posterior orange). Probability: anterior red, posterior green.", fontsize=6, y=0.01)
    fig.savefig(folder / "valve_slices_labels_probabilities.png")
    plt.close(fig)
    fig = plt.figure(figsize=(6.6, 2.4))
    everything = np.concatenate([data[k] for k in data if k.endswith("_vertices")])
    mid, half = everything.mean(axis=0), 0.6 * np.max(np.ptp(everything, axis=0))
    limits = [(c - half, c + half) for c in mid]
    for index, (source, title) in enumerate((("predicted", "fitted to the nnU-Net labels"),
                                             ("reference", "fitted to the reference labels"))):
        ax = q._axes3d(fig, int(f"13{index + 1}"), title, limits)
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        for leaflet, colour in (("anterior", ANTERIOR_C), ("posterior", POSTERIOR_C)):
            q._surface(ax, data[f"{source}_{leaflet}_vertices"], data[f"{source}_{leaflet}_faces"], colour, 0.7)
    ax = q._axes3d(fig, 133, "reference voxels: distance to the\nsheet fitted to the nnU-Net labels (mm)", limits)
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    points = np.concatenate([data["anterior_points"], data["posterior_points"]])
    values = np.concatenate([data["anterior_distance_mm"], data["posterior_distance_mm"]])
    step = max(1, len(points) // 6000)
    scatter = ax.scatter(*points[::step].T, c=values[::step], s=0.5, cmap="viridis", vmin=0,
                         vmax=float(np.percentile(values, 99)))
    fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.0)
    fig.savefig(folder / "valve_sheets_and_distances.png")
    plt.close(fig)


# ----------------------------------------------------------------- chamber ----

def run_chamber() -> list:
    import nibabel as nib

    import e76_chamber_shape_fit as stage_c
    from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
    from rooted4d.experiments.component_eval import cavity_boundary_points, point_to_mesh_mm
    from rooted4d.segmentation import load_patient_splits

    e77 = json.loads(Path(os.environ["R4D_E77"]).read_text())
    by_patient = {}
    for r in e77["rows"]:
        if r.get("fitted"):
            by_patient.setdefault(r["patient"], []).append(r)
    errors = {}
    for patient, frames in by_patient.items():
        if len(frames) == 2:
            ref = sorted(f["reference_volume_ml"] for f in frames)
            fit = sorted(f["direct"]["volume_ml"] for f in frames)
            errors[patient] = abs(100.0 * (fit[1] - fit[0]) / fit[1] - 100.0 * (ref[1] - ref[0]) / ref[1])
    picks = _pick(errors, worst=True)
    root = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
    splits = load_patient_splits(os.environ["R4D_SPLITS"])
    segmenter = stage_c.Segmenter(Path(os.environ["R4D_SEGMENTER"]))
    faces = lv_topology_faces(8, 32)
    report = []
    for label, patient in picks.items():
        frames = [p for p in stage_c.frames_of(root, splits, "test") if p.parent.name == patient]
        arrays, volumes = {}, {}
        for path in frames:
            truth, zooms, frame, truth_fit = stage_c.reference_fit(path)
            image = np.asarray(nib.load(str(path).replace("_gt.nii.gz", ".nii.gz")).dataobj, dtype=np.float32)
            predicted = segmenter(Path(str(path).replace("_gt.nii.gz", ".nii.gz")), truth.shape, zooms[:2])
            direct = fit_lv_surface(predicted, zooms, n_rings=8, n_angles=32, base_first=True)
            points = cavity_boundary_points(truth, zooms, cavity_label=3)
            tag = f"frame{frame:02d}"
            arrays.update({f"{tag}_image": image, f"{tag}_reference": truth.astype(np.uint8),
                           f"{tag}_predicted": predicted.astype(np.uint8), f"{tag}_direct": direct.vertices,
                           f"{tag}_reference_surface": truth_fit.vertices, f"{tag}_points": points,
                           f"{tag}_distance_mm": point_to_mesh_mm(points, direct.vertices, faces)})
            volumes[tag] = {"reference_ml": stage_c.score(points, truth_fit.vertices, faces)["volume_ml"],
                            "direct_ml": stage_c.score(points, direct.vertices, faces)["volume_ml"]}
        destination = OUT / "chamber_pairs" / f"{label}_{patient}"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "data.npz", faces=faces, volumes=np.asarray(json.dumps(volumes)), **arrays)
        entry = {"label": label, "patient": patient, "ef_error_points": errors[patient], "volumes_ml": volumes}
        report.append(entry)
        print(json.dumps(entry), flush=True)
    return report


def render_chamber(folder: Path) -> None:
    plt = q._plotting()
    with np.load(folder / "data.npz") as z:
        data = {k: z[k] for k in z.files}
    volumes = json.loads(str(data["volumes"]))
    tags = sorted(volumes, key=lambda t: -volumes[t]["reference_ml"])  # end diastole first
    ef = {kind: 100.0 * (volumes[tags[0]][kind] - volumes[tags[1]][kind]) / volumes[tags[0]][kind]
          for kind in ("reference_ml", "direct_ml")}
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.9))
    for row, tag in enumerate(tags):
        truth, predicted, image = data[f"{tag}_reference"], data[f"{tag}_predicted"], data[f"{tag}_image"]
        phase = "end diastole" if row == 0 else "end systole"
        slices = [int(i) for i in np.nonzero((truth == 3).any(axis=(0, 1)))[0]]
        shown = {slices[0]: "basal", slices[len(slices) // 2]: "mid", slices[-1]: "apical"}
        # The reference's basal, mid and apical slices can all agree while the error sits elsewhere,
        # so the slice where the two cavities disagree most is always shown as well.
        disagreement = ((truth == 3) ^ (predicted == 3)).sum(axis=(0, 1))
        order = [int(i) for i in np.argsort(-disagreement, kind="stable")]
        if order[0] in shown:
            shown[order[0]] += ", largest disagreement"
            extra = next((i for i in order if i not in shown), None)
            if extra is not None:
                shown[extra] = "next-largest disagreement"
        else:
            shown[order[0]] = "largest disagreement"
        # Crop around the reference heart and every predicted cavity voxel, so false positives stay in view.
        heart = np.argwhere(((truth > 0) | (predicted == 3)).any(axis=2))
        low, high = heart.min(axis=0), heart.max(axis=0)
        margin = (0.25 * (high - low)).astype(int) + 4
        (x0, y0), (x1, y1) = np.maximum(low - margin, 0), np.minimum(high + margin, np.array(truth.shape[:2]) - 1)
        for column in range(4):
            axes[row, column].set_axis_off()
        for column, index in enumerate(sorted(shown)[:4]):
            ax = axes[row, column]
            ax.imshow(image[x0:x1 + 1, y0:y1 + 1, index].T, cmap="gray", origin="lower")
            for labels, colour, style in ((truth, "#2ca02c", "-"), (predicted, "#d62728", "--")):
                if (labels[x0:x1 + 1, y0:y1 + 1, index] == 3).any():
                    ax.contour((labels[x0:x1 + 1, y0:y1 + 1, index] == 3).T, levels=[0.5], colors=[colour],
                               linewidths=0.9, linestyles=style, origin="lower")
            ax.set_title("%s, slice %d\n%s" % (phase, index, shown[index]), fontsize=6)
    fig.suptitle("Ejection fraction: reference %.1f%%, image input %.1f%%. Green: reference cavity; red, dashed: "
                 "prediction." % (ef["reference_ml"], ef["direct_ml"]), fontsize=6.5, y=0.02)
    fig.savefig(folder / "ed_es_slices.png")
    plt.close(fig)
    fig = plt.figure(figsize=(6.6, 4.4))
    everything = np.concatenate([data[f"{t}_reference_surface"] for t in tags])
    centre, half = everything.mean(axis=0), 0.55 * np.max(np.ptp(everything, axis=0))
    limits = [(c - half, c + half) for c in centre]
    for row, tag in enumerate(tags):
        phase = "end diastole" if row == 0 else "end systole"
        ax = q._axes3d(fig, int(f"23{row * 3 + 1}"), f"{phase}\nimage input, {volumes[tag]['direct_ml']:.0f} ml", limits)
        q._surface(ax, data[f"{tag}_direct"], data["faces"], CHAMBER_C, 0.6)
        ax = q._axes3d(fig, int(f"23{row * 3 + 2}"), f"reference\n{volumes[tag]['reference_ml']:.0f} ml", limits)
        q._surface(ax, data[f"{tag}_reference_surface"], data["faces"], "0.6", 0.6)
        ax = q._axes3d(fig, int(f"23{row * 3 + 3}"), "reference points\ndistance to the prediction", limits)
        scatter = ax.scatter(*data[f"{tag}_points"].T, c=data[f"{tag}_distance_mm"], s=1, cmap="viridis", vmin=0,
                             vmax=6.0)
    fig.colorbar(scatter, ax=fig.axes, shrink=0.5, pad=0.02, label="mm, saturating at 6")
    fig.savefig(folder / "ed_es_surfaces.png")
    plt.close(fig)


# ------------------------------------------------------------------ driver ----

RUN = {"arms": run_arms, "image": run_image, "modes": run_modes, "valve": run_valve, "chamber": run_chamber}
RENDER = {"arms": render_arms, "image": render_image, "modes": render_modes, "valve": render_valve,
          "chamber_pairs": render_chamber}


def main() -> int:
    part = os.environ.get("R4D_PART", "")
    if part == "render":
        root = Path(os.environ["R4D_IN"])
        count = 0
        for kind, draw in RENDER.items():
            folders = [root / kind] if kind == "modes" else sorted((root / kind).glob("*"))
            for folder in folders:
                if (folder / "data.npz").exists():
                    draw(folder)
                    count += 1
        print("rendered", count, "folders under", root, flush=True)
        return 0
    if os.environ.get("R4D_ALLOW_TEST") != "1" and part != "modes":
        print("refusing to read test samples without R4D_ALLOW_TEST=1", flush=True)
        return 2
    report = {"part": part, "samples": RUN[part]()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"e81_{part}_index.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("wrote", OUT / f"e81_{part}_index.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
