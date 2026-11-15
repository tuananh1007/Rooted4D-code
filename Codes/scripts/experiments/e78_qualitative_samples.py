"""E78: qualitative samples for the figures -- inputs, intermediate states, predictions, truth.

Not an experiment: nothing here is selected or scored for the paper. Samples are
chosen by rule, the test samples at the 25th, 50th and 75th percentile of an
error computed here, so the figures show typical cases rather than chosen ones.
Every array is saved (``data.npz``) next to rendered panels (PNG), so figures
can be restyled without rerunning anything.

* ``R4D_PART=fusion`` (v4 test, the frozen S configuration): the CMR slice and
  TEE sector likelihoods the generator renders, the observed point sets, the
  input-derived start, the two independent reconstructions that late fusion
  registers (and the annular gap between them), the Rooted4D reconstruction
  over the cycle, and the true geometry, all in the CMR frame. Samples are
  ranked by E64's Rooted4D held-out error.
* ``R4D_PART=chamber`` (ACDC test): short-axis slices, feature maps of the
  frozen E31 segmenter (first and third encoder blocks, bottleneck, last
  decoder block) for the mid-ventricular slice in the network's input frame,
  the cavity probability, predicted and reference labels, and the direct and
  reference surfaces. Frames are ranked by the direct surface's symmetric
  distance.

    R4D_ALLOW_TEST=1 R4D_PART=fusion|chamber [R4D_RENDER=0] python e78_qualitative_samples.py
    R4D_PART=render R4D_IN=<qualitative dir> python e78_qualitative_samples.py

The cluster environment has no plotting library, so the cluster run saves the
arrays with ``R4D_RENDER=0`` and the panels are drawn afterwards from them.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

QUANTILES = (0.25, 0.50, 0.75)
PHASES = np.linspace(0.0, 1.0, 8, endpoint=False)
OUT = Path(os.environ.get("R4D_OUT", "results")) / "qualitative"
RENDER = os.environ.get("R4D_RENDER", "1") == "1"
plt = None


def _plotting():
    """Import matplotlib only where panels are drawn."""

    global plt
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        pyplot.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 200,
                                "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
        plt = pyplot
    return plt
CHAMBER_C, ANTERIOR_C, POSTERIOR_C, TRUTH_C = "#3b6ea5", "#c05746", "#d99b3c", "0.25"
ZOOM = 1.3  # E78's 3D panels


def by_quantile(scores: dict) -> list:
    """The keys at the nearest ranks of the fixed quantiles."""

    ordered = sorted(scores, key=lambda k: (scores[k], k))
    return [ordered[int(round(q * (len(ordered) - 1)))] for q in QUANTILES]


# ------------------------------------------------------------------ fusion ----

def _truth_cmr_frame(blob, phases):
    from rooted4d.motion.fourier import FourierMotion
    from rooted4d.synthetic.benchmark import session_residual

    motion = FourierMotion(blob["template_vertices"], blob["motion_coefficients"])
    return motion.evaluate(phases) - 0.5 * session_residual(phases, blob["session_static"], blob["session_dynamic"])


def _late_fusion_internals(problem, config, alignment, overrides):
    """``baselines.run_late_fusion`` step by step, keeping what it discards.

    The interface gap computed here is checked against the arm's own, so the
    figures cannot drift from the reported numbers.
    """

    from rooted4d.inference.baselines import _start
    from rooted4d.inference.joint import fit_joint_scene, predict
    from rooted4d.registration import align_periodic_features, register_point_sets

    observations, model = problem.observations, problem.model
    cmr_config = replace(config, use_tee=False, estimate_pose=False, estimate_phase_warp=False, use_session=False,
                         **overrides.get("cmr", {}))
    cmr_fit = fit_joint_scene(observations, model, cmr_config)
    tee_config = replace(config, use_cmr=False, use_session=False, **overrides.get("tee", {}))
    tee_fit = fit_joint_scene(observations, model, tee_config, start=_start(alignment, model, tee_config))
    registration = np.union1d(np.intersect1d(observations.cmr_vertices, observations.tee_vertices),
                              np.sort(model.groups["annulus"]))
    temporal_vertices = np.intersect1d(observations.cmr_vertices, observations.tee_vertices)

    def features(points):
        centred = points - points.mean(axis=1, keepdims=True)
        radii = np.linalg.norm(centred, axis=2)
        return np.stack([radii.mean(axis=1), radii.std(axis=1), centred[:, :, 2].mean(axis=1),
                         np.linalg.norm(points.mean(axis=1), axis=1)], axis=1)

    temporal = align_periodic_features(
        features(observations.cmr_points[:, np.searchsorted(observations.cmr_vertices, temporal_vertices), :]),
        features(observations.tee_points[:, np.searchsorted(observations.tee_vertices, temporal_vertices), :]))
    unposed = replace(tee_fit.parameters, pose_axis_angle=np.zeros(3), pose_translation=np.zeros(3))
    cmr_geometry, _ = predict(cmr_fit.parameters, observations, model, cmr_config, cmr_times=np.array([0.0]),
                              cmr_vertices=registration, tee_times=np.empty(0))
    tee_geometry, _ = predict(unposed, observations, model, cmr_config,
                              cmr_times=np.array([float(temporal.offset_cycles)]),
                              cmr_vertices=registration, tee_times=np.empty(0))
    spatial = register_point_sets(cmr_geometry.reshape(-1, 3),
                                  tee_fit.parameters.transform().apply(tee_geometry.reshape(-1, 3)))
    return cmr_fit, cmr_config, tee_fit, unposed, float(temporal.offset_cycles), spatial.transform, temporal


def _inverse(transform, points):
    return (np.asarray(points) - transform.translation) @ transform.rotation


def fusion_sample(path: Path, model, config, independent, options, e64_row, destination: Path) -> dict:
    """``config`` is the frozen joint arm's; ``independent`` and ``options`` are E64's late-fusion arm's."""
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.geometry.toy import make_toy_complex
    from rooted4d.inference.baselines import run_arm
    from rooted4d.inference.initialization import input_derived_alignment, start_from
    from rooted4d.inference.joint import fit_joint_scene, predict

    problem = benchmark_problem(path, model, config)
    observations = problem.observations
    alignment = input_derived_alignment(observations)
    everything = np.arange(model.n_physical)
    joint = fit_joint_scene(observations, model, config, start=start_from(alignment, model, config))
    held = joint.heldout_errors()
    # The fit must be E64's own, at re-run precision (re-runs are not bitwise).
    for key in ("heldout_cmr_mm", "heldout_tee_mm"):
        if e64_row is not None:  # None only for a validation code-path check
            assert abs(held[key] - e64_row[key]) < 1e-3, (key, held[key], e64_row[key])
    rooted, _ = predict(joint.parameters, observations, model, config, cmr_times=PHASES,
                        cmr_vertices=everything, tee_times=np.empty(0))
    cmr_fit, cmr_config, tee_fit, unposed, offset, spatial, _ = _late_fusion_internals(
        problem, independent, alignment, options.get("modality_overrides", {}))
    chamber_side, _ = predict(cmr_fit.parameters, observations, model, cmr_config, cmr_times=PHASES,
                              cmr_vertices=everything, tee_times=np.empty(0))
    valve_own, _ = predict(unposed, observations, model, cmr_config, cmr_times=(PHASES + offset) % 1.0,
                           cmr_vertices=everything, tee_times=np.empty(0))
    valve_side = _inverse(spatial, tee_fit.parameters.transform().apply(valve_own.reshape(-1, 3))).reshape(valve_own.shape)
    annulus = np.sort(model.groups["annulus"])
    # Check against the arm itself, on its own 32-phase grid, in the TEE frame.
    grid = np.linspace(0.0, 1.0, 32, endpoint=False)
    c32, _ = predict(cmr_fit.parameters, observations, model, cmr_config, cmr_times=grid, cmr_vertices=annulus,
                     tee_times=np.empty(0))
    v32, _ = predict(unposed, observations, model, cmr_config, cmr_times=(grid + offset) % 1.0, cmr_vertices=annulus,
                     tee_times=np.empty(0))
    gap = float(np.abs(spatial.apply(c32.reshape(-1, 3)).reshape(c32.shape)
                       - tee_fit.parameters.transform().apply(v32)).max())
    arm = run_arm("late_fusion", problem, independent, alignment=alignment, options=options)
    assert np.isclose(gap, arm.attachment_residual_mm, rtol=1e-9, atol=1e-9), (gap, arm.attachment_residual_mm)

    with np.load(path, allow_pickle=True) as blob:
        truth = _truth_cmr_frame(blob, PHASES)
        cmr_images = np.asarray(blob["cmr_surface_likelihood"], dtype=np.float32)
        tee_images = np.asarray(blob["tee_surface_likelihood"], dtype=np.float32)
        tee_true_phases = np.asarray(blob["tee_true_phases"])
        true_pose = (np.asarray(blob["pose_rotation"]), np.asarray(blob["pose_translation"]))
    start = start_from(alignment, model, config).transform()
    tee_in_cmr_start = _inverse(start, observations.tee_points.reshape(-1, 3)).reshape(observations.tee_points.shape)
    complex_ = make_toy_complex(len(annulus))
    faces = {k: complex_.faces[k] for k in ("chamber", "anterior_leaflet", "posterior_leaflet")}
    errors = {"rooted4d": np.linalg.norm(rooted - truth, axis=-1),
              "late_fusion_chamber_side": np.linalg.norm(chamber_side - truth, axis=-1),
              "late_fusion_valve_side": np.linalg.norm(valve_side - truth, axis=-1)}
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "data.npz", phases=PHASES, truth=truth, rooted4d=rooted,
        late_fusion_chamber_side=chamber_side, late_fusion_valve_side=valve_side,
        cmr_points=observations.cmr_points, cmr_times=observations.cmr_times, cmr_vertices=observations.cmr_vertices,
        tee_points=observations.tee_points, tee_times=observations.tee_times, tee_vertices=observations.tee_vertices,
        tee_true_phases=tee_true_phases, tee_points_in_cmr_frame_at_start=tee_in_cmr_start,
        cmr_slice_likelihood=cmr_images, tee_sector_likelihood=tee_images,
        true_pose_rotation=true_pose[0], true_pose_translation=true_pose[1],
        annulus=annulus, chamber_faces=faces["chamber"], anterior_faces=faces["anterior_leaflet"],
        posterior_faces=faces["posterior_leaflet"], **{f"error_{k}": v for k, v in errors.items()})
    if RENDER:
        _render_fusion(destination, truth, rooted, chamber_side, valve_side, observations.cmr_points,
                       observations.tee_points, tee_in_cmr_start, cmr_images, tee_images, faces, annulus, errors)
    return {"sample": path.name, "late_fusion_gap_mm": gap, "rooted4d_heldout": held,
            "median_vertex_error_mm": {k: float(np.median(v)) for k, v in errors.items()}}


def _axes3d(fig, position, title, limits, zoom=1.0):
    ax = fig.add_subplot(position, projection="3d")
    ax.set_title(title, fontsize=7, pad=0)
    for setter, (low, high) in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
        setter(low, high)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off()
    return ax


def _surface(ax, vertices, faces, colour, alpha, values=None, cmap="viridis", vmax=None):
    if values is None:
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color=colour,
                        alpha=alpha, linewidth=0.1, edgecolor="0.4", shade=True)
    else:
        collection = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces,
                                     cmap=cmap, alpha=alpha, linewidth=0.0, shade=False)
        collection.set_array(values[faces].mean(axis=1))
        collection.set_clim(0.0, vmax)
        return collection
    return None


def _ring(ax, vertices, annulus, colour, width=1.4, style="-"):
    loop = vertices[np.append(annulus, annulus[0])]
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], style, color=colour, lw=width)


def _render_fusion(destination, truth, rooted, chamber_side, valve_side, cmr_points, tee_points, tee_in_cmr_start,
                   cmr_images, tee_images, faces, annulus, errors):
    _plotting()
    everything = np.concatenate([truth.reshape(-1, 3), rooted.reshape(-1, 3)])
    centre = everything.mean(axis=0)
    half = 0.55 * np.max(everything.max(axis=0) - everything.min(axis=0))
    limits = [(c - half, c + half) for c in centre]
    phase = 0
    # Inputs: rendered likelihoods.
    fig, axes = plt.subplots(2, 5, figsize=(6.6, 2.8))
    for column in range(5):
        axes[0, column].imshow(cmr_images[0, column], cmap="gray")
        axes[0, column].set_title(f"CMR slice {column + 1}", fontsize=7)
    # The TEE grid is depth x azimuth x elevation; sector cuts read like B-mode.
    frames = (0, tee_images.shape[0] // 2)
    middle_az, middle_el = tee_images.shape[2] // 2, tee_images.shape[3] // 2
    panels = [(tee_images[frames[0], :, :, middle_el], "TEE, depth x azimuth"),
              (tee_images[frames[0], :, middle_az, :], "TEE, depth x elevation"),
              (tee_images[frames[0]].max(axis=0), "TEE, max over depth"),
              (tee_images[frames[1], :, :, middle_el], "TEE, later frame"),
              (tee_images[frames[1]].max(axis=0), "later frame, max")]
    for column, (panel, title) in enumerate(panels):
        axes[1, column].imshow(panel, cmap="gray", aspect="auto")
        axes[1, column].set_title(title, fontsize=6.5)
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.savefig(destination / "input_images.png")
    plt.close(fig)
    # Inputs: observed points, each in its own frame, and the input-derived start.
    fig = plt.figure(figsize=(6.6, 2.4))
    ax = _axes3d(fig, 131, "input: CMR points (frame 1)", limits, zoom=ZOOM)
    ax.scatter(*cmr_points[0].T, s=4, color=CHAMBER_C)
    tee = tee_points[0]
    tee_centre, tee_half = tee.mean(axis=0), half
    ax = _axes3d(fig, 132, "input: TEE points (own frame)", [(c - tee_half, c + tee_half) for c in tee_centre],
                 zoom=ZOOM)
    ax.scatter(*tee.T, s=4, color=ANTERIOR_C)
    ax = _axes3d(fig, 133, "input-derived start (CMR frame)", limits, zoom=ZOOM)
    ax.scatter(*cmr_points[0].T, s=4, color=CHAMBER_C, label="CMR")
    ax.scatter(*tee_in_cmr_start[0].T, s=4, color=ANTERIOR_C, label="TEE at start pose")
    fig.savefig(destination / "input_points_and_start.png")
    plt.close(fig)
    # Intermediate, prediction, truth, error.
    vmax = 5.0  # a fixed scale, shared by both methods
    fig = plt.figure(figsize=(6.6, 2.4))
    ax = _axes3d(fig, 141, "late fusion (two annuli)", limits, zoom=ZOOM)
    _surface(ax, chamber_side[phase], faces["chamber"], CHAMBER_C, 0.18)
    _surface(ax, valve_side[phase], faces["anterior_leaflet"], ANTERIOR_C, 0.7)
    _surface(ax, valve_side[phase], faces["posterior_leaflet"], POSTERIOR_C, 0.7)
    _ring(ax, chamber_side[phase], annulus, CHAMBER_C)
    _ring(ax, valve_side[phase], annulus, ANTERIOR_C, style="--")
    ax = _axes3d(fig, 142, "Rooted4D (one annulus)", limits, zoom=ZOOM)
    _surface(ax, rooted[phase], faces["chamber"], CHAMBER_C, 0.18)
    _surface(ax, rooted[phase], faces["anterior_leaflet"], ANTERIOR_C, 0.7)
    _surface(ax, rooted[phase], faces["posterior_leaflet"], POSTERIOR_C, 0.7)
    _ring(ax, rooted[phase], annulus, "k")
    ax = _axes3d(fig, 143, "ground truth", limits, zoom=ZOOM)
    _surface(ax, truth[phase], faces["chamber"], "0.6", 0.18)
    _surface(ax, truth[phase], faces["anterior_leaflet"], "0.45", 0.7)
    _surface(ax, truth[phase], faces["posterior_leaflet"], "0.55", 0.7)
    _ring(ax, truth[phase], annulus, "k")
    ax = _axes3d(fig, 144, "Rooted4D error", limits, zoom=ZOOM)
    for key in ("chamber", "anterior_leaflet", "posterior_leaflet"):
        collection = _surface(ax, rooted[phase], faces[key], None, 0.9, errors["rooted4d"][phase], vmax=vmax)
    fig.colorbar(collection, ax=ax, shrink=0.6, pad=0.0, label="mm")
    fig.savefig(destination / "late_fusion_rooted4d_truth_error.png")
    plt.close(fig)
    # Errors of both methods on one fixed scale: late fusion's chamber from its CMR fit, its
    # leaflets from its registered TEE fit (the parts each fit observed).
    fig = plt.figure(figsize=(4.4, 2.4))
    ax = _axes3d(fig, 121, "late fusion error (mm)", limits, zoom=ZOOM)
    _surface(ax, chamber_side[phase], faces["chamber"], None, 0.9, errors["late_fusion_chamber_side"][phase], vmax=vmax)
    for key in ("anterior_leaflet", "posterior_leaflet"):
        _surface(ax, valve_side[phase], faces[key], None, 0.9, errors["late_fusion_valve_side"][phase], vmax=vmax)
    ax = _axes3d(fig, 122, "Rooted4D error (mm)", limits, zoom=ZOOM)
    for key in ("chamber", "anterior_leaflet", "posterior_leaflet"):
        collection = _surface(ax, rooted[phase], faces[key], None, 0.9, errors["rooted4d"][phase], vmax=vmax)
    fig.colorbar(collection, ax=fig.axes, shrink=0.6, pad=0.02, label="error (mm)")
    fig.savefig(destination / "error_maps.png")
    plt.close(fig)
    # Looking down the annular axis: the annulus plane from the true ring at this phase.
    ring = truth[phase][annulus]
    centre_ring = ring.mean(axis=0)
    _, _, vt = np.linalg.svd(ring - centre_ring)
    plane = vt[:2]

    def flat(points):
        return (np.asarray(points) - centre_ring) @ plane.T

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.4), sharex=True, sharey=True)
    for ax, title in zip(axes, ("late fusion", "Rooted4D")):
        loop = flat(ring[np.append(np.arange(len(annulus)), 0)])
        ax.plot(*loop.T, color="0.55", lw=2.2, label="true annulus")
        ax.set_title(title, fontsize=7)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    ordered = np.append(annulus, annulus[0])
    axes[0].plot(*flat(chamber_side[phase][ordered]).T, color=CHAMBER_C, lw=1.2, label="from CMR fit")
    axes[0].plot(*flat(valve_side[phase][ordered]).T, "--", color=ANTERIOR_C, lw=1.2, label="from TEE fit")
    axes[1].plot(*flat(rooted[phase][ordered]).T, color="k", lw=1.2, label="shared ring")
    for ax in axes:
        ax.legend(fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.savefig(destination / "annulus_top_view.png")
    plt.close(fig)
    # The cycle: Rooted4D against the truth at each of the eight phases.
    fig = plt.figure(figsize=(6.6, 3.4))
    for index in range(len(PHASES)):
        ax = _axes3d(fig, int(f"24{index + 1}"), f"phase {PHASES[index]:.3f}", limits, zoom=ZOOM)
        _surface(ax, truth[index], faces["chamber"], "0.7", 0.12)
        _ring(ax, truth[index], annulus, "0.4", width=1.0, style="--")
        for key, colour in (("anterior_leaflet", ANTERIOR_C), ("posterior_leaflet", POSTERIOR_C)):
            _surface(ax, rooted[index], faces[key], colour, 0.6)
        _ring(ax, rooted[index], annulus, "k", width=1.0)
    fig.savefig(destination / "rooted4d_cycle_vs_truth.png")
    plt.close(fig)


def run_fusion() -> list:
    import e59_unknown_alignment as e59
    from uniformity_runner import frozen_s
    from rooted4d.inference import SolverConfig

    recipe, base, weight, root = frozen_s("v4")
    model = e59.build_model(Path(root), recipe)[0]
    config = SolverConfig(**{**base, "identity_prior_weight": weight})
    if os.environ.get("R4D_SESSION_PRIOR"):
        # E75's configuration: the session weight E73 selected on validation. The
        # rows then come from E75 (R4D_E64_ROWS), which the fits are checked against.
        config = replace(config, session_prior_weight=float(os.environ["R4D_SESSION_PRIOR"]))
    frozen = {v["name"]: v for v in json.loads(Path("Codes/configs/e62/v4_test.json").read_text())["variants"]}
    independent = SolverConfig(**{**base, **(frozen["S/independent"].get("overrides") or {})})
    options = frozen["S/independent"].get("options") or {}
    rows = []
    for shard in sorted(Path(os.environ.get("R4D_E64_ROWS", "results/benchmark/E64/v4_test")).glob("shard_*.jsonl")):
        rows += [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    scores = {r["sample"]: r["variants"]["S/joint"]["heldout_cmr_mm"] + r["variants"]["S/joint"]["heldout_tee_mm"]
              for r in rows if r["variants"]["S/joint"].get("succeeded")}
    chosen = by_quantile(scores)
    paths = {p.name: p for p in e59.sample_paths(Path(root), "test")}
    report = []
    for quantile, name in zip(QUANTILES, chosen):
        row = next(r for r in rows if r["sample"] == name)["variants"]["S/joint"]
        entry = fusion_sample(paths[name], model, config, independent, options, row,
                              OUT / "fusion" / f"q{int(quantile * 100):02d}_{Path(name).stem}")
        entry.update(quantile=quantile, e64_heldout_sum_mm=scores[name])
        report.append(entry)
        print(json.dumps(entry), flush=True)
    return report


# ----------------------------------------------------------------- chamber ----

def run_chamber() -> list:
    import nibabel as nib
    import torch

    import e76_chamber_shape_fit as stage_c
    from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
    from rooted4d.experiments.component_eval import cavity_boundary_points, point_to_mesh_mm
    from rooted4d.segmentation import load_patient_splits
    from rooted4d.segmentation.data import PATCH, TARGET_SPACING_MM

    root = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
    splits = load_patient_splits(os.environ["R4D_SPLITS"])
    segmenter = stage_c.Segmenter(Path(os.environ["R4D_SEGMENTER"]))
    faces = lv_topology_faces(stage_c.N_RINGS, stage_c.N_ANGLES)
    candidates = {}
    for path in stage_c.frames_of(root, splits, "test"):
        image_path = Path(str(path).replace("_gt.nii.gz", ".nii.gz"))
        truth, zooms, frame, truth_fit = stage_c.reference_fit(path)
        predicted = segmenter(image_path, truth.shape, zooms[:2])
        direct = fit_lv_surface(predicted, zooms, n_rings=8, n_angles=32, base_first=True)
        if truth_fit is None or direct is None:
            continue
        points = cavity_boundary_points(truth, zooms, cavity_label=3)
        candidates[str(path)] = stage_c.score(points, direct.vertices, faces)["symmetric_mm"]
    report = []
    for quantile, key in zip(QUANTILES, by_quantile(candidates)):
        path = Path(key)
        entry = chamber_sample(path, segmenter, faces,
                               OUT / "chamber" / f"q{int(quantile * 100):02d}_{path.parent.name}_frame"
                               f"{int(path.name.split('_frame')[-1].split('_')[0]):02d}")
        entry.update(quantile=quantile, direct_symmetric_mm=candidates[key])
        report.append(entry)
        print(json.dumps(entry), flush=True)
    return report


def chamber_sample(path: Path, segmenter, faces, destination: Path) -> dict:
    import nibabel as nib
    import torch

    import e76_chamber_shape_fit as stage_c
    from rooted4d.active.acdc import fit_lv_surface
    from rooted4d.experiments.component_eval import cavity_boundary_points, point_to_mesh_mm
    from rooted4d.segmentation.data import PATCH, TARGET_SPACING_MM

    image_path = Path(str(path).replace("_gt.nii.gz", ".nii.gz"))
    truth, zooms, frame, truth_fit = stage_c.reference_fit(path)
    image = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
    predicted = segmenter(image_path, truth.shape, zooms[:2])
    direct = fit_lv_surface(predicted, zooms, n_rings=8, n_angles=32, base_first=True)
    points = cavity_boundary_points(truth, zooms, cavity_label=3)
    slices = [int(i) for i in np.nonzero((truth == 3).any(axis=(0, 1)))[0]]
    mid = slices[len(slices) // 2]
    # Feature maps of the mid slice, in the network's input frame (E36's preprocessing).
    plane = image[:, :, mid]
    low, high = np.percentile(plane[np.isfinite(plane)], [1.0, 99.0])
    plane = np.clip((plane - low) / max(high - low, 1e-6), 0.0, 1.0)
    tensor = torch.as_tensor(plane, dtype=torch.float32)[None, None]
    size = (max(1, int(round(plane.shape[0] * zooms[0] / TARGET_SPACING_MM))),
            max(1, int(round(plane.shape[1] * zooms[1] / TARGET_SPACING_MM))))
    resampled = torch.nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    canvas = torch.zeros((1, 1, PATCH, PATCH))
    rows_, cols = min(PATCH, size[0]), min(PATCH, size[1])
    sr, sc = (size[0] - rows_) // 2, (size[1] - cols) // 2
    dr, dc = (PATCH - rows_) // 2, (PATCH - cols) // 2
    canvas[0, 0, dr:dr + rows_, dc:dc + cols] = resampled[0, 0, sr:sr + rows_, sc:sc + cols]
    captured = {}
    hooks = [getattr(segmenter.net, name).register_forward_hook(
        lambda _m, _i, output, name=name: captured.__setitem__(name, output.detach()[0].numpy()))
        for name in ("down1", "down3", "bottom", "up1")]
    with torch.no_grad():
        logits = segmenter.net(canvas)
    for hook in hooks:
        hook.remove()
    probability = torch.softmax(logits, dim=1)[0].numpy()
    distances = point_to_mesh_mm(points, direct.vertices, faces)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "data.npz", image=image, zooms=np.asarray(zooms), labels_reference=truth,
        labels_predicted=predicted, labelled_slices=np.asarray(slices), mid_slice=mid,
        network_input=canvas[0, 0].numpy(), probability=probability,
        **{f"features_{k}": v for k, v in captured.items()},
        direct_vertices=direct.vertices, reference_vertices=truth_fit.vertices, faces=faces,
        reference_boundary_points=points, reference_points_to_direct_mm=distances)
    if RENDER:
        _render_chamber(destination, image, truth, predicted, slices, canvas[0, 0].numpy(), captured,
                        probability, direct.vertices, truth_fit.vertices, faces, points, distances)
    return {"frame": str(path), "mid_slice": mid}


def _render_chamber(destination, image, truth, predicted, slices, network_input, captured, probability,
                    direct, reference, faces, points, distances):
    _plotting()
    picks = [slices[0], slices[len(slices) // 2], slices[-1]]
    heart = np.argwhere((truth > 0).any(axis=2))
    low, high = heart.min(axis=0), heart.max(axis=0)
    margin = (0.35 * (high - low)).astype(int) + 4
    x0, y0 = np.maximum(low - margin, 0)
    x1, y1 = np.minimum(high + margin, np.array(truth.shape[:2]) - 1)
    for name, window in (("input_and_labels", (0, truth.shape[0] - 1, 0, truth.shape[1] - 1)),
                         ("input_and_labels_cropped", (x0, x1, y0, y1))):
        a, b, c, d = window
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
        for ax, index, label in zip(axes, picks, ("basal", "mid", "apical")):
            ax.imshow(image[a:b + 1, c:d + 1, index].T, cmap="gray", origin="lower")
            ax.contour((truth[a:b + 1, c:d + 1, index] == 3).T, levels=[0.5], colors=["#2ca02c"], linewidths=0.9,
                       origin="lower")
            ax.contour((predicted[a:b + 1, c:d + 1, index] == 3).T, levels=[0.5], colors=["#d62728"],
                       linewidths=0.9, linestyles="--", origin="lower")
            ax.set_title(f"{label} slice {index}", fontsize=7)
            ax.set_axis_off()
        fig.suptitle("LV cavity: reference label (green) and E31 prediction (red, dashed)", fontsize=7, y=0.02)
        fig.savefig(destination / f"{name}.png")
        plt.close(fig)
    fig, axes = plt.subplots(4, 9, figsize=(6.6, 3.2))
    for row, name in enumerate(("down1", "down3", "bottom", "up1")):
        maps = captured[name]
        axes[row, 0].imshow(network_input, cmap="gray")
        axes[row, 0].set_ylabel(name, fontsize=6)
        axes[row, 0].set_title("input" if row == 0 else "", fontsize=6)
        order = np.argsort(maps.reshape(len(maps), -1).std(axis=1))[::-1][:7]
        for column, channel in enumerate(order, start=1):
            axes[row, column].imshow(maps[channel], cmap="magma")
            axes[row, column].set_title(f"{name} ch{channel}", fontsize=5)
        flat = maps.reshape(len(maps), -1).T
        flat = flat - flat.mean(axis=0)
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        rgb = (flat @ vt[:3].T).reshape(maps.shape[1], maps.shape[2], 3)
        rgb = (rgb - rgb.min(axis=(0, 1))) / np.maximum(np.ptp(rgb, axis=(0, 1)), 1e-9)
        axes[row, 8].imshow(rgb)
        axes[row, 8].set_title("PCA RGB", fontsize=5)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(destination / "segmenter_features_mid_slice.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.2))
    axes[0].imshow(network_input, cmap="gray"); axes[0].set_title("network input", fontsize=7)
    axes[1].imshow(probability[3], cmap="viridis", vmin=0, vmax=1); axes[1].set_title("cavity probability", fontsize=7)
    for ax in axes:
        ax.set_axis_off()
    fig.savefig(destination / "cavity_probability_mid_slice.png")
    plt.close(fig)
    fig = plt.figure(figsize=(6.6, 2.4))
    both = np.vstack([direct, reference])
    centre, half = both.mean(axis=0), 0.55 * np.max(both.max(axis=0) - both.min(axis=0))
    limits = [(c - half, c + half) for c in centre]
    ax = _axes3d(fig, 131, "prediction: direct surface", limits, zoom=ZOOM)
    _surface(ax, direct, faces, CHAMBER_C, 0.6)
    ax = _axes3d(fig, 132, "reference surface", limits, zoom=ZOOM)
    _surface(ax, reference, faces, "0.6", 0.6)
    ax = _axes3d(fig, 133, "reference points\ndistance to the prediction (mm)", limits, zoom=ZOOM)
    scatter = ax.scatter(*points.T, c=distances, s=1, cmap="viridis", vmin=0, vmax=float(np.percentile(distances, 99)))
    fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.0)
    fig.savefig(destination / "surfaces.png")
    plt.close(fig)


def render_saved(root: Path) -> int:
    """Draw every panel from the saved arrays of an earlier run."""

    count = 0
    for data in sorted(root.glob("fusion/*/data.npz")):
        with np.load(data) as z:
            faces = {"chamber": z["chamber_faces"], "anterior_leaflet": z["anterior_faces"],
                     "posterior_leaflet": z["posterior_faces"]}
            errors = {k[len("error_"):]: z[k] for k in z.files if k.startswith("error_")}
            _render_fusion(data.parent, z["truth"], z["rooted4d"], z["late_fusion_chamber_side"],
                           z["late_fusion_valve_side"], z["cmr_points"], z["tee_points"],
                           z["tee_points_in_cmr_frame_at_start"], z["cmr_slice_likelihood"],
                           z["tee_sector_likelihood"], faces, z["annulus"], errors)
        count += 1
    for data in sorted(root.glob("chamber/*/data.npz")):
        with np.load(data) as z:
            captured = {k[len("features_"):]: z[k] for k in z.files if k.startswith("features_")}
            _render_chamber(data.parent, z["image"], z["labels_reference"], z["labels_predicted"],
                            [int(i) for i in z["labelled_slices"]], z["network_input"], captured, z["probability"],
                            z["direct_vertices"], z["reference_vertices"], z["faces"],
                            z["reference_boundary_points"], z["reference_points_to_direct_mm"])
        count += 1
    print("rendered", count, "samples under", root, flush=True)
    return 0


def main() -> int:
    if os.environ.get("R4D_PART") == "render":
        return render_saved(Path(os.environ["R4D_IN"]))
    if os.environ.get("R4D_ALLOW_TEST") != "1":
        print("refusing to read test samples without R4D_ALLOW_TEST=1", flush=True)
        return 2
    part = os.environ.get("R4D_PART", "fusion")
    report = {"part": part, "quantiles": QUANTILES,
              "samples": run_fusion() if part == "fusion" else run_chamber()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{part}_index.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("wrote", OUT / f"{part}_index.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
