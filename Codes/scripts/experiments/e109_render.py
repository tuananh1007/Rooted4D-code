"""E109's panels, drawn locally from the arrays the cluster parts saved.

Every panel uses the paper's reader-facing names and colours. The chamber parts'
scores are checked here against the committed rows (E36, E84, E49g, E37), so the
index states how each drawn surface relates to the number the paper reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "0.25", "axes.linewidth": 0.7, "xtick.color": "0.25", "ytick.color": "0.25",
    "axes.labelcolor": "0.1", "grid.color": "0.9", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

CHAMBER_C = "#3b6ea5"
ANTERIOR_C = "#c05746"
POSTERIOR_C = "#d99b3c"
ANNULUS_C = "#1a1a1a"
VALVE_RING_C = "#c0392b"
TRUTH_C = "0.55"
ERROR_CMAP = "viridis"
FUSION_VMAX = 5.0       # mm, saturating, as in E78 and E81
CHAMBER_VMAX = 8.0      # mm, saturating
LIGHT = np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])

SHORT_TITLES = {"soft": "Soft attachment,\n$\\lambda=1$", "rooted4d_global": "Rooted4D,\nglobal basis",
                "late_fusion": "Late fusion\n", "exact_repair": "Exact repair\n", "rooted4d": "Rooted4D\n",
                "truth": "Ground truth\n"}
#: The second copy of the annulus a method keeps, keyed by its chamber-side geometry: late
#: fusion's and exact repair's TEE reconstruction, and soft attachment's ring component, the
#: copy its ring observations land on and where its interface gap opens.
SECOND_RING = {"late_fusion_chamber": "late_fusion_valve", "exact_repair_chamber": "exact_repair_valve",
               "soft_chamber": "soft_ring"}
FUSION_METHODS = [  # key, title, chamber-side geometry, valve-side geometry
    ("late_fusion", "Late fusion", "late_fusion_chamber", "late_fusion_valve"),
    ("exact_repair", "Exact repair", "exact_repair_chamber", "exact_repair_valve"),
    ("soft", "Soft attachment, $\\lambda=1$", "soft_chamber", "soft_valve"),
    ("rooted4d_global", "Rooted4D, global basis", "rooted4d_global", "rooted4d_global"),
    ("rooted4d", "Rooted4D", "rooted4d", "rooted4d"),
]
PHASE_STYLE = {"late_fusion": ("0.45", "--"), "exact_repair": (POSTERIOR_C, ":"),
               "soft": ("#7b6fb0", (0, (5, 1.5, 1, 1.5))), "rooted4d_global": (ANTERIOR_C, "-."),
               "rooted4d": ("k", "-")}
CHAMBER_METHODS = [  # key, title, colour for contours
    ("rooted4d_label", "Rooted4D,\nlabel input", "#00b4d8"),
    ("rooted4d_compact", "Rooted4D +\ncompact U-Net", "#3b6ea5"),
    ("rooted4d_nnunet", "Rooted4D +\nnnU-Net", "#2a9d8f"),
    ("tetheart", "TetHeart\n", "#c05746"),
    ("ghdheart", "GHDHeart\n", "#d99b3c"),
]
REFERENCE_C = "#f4f1de"
CHAMBER_BAR = "distance to the reference contour (mm; in-plane, interpolated between slices)"
RANK_NAMES = {"q25": "25th percentile", "q50": "median", "q75": "75th percentile", "worst": "worst"}


def _save(fig, path: Path, pdf: bool = False) -> None:
    fig.savefig(path.with_suffix(".png"))
    if pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------- 3D helpers ----

def _shade(rgba: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Lambertian shading of per-face colours, two-sided, so curvature reads in print."""

    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    intensity = 0.58 + 0.42 * np.abs(normals @ LIGHT)
    out = rgba.copy()
    out[:, :3] = np.clip(out[:, :3] * intensity[:, None], 0.0, 1.0)
    return out


def _mesh(ax, vertices, faces, colour=None, values=None, alpha=1.0, vmax=1.0, edge=None, lw=0.0):
    faces = np.asarray(faces)
    if values is None:
        rgba = np.tile(colors.to_rgba(colour, alpha), (len(faces), 1))
    else:
        mapped = matplotlib.colormaps[ERROR_CMAP](np.clip(np.asarray(values)[faces].mean(axis=1) / vmax, 0.0, 1.0))
        mapped[:, 3] = alpha
        rgba = mapped
    poly = Poly3DCollection(vertices[faces], facecolors=_shade(rgba, vertices, faces),
                            edgecolors=edge if edge is not None else "none", linewidths=lw)
    ax.add_collection3d(poly)
    return poly


def _ring3d(ax, vertices, annulus, colour, lw=1.3, style="-", zorder=10):
    loop = vertices[np.append(annulus, annulus[0])]
    ax.plot(*loop.T, style, color=colour, lw=lw, zorder=zorder)


def _axes3d(fig, rect_or_spec, limits, elev, azim, zoom=1.0):
    ax = fig.add_subplot(rect_or_spec, projection="3d")
    for setter, (low, high) in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
        setter(low, high)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return ax


def _place3d(fig, rect, limits, elev, azim, zoom=1.2, base_up=False, title=None, title_size=6.8):
    """A 3D axes at ``rect`` with equal scaling; its title sits above the box, clear of the render."""

    ax = fig.add_axes(rect, projection="3d")
    for setter, (low, high) in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
        setter(low, high)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    if base_up:
        ax.invert_zaxis()
    ax.set_axis_off()
    if title:
        fig.text(rect[0] + 0.5 * rect[2], rect[1] + rect[3] + 0.004, title, ha="center", va="bottom",
                 fontsize=title_size, linespacing=1.1)
    return ax


def _limits(points: np.ndarray, pad: float = 0.52):
    centre = 0.5 * (points.max(axis=0) + points.min(axis=0))
    half = pad * float(np.max(np.ptp(points, axis=0)))
    return [(c - half, c + half) for c in centre]


def _colorbar(fig, rect, vmax, label):
    ax = fig.add_axes(rect)
    bar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0.0, vmax), cmap=ERROR_CMAP), cax=ax,
                       orientation="horizontal", extend="max")
    bar.set_label(label, fontsize=6.8, labelpad=1.5)
    bar.ax.tick_params(labelsize=6.2, length=2, pad=1)
    bar.outline.set_linewidth(0.4)
    return bar


# ------------------------------------------------------------------ fusion ----

def _canonical_frame(truth: np.ndarray, annulus: np.ndarray, apex: int) -> np.ndarray:
    """A rotation putting the true long axis (apex to annulus) up and vertex 0 toward +x."""

    centre = truth[annulus].mean(axis=0)
    up = centre - truth[apex]
    up /= np.linalg.norm(up)
    x = truth[annulus[0]] - centre
    x -= (x @ up) * up
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


def _fusion_arrays(folder: Path) -> dict:
    data = _load(folder / "data.npz")
    annulus = data["annulus"]
    n = len(annulus)
    apex = 2 * n
    rotation = _canonical_frame(data["truth"][0], annulus, apex)
    centre = data["truth"][0][annulus].mean(axis=0)
    for key in list(data):
        if key == "truth" or key.startswith("geometry_"):
            data[key] = (data[key] - centre) @ rotation.T
    data["_rotation"], data["_centre"] = rotation, centre
    return data


def _fusion_faces(data):
    return {"chamber": data["chamber_faces"], "anterior_leaflet": data["anterior_faces"],
            "posterior_leaflet": data["posterior_faces"]}


def _draw_fusion_method(ax, data, chamber_key, valve_key, phase, coloured_by_error=True, vmax=FUSION_VMAX):
    faces, annulus, truth = _fusion_faces(data), data["annulus"], data["truth"][phase]
    chamber = data[f"geometry_{chamber_key}"][phase]
    valve = data[f"geometry_{valve_key}"][phase]
    second = SECOND_RING.get(chamber_key)
    if coloured_by_error:
        _mesh(ax, chamber, faces["chamber"], values=np.linalg.norm(chamber - truth, axis=1), alpha=0.92, vmax=vmax)
        for key in ("anterior_leaflet", "posterior_leaflet"):
            _mesh(ax, valve, faces[key], values=np.linalg.norm(valve - truth, axis=1), alpha=0.97, vmax=vmax)
    else:
        _mesh(ax, chamber, faces["chamber"], CHAMBER_C, alpha=0.28)
        _mesh(ax, valve, faces["anterior_leaflet"], ANTERIOR_C, alpha=0.9)
        _mesh(ax, valve, faces["posterior_leaflet"], POSTERIOR_C, alpha=0.9)
    _ring3d(ax, chamber, annulus, ANNULUS_C, lw=1.2)
    if second and f"geometry_{second}" in data:
        _ring3d(ax, data[f"geometry_{second}"][phase], annulus, VALVE_RING_C, lw=1.1, style="--")


def _draw_truth(ax, data, phase):
    faces, annulus, truth = _fusion_faces(data), data["annulus"], data["truth"][phase]
    _mesh(ax, truth, faces["chamber"], "0.70", alpha=0.35)
    _mesh(ax, truth, faces["anterior_leaflet"], "0.48", alpha=0.95)
    _mesh(ax, truth, faces["posterior_leaflet"], "0.62", alpha=0.95)
    _ring3d(ax, truth, annulus, ANNULUS_C, lw=1.2)


def _ring_gap(data, chamber_key, valve_key) -> float:
    annulus = data["annulus"]
    return float(np.abs(data[f"geometry_{chamber_key}"][:, annulus] - data[f"geometry_{valve_key}"][:, annulus]).max())


def _top_view(ax, data, chamber_key, valve_key, phase, title=None, extent=None):
    """Down the true annular axis: the leaflets filled, the true ring in grey, the fitted ring(s) on top."""

    annulus, truth = data["annulus"], data["truth"][phase]
    loop = np.append(annulus, annulus[0])
    chamber = data[f"geometry_{chamber_key}"][phase]
    valve = data[f"geometry_{valve_key}"][phase]
    for faces, colour in ((data["anterior_faces"], ANTERIOR_C), (data["posterior_faces"], POSTERIOR_C)):
        for face in faces:
            ax.fill(*valve[face][:, :2].T, color=colour, alpha=0.30, lw=0)
    ax.plot(*truth[loop][:, :2].T, color=TRUTH_C, lw=2.6, solid_capstyle="round", zorder=2)
    ax.plot(*chamber[loop][:, :2].T, color=ANNULUS_C, lw=0.9, zorder=3)
    second = SECOND_RING.get(chamber_key)
    if second and f"geometry_{second}" in data:
        ax.plot(*data[f"geometry_{second}"][phase][loop][:, :2].T, "--", color=VALVE_RING_C, lw=0.9, zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if extent is not None:
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
    if title:
        ax.set_title(title, fontsize=6.6, pad=2)


def render_fusion_sample(folder: Path, entry: dict) -> None:
    data = _fusion_arrays(folder)
    phase = 0
    points = np.concatenate([data["truth"].reshape(-1, 3), data["geometry_rooted4d"].reshape(-1, 3)])
    limits = _limits(points)
    annulus = data["annulus"]
    ring_extent = 1.18 * float(np.abs(data["truth"][:, annulus, :2]).max())

    # (1) The compared methods, coloured by error, with the annulus from above.
    fig = plt.figure(figsize=(7.2, 2.75))
    widths = 1.0 / 6
    for column, (key, title, chamber_key, valve_key) in enumerate(FUSION_METHODS + [("truth", "Ground truth", "", "")]):
        ax = _place3d(fig, [column * widths, 0.37, widths, 0.50], limits, 42, -62, zoom=1.42,
                      title=SHORT_TITLES.get(key, title))
        if key == "truth":
            _draw_truth(ax, data, phase)
        else:
            _draw_fusion_method(ax, data, chamber_key, valve_key, phase)
        top = fig.add_axes([column * widths + 0.014, 0.11, widths - 0.028, 0.27])
        if key == "truth":
            _top_view(top, data, "rooted4d", "rooted4d", phase, extent=ring_extent)
            top.lines[-1].set_visible(False)
            top.text(0.5, -0.04, "true annulus", transform=top.transAxes, ha="center", va="top", fontsize=6.2)
        else:
            _top_view(top, data, chamber_key, valve_key, phase, extent=ring_extent)
            gap = entry["rows"][key]["attachment_residual_mm"]
            held = entry["rows"][key]
            top.text(0.5, -0.04, "gap %.2f mm\nCMR %.2f, TEE %.2f mm" % (gap, held["heldout_cmr_mm"],
                                                                         held["heldout_tee_mm"]),
                     transform=top.transAxes, ha="center", va="top", fontsize=6.0, linespacing=1.15)
    _colorbar(fig, [0.30, 1.05, 0.40, 0.022], FUSION_VMAX, "distance to the true anatomy (mm)")
    handles = [Line2D([], [], color=TRUTH_C, lw=2.4, label="true annulus"),
               Line2D([], [], color=ANNULUS_C, lw=1.0, label="annulus of the chamber (or the shared annulus)"),
               Line2D([], [], color=VALVE_RING_C, lw=1.0, ls="--", label="second copy of the annulus, where a method keeps two")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, folder / "methods_error_and_annulus", pdf=True)

    # (2) The headline pair: two reconstructions registered afterwards, against one anatomy.
    fig = plt.figure(figsize=(7.2, 2.35))
    panels = [("(a) Late fusion: two annuli", "late_fusion_chamber", "late_fusion_valve"),
              ("(b) Rooted4D: one shared annulus", "rooted4d", "rooted4d"), ("(c) Ground truth", None, None)]
    for column, (title, chamber_key, valve_key) in enumerate(panels):
        ax = _place3d(fig, [column * 0.215, 0.12, 0.215, 0.76], limits, 42, -62, zoom=1.42, title=title,
                      title_size=7.0)
        if chamber_key is None:
            _draw_truth(ax, data, phase)
        else:
            _draw_fusion_method(ax, data, chamber_key, valve_key, phase, coloured_by_error=False)
    late_gap = entry["rows"]["late_fusion"]["attachment_residual_mm"]
    for column, (title, chamber_key, valve_key) in enumerate(panels[:2]):
        ax = fig.add_axes([0.665 + column * 0.168, 0.22, 0.155, 0.58])
        _top_view(ax, data, chamber_key, valve_key, phase, extent=ring_extent,
                  title=("(d) late fusion,\nseen from the atrium" if column == 0 else "(e) Rooted4D,\nseen from the atrium"))
        ax.text(0.5, -0.03, "gap %.2f mm" % (late_gap if column == 0 else 0.0), transform=ax.transAxes,
                ha="center", va="top", fontsize=6.4)
    handles = [Patch(facecolor=CHAMBER_C, alpha=0.4, label="chamber"),
               Patch(facecolor=ANTERIOR_C, alpha=0.8, label="anterior leaflet"),
               Patch(facecolor=POSTERIOR_C, alpha=0.8, label="posterior leaflet"),
               Line2D([], [], color=ANNULUS_C, lw=1.0, label="chamber's annulus"),
               Line2D([], [], color=VALVE_RING_C, lw=1.0, ls="--", label="leaflets' annulus"),
               Line2D([], [], color=TRUTH_C, lw=2.4, label="true annulus")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.07))
    _save(fig, folder / "late_fusion_vs_rooted4d", pdf=True)

    # (3) The annulus along its length: where the two annuli part, and how far each sits from the truth.
    angle = np.linspace(0.0, 360.0, len(annulus), endpoint=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for key, title, chamber_key, valve_key in FUSION_METHODS:
        colour, style = PHASE_STYLE[key]
        chamber = data[f"geometry_{chamber_key}"][:, annulus]
        second = SECOND_RING.get(chamber_key)
        other = data[f"geometry_{second}"][:, annulus] if second and f"geometry_{second}" in data else chamber
        between = np.linalg.norm(chamber - other, axis=2).max(axis=0)
        to_truth = np.linalg.norm(chamber - data["truth"][:, annulus], axis=2).mean(axis=0)
        axes[0].plot(angle, between, ls=style, color=colour, lw=1.1, label=title)
        axes[1].plot(angle, to_truth, ls=style, color=colour, lw=1.1, label=title)
    axes[0].set_ylabel("gap between the two\nannulus copies (mm)")
    axes[1].set_ylabel("distance to the\ntrue annulus (mm)")
    for ax in axes:
        ax.set_xlabel("position around the annulus (degrees from vertex 0)")
        ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360]); ax.set_ylim(bottom=0)
        ax.grid(True, axis="y")
        for commissure in (0.0, 180.0):
            ax.axvline(commissure, color="0.8", lw=0.6, zorder=0)
    axes[0].set_title("(a) largest over the eight drawn phases", loc="left", fontsize=7)
    axes[1].set_title("(b) mean over the eight drawn phases", loc="left", fontsize=7)
    for ax in axes:
        ax.text(182, 0.02, "commissure", transform=ax.get_xaxis_transform(), fontsize=5.8, color="0.5")
    axes[1].legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(w_pad=2.0)
    _save(fig, folder / "annulus_profile")

    # (4) Rooted4D over the cycle, against the true anatomy.
    fig = plt.figure(figsize=(7.2, 3.5))
    for index in range(len(data["phases"])):
        row, column = divmod(index, 4)
        ax = _place3d(fig, [column * 0.25, 0.53 - row * 0.47, 0.25, 0.41], limits, 42, -62, zoom=1.25,
                      title="phase %.3f" % data["phases"][index])
        faces, truth = _fusion_faces(data), data["truth"][index]
        geometry = data["geometry_rooted4d"][index]
        _mesh(ax, truth, faces["chamber"], "0.75", alpha=0.18)
        _ring3d(ax, truth, annulus, TRUTH_C, lw=2.2)
        _mesh(ax, geometry, faces["chamber"], CHAMBER_C, alpha=0.22)
        _mesh(ax, geometry, faces["anterior_leaflet"], ANTERIOR_C, alpha=0.9)
        _mesh(ax, geometry, faces["posterior_leaflet"], POSTERIOR_C, alpha=0.9)
        _ring3d(ax, geometry, annulus, ANNULUS_C, lw=1.0)
    handles = [Patch(facecolor=CHAMBER_C, alpha=0.4, label="Rooted4D chamber"),
               Patch(facecolor=ANTERIOR_C, alpha=0.8, label="anterior leaflet"),
               Patch(facecolor=POSTERIOR_C, alpha=0.8, label="posterior leaflet"),
               Line2D([], [], color=ANNULUS_C, lw=1.0, label="Rooted4D annulus"),
               Patch(facecolor="0.75", alpha=0.5, label="true chamber"),
               Line2D([], [], color=TRUTH_C, lw=2.2, label="true annulus")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, folder / "rooted4d_cycle_vs_truth")

    # (5) Inputs: rendered likelihoods and observed points, and the input-derived start.
    render_fusion_inputs(folder, data)

    # (6) Each method's estimated TEE time-to-phase map, minus the true one.
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    times = data["tee_times"]
    for key, title, _, _ in FUSION_METHODS:
        colour, style = PHASE_STYLE[key]
        error = (data[f"phase_map_{key}"] - data["phase_map_truth"] + 0.5) % 1.0 - 0.5
        ax.plot(times, error, ls=style, color=colour, lw=1.1,
                label="%s (%.3f)" % (title, entry["rows"][key]["phase_map_cycles"]))
    ax.axhline(0.0, color="0.75", lw=0.7, zorder=0)
    ax.grid(True, axis="y")
    ax.set_xlabel("TEE acquisition time (normalized)")
    ax.set_ylabel("phase-map error (cycles)")
    ax.legend(frameon=False, fontsize=6.0, title="median absolute error, cycles", title_fontsize=6.0,
              loc="upper left", bbox_to_anchor=(1.0, 1.02))
    _save(fig, folder / "phase_map_errors")


def render_fusion_inputs(folder: Path, data: dict) -> None:
    cmr_images, tee_images = data["cmr_slice_likelihood"], data["tee_sector_likelihood"]
    fig = plt.figure(figsize=(7.2, 3.3))
    for column in range(cmr_images.shape[1]):
        ax = fig.add_axes([0.005 + column * 0.12, 0.55, 0.115, 0.38])
        ax.imshow(cmr_images[0, column], cmap="gray", interpolation="nearest")
        ax.set_title("CMR slice %d" % (column + 1), fontsize=6.6, pad=2)
        ax.set_axis_off()
    frame = 0
    middle_az, middle_el = tee_images.shape[2] // 2, tee_images.shape[3] // 2
    for column, (panel, title) in enumerate(((tee_images[frame, :, :, middle_el], "TEE depth \u00d7 azimuth"),
                                             (tee_images[frame, :, middle_az, :], "TEE depth \u00d7 elevation"),
                                             (tee_images[frame].max(axis=0), "TEE, max over depth"))):
        ax = fig.add_axes([0.005 + column * 0.12, 0.08, 0.115, 0.38])
        ax.imshow(panel, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=5.9, pad=2)
        ax.set_axis_off()
    fig.text(0.30, 0.985, "(a) image likelihoods the generator renders, first frame", ha="center", va="top",
             fontsize=7)
    # Points in the original CMR frame (not the drawing frame), which is where the inputs live.
    cmr_points, tee_points = data["cmr_points"][0], data["tee_points"][0]
    start = data["tee_points_in_cmr_frame_at_start"][0]
    for column, (title, sets) in enumerate((
            ("(b) CMR points, frame 1", [(cmr_points, CHAMBER_C)]),
            ("(c) TEE points, own frame", [(tee_points, ANTERIOR_C)]),
            ("(d) start: TEE mapped to CMR", [(cmr_points, CHAMBER_C), (start, ANTERIOR_C)]))):
        stacked = np.concatenate([s for s, _ in sets])
        ax = fig.add_axes([0.62 + (column % 2) * 0.19, 0.50 - (column // 2) * 0.47, 0.19, 0.47], projection="3d")
        for setter, (low, high) in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), _limits(stacked, 0.6)):
            setter(low, high)
        ax.set_box_aspect((1, 1, 1), zoom=1.2)
        ax.view_init(elev=24, azim=-60)
        ax.set_axis_off()
        for points, colour in sets:
            ax.scatter(*points.T, s=2.5, color=colour, depthshade=False, linewidths=0)
        ax.text2D(0.5, 0.98, title, transform=ax.transAxes, ha="center", va="top", fontsize=6.6)
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=CHAMBER_C, label="CMR"),
               Line2D([], [], ls="none", marker="o", ms=3, color=ANTERIOR_C, label="TEE")]
    fig.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.4, bbox_to_anchor=(0.995, 0.02))
    _save(fig, folder / "inputs")


def render_fusion_summary(root: Path, entries: list) -> None:
    rows = len(entries)
    fig = plt.figure(figsize=(7.2, 1.38 * rows + 0.45))
    height = 1.0 / (rows + 0.35)
    widths = 1.0 / 6
    for row, entry in enumerate(entries):
        folder = root / "fusion" / f"{entry['label']}_{Path(entry['sample']).stem}"
        data = _fusion_arrays(folder)
        points = np.concatenate([data["truth"].reshape(-1, 3), data["geometry_rooted4d"].reshape(-1, 3)])
        limits = _limits(points)
        bottom = 1.0 - (row + 1) * height
        for column, (key, title, chamber_key, valve_key) in enumerate(
                FUSION_METHODS + [("truth", "Ground truth", "", "")]):
            ax = _place3d(fig, [column * widths, bottom + 0.1 * height, widths, 0.9 * height], limits, 42, -62,
                          zoom=1.25, title=SHORT_TITLES.get(key, title) if row == 0 else None)
            if key == "truth":
                _draw_truth(ax, data, 0)
            else:
                _draw_fusion_method(ax, data, chamber_key, valve_key, 0)
                held = entry["rows"][key]
                fig.text(column * widths + 0.5 * widths, bottom + 0.05 * height,
                         "%.2f + %.2f mm" % (held["heldout_cmr_mm"], held["heldout_tee_mm"]), ha="center",
                         fontsize=5.9, color="0.25")
        fig.text(0.002, bottom + 0.5 * height, RANK_NAMES[entry["label"]], rotation=90, ha="left",
                 va="center", fontsize=6.8)
    _colorbar(fig, [0.30, 0.012, 0.40, 0.012], FUSION_VMAX, "distance to the true anatomy (mm)")
    _save(fig, root / "summary_fusion_methods", pdf=True)


# ------------------------------------------------------------------- image ----

IMAGE_NAMES = {"point_fit": "point fit\n(the warm start)", "image_cold": "image,\nzero start",
               "commissures_cold": "image + 2 commissures,\nzero start",
               "landmarks_8_cold": "image + 8 landmarks,\nzero start", "image_warm": "image,\nwarm start"}


def _image_panels(axes, data):
    truth, annulus = data["truth"], data["annulus"]
    plane = np.linalg.svd(truth[annulus] - truth[annulus].mean(axis=0))[2][:2]

    def flat(points):
        points = np.asarray(points)
        return (points - points[annulus].mean(axis=0)) @ plane.T

    loop = np.append(annulus, annulus[0])
    keys, rotations = [str(k) for k in data["arm_keys"]], data["rotations"]
    reference = flat(truth)
    shapes = {key: flat(data[f"geometry_{key}"]) for key in keys}
    extent = 1.06 * np.abs(np.concatenate([reference[annulus]] + [s[annulus] for s in shapes.values()])).max()
    for ax, key, rotation in zip(axes, keys, rotations):
        shape = shapes[key]
        for faces, colour in ((data["anterior_faces"], "0.78"), (data["posterior_faces"], "0.88")):
            for face in faces:
                ax.fill(*reference[face].T, color=colour, lw=0)
        ax.plot(*reference[loop].T, color=TRUTH_C, lw=2.2)
        for faces, colour in ((data["anterior_faces"], ANTERIOR_C), (data["posterior_faces"], POSTERIOR_C)):
            for face in faces:
                ax.fill(*shape[face].T, color=colour, alpha=0.42, lw=0)
        ax.plot(*shape[loop].T, color=ANNULUS_C, lw=0.9)
        ax.plot(*shape[annulus[:1]].T, "o", color=ANNULUS_C, ms=2.6)
        ax.plot(*reference[annulus[:1]].T, "o", color=TRUTH_C, ms=2.6)
        if "commissures" in key:
            ax.plot(*reference[data["commissures"]].T, "^", color="#2a9d8f", ms=3.6)
        if "landmarks_8" in key:
            ax.plot(*reference[data["landmarks_8"]].T, "^", color="#2a9d8f", ms=3.0)
        good = rotation < 5.0
        ax.text(0.5, -0.03, "%.1f°" % rotation, transform=ax.transAxes, ha="center", va="top",
                fontsize=7, fontweight="bold" if good else "normal", color="#1b7a3a" if good else "#a33")
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    return keys


def render_image_sample(folder: Path) -> None:
    data = _load(folder / "data.npz")
    fig, axes = plt.subplots(1, len(data["arm_keys"]), figsize=(7.2, 1.9))
    keys = _image_panels(axes, data)
    for ax, key in zip(axes, keys):
        ax.set_title(IMAGE_NAMES.get(key, key), fontsize=6.5, pad=2)
    _image_legend(fig)
    _save(fig, folder / "image_evidence_top_view")


def _image_legend(fig, y=-0.08):
    handles = [Line2D([], [], color=TRUTH_C, lw=2.2, label="true annulus (leaflets in grey)"),
               Line2D([], [], color=ANNULUS_C, lw=0.9, label="fitted annulus"),
               Patch(facecolor=ANTERIOR_C, alpha=0.5, label="fitted anterior leaflet"),
               Patch(facecolor=POSTERIOR_C, alpha=0.5, label="fitted posterior leaflet"),
               Line2D([], [], ls="none", marker="o", ms=3, color=ANNULUS_C, label="vertex 0"),
               Line2D([], [], ls="none", marker="^", ms=3.5, color="#2a9d8f", label="landmarks used")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.0, bbox_to_anchor=(0.5, y))


def render_image_summary(root: Path, entries: list) -> None:
    fig, axes = plt.subplots(len(entries), 5, figsize=(7.2, 1.55 * len(entries) + 0.3), squeeze=False)
    for row, entry in enumerate(entries):
        data = _load(root / "image" / f"{entry['label']}_{Path(entry['sample']).stem}" / "data.npz")
        keys = _image_panels(axes[row], data)
        axes[row, 0].text(-0.08, 0.5, RANK_NAMES[entry["label"]], rotation=90, transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=6.8)
        if row == 0:
            for ax, key in zip(axes[row], keys):
                ax.set_title(IMAGE_NAMES.get(key, key), fontsize=6.5, pad=3)
    fig.subplots_adjust(hspace=0.32, wspace=0.05)
    _image_legend(fig, y=-0.04)
    fig.text(0.5, -0.07, "Seen down the true annular axis, each fit centred on the truth so that only its rotation "
             "about the axis shows; the number is the rotation error.", ha="center", fontsize=6.2)
    _save(fig, root / "summary_image_evidence", pdf=True)


# ------------------------------------------------------------------- modes ----

def render_modes(folder: Path) -> None:
    data = _load(folder / "data.npz")
    summary = {(r["space"], r["sweep"]): r for r in json.loads(str(data["summary"]))}
    faces = {"chamber": data["chamber_faces"], "anterior_leaflet": data["anterior_faces"],
             "posterior_leaflet": data["posterior_faces"]}
    annulus = data["annulus"]
    rows = [("S", "annulus_preserving", "(a) structured space: largest direction that keeps the annulus fixed"),
            ("S", "leading", "(b) structured space: leading mode"),
            ("G", "leading", "(c) global space: leading mode (no direction keeps the annulus fixed)")]
    reference = data["S_leading"][1]
    rotation = _canonical_frame(reference, annulus, 2 * len(annulus))
    centre = reference[annulus].mean(axis=0)

    def frame(points):
        return (points - centre) @ rotation.T

    everything = np.concatenate([frame(data[f"{s}_{n}"].reshape(-1, 3)) for s, n, _ in rows])
    limits = _limits(everything, 0.5)
    fig = plt.figure(figsize=(5.4, 6.2))
    for row, (space, sweep, title) in enumerate(rows):
        shapes = frame(data[f"{space}_{sweep}"])
        bottom = 0.67 - row * 0.325
        fig.text(0.01, bottom + 0.30, title, fontsize=7, ha="left")
        moved = summary[(space, sweep)]["annulus_moves_mm"]
        fig.text(0.99, bottom + 0.30, "annulus moves %s" % ("%.0e mm" % moved if moved < 1e-6 else "%.1f mm" % moved),
                 fontsize=6.4, ha="right", color="0.3")
        for column, label in enumerate(("$-2$ SD", "mean", "$+2$ SD")):
            ax = fig.add_axes([column / 3, bottom, 1 / 3, 0.29], projection="3d")
            for setter, (low, high) in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
                setter(low, high)
            ax.set_box_aspect((1, 1, 1), zoom=1.25)
            ax.view_init(elev=24, azim=-62)
            ax.set_axis_off()
            _mesh(ax, shapes[column], faces["chamber"], CHAMBER_C, alpha=0.22)
            _mesh(ax, shapes[column], faces["anterior_leaflet"], ANTERIOR_C, alpha=0.9)
            _mesh(ax, shapes[column], faces["posterior_leaflet"], POSTERIOR_C, alpha=0.9)
            _ring3d(ax, shapes[1], annulus, TRUTH_C, lw=2.2)
            _ring3d(ax, shapes[column], annulus, ANNULUS_C, lw=1.0)
            ax.text2D(0.5, 0.02, label, transform=ax.transAxes, ha="center", fontsize=6.8)
    handles = [Line2D([], [], color=ANNULUS_C, lw=1.0, label="annulus of the swept shape"),
               Line2D([], [], color=TRUTH_C, lw=2.2, label="annulus of the mean shape")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.01))
    _save(fig, folder / "shape_space_sweeps", pdf=True)


# ----------------------------------------------------------------- chamber ----

def _section(vertices: np.ndarray, faces: np.ndarray, axis: int, value: float) -> np.ndarray:
    """Segments where a triangulated surface crosses the plane ``x[axis] = value``, shape (k, 2, 3)."""

    d = vertices[:, axis] - value
    tri = np.asarray(faces)
    sd = d[tri]
    above = sd > 0
    count = above.sum(axis=1)
    crossing = (count == 1) | (count == 2)
    segments = []
    for face, signs, dist in zip(tri[crossing], above[crossing], sd[crossing]):
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if signs[i] != signs[j]:
                t = dist[i] / (dist[i] - dist[j])
                points.append(vertices[face[i]] + t * (vertices[face[j]] - vertices[face[i]]))
        if len(points) == 2:
            segments.append(points)
    return np.asarray(segments).reshape(-1, 2, 3)


def _chamber_frame_arrays(folder: Path, frame: int) -> dict:
    """Every method's surface for one frame, in the label's index-times-spacing millimetres."""

    ours = _load(folder / f"frame{frame:02d}_rooted4d.npz")
    out = {"image": ours["image"], "zooms": ours["zooms"], "labels": ours["labels_reference"],
           "points": ours["reference_points"], "surfaces": {}}
    for key in ("rooted4d_label", "rooted4d_compact", "rooted4d_nnunet"):
        if f"{key}_vertices" in ours:
            out["surfaces"][key] = (ours[f"{key}_vertices"], ours["faces"])
    tetheart = folder / f"frame{frame:02d}_tetheart.npz"
    if tetheart.exists():
        t = _load(tetheart)
        out["surfaces"]["tetheart"] = (t["clipped_vertices"], t["clipped_faces"])
        out["tetheart_raw"] = (t["raw_vertices"], t["raw_faces"])
    ghd = folder / f"frame{frame:02d}_ghdheart.npz"
    if ghd.exists():
        g = _load(ghd)
        out["surfaces"]["ghdheart"] = (g["vertices"], g["faces"])
    return out


def _inplane_distances(vertices, labels, zooms):
    """Each vertex's distance to the reference contour, in its short-axis plane, between slices interpolated.

    The label exists only on slices 5 to 10 mm apart, so the 3D distance to the nearest
    labelled voxel grows between slices for every surface, and a finely meshed one is
    striped by it. Measured in-plane on the two slices that bracket a vertex and
    interpolated linearly, the colour shows where a surface leaves the anatomy. The
    first and last slices are filled discs, as the reference's caps are; beyond them
    the distance along the stack is added, so an overhang shows.
    """
    from scipy.spatial import cKDTree

    cavity = np.asarray(labels) == 3
    slices = [k for k in range(cavity.shape[2]) if cavity[:, :, k].any()]
    trees, heights = [], []
    for k in slices:
        plane = cavity[:, :, k]
        if k in (slices[0], slices[-1]):
            rows, columns = np.nonzero(plane)       # the end slices are the reference's caps
        else:
            padded = np.pad(plane, 1)
            interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2]
                        & padded[1:-1, 2:])
            rows, columns = np.nonzero(plane & ~interior)
        trees.append(cKDTree(np.stack([rows * zooms[0], columns * zooms[1]], axis=1)))
        heights.append(k * zooms[2])
    heights = np.asarray(heights)
    xy, z = vertices[:, :2], vertices[:, 2]
    per_slice = np.stack([tree.query(xy)[0] for tree in trees], axis=1)
    upper = np.clip(np.searchsorted(heights, z), 1, len(heights) - 1)
    lower = upper - 1
    t = np.clip((z - heights[lower]) / (heights[upper] - heights[lower]), 0.0, 1.0)
    rows = np.arange(len(z))
    distance = (1 - t) * per_slice[rows, lower] + t * per_slice[rows, upper]
    below, above = z < heights[0], z > heights[-1]
    distance[below] = np.hypot(per_slice[below, 0], heights[0] - z[below])
    distance[above] = np.hypot(per_slice[above, -1], z[above] - heights[-1])
    return distance


def _chamber_view_limits(arrays):
    points = arrays["points"]
    return _limits(points, 0.62)


def _draw_chamber_surface(ax, arrays, key):
    points = arrays["points"]
    if key == "reference":
        ax.scatter(*points.T, s=0.35, color="0.35", depthshade=False, linewidths=0, alpha=0.8)
        return
    vertices, faces = arrays["surfaces"][key]
    if key == "tetheart" and "tetheart_raw" in arrays:
        raw_vertices, raw_faces = arrays["tetheart_raw"]
        z = raw_vertices[:, 2]
        outside = ((z[raw_faces] < points[:, 2].min()) | (z[raw_faces] > points[:, 2].max())).any(axis=1)
        if outside.any():
            _mesh(ax, raw_vertices, raw_faces[outside], "0.72", alpha=0.35)
    _mesh(ax, vertices, faces, values=_inplane_distances(vertices, arrays["labels"], arrays["zooms"]), alpha=0.97,
          vmax=CHAMBER_VMAX)


def _slice_image(ax, arrays, axis, index, crop):
    image, zooms = arrays["image"], arrays["zooms"]
    if axis == 2:
        plane = image[:, :, index]
        extent = (-0.5 * zooms[0], (image.shape[0] - 0.5) * zooms[0], -0.5 * zooms[1], (image.shape[1] - 0.5) * zooms[1])
        label = arrays["labels"][:, :, index] == 3
    else:
        plane = image[index, :, :]
        extent = (-0.5 * zooms[1], (image.shape[1] - 0.5) * zooms[1], -0.5 * zooms[2], (image.shape[2] - 0.5) * zooms[2])
        label = arrays["labels"][index, :, :] == 3
    finite = plane[np.isfinite(plane)]
    low, high = np.percentile(finite, [1.0, 99.5])
    ax.imshow(np.clip((plane.T - low) / max(high - low, 1e-6), 0, 1), cmap="gray", origin="lower", extent=extent,
              interpolation="nearest" if axis != 2 else "bilinear", aspect="equal")
    if label.any():
        ax.contour(label.T.astype(float), levels=[0.5], colors=[REFERENCE_C], linewidths=1.0, linestyles="-",
                   origin="lower", extent=extent)
    ax.set_xlim(crop[0]); ax.set_ylim(crop[1])
    if axis != 2:
        ax.invert_yaxis()       # slice 0 is basal in ACDC: base up, apex down
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_section(ax, vertices, faces, axis, value, colour, lw=1.1, style="-"):
    segments = _section(vertices, faces, axis, value)
    if len(segments) == 0:
        return
    keep = [a for a in range(3) if a != axis]
    ax.add_collection(LineCollection(segments[:, :, keep], colors=[colour], linewidths=lw, linestyles=style,
                                     capstyle="round"))


def _slice_geometry(arrays):
    labels, zooms = arrays["labels"], arrays["zooms"]
    cavity = np.argwhere(labels == 3)
    slices = np.unique(cavity[:, 2])
    mid = int(slices[len(slices) // 2])
    centre = cavity.mean(axis=0)
    column = int(round(centre[0]))
    half = 0.5 * max(np.ptp(cavity[:, 0]) * zooms[0], np.ptp(cavity[:, 1]) * zooms[1]) + 18.0
    short_crop = ((centre[0] * zooms[0] - half, centre[0] * zooms[0] + half),
                  (centre[1] * zooms[1] - half, centre[1] * zooms[1] + half))
    long_half = 0.5 * np.ptp(cavity[:, 1]) * zooms[1] + 18.0
    z_low, z_high = -0.5 * zooms[2], (labels.shape[2] - 0.5) * zooms[2]
    long_crop = ((centre[1] * zooms[1] - long_half, centre[1] * zooms[1] + long_half),
                 (min(z_low, -25.0), max(z_high, labels.shape[2] * zooms[2] + 25.0)))
    return mid, column, short_crop, long_crop


def render_chamber_frame(folder: Path, entry: dict, frame: int, scores: dict, title: str) -> None:
    arrays = _chamber_frame_arrays(folder, frame)
    zooms = arrays["zooms"]
    mid, column, short_crop, long_crop = _slice_geometry(arrays)
    methods = [("reference", "Reference\nlabel", REFERENCE_C)] + [m for m in CHAMBER_METHODS
                                                                    if m[0] in arrays["surfaces"]]
    count = len(methods)
    fig = plt.figure(figsize=(7.2, 4.15))
    width = 1.0 / count
    limits = _chamber_view_limits(arrays)
    for index, (key, name, colour) in enumerate(methods):
        ax = _place3d(fig, [index * width, 0.62, width, 0.27], limits, 18, -58, zoom=1.22, base_up=True,
                      title=name)
        _draw_chamber_surface(ax, arrays, key)
        if key != "reference" and key in scores:
            fig.text(index * width + 0.5 * width, 0.605, "%.2f mm" % scores[key], ha="center", va="top",
                     fontsize=6.4, color="0.2")
        short = fig.add_axes([index * width + 0.004, 0.30, width - 0.008, 0.27])
        _slice_image(short, arrays, 2, mid, short_crop)
        long_ = fig.add_axes([index * width + 0.004, 0.01, width - 0.008, 0.27])
        _slice_image(long_, arrays, 0, column, long_crop)
        if key != "reference":
            vertices, faces = arrays["surfaces"][key]
            _draw_section(short, vertices, faces, 2, mid * zooms[2], colour)
            if key == "tetheart" and "tetheart_raw" in arrays:
                _draw_section(long_, *arrays["tetheart_raw"], 0, column * zooms[0], colour, lw=0.9, style=":")
            _draw_section(long_, vertices, faces, 0, column * zooms[0], colour)
        if index == 0:
            short.text(-0.06, 0.5, "short axis, slice %d" % mid, rotation=90, transform=short.transAxes,
                       ha="right", va="center", fontsize=6.3)
            long_.text(-0.06, 0.5, "long-axis reformat", rotation=90, transform=long_.transAxes, ha="right",
                       va="center", fontsize=6.3)
    _colorbar(fig, [0.30, -0.03, 0.40, 0.016], CHAMBER_VMAX, CHAMBER_BAR)
    fig.text(0.5, -0.12, "%s. Under each surface: its symmetric surface distance to the reference label, the "
             "metric of the comparison table. Reference contour in cream; each method's cut through its surface "
             "in colour (TetHeart's unclipped surface dotted)." % title, ha="center", fontsize=6.0, wrap=True)
    _save(fig, folder / f"chamber_methods_frame{frame:02d}", pdf=True)


def render_external_summary(root: Path, entries: list, scores: dict) -> None:
    rows = len(entries)
    methods = [("reference", "Reference\nlabel")] + [(k, n) for k, n, _ in CHAMBER_METHODS]
    fig = plt.figure(figsize=(7.2, 1.45 * rows + 0.5))
    height = 1.0 / (rows + 0.4)
    width = 1.0 / len(methods)
    for row, entry in enumerate(entries):
        folder = root / "external" / f"{entry['label']}_{entry['patient']}"
        arrays = _chamber_frame_arrays(folder, entry["donor_frame"])
        limits = _chamber_view_limits(arrays)
        bottom = 1.0 - (row + 1) * height
        for column, (key, name) in enumerate(methods):
            if key != "reference" and key not in arrays["surfaces"]:
                continue
            ax = _place3d(fig, [column * width, bottom + 0.1 * height, width, 0.9 * height], limits, 18, -58,
                          zoom=1.2, base_up=True, title=name if row == 0 else None)
            _draw_chamber_surface(ax, arrays, key)
            value = scores.get((entry["patient"], entry["donor_frame"]), {}).get(key)
            if value is not None:
                fig.text(column * width + 0.5 * width, bottom + 0.06 * height, "%.2f mm" % value, ha="center",
                         fontsize=6.2, color="0.2")
        fig.text(0.002, bottom + 0.5 * height, "%s\n%s" % (RANK_NAMES[entry["label"]], entry["patient"]),
                 rotation=90, ha="left", va="center", fontsize=6.4, linespacing=1.1)
    _colorbar(fig, [0.30, 0.012, 0.40, 0.010], CHAMBER_VMAX, CHAMBER_BAR)
    _save(fig, root / "summary_external_chamber", pdf=True)


def render_pair(root: Path, fusion_entry: dict, picks_entry: dict, scores: dict) -> None:
    """One fusion sample and its chamber donor's frame: the qualitative results side by side."""

    folder = root / "fusion" / f"{fusion_entry['label']}_{Path(fusion_entry['sample']).stem}"
    data = _fusion_arrays(folder)
    fig = plt.figure(figsize=(7.2, 3.9))
    points = np.concatenate([data["truth"].reshape(-1, 3), data["geometry_rooted4d"].reshape(-1, 3)])
    limits = _limits(points)
    fig.text(0.0, 0.985, "(a) Joint CMR\u2013TEE reconstruction, realistic-session benchmark at 96 ring nodes:\nthe "
             "test sample at the %s of Rooted4D's error (chamber from ACDC %s, valve from MVSeg2023)"
             % (RANK_NAMES[fusion_entry["label"]], picks_entry["patient"]), fontsize=6.6, va="top")
    for column, (title, chamber_key, valve_key) in enumerate(
            [("Late fusion", "late_fusion_chamber", "late_fusion_valve"),
             ("Rooted4D", "rooted4d", "rooted4d"), ("Ground truth", None, None)]):
        ax = _place3d(fig, [0.02 + column * 0.21, 0.55, 0.21, 0.36], limits, 42, -62, zoom=1.22, title=title)
        if chamber_key is None:
            _draw_truth(ax, data, 0)
        else:
            _draw_fusion_method(ax, data, chamber_key, valve_key, 0)
    rows = fusion_entry["rows"]
    fig.text(0.68, 0.86, "held-out error, CMR + TEE\nlate fusion: %.2f + %.2f mm, gap %.2f mm\n"
             "Rooted4D: %.2f + %.2f mm, gap 0" % (rows["late_fusion"]["heldout_cmr_mm"],
                                                rows["late_fusion"]["heldout_tee_mm"],
                                                rows["late_fusion"]["attachment_residual_mm"],
                                                rows["rooted4d"]["heldout_cmr_mm"], rows["rooted4d"]["heldout_tee_mm"]),
             fontsize=6.5, va="top", linespacing=1.3)
    _colorbar(fig, [0.68, 0.64, 0.27, 0.018], FUSION_VMAX, "distance to the true anatomy (mm)")
    arrays = _chamber_frame_arrays(root / "external" / f"{picks_entry['label']}_{picks_entry['patient']}",
                                   picks_entry["donor_frame"])
    fig.text(0.0, 0.485, "(b) The same chamber donor in ACDC (%s, frame %d): chamber reconstruction from cine "
             "CMR,\ncoloured by in-plane distance to the reference contour (0\u2013%g mm); below each, the symmetric "
             "surface distance" % (picks_entry["patient"], picks_entry["donor_frame"], CHAMBER_VMAX),
             fontsize=6.6, va="top")
    methods = [("reference", "Reference\nlabel")] + [(k, n) for k, n, _ in CHAMBER_METHODS if k in arrays["surfaces"]]
    width = 1.0 / len(methods)
    limits = _chamber_view_limits(arrays)
    frame_scores = scores.get((picks_entry["patient"], picks_entry["donor_frame"]), {})
    for column, (key, name) in enumerate(methods):
        ax = _place3d(fig, [column * width, 0.07, width, 0.30], limits, 18, -58, zoom=1.2, base_up=True,
                      title=name, title_size=6.4)
        _draw_chamber_surface(ax, arrays, key)
        if key in frame_scores:
            fig.text(column * width + 0.5 * width, 0.055, "%.2f mm" % frame_scores[key], ha="center", va="top",
                     fontsize=6.2, color="0.2")
    _save(fig, root / "pairs" / f"{fusion_entry['label']}_{picks_entry['patient']}", pdf=True)


# ------------------------------------------------------------------ checks ----

def chamber_checks(root: Path, workspace: Path) -> dict:
    """Each drawn chamber surface's score next to the committed row it should match."""

    e36 = {(r["patient"], r["frame"]): r for r in
           json.loads((workspace / "results/21931418/E36_image_input_chamber.json").read_text())["rows"]}
    e84 = {(r["patient"], r["frame"]): r for r in
           json.loads((workspace / "results/benchmark/E84/E84_nnunet_chamber.json").read_text())["rows"]}
    e49g = {(r["patient"], r["frame"]): r for r in
            json.loads((workspace / "results/22102244_thfinal/E49g_test_iter_60000_matched.json").read_text())["rows"]
            if r.get("status") == "scored"}
    e37 = {(r["patient"], r["frame"]): r for r in
           json.loads((workspace / "results/21959021_test/E37_test_B64.json").read_text())["results"]["64"]["rows"]}
    ours = {(r["patient"], r["frame"]): r for r in json.loads((root / "chamber_index.json").read_text())["samples"]}
    tet_path, ghd_path = root / "tetheart_index.json", root / "ghdheart_index.json"
    tet = ({(r["patient"], r["frame"]): r for r in json.loads(tet_path.read_text())["samples"]}
           if tet_path.exists() else {})
    ghd = ({(r["patient"], r["frame"]): r for r in json.loads(ghd_path.read_text())["samples"]}
           if ghd_path.exists() else {})
    report, scores = [], {}
    for key, entry in sorted(ours.items()):
        record = {"patient": key[0], "frame": key[1], "label": entry["label"]}

        def pair(name, drawn, published):
            record[name] = {"drawn_mm": drawn, "published_mm": published,
                            "difference_mm": None if drawn is None or published is None else drawn - published}
            if drawn is not None:
                scores.setdefault(key, {})[name] = drawn

        pair("rooted4d_label", (entry.get("rooted4d_label") or {}).get("symmetric_mm"),
             e36[key]["label_input"]["symmetric_mean_mm"] if key in e36 else None)
        pair("rooted4d_compact", (entry.get("rooted4d_compact") or {}).get("symmetric_mm"),
             e84[key]["compact_unet"]["symmetric_mm"] if key in e84 else None)
        pair("rooted4d_nnunet", (entry.get("rooted4d_nnunet") or {}).get("symmetric_mm"),
             e84[key]["nnunet"]["symmetric_mm"] if key in e84 else None)
        if key in tet:
            pair("tetheart", tet[key]["clipped"]["symmetric_mm"],
                 e49g[key]["symmetric_mm_clipped_with_endcaps"] if key in e49g else None)
            record["tetheart_unclipped"] = {"drawn_mm": tet[key]["raw"]["symmetric_mm"],
                                            "published_mm": e49g.get(key, {}).get("symmetric_mm_raw_with_endcaps"),
                                            "stack_extent_mm": tet[key]["stack_extent_mm"]}
        if key in ghd and not ghd[key].get("failed"):
            pair("ghdheart", ghd[key]["symmetric_mm"], e37[key]["symmetric_mean_mm"] if key in e37 else None)
        report.append(record)
    (root / "external_index.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", root / "external_index.json", flush=True)
    return scores


# -------------------------------------------------------------------- main ----

def render_all(root: Path) -> int:
    workspace = Path.cwd()
    picks = json.loads((root / "picks.json").read_text())["samples"]
    fusion = json.loads((root / "fusion_index.json").read_text())["samples"] if (root / "fusion_index.json").exists() else []
    for entry in fusion:
        render_fusion_sample(root / "fusion" / f"{entry['label']}_{Path(entry['sample']).stem}", entry)
        print("fusion", entry["label"], flush=True)
    if fusion:
        render_fusion_summary(root, fusion)
    if (root / "image_index.json").exists():
        image = json.loads((root / "image_index.json").read_text())["samples"]
        for entry in image:
            render_image_sample(root / "image" / f"{entry['label']}_{Path(entry['sample']).stem}")
        render_image_summary(root, image)
        print("image", flush=True)
    if (root / "modes" / "data.npz").exists():
        render_modes(root / "modes")
        print("modes", flush=True)
    if (root / "chamber_index.json").exists():
        scores = chamber_checks(root, workspace)
        for entry in picks:
            folder = root / "external" / f"{entry['label']}_{entry['patient']}"
            for frame in entry["frames"]:
                title = "ACDC %s, frame %d (%s)" % (entry["patient"], frame["frame"],
                                                    "the frame the chamber donor was built from" if frame["donor"]
                                                    else "the other labelled frame")
                render_chamber_frame(folder, entry, frame["frame"], scores.get((entry["patient"], frame["frame"]), {}),
                                     title)
            print("external", entry["label"], flush=True)
        render_external_summary(root, picks, scores)
        if fusion:
            (root / "pairs").mkdir(exist_ok=True)
            by_label = {e["label"]: e for e in picks}
            for entry in fusion:
                render_pair(root, entry, by_label[entry["label"]], scores)
    return 0
