"""E118's panels: Figure 5's qualitative comparison with the methods added after E109.

Draws, locally and from the arrays E109 and E118 saved, every row of the
paper's attachment figure (reference, late fusion, late fusion + BCPD, GPMM
fusion, Rooted4D's joint fit, Rooted4D) in the published figure's style, so the
new rows sit beside the old ones. The published panels came from a renderer
that is not in this repository; their camera was recovered here by fitting the
drawn annulus of the four reference panels (shared camera, orthographic), which
reproduces the reference silhouettes at an intersection over union of 0.98 to
0.99 and the leaflet colours at 0.90 to 0.98. Colours are the published
legend's; the inset is the published crop.

Writes, under ``--out``:

* ``panels/<row>_phase<p>_{main,inset}.png``: drop-in panels at the published
  pixel size (345 x 255 and 85 x 62, as embedded in the draw.io source), and
  ``_hires`` copies at four times that size;
* ``fig_qualitative_comparison_extended.{pdf,png}``: all six rows in a compact
  layout, the inset beside each panel, labelled with each phase's mean annulus
  separation;
* ``index.json``: the drawn separations and the camera.

    python e118_render.py --sample results/qualitative96/fusion/q75_sample-66fc7a9456bbb0f4 --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

#: The published figure's colours (its draw.io legend).
ANTERIOR = "#e9ac42"
POSTERIOR = "#289d93"
SHARED_RING = "#254d70"
VALVE_RING = "#b94453"
#: The drawn rings are darker than their legend swatches in the published panels.
RING_SHADE = 0.78
#: The published camera, as (rotation vector, pixels per mm, centre column, centre row), about the
#: reference annulus' centroid at phase 0, for a 345 x 255 panel.
CAMERA = (0.3233, 2.6568, -1.4172, 4.1585, 197.8096, 132.6513)
LIGHT_IN_VIEW = np.array([0.5, 0.5, 1.0])     # upper right, toward the viewer
SHADING = {"ambient": 0.6, "diffuse": 0.3, "specular": 0.1, "shininess": 20.0}
PANEL = (345, 255)
INSET = (56, 132, 85, 62)                     # column, row, width, height in the panel
DRAWN = (0, 2, 4, 6)                          # phases 0, 0.25, 0.5, 0.75 of E109's eight
#: Line widths in panel pixels; the separate valve annulus is drawn thinner, on top, so that where it
#: meets the chamber's annulus both stay visible.
SHARED_PX, SEPARATE_PX = 2.6, 1.7
DEPTH_TOLERANCE_MM = 1.2
ROWS = [  # key, label, leaflet geometry, chamber-side ring geometry, separate valve ring or None
    ("reference", "Reference", "truth", "truth", None),
    ("late_fusion", "Late fusion", "geometry_late_fusion_valve", "geometry_late_fusion_chamber",
     "geometry_late_fusion_valve"),
    ("lf_bcpd", "Late fusion\n+ BCPD", "geometry_lf_bcpd_valve", "geometry_lf_bcpd_chamber", "geometry_lf_bcpd_valve"),
    ("gpmm", "GPMM fusion", "geometry_gpmm", "geometry_gpmm", None),
    ("joint", "Rooted4D,\njoint fit", "geometry_rooted4d", "geometry_rooted4d", None),
    ("rooted4d", "Rooted4D", "geometry_rooted4d_field", "geometry_rooted4d_field", None),
]


def _rgb(hex_colour: str, shade: float = 1.0) -> np.ndarray:
    return np.array(matplotlib.colors.to_rgb(hex_colour)) * shade


# ----------------------------------------------------------------- renderer ----

class Camera:
    """Orthographic, about ``centre``; screen columns to the right, rows down, depth toward the viewer."""

    def __init__(self, centre: np.ndarray, supersample: int = 1):
        self.rotation = Rotation.from_rotvec(CAMERA[:3]).as_matrix()
        self.scale, self.column, self.row = (v * supersample for v in CAMERA[3:])
        self.centre = np.asarray(centre, dtype=np.float64)
        self.width, self.height = PANEL[0] * supersample, PANEL[1] * supersample
        self.supersample = supersample

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = (np.asarray(points) - self.centre) @ self.rotation.T
        return np.stack([self.column + self.scale * q[:, 0], self.row - self.scale * q[:, 1]], 1), q[:, 2]

    def toward_viewer(self) -> np.ndarray:
        return self.rotation.T @ np.array([0.0, 0.0, 1.0])


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, faces[:, k], normals)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def render(camera: Camera, meshes, lines) -> np.ndarray:
    """A z-buffer with smooth two-sided shading; ``lines`` are drawn where no surface hides them."""

    height, width = camera.height, camera.width
    depth_buffer = np.full((height, width), -np.inf)
    image = np.ones((height, width, 3))
    view = camera.toward_viewer()
    light = camera.rotation.T @ LIGHT_IN_VIEW
    light /= np.linalg.norm(light)
    half = (light + view) / np.linalg.norm(light + view)
    for vertices, faces, colour in meshes:
        projected, depth = camera.project(vertices)
        normals = _vertex_normals(vertices, faces)
        for triangle in faces:
            a, b, c = projected[triangle]
            x0, x1 = max(int(np.floor(min(a[0], b[0], c[0]))), 0), min(int(np.ceil(max(a[0], b[0], c[0]))), width - 1)
            y0, y1 = max(int(np.floor(min(a[1], b[1], c[1]))), 0), min(int(np.ceil(max(a[1], b[1], c[1]))), height - 1)
            area = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if x1 < x0 or y1 < y0 or abs(area) < 1e-12:
                continue
            ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            px, py = xs + 0.5, ys + 0.5
            w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / area
            w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / area
            weights = np.stack([w0, w1, 1.0 - w0 - w1], -1)
            inside = (weights >= -1e-9).all(-1)
            z = weights @ depth[triangle]
            inside &= z > depth_buffer[y0:y1 + 1, x0:x1 + 1]
            if not inside.any():
                continue
            yy, xx, w = ys[inside], xs[inside], weights[inside]
            normal = w @ normals[triangle]
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            normal *= np.where(normal @ view < 0, -1.0, 1.0)[:, None]
            diffuse = np.clip(normal @ light, 0.0, 1.0)
            specular = np.clip(normal @ half, 0.0, 1.0) ** SHADING["shininess"]
            depth_buffer[yy, xx] = z[inside]
            image[yy, xx] = np.clip(colour[None] * (SHADING["ambient"] + SHADING["diffuse"] * diffuse[:, None])
                                    + SHADING["specular"] * specular[:, None], 0.0, 1.0)
    for points, colour, line_px in lines:
        radius = line_px * camera.supersample / 2.0
        projected, depth = camera.project(np.vstack([points, points[:1]]))
        for i in range(len(projected) - 1):
            a, b = projected[i], projected[i + 1]
            segment = b - a
            x0, x1 = max(int(np.floor(min(a[0], b[0]) - radius - 1)), 0), min(int(np.ceil(max(a[0], b[0]) + radius + 1)), width - 1)
            y0, y1 = max(int(np.floor(min(a[1], b[1]) - radius - 1)), 0), min(int(np.ceil(max(a[1], b[1]) + radius + 1)), height - 1)
            if x1 < x0 or y1 < y0:
                continue
            ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            px, py = xs + 0.5, ys + 0.5
            t = np.clip(((px - a[0]) * segment[0] + (py - a[1]) * segment[1]) / max(segment @ segment, 1e-12), 0.0, 1.0)
            cover = np.clip(radius + 0.5 - np.hypot(px - a[0] - t * segment[0], py - a[1] - t * segment[1]), 0.0, 1.0)
            z = depth[i] + t * (depth[i + 1] - depth[i])
            visible = (cover > 0) & (z >= depth_buffer[ys, xs] - DEPTH_TOLERANCE_MM)
            yy, xx, alpha = ys[visible], xs[visible], cover[visible][:, None]
            image[yy, xx] = image[yy, xx] * (1.0 - alpha) + colour[None] * alpha
    return (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)


# ------------------------------------------------------------------ panels ----

def _arrays(sample: Path) -> dict:
    data = dict(np.load(sample / "data.npz", allow_pickle=True))
    with np.load(sample / "recent.npz") as recent:
        data.update({k: recent[k] for k in recent.files if k.startswith("geometry_")})
    if not np.allclose(data["phases"], np.linspace(0.0, 1.0, 8, endpoint=False)):
        raise SystemExit("unexpected phases")
    return data


def _separation(data: dict, ring_key: str, valve_key: str | None) -> np.ndarray:
    """Whole-annulus mean separation per drawn phase (the published labels' quantity)."""

    annulus = data["annulus"]
    if valve_key is None:
        return np.zeros(len(DRAWN))
    gap = np.linalg.norm(data[ring_key][:, annulus] - data[valve_key][:, annulus], axis=2).mean(axis=1)
    return gap[list(DRAWN)]


def draw_panels(data: dict, out: Path, supersample: int = 4) -> dict:
    annulus = data["annulus"]
    camera = Camera(data["truth"][0][annulus].mean(axis=0), supersample)
    anterior, posterior = _rgb(ANTERIOR), _rgb(POSTERIOR)
    shared, separate = _rgb(SHARED_RING, RING_SHADE), _rgb(VALVE_RING, RING_SHADE)
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    images, separations = {}, {}
    column, row, w, h = (v * supersample for v in INSET)
    for key, _, leaflets, ring, valve_ring in ROWS:
        separations[key] = _separation(data, ring, valve_ring).tolist()
        for phase in DRAWN:
            geometry = data[leaflets][phase]
            lines = [(data[ring][phase][annulus], shared, SHARED_PX)]
            if valve_ring is not None:
                lines.append((data[valve_ring][phase][annulus], separate, SEPARATE_PX))
            image = render(camera, [(geometry, data["anterior_faces"], anterior),
                                    (geometry, data["posterior_faces"], posterior)], lines)
            images[(key, phase)] = image
            stem = f"{key}_phase{data['phases'][phase]:.2f}"
            hires = Image.fromarray(image)
            hires.save(panels / f"{stem}_main_hires.png")
            hires.crop((column, row, column + w, row + h)).save(panels / f"{stem}_inset_hires.png")
            small = hires.resize(PANEL, Image.LANCZOS)
            small.save(panels / f"{stem}_main.png")
            small.crop(INSET[:2] + (INSET[0] + INSET[2], INSET[1] + INSET[3])).save(panels / f"{stem}_inset.png")
    return {"images": images, "separations": separations, "supersample": supersample}


def _check_reference(out: Path, published: Path | None, drawn: dict) -> dict:
    """Agreement of the redrawn reference and late-fusion panels with the published ones, if given."""

    if published is None:
        return {}
    report = {}
    for key, cells in (("reference", (15, 21, 27, 33)), ("late_fusion", (40, 47, 54, 61))):
        for phase, cell in zip(DRAWN, cells):
            ours = np.asarray(Image.fromarray(drawn["images"][(key, phase)]).resize(PANEL, Image.LANCZOS)).astype(int)
            theirs = np.asarray(Image.open(published / f"cell_{cell}.png").convert("RGB")).astype(int)
            a, b = ours.sum(2) < 700, theirs.sum(2) < 700
            report[f"{key}_phase{phase}"] = round(float((a & b).sum() / (a | b).sum()), 4)
    return report


# ------------------------------------------------------------------ figure ----

def compose(data: dict, drawn: dict, out: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    s = drawn["supersample"]
    # One crop for every panel: the union of drawn pixels, padded.
    mask = np.zeros(drawn["images"][(ROWS[0][0], 0)].shape[:2], dtype=bool)
    for image in drawn["images"].values():
        mask |= image.astype(int).sum(2) < 760
    ys, xs = np.nonzero(mask)
    pad = 4 * s
    top, bottom = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    left, right = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    crop_w, crop_h = right - left, bottom - top

    width = 7.0
    label_w, gap, margin = 0.80, 0.04, 0.06
    cell_w = (width - label_w - margin) / 4
    inset_w = 0.52
    main_w = cell_w - inset_w - 2 * gap
    main_h = main_w * crop_h / crop_w
    inset_h = inset_w * INSET[3] / INSET[2]
    row_h = max(main_h, inset_h) + 0.16
    header = 0.42
    height = header + row_h * len(ROWS) + 0.02
    fig = plt.figure(figsize=(width, height))

    def box(x, y, w, h):  # inches from the top left
        return [x / width, 1 - (y + h) / height, w / width, h / height]

    handles = [Line2D([], [], color=ANTERIOR, lw=4, label="Anterior leaflet"),
               Line2D([], [], color=POSTERIOR, lw=4, label="Posterior leaflet"),
               Line2D([], [], color=SHARED_RING, lw=2.2, label="Chamber / shared annulus"),
               Line2D([], [], color=VALVE_RING, lw=2.2, label="Separate valve annulus")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5 + 0.5 * label_w / width, 1.0), handlelength=1.6, columnspacing=1.6)
    for k, phase in enumerate(DRAWN):
        fig.text((label_w + (k + 0.5) * cell_w) / width, 1 - 0.32 / height, f"Phase {data['phases'][phase]:.2f}",
                 ha="center", va="center", fontsize=8)
    column, row, w, h = (v * s for v in INSET)
    for r, (key, label, *_rest, valve_ring) in enumerate(ROWS):
        y = header + r * row_h
        fig.text(0.02 / width, 1 - (y + main_h / 2) / height, label, ha="left", va="center", fontsize=8,
                 fontweight="bold", linespacing=1.05)
        edge = "#666666" if key == "reference" else ("#b85450" if valve_ring else "#82b366")
        for k, phase in enumerate(DRAWN):
            image = drawn["images"][(key, phase)]
            x = label_w + k * cell_w + gap
            main = fig.add_axes(box(x, y, main_w, main_h))
            main.imshow(Image.fromarray(image[top:bottom, left:right]).resize(
                (int(main_w * 600), int(main_h * 600)), Image.LANCZOS), interpolation="none")
            main.set_axis_off()
            scale = (int(main_w * 600)) / crop_w
            main.add_patch(Rectangle(((column - left) * scale, (row - top) * scale), w * scale, h * scale,
                                     fill=False, ec=edge, lw=0.7))
            inset = fig.add_axes(box(x + main_w + gap, y + (main_h - inset_h) / 2, inset_w, inset_h))
            inset.imshow(image[row:row + h, column:column + w], interpolation="none")
            inset.set_xticks([]); inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_edgecolor(edge); spine.set_linewidth(0.9)
            for corner_main, corner_inset in (((column - left + w) * scale, (row - top) * scale), (-0.5, -0.5)), \
                                             (((column - left + w) * scale, (row - top + h) * scale), (-0.5, h - 0.5)):
                inset.add_artist(ConnectionPatch(xyA=corner_main, coordsA=main.transData, xyB=corner_inset,
                                                 coordsB=inset.transData, color=edge, lw=0.5))
            if key != "reference":
                fig.text((label_w + (k + 0.5) * cell_w) / width, 1 - (y + main_h + 0.02) / height,
                         "Mean gap %.2f mm" % drawn["separations"][key][k], ha="center", va="top", fontsize=6.8)
    fig.savefig(out / "fig_qualitative_comparison_extended.pdf", dpi=600)
    fig.savefig(out / "fig_qualitative_comparison_extended.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("results/qualitative96/fusion/q75_sample-66fc7a9456bbb0f4"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--published", type=Path, default=None,
                        help="a folder of the published panels as cell_<id>.png, for the agreement check")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    data = _arrays(arguments.sample)
    drawn = draw_panels(data, arguments.out)
    compose(data, drawn, arguments.out)
    index = {"sample": str(arguments.sample), "camera": list(CAMERA), "inset": list(INSET),
             "phases": [float(data["phases"][p]) for p in DRAWN],
             "mean_separation_mm": drawn["separations"],
             "agreement_with_published": _check_reference(arguments.out, arguments.published, drawn)}
    (arguments.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
