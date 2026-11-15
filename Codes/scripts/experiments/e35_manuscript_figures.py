"""Generate the manuscript figures from stored result artifacts.

Every figure is drawn from a JSON or NPZ already under results/ or data/, so
nothing here recomputes a number that the paper reports. If a figure and the
text ever disagree, the figure is wrong and this script is where to look.

Titles are omitted deliberately: the captions live in LaTeX, and a title baked
into the PDF would duplicate them and drift.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401  (registers 3d)

OUT = Path("Manuscript/figures")
# The repaired ladder. E13_ladder_test_v3_m32.json is retained for the
# before-and-after comparison but must not feed a figure: its late-fusion
# rows come from the defective temporal lookup.
LADDER = Path("results/benchmark/E51_ladder_test_v3_m32_repaired.json")
# The trade figure plots the soft-attachment arm, whose published rows came
# from a detached model with renormalised coordinates (E60). E61 re-scored the
# corrected arm on the same test samples; its ladder is E51's rows with only the
# soft-attachment rows replaced, and the defective rows kept aside under
# joint_soft_attachment_historical, which this figure does not draw.
TRADE_LADDER = Path("results/benchmark/E61_ladder_test_v3_m32_soft_corrected.json")
CHAMBER = Path("results/benchmark/component_chamber_acdc_full.json")
SEGMENTATION = Path("results/21900728/E31_segmentation.json")
LANDMARKS = Path("results/21920159/E32_image_landmarks.json")
# The paired statistics of Table tab:paired, computed by e33_paired_ladder.py
# from TRADE_LADDER's rows with the composite identity as the analysis unit.
# The per-sample figure annotates these numbers rather than recomputing them.
PAIRED = Path("results/benchmark/E61_paired_ladder_v3_soft_corrected.json")
RESOLUTION = Path("results/benchmark/E22_annular_resolution.json")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "0.25",
    "axes.linewidth": 0.7,
    "xtick.color": "0.25",
    "ytick.color": "0.25",
    "axes.labelcolor": "0.1",
    "grid.color": "0.9",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

CHAMBER_C = "#3b6ea5"
ANTERIOR_C = "#c05746"
POSTERIOR_C = "#d99b3c"
ANNULUS_C = "#1a1a1a"
GREEN_C = "#5b8c5a"
GRID_KW = dict(color="0.92", lw=0.5)


def _panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=7.5, pad=4)


def sample_path() -> Path:
    # E100's small-session benchmark at 24 ring nodes, with the valve oriented to its chamber.
    root = Path("data/derived/e100-v3o/samples/test")
    return sorted(root.glob("*.npz"))[0]



def figure_template():
    """The heterogeneous template, with the annular vertices that all parts share.

    The surfaces are the reduced instance's own faces (``make_toy_complex``), so
    the chamber, the two leaflets and the ring are drawn as the fitter sees
    them; the vertices shown as markers are the physical degrees of freedom.
    """
    import sys

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sys.path.insert(0, "Codes")
    from rooted4d.geometry.toy import make_toy_complex

    data = np.load(sample_path())
    # A half-turn about x, a proper rotation, so that the apex is drawn below the
    # annulus as in the anatomical convention; nothing is mirrored.
    v = data["template_vertices"] * np.array([1.0, -1.0, -1.0])
    idx = {name: data["local_to_global__" + name]
           for name in ("chamber", "annulus", "anterior_leaflet", "posterior_leaflet")}
    annulus = idx["annulus"]
    faces = make_toy_complex(n_annulus=int(annulus.size)).faces
    if max(int(f.max()) for f in faces.values()) >= v.shape[0]:
        raise SystemExit("the toy complex's faces do not index this sample's vertices")
    shared_with_chamber = np.intersect1d(annulus, idx["chamber"]).size
    shared_with_anterior = np.intersect1d(annulus, idx["anterior_leaflet"]).size
    shared_with_posterior = np.intersect1d(annulus, idx["posterior_leaflet"]).size
    half = annulus.size // 2
    commissures = annulus[[0, half]]

    chamber_only = np.setdiff1d(idx["chamber"], annulus)
    ant_only = np.setdiff1d(idx["anterior_leaflet"], annulus)
    post_only = np.setdiff1d(idx["posterior_leaflet"], annulus)
    centre = v.mean(axis=0)
    span = np.abs(v - centre).max()

    fig = plt.figure(figsize=(6.6, 3.1))
    views = [("(a) oblique view", 22, -58), ("(b) along the annular axis, from the atrial side", 88, -58)]
    for panel, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        for name, color, alpha, lw in (("chamber", CHAMBER_C, 0.10, 0.2),
                                       ("anterior_leaflet", ANTERIOR_C, 0.30, 0.25),
                                       ("posterior_leaflet", POSTERIOR_C, 0.30, 0.25)):
            poly = Poly3DCollection(v[faces[name]], facecolor=color, alpha=alpha,
                                    edgecolor=color, linewidths=lw)
            ax.add_collection3d(poly)
        ax.scatter(*v[chamber_only].T, s=9, c=CHAMBER_C, depthshade=False, zorder=4)
        ax.scatter(*v[ant_only].T, s=9, c=ANTERIOR_C, depthshade=False, zorder=4)
        ax.scatter(*v[post_only].T, s=9, c=POSTERIOR_C, depthshade=False, zorder=4)
        ring = v[np.append(annulus, annulus[0])]
        ax.plot(*ring.T, color=ANNULUS_C, lw=1.7, zorder=5)
        ax.scatter(*v[annulus].T, s=24, facecolor="white", edgecolor=ANNULUS_C,
                   linewidths=1.0, zorder=6, depthshade=False)
        ax.scatter(*v[commissures].T, s=34, facecolor=ANNULUS_C, edgecolor="white",
                   linewidths=0.8, zorder=7, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(centre[0] - span, centre[0] + span)
        ax.set_ylim(centre[1] - span, centre[1] + span)
        ax.set_zlim(centre[2] - span, centre[2] + span)
        ax.set_box_aspect((1, 1, 1), zoom=1.55 if panel == 0 else 1.45)
        ax.set_axis_off()
        ax.text2D(0.0, 0.98, title, transform=ax.transAxes, fontsize=7.5, va="top", ha="left")
    handles = [Patch(facecolor=CHAMBER_C, alpha=0.35, edgecolor=CHAMBER_C, label="chamber boundary"),
               Patch(facecolor=ANTERIOR_C, alpha=0.5, edgecolor=ANTERIOR_C, label="anterior leaflet"),
               Patch(facecolor=POSTERIOR_C, alpha=0.5, edgecolor=POSTERIOR_C, label="posterior leaflet"),
               Line2D([], [], color=ANNULUS_C, lw=1.5, marker="o", mfc="white", mec=ANNULUS_C,
                      ms=5, label="annular ring: vertices shared by all three"),
               Line2D([], [], ls="none", marker="o", mfc=ANNULUS_C, mec="white", ms=5.5,
                      label="commissures (tagged ring vertices)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.8,
               handletextpad=0.5, columnspacing=1.4, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, bottom=0.12, top=1.0)
    fig.savefig(OUT / "fig_template.pdf")
    plt.close(fig)
    return {"annulus_nodes": int(annulus.size),
            "shared_with_chamber": int(shared_with_chamber),
            "shared_with_anterior": int(shared_with_anterior),
            "shared_with_posterior": int(shared_with_posterior),
            "physical_vertices": int(v.shape[0])}


def figure_trade_global():
    """Interface validity against per-modality prediction, one point per arm (earlier configuration)."""
    rows = json.loads(TRADE_LADDER.read_text())["rows"]
    by_arm: dict = {}
    for r in rows:
        by_arm.setdefault(r["baseline_id"], []).append(r)

    label = {
        "late_fusion": "Late fusion", "joint_soft_attachment": "Soft attachment",
        "joint_exact_no_gauges": "Exact, no gauges", "rooted4d_full": "Rooted4D",
    }
    marker = {"late_fusion": "s", "joint_soft_attachment": "D",
              "joint_exact_no_gauges": "o", "rooted4d_full": "*"}

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    floor = 1e-5  # so exact zeros are drawable on a log axis, marked in the caption
    # A legend rather than labels beside the markers: Rooted4D and exact
    # assembly without gauges sit 0.04 mm apart at the same residual, and soft
    # attachment now sits level with Rooted4D, so free-floating labels could not
    # be placed without appearing to belong to the wrong marker -- which is what
    # an earlier version of this figure did.
    for arm in ("rooted4d_full", "joint_exact_no_gauges", "joint_soft_attachment", "late_fusion"):
        rs = by_arm[arm]
        cmr = np.median([r["heldout_cmr_mm"] for r in rs])
        tee = np.median([r["heldout_tee_mm"] for r in rs])
        gap = np.median([r["attachment_residual_mm"] for r in rs])
        y = 0.5 * (cmr + tee)
        ax.scatter(max(gap, floor), y, s=110 if arm == "rooted4d_full" else 38,
                   marker=marker[arm], label=label[arm],
                   color=ANNULUS_C if arm == "rooted4d_full" else CHAMBER_C,
                   edgecolors="white", linewidths=0.6, zorder=5 if arm == "rooted4d_full" else 4,
                   clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.5, 60)
    ax.margins(y=0.18)
    ax.set_xlabel("interface residual at the annulus (mm)")
    ax.set_ylabel("mean held-out point error (mm)")
    ax.axvline(floor, color="0.7", lw=0.8, ls=":")
    ax.text(floor, ax.get_ylim()[1], " exact (0)", fontsize=6.5, color="0.45",
            va="top", ha="left")
    ax.legend(loc="center", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(OUT / "fig_trade_global.pdf")
    plt.close(fig)


def _bland_altman(ax, truth, pred, color):
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    mean = 0.5 * (truth + pred)
    diff = pred - truth
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    ax.axhspan(lo, hi, color=color, alpha=0.07, lw=0, zorder=0)
    ax.scatter(mean, diff, s=13, color=color, alpha=0.8, edgecolor="white", linewidths=0.3, zorder=3)
    ax.axhline(bias, color=ANNULUS_C, lw=1.0, zorder=2)
    ax.axhline(0.0, color="0.75", lw=0.6, zorder=1)
    for y, tag, dy, va in ((lo, "\u22121.96 SD", -2, "top"), (hi, "+1.96 SD", 2, "bottom")):
        ax.axhline(y, color=ANNULUS_C, lw=0.8, ls="--", zorder=2)
        ax.annotate(tag, (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, dy),
                    textcoords="offset points", fontsize=5.8, color="0.35", va=va)
    ax.annotate("bias", (0.01, bias), xycoords=("axes fraction", "data"), xytext=(0, 2),
                textcoords="offset points", fontsize=5.8, color="0.35", va="bottom")
    ax.grid(True, axis="y", **GRID_KW)
    ax.set_xlabel("mean ejection fraction (\\%)" if False else "mean ejection fraction (%)")
    return bias, lo, hi


def figure_bland_altman():
    """Ejection-fraction agreement given reference labels, and behind a segmenter."""
    chamber = json.loads(CHAMBER.read_text())["ejection_fraction"]["rows"]
    seg = json.loads(SEGMENTATION.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    b1, l1, h1 = _bland_altman(
        axes[0],
        [r["segmentation_ef_percent"] for r in chamber],
        [r["fitted_ef_percent"] for r in chamber],
        CHAMBER_C)
    b2, l2, h2 = _bland_altman(
        axes[1],
        [r["ef_truth"] for r in seg["ef_rows"]],
        [r["ef_pred"] for r in seg["ef_rows"]],
        ANTERIOR_C)

    axes[0].set_ylabel("fitted $-$ reference (points)")
    _panel_title(axes[0], "(a) reference labels, $n=%d$ patients" % len(chamber))
    _panel_title(axes[1], "(b) behind a %.3f-Dice segmenter, $n=%d$ patients"
                 % (seg["test_dice"]["lv_cavity"], len(seg["ef_rows"])))
    for ax, (b, lo, hi) in zip(axes, [(b1, l1, h1), (b2, l2, h2)]):
        ax.text(0.98, 0.03, "bias %+.2f\nLoA [%.2f, %+.2f]" % (b, lo, hi),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                color="0.15", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", lw=0.5))
    for ax in axes:
        ax.margins(y=0.12)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(OUT / "fig_bland_altman.pdf")
    plt.close(fig)
    return {"reference": [b1, l1, h1], "segmented": [b2, l2, h2]}


def figure_landmarks_global():
    """Rotation error against the number of identified annular landmarks (earlier configuration)."""
    data = json.loads(LANDMARKS.read_text())
    counts, values = [], []
    for key, stat in sorted(data["by_landmark_count"].items(), key=lambda kv: int(kv[0])):
        counts.append(int(key))
        values.append(float(stat["rotation_deg"]))
    reference = data["point_reference"]["point_rotation_deg"]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(counts, values, "-o", color=ANNULUS_C, lw=1.2, ms=5, zorder=4)
    ax.axhline(reference, color=CHAMBER_C, lw=1.0, ls="--")
    ax.text(8, reference, "full point correspondence ", color=CHAMBER_C,
            fontsize=6.5, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(counts)
    ax.set_xlabel("identified annular landmarks")
    ax.set_ylabel("rotation error (deg)")
    ax.annotate("%.1f$^\\circ$" % values[0], (counts[0], values[0]),
                textcoords="offset points", xytext=(6, -2), fontsize=7)
    ax.annotate("%.2f$^\\circ$" % values[1], (counts[1], values[1]),
                textcoords="offset points", xytext=(4, 6), fontsize=7)
    fig.savefig(OUT / "fig_landmarks_global.pdf")
    plt.close(fig)
    return {"counts": counts, "rotation_deg": values, "point_reference": reference}


# E100, the rebuild (oriented valves, 96 ring nodes), is the manuscript's evaluation
# since 2 October: each primary's chain test rows, which carry trajectory late fusion
# as an arm, and the small-session primary's E70 and E74 rows.
E100 = Path("results/benchmark/E100")
E100_ROWS = E100 / "rows"
E100_MAIN = E100 / "E100_main.json"
PRIMARY = "v4n96o"
# Since 13 November the image studies are E120's rerun with the session-gauge fix
# (Docs/E120_RESULT_2026-11-13.md); E100's own E70 and E74 rows predate it.
E120 = Path("results/benchmark/E120")
E70_ROWS = E120 / "rows/v3n96o/components/E70_v3n96o_test"
E70_ANALYSIS = E120 / "E120_studies.json"
E74_ROWS = E120 / "rows/v3n96o/components/E74_v3n96o_test"
E74_ANALYSIS = E120 / "E120_E74.json"


def figure_landmarks():
    """E70 and E74: rotation error per test sample under image evidence, small-session primary (E100).

    One row per arm, one point per sample, the median as a bar and printed at
    the right. Zero-start arms (E74, and E70's image-only arm) sit above the
    warm-started ones (E70, and E74's commissure arm). Medians are checked
    against the pre-registered analyses rather than recomputed.
    """
    def rows_of(directory):
        rows = {}
        for shard in sorted(directory.glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    rows[record["sample"]] = record["variants"]
        return rows

    e70, e74 = rows_of(E70_ROWS), rows_of(E74_ROWS)
    medians = {**{("e70", k): v["rotation_deg"] for k, v in json.loads(E70_ANALYSIS.read_text())["E70"]["medians"].items()},
               **{("e74", k): v["rotation_deg"] for k, v in json.loads(E74_ANALYSIS.read_text())["medians"].items()}}
    arms = [("e70", "point_fit", "point fit", CHAMBER_C),
            ("e74", "image_cold", "image", ANTERIOR_C),
            ("e74", "landmarks_2_cold", "+ 2 neighboring", "0.35"),
            ("e74", "commissures_cold", "+ 2 commissures", "0.35"),
            ("e74", "landmarks_4_cold", "+ 4 landmarks", "0.35"),
            ("e74", "landmarks_8_cold", "+ 8 landmarks", "0.35"),
            ("e70", "image_warm", "image", "0.35"),
            ("e74", "commissures_warm", "+ 2 commissures", "0.35")]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.95))
    ax.axvspan(0.05, 5.0, color="#eef3f8", lw=0, zorder=0)
    ax.axvline(5.0, color=CHAMBER_C, lw=0.6, ls=":", zorder=1)
    ax.text(4.6, -0.95, "within 5\u00b0", fontsize=6, color=CHAMBER_C, ha="right", va="center")
    ax.axvline(90.0, color="0.6", lw=0.6, ls=":", zorder=1)
    ax.text(85, -0.95, "beyond 90\u00b0", fontsize=6, color="0.45", ha="right", va="center")
    out = {}
    for y, (run, arm, _, colour) in enumerate(arms):
        source = e70 if run == "e70" else e74
        values = np.array([variants[arm]["rotation_deg"] for variants in source.values()])
        assert len(values) == 60 and np.isclose(np.median(values), medians[(run, arm)])
        ax.scatter(values, y + rng.uniform(-0.22, 0.22, values.size), s=7, color=colour,
                   alpha=0.6, linewidths=0, zorder=3)
        m = float(np.median(values))
        ax.plot([m, m], [y - 0.34, y + 0.34], color=ANNULUS_C, lw=1.6, zorder=4)
        ax.text(1.02, y, ("%.0f\u00b0" if m >= 10 else "%.2f\u00b0") % m, fontsize=6.5,
                va="center", ha="left", transform=ax.get_yaxis_transform(),
                fontweight="bold" if arm in ("image_cold", "commissures_cold") else "normal")
        out[f"{run}/{arm}"] = {"median": m, "below_5deg": int((values < 5.0).sum())}
    ax.text(1.02, -0.95, "median", fontsize=6.5, color="0.45", va="center", ha="left",
            transform=ax.get_yaxis_transform())
    for y, label in ((0.5, "zero start"), (5.5, "warm start")):
        ax.axhline(y, color="0.8", lw=0.6, zorder=1)
        ax.text(0.055, y + 0.12, label, fontsize=6.0, color="0.45", va="top", ha="left",
                fontstyle="italic")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 300)
    ax.set_ylim(len(arms) - 0.5, -1.2)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([label for _, _, label, _ in arms])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("rotation error (deg)")
    fig.savefig(OUT / "fig_landmarks.pdf")
    plt.close(fig)
    return out


def figure_fusion_pairs_global():
    """Rooted4D against late fusion, one point per test sample, per endpoint (earlier configuration).

    The table reports medians of paired differences; this shows the samples
    behind them, so a reader can see whether a median difference is carried by
    most samples or by a few. The annotations are the table's own statistics,
    read from PAIRED, and the counts are computed from the same rows.
    """
    rows = json.loads(TRADE_LADDER.read_text())["rows"]
    paired = json.loads(PAIRED.read_text())["comparisons"]["late_fusion"]
    by_arm: dict = {}
    for r in rows:
        by_arm.setdefault(r["baseline_id"], {})[r["sample"]] = r
    samples = sorted(by_arm["rooted4d_full"])
    if sorted(by_arm["late_fusion"]) != samples:
        raise SystemExit("late fusion and Rooted4D rows cover different samples")

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    panels = [("heldout_cmr_mm", "held-out CMR (mm)", "mm", (1, 2, 4)),
              ("heldout_tee_mm", "held-out TEE (mm)", "mm", (2, 4, 8)),
              ("rotation_deg", "rotation (deg)", "deg", (0.2, 1, 5)),
              ("translation_mm", "translation (mm)", "mm", (1, 3, 10, 30))]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.05))
    counts = {}
    for ax, (key, label, unit, ticks), letter in zip(axes, panels, "abcd"):
        x = np.array([by_arm["late_fusion"][s][key] for s in samples], float)
        y = np.array([by_arm["rooted4d_full"][s][key] for s in samples], float)
        # Recomputed only as a guard: the annotation must be the table's number.
        if abs(float(np.median(y - x)) - paired[key]["difference"]) > 1e-9:
            raise SystemExit("paired median for %s disagrees with %s" % (key, PAIRED))
        low = min(x.min(), y.min()) / 1.3
        high = max(x.max(), y.max()) * 1.3
        ax.plot([low, high], [low, high], color="0.6", lw=0.8, ls="--", zorder=1)
        better = y < x
        ax.scatter(x[~better], y[~better], s=8, color=ANTERIOR_C, edgecolor="none",
                   alpha=0.85, zorder=3, label="late fusion lower")
        ax.scatter(x[better], y[better], s=8, color=CHAMBER_C, edgecolor="none",
                   alpha=0.85, zorder=3, label="Rooted4D lower")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
            axis.set_minor_locator(NullLocator())
        ax.set_xlabel("late fusion")
        if letter == "a":
            ax.set_ylabel("Rooted4D")
        ax.set_title("(%s) %s" % (letter, label), loc="left", fontsize=7)
        stat = paired[key]
        # A typographic minus, as in the text; "%+.2f" writes a hyphen.
        ax.text(0.04, 0.97, ("$\\Delta$ %+.2f %s\n[%+.2f, %+.2f]"
                             % (stat["difference"], unit, stat["ci_low"], stat["ci_high"])
                             ).replace("-", "\u2212"),
                transform=ax.transAxes, ha="left", va="top", fontsize=6, color="0.2")
        ax.text(0.97, 0.03, "%d/%d lower" % (int(better.sum()), len(samples)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=CHAMBER_C)
        counts[key] = {"rooted4d_lower": int(better.sum()), "samples": len(samples)}
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.2,
               fontsize=6.5, markerscale=1.3)
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(OUT / "fig_fusion_pairs_global.pdf")
    plt.close(fig)
    return counts


# E64: the frozen structured configuration on test. E75: the second test
# evaluation, at the session weights E73 selected on validation, which changed
# the configuration on v4 and v3 at 48 nodes; on v3 E64 stands.
E64 = Path("results/benchmark/E64")
E64_REPORT = Path("results/benchmark/E64_test.json")
E75 = Path("results/benchmark/E75")
E75_REPORT = Path("results/benchmark/E75_test.json")
# E87: late fusion with trajectory registration, the paper's late fusion. Its
# fits are the feature-registered arm's, so its held-out errors are the same.
E87 = Path("results/benchmark/E87")
TRAJECTORY = "S/late_fusion_trajectory"
STAGE = Path("Codes/scripts/experiments/e62_stage_b.py")


def main_evaluation(bench):
    """Rows and report of the evaluation the manuscript reports for a benchmark."""
    if bench in ("v4", "v3n48"):
        return E75 / ("%s_test" % bench), E75_REPORT
    return E64 / ("%s_test" % bench), E64_REPORT


def _stage():
    """The E62/E64 stage module, so every statistic here is computed by its code."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("e62_stage_b", STAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_difference(stage, rows, first, second, key):
    """E62's statistic for any endpoint: difference of identity medians, 95% interval.

    The resampling reproduces ``paired`` exactly -- same seed, same draw order --
    so for CMR and TEE it returns the numbers the E64 report holds, which the
    caller asserts.
    """
    ids_a, a = stage.per_identity(rows, first, key)
    ids_b, b = stage.per_identity(rows, second, key)
    if ids_a != ids_b:
        raise SystemExit("identities differ between %s and %s" % (first, second))
    rng = np.random.default_rng(stage.SEED)
    draws = np.empty(stage.RESAMPLES)
    for r in range(stage.RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return (float(np.nanmedian(a) - np.nanmedian(b)),
            float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _nice_ticks(low, high):
    candidates = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    inside = [c for c in candidates if low <= c <= high]
    return inside if len(inside) >= 2 else [round(low, 2), round(high, 2)]


def figure_fusion_pairs(bench=PRIMARY):
    """Rooted4D (structured) against late fusion, one point per test sample.

    The annotation is E62's statistic -- the difference of identity medians with
    its 95% bootstrap interval -- computed by the stage module's own code; for
    CMR and TEE it must equal the E64 report, and the script refuses otherwise.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    stage = _stage()
    rows = stage.load_rows([E100_ROWS / ("%s_test" % bench)])
    report_path = E100_MAIN
    report = json.loads(report_path.read_text())["instances"][bench]["T3_fusion_cost"]["comparison"]
    first, second = "S/joint", TRAJECTORY
    panels = [("heldout_cmr_mm", "held-out CMR (mm)", "mm", "cmr"),
              ("heldout_tee_mm", "held-out TEE (mm)", "mm", "tee"),
              ("rotation_deg", "rotation (deg)", "deg", None),
              ("translation_mm", "translation (mm)", "mm", None)]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.15))
    info = {}
    for ax, (key, label, unit, check), letter in zip(axes, panels, "abcd"):
        ax.grid(True, **GRID_KW)
        x = np.array([r["variants"][second][key] for r in rows], float)
        y = np.array([r["variants"][first][key] for r in rows], float)
        observed, lower, upper = _identity_difference(stage, rows, first, second, key)
        if check is not None:
            expected = report[check]
            if max(abs(observed - expected["observed"]), abs(lower - expected["lower"]),
                   abs(upper - expected["upper"])) > 1e-12:
                raise SystemExit("%s interval disagrees with %s" % (key, report_path))
        low, high = min(x.min(), y.min()) / 1.3, max(x.max(), y.max()) * 1.3
        ax.plot([low, high], [low, high], color="0.6", lw=0.8, ls="--", zorder=1)
        better = y < x
        ax.scatter(x[~better], y[~better], s=13, color=ANTERIOR_C, edgecolor="white",
                   linewidths=0.3, alpha=0.9, zorder=3, label="late fusion lower")
        ax.scatter(x[better], y[better], s=13, color=CHAMBER_C, edgecolor="white",
                   linewidths=0.3, alpha=0.9, zorder=3, label="Rooted4D lower")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ticks = _nice_ticks(low, high)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
            axis.set_minor_locator(NullLocator())
        ax.set_xlabel("late fusion")
        if letter == "a":
            ax.set_ylabel("Rooted4D")
        _panel_title(ax, "(%s) %s" % (letter, label))
        ax.text(0.05, 0.96, ("$\\Delta$ %+.2f %s\n[%+.2f, %+.2f]" % (observed, unit, lower, upper)
                             ).replace("-", "\u2212"),
                transform=ax.transAxes, ha="left", va="top", fontsize=6, color="0.15",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", lw=0.5))
        ax.text(0.96, 0.04, "%d/%d lower" % (int(better.sum()), len(rows)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=CHAMBER_C,
                fontweight="bold")
        info[key] = {"difference": observed, "interval": [lower, upper],
                     "rooted4d_lower": int(better.sum()), "samples": len(rows)}
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.2,
               fontsize=6.5, markerscale=1.3)
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(OUT / "fig_fusion_pairs.pdf")
    plt.close(fig)
    return info



def figure_trade(bench=PRIMARY):
    """Interface residual against mean held-out error, one point per arm (the main evaluation, test).

    The x axis is symmetric-log so that exactly assembled arms sit at a true zero,
    in a shaded "exact" band, rather than at a drawn floor. Every point is labeled
    in place; there is no legend to read across.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    stage = _stage()
    rows = stage.load_rows([E100_ROWS / ("%s_test" % bench)])
    # (arm, label, marker, color, size, label offset in points, text alignment)
    arms = [("S/joint", "Rooted4D", "*", ANNULUS_C, 200, (12, 30), "left"),
            ("S/exact_no_gauges", "exact, no gauges", "o", CHAMBER_C, 34, None, None),
            ("S/penalty_repair", "penalty repair", "P", CHAMBER_C, 40, None, None),
            ("S/soft_matched_1e4", "soft attachment, $\\lambda=10^4$", "D", CHAMBER_C, 30, (0, -13), "center"),
            ("S/soft_matched", "soft attachment, $\\lambda=1$ (selected)", "d", POSTERIOR_C, 46, (0, 12), "center"),
            ("S/exact_repair", "exact repair", "X", ANTERIOR_C, 44, (8, 0), "left"),
            (TRAJECTORY, "late fusion", "s", ANTERIOR_C, 40, (8, 0), "left"),
            ("G/joint", "Rooted4D, global basis", "*", "0.55", 110, (8, 0), "left")]
    linthresh = 1e-5
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.45)
    ax.grid(True, axis="y", **GRID_KW)
    ax.axvspan(-linthresh, linthresh, color="0.93", lw=0, zorder=0)
    info = {}
    for name, label, marker, color, size, offset, align in arms:
        med = stage.sample_medians(rows, name)
        y = 0.5 * (med["heldout_cmr_mm"] + med["heldout_tee_mm"])
        gap = med["attachment_residual_mm"]
        # Penalty repair and exact assembly without gauges land on Rooted4D
        # (Table tab:paired: within 0.004 in the sum); they are drawn over the
        # star and named in its label rather than given labels of their own.
        ax.scatter(gap, y, s=size, marker=marker, color=color,
                   edgecolors="white", linewidths=0.5,
                   zorder={"S/penalty_repair": 6, "S/joint": 5}.get(name, 4), clip_on=False)
        if offset is not None:
            text = label
            if name == "S/joint":
                text = "Rooted4D\n(exact, no gauges and penalty\nrepair coincide with it)"
            ax.annotate(text, (gap, y), xytext=offset, textcoords="offset points",
                        fontsize=6.3, ha=align, va="center", color="0.1",
                        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5, shrinkA=0, shrinkB=3)
                        if name in ("S/joint",) else None)
        info[name] = {"gap": gap, "mean_error": y}
    ax.set_xlim(-0.6 * linthresh, 60)
    ax.set_ylim(1.30, 2.0)
    ax.xaxis.set_major_locator(FixedLocator([0, 1e-4, 1e-2, 1, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: "0" if x == 0 else ("1" if x == 1 else "10" if x == 10 else "$10^{%d}$" % round(np.log10(x)))))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("interface residual at the annulus (mm)")
    ax.set_ylabel("mean held-out point error (mm)")
    ax.text(0, 1.76, "exact attachment", fontsize=6, color="0.4", ha="center", va="center", rotation=90)
    fig.savefig(OUT / "fig_trade.pdf")
    plt.close(fig)
    return info


def figure_resolution():
    """Annular representation error against ring density, with the fixed rule."""
    data = json.loads(RESOLUTION.read_text())
    rule = data["decision_rule"]
    nodes = [int(n) for n in data["candidates"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for side, color, name in (("chamber", CHAMBER_C, "chamber ring"), ("valve", ANTERIOR_C, "valve ring")):
        median = [data[side][str(n)]["p95_error_mm"]["median"] for n in nodes]
        upper = [data[side][str(n)]["p95_error_mm"]["p95"] for n in nodes]
        axes[0].plot(nodes, median, "-o", color=color, ms=4, lw=1.3, label=name, mec="white", mew=0.5)
        axes[0].fill_between(nodes, median, upper, color=color, alpha=0.12, lw=0)
    saddle = [100.0 * data["valve"][str(n)]["saddle_error_fraction"]["median"] for n in nodes]
    saddle_hi = [100.0 * data["valve"][str(n)]["saddle_error_fraction"]["p95"] for n in nodes]
    axes[1].plot(nodes, saddle, "-o", color=ANTERIOR_C, ms=4, lw=1.3, label="valve ring", mec="white", mew=0.5)
    axes[1].fill_between(nodes, saddle, saddle_hi, color=ANTERIOR_C, alpha=0.12, lw=0)
    thresholds = (rule["p95_representation_error_mm_below"], 100.0 * rule["median_saddle_error_fraction_below"])
    for ax, threshold, tag, x, ha in zip(axes, thresholds,
                                         ("rule: p95 below %.2f mm", "rule: median below %.0f%%"),
                                         (16.5, 16.5), ("left", "left")):
        ax.axhline(threshold, color=ANNULUS_C, lw=0.8, ls="--")
        ax.axhspan(0, threshold, color=GREEN_C, alpha=0.07, lw=0)
        ax.text(x, threshold * 0.95, tag % threshold, fontsize=6, color="0.3", va="top", ha=ha)
        ax.grid(True, axis="y", **GRID_KW)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(nodes)
        ax.set_xticklabels([str(n) for n in nodes])
        ax.minorticks_off()
        ax.set_xlabel("annular ring nodes")
        ax.axvline(24, color="0.6", lw=0.8, ls=":")
        ax.axvline(data["selected_nodes"], color="0.6", lw=0.8, ls=":")
        top = ax.get_ylim()[1]
        ax.text(24, top, " benchmark", fontsize=6, color="0.45", va="top", ha="left")
        ax.text(data["selected_nodes"], top, "selected ", fontsize=6, color="0.45", va="top", ha="right")
    axes[0].set_ylabel("p95 ring error (mm)")
    _panel_title(axes[0], "(a) representation error of the ring")
    axes[1].set_ylabel("saddle-amplitude error (%)")
    _panel_title(axes[1], "(b) valve saddle")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=False, handletextpad=0.3,
                   title="line: median; band: to the 95th percentile over cases", title_fontsize=6)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(OUT / "fig_resolution.pdf")
    plt.close(fig)
    return {side: {n: data[side][str(n)]["p95_error_mm"]["median"] for n in nodes}
            for side in ("chamber", "valve")}


DRIFT = Path("results/benchmark/E63")


def _drift_rows():
    rows = [json.loads(line) for shard in sorted(DRIFT.glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("no E63 rows under %s" % DRIFT)
    return rows


def _penalties(rows, family):
    """(penalty, variant name) pairs, by the names the spec actually wrote."""
    names = [n for n in rows[0]["variants"] if n.startswith(family + "/l=")]
    return sorted((float(n.split("l=")[1]), n) for n in names)


def figure_drift():
    """Interface residual and accuracy gap of soft attachment against its penalty.

    E63 fits the published configuration on the v3 validation split, so it
    measures Eq. (soft-drift) on the reconstruction model rather than on T2's
    synthetic quadratic. Panel (a) is the interface residual itself; panel (b) is
    how far each soft arm's held-out error sits from the exact arm's on the same
    sample, which is where the renormalised (historical) arm differs.
    """
    rows = _drift_rows()
    if any(not v.get("succeeded") for r in rows for v in r["variants"].values()):
        raise SystemExit("E63 has failed fits; the figure would silently drop them")
    exact = {r["sample"]: r["variants"]["rooted4d_full"] for r in rows}
    exact_gap = max(v["attachment_residual_mm"] for v in exact.values())

    def series(family, key):
        values = []
        for lam, name in _penalties(rows, family):
            if key == "gap":
                x = [r["variants"][name]["attachment_residual_mm"] for r in rows]
            else:
                x = [abs(r["variants"][name][key] - exact[r["sample"]][key]) for r in rows]
            values.append((lam, np.median(x), np.percentile(x, 10), np.percentile(x, 90)))
        return np.array(values)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45))
    ax = axes[0]
    matched, historical = series("matched", "gap"), series("historical", "gap")
    ax.fill_between(matched[:, 0], matched[:, 2], matched[:, 3], color=CHAMBER_C, alpha=0.15, lw=0)
    ax.plot(matched[:, 0], matched[:, 1], "-o", color=CHAMBER_C, ms=3.5, lw=1.1,
            label="soft attachment, matched")
    ax.plot(historical[:, 0], historical[:, 1], "s", color=ANTERIOR_C, ms=4, mfc="none", mew=1.0,
            label="soft attachment, renormalized")
    anchor = matched[matched[:, 0] == 1e4][0]
    # Offset by a decade: the data follow the reference so closely that a line
    # through them would be invisible.
    grid = np.array([3e2, 1e8])
    ax.plot(grid, 10.0 * anchor[1] * 1e4 / grid, color="0.55", lw=0.8, ls="--", zorder=1)
    ax.text(2e5, 10.0 * anchor[1] * 1e4 / 2e5 * 2.2, "slope $-1$", fontsize=6.5, color="0.4",
            rotation=-27)
    floor = 1e-10
    ax.axhline(floor, color=ANNULUS_C, lw=1.0)
    ax.text(1.2e0, floor * 2.2, "Rooted4D: exactly 0 at every $\\lambda$ (drawn at $10^{-10}$)",
            fontsize=6.5, color=ANNULUS_C, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor / 4, 8)
    ax.set_xlabel("penalty weight $\\lambda$")
    ax.set_ylabel("interface residual (mm)")
    _panel_title(ax, "(a) attachment violation")
    ax.grid(True, **GRID_KW)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.3, fontsize=6.5)

    ax = axes[1]
    for key, style, label in (("heldout_cmr_mm", "-", "CMR"), ("heldout_tee_mm", ":", "TEE")):
        m, h = series("matched", key), series("historical", key)
        ax.plot(m[:, 0], m[:, 1], style, marker="o", color=CHAMBER_C, ms=3.2, lw=1.1,
                label="matched, %s" % label)
        ax.plot(h[:, 0], h[:, 1], style, marker="s", color=ANTERIOR_C, ms=3.6, lw=1.0, mfc="none",
                mew=1.0, label="renormalized, %s" % label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("penalty weight $\\lambda$")
    ax.set_ylabel("$|$held-out error $-$ Rooted4D$|$ (mm)")
    _panel_title(ax, "(b) accuracy gap to exact assembly")
    ax.grid(True, **GRID_KW)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.3, fontsize=6.2)
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(OUT / "fig_drift.pdf")
    plt.close(fig)

    # Per-sample log-log slope of the matched arm over the asymptotic range.
    asymptotic = [(l, n) for l, n in _penalties(rows, "matched") if l >= 1e2]
    lams = np.array([l for l, _ in asymptotic])
    slopes = []
    for r in rows:
        gaps = np.array([r["variants"][n]["attachment_residual_mm"] for _, n in asymptotic])
        slopes.append(np.polyfit(np.log10(lams), np.log10(gaps), 1)[0])
    return {"samples": len(rows), "exact_max_gap_mm": exact_gap,
            "matched_gap_median_at_1e4": float(anchor[1]),
            "slope_median": float(np.median(slopes)), "slope_range": [float(min(slopes)), float(max(slopes))],
            "historical_cmr_gap_median": series("historical", "heldout_cmr_mm")[:, 1].tolist(),
            "historical_tee_gap_median": series("historical", "heldout_tee_mm")[:, 1].tolist(),
            "matched_cmr_gap_median": series("matched", "heldout_cmr_mm")[:, 1].tolist()}


FIGURES = {
    "template": figure_template,
    "drift": figure_drift,
    "trade": figure_trade,
    "trade_global": figure_trade_global,
    "fusion_pairs_global": figure_fusion_pairs_global,
    "bland_altman": figure_bland_altman,
    "landmarks": figure_landmarks,
    "landmarks_global": figure_landmarks_global,
    "fusion_pairs": figure_fusion_pairs,
    "resolution": figure_resolution,
}


def main(argv: list[str]) -> int:
    """Draw the named figures, or all of them.

    Naming figures matters on a manuscript synced by hand: redrawing an
    unchanged figure rewrites its PDF, which then looks changed to whoever
    uploads it.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    names = argv or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit("unknown figure(s): %s; known: %s" % (unknown, list(FIGURES)))
    info = {name: FIGURES[name]() for name in names}
    print(json.dumps(info, indent=2, default=str))
    for f in sorted(OUT.glob("*.pdf")):
        print("  %-28s %6.1f kB" % (f.name, f.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
