"""Write the manuscript's tables from E100's reports (``e100_components.py analyze``).

Nothing is computed afresh: every number is read from ``results/benchmark/E100/``
and printed as LaTeX rows in the formats of ``e64_manuscript_tables.py``, whose
helpers it reuses. The tables in the manuscript are pasted from this output, so a
table and its report cannot drift apart unnoticed: rerun this and diff.

    python e100_manuscript_tables.py [<reports dir>] > tables.tex
    python e100_manuscript_tables.py e120 [<E120 dir>] > tables.tex   # image-domain tables, session-gauge rerun
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e64_manuscript_tables", HERE / "e64_manuscript_tables.py")
E64 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E64)
number, gap, signed, interval = E64.number, E64.gap, E64.signed, E64.interval

PRIMARY = (("v4n96o", "Realistic session (primary)"), ("v3n96o", "Small session"))
ROBUSTNESS = (("v4n48o", "Realistic session, 48 nodes"), ("v4oriented", "Realistic session, 24 nodes"),
              ("v3n48o", "Small session, 48 nodes"), ("v3o", "Small session, 24 nodes"))
#: Row labels in the alignment and representation tables.
LABELS = dict(PRIMARY + ROBUSTNESS)


def load(directory: Path, name: str) -> dict | None:
    path = directory / f"E100_{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main_table(instance: dict) -> str:
    """Every method on one instance's test split: medians over samples (E64's rows and format)."""

    lines = []
    for name, label in E64.ROWS:
        m = instance["medians"][name]
        fail = 100.0 * m["failures"] / max(m["samples"], 1)
        lines.append("%s & %s & %s & %s & %s & %s & %s & %.0f\\\\" % (
            label, number(m["heldout_cmr_mm"]), number(m["heldout_tee_mm"]), number(m["rotation_deg"]),
            number(m["translation_mm"]), number(m["phase_map_cycles"], 3), gap(m["attachment_residual_mm"]), fail))
    return "\n".join(lines)


def paired_table(instance: dict) -> str:
    lines = []
    for _, label, path in E64.COMPARATORS:
        entry = instance[path[0]][path[1]]
        lines.append("%s & %s & %s & %s\\\\" % (label, interval(entry["sum"]), interval(entry["cmr"]),
                                                interval(entry["tee"])))
    return "\n".join(lines)


def alignment_table(instances: dict, keys) -> str:
    """Rooted4D minus trajectory late fusion: rotation, translation, phase map, interface gap."""

    lines = []
    for key in keys:
        if "rooted4d_minus_trajectory" not in instances.get(key, {}):
            continue
        entry = instances[key]["rooted4d_minus_trajectory"]
        lines.append("%s & %s & %s & %s & %s\\\\" % (
            LABELS[key], interval(entry["rotation_deg"], 2), interval(entry["translation_mm"], 2),
            interval(entry["phase_map_cycles"], 3), interval(entry["attachment_residual_mm"], 2)))
    return "\n".join(lines)


def representation_table(gate: dict, instances: dict, keys) -> str:
    """Structured minus global: cross-validation on training, and test."""

    lines = []
    for key in keys:
        cv = gate["instances"].get(key, {}).get("representation")
        test = instances.get(key, {}).get("T1_representation", {}).get("comparison")
        if cv is None or test is None:
            continue
        lines.append("%s & %s & %s & %s & %s & %s & %s\\\\" % (
            LABELS[key], interval(cv["sum"]), signed(cv["cmr"]["observed"]), signed(cv["tee"]["observed"]),
            interval(test["sum"]), signed(test["cmr"]["observed"]), signed(test["tee"]["observed"])))
    return "\n".join(lines)


def temporal_table(temporal: dict) -> str:
    """E66's motion and warp variants (small session) and E67's session variants (realistic session)."""

    def row(label, m):
        return "%s & %.3f & %.3f & %.3f & %.3f & %.4f\\\\" % (
            label, m["heldout_cmr_mm"], m["heldout_tee_mm"], m["rotation_deg"], m["translation_mm"],
            m["phase_map_cycles"])

    lines = []
    if "E66" in temporal:
        m = temporal["E66"]["medians"]
        lines += [r"\multicolumn{6}{@{}l}{\textit{Small session: motion and warp}}\\",
                  row("Periodic", m["S/periodic"]), row("Per-frame", m["S/per_frame"]),
                  row("Warp frozen", m["S/fixed_warp"])]
    if "E67" in temporal:
        m = temporal["E67"]["medians"]
        lines += [r"\midrule", r"\multicolumn{6}{@{}l}{\textit{Realistic session: session term}}\\",
                  row("Constant + oscillation", m["S/frozen"]), row("Oscillation only", m["S/oscillatory"]),
                  row("None", m["S/no_session"])]
    return "\n".join(lines)


def scarcity_table(temporal: dict) -> str:
    """E68: the selected session weight against the comparison arm, under thinned evidence."""

    entry = temporal.get("E68")
    if entry is None:
        return ""
    weak = next(name for name in entry["arms"] if name != "frozen")
    labels = {"full": "All", "half": "Half", "quarter": "Quarter", "eighth": "Eighth",
              "eighth_noisy": "Eighth, noise"}
    lines = []
    for level, label in labels.items():
        block = entry["levels"][level]
        medians = block["medians"]
        frozen, other = medians[f"{level}/frozen"], medians[f"{level}/{weak}"]
        visible = block["visible"]
        lines.append("%s (%d / %d) & %.2f / %.2f & %.2f / %.2f & %s\\\\" % (
            label, round(visible["visible_cmr_frames"]), round(visible["visible_tee_vertices"]),
            frozen["heldout_cmr_mm"], frozen["heldout_tee_mm"], other["heldout_cmr_mm"], other["heldout_tee_mm"],
            interval(block[f"frozen_minus_{weak}"]["sum"])))
    return "\n".join(lines)


def _rotation(value: float) -> str:
    return "%.1f" % value if value >= 20.0 else "%.2f" % value


def _phase(value: float) -> str:
    return "%.3f" % value if value >= 0.1 else "%.4f" % value


def landmarks_table(images: dict, e74: dict, e91: dict, e96: dict, key: str = "v3n96o") -> str:
    """Image-domain fitting: point fit, zero start (E74), search start (E91, E96), warm start (E70, E74)."""

    e70 = images["E70"]

    def row(label, medians, percent, italic=False):
        cells = [_rotation(medians["rotation_deg"]), "%.0f" % percent, "%.2f" % medians["translation_mm"],
                 _phase(medians["phase_map_cycles"])]
        if italic:
            label, cells = r"\textit{%s}" % label, [r"\textit{%s}" % c for c in cells]
        return "%s & %s\\\\" % (label, " & ".join(cells))

    def from_e70(arm):
        return e70["medians"][arm], 100.0 * e70["within_5deg"][arm] / e70["samples"]

    def from_e74(arm):
        return e74["medians"][arm], 100.0 * e74["counts"][arm]["within_5deg"] / e74["samples"]

    def from_summary(summary):
        return summary, 100.0 * summary["within_5deg"]

    search = e91["benchmarks"][key]["arms"]
    lines = [row("Point fit", *from_e70("point_fit"), italic=True),
             r"\multicolumn{5}{@{}l}{\textit{Zero start}}\\",
             row("Image", *from_e74("image_cold")),
             row(r"\quad + 2 neighboring", *from_e74("landmarks_2_cold")),
             row(r"\quad + 2 commissures", *from_e74("commissures_cold")),
             row(r"\quad + 4 landmarks", *from_e74("landmarks_4_cold")),
             row(r"\quad + 8 landmarks", *from_e74("landmarks_8_cold")),
             r"\multicolumn{5}{@{}l}{\textit{Search start}}\\",
             row("Image", *from_summary(search["image_search"])),
             row(r"\quad + 2 commissures", *from_summary(search["commissures_search"])),
             row(r"\qquad \SI{8}{\milli\meter} error", *from_summary(e96["arms"]["noise_8mm"])),
             row(r"\qquad one only", *from_summary(e96["arms"]["one_only"])),
             row(r"\qquad \ang{15} along the ring", *from_summary(e96["arms"]["slide_one"])),
             row(r"\qquad swapped", *from_summary(e96["arms"]["swap"])),
             r"\multicolumn{5}{@{}l}{\textit{Warm start}}\\",
             row("Image", *from_e70("image_warm")),
             row(r"\quad + 2 neighboring", *from_e70("landmarks_2")),
             row(r"\quad + 2 commissures", *from_e74("commissures_warm")),
             row(r"\quad + 4 landmarks", *from_e70("landmarks_4")),
             row(r"\quad + 8 landmarks", *from_e70("landmarks_8"))]
    return "\n".join(lines)


def channel_table(images: dict) -> str:
    e71 = images["E71"]
    labels = (("point_fit", "Point only"), ("cmr_image_tee_point", "CMR image, TEE points"),
              ("cmr_point_tee_image", "CMR points, TEE image"), ("both_image", "Both image"))
    return "\n".join("%s & %.2f & %.2f & %.4f & %d\\\\" % (
        label, e71["medians"][arm]["rotation_deg"], e71["medians"][arm]["translation_mm"],
        e71["medians"][arm]["phase_map_cycles"], e71["within_5deg"][arm]) for arm, label in labels)


def conformal_table(images: dict) -> str:
    quantities = images["E69"]["quantities"]
    formats = (("phase_offset_cycles", "Phase offset", lambda v: "%.3f cycles" % v),
               ("translation_norm_mm", "Translation", lambda v: r"\SI{%.2f}{\milli\meter}" % v),
               ("rotation_angle_deg", "Rotation", lambda v: r"\ang{%.2f}" % v),
               ("annulus_diameter_mm", "Annulus diameter", lambda v: r"\SI{%.2f}{\milli\meter}" % v))
    lines = []
    for name, label, fmt in formats:
        for i, target in enumerate(("0.8", "0.9")):
            entry = quantities[name][target]
            lines.append("%s & %.2f & %s & %.3f & %.3f\\\\" % (
                label if i == 0 else " " * len(label), float(target), fmt(entry["half_width"]),
                entry["coverage_samples"], entry["coverage_identities"]))
    return "\n".join(lines)


def reliability_table(e95: dict, bench: str = "v4") -> str:
    """E95's signals against the errors: Spearman rho with its interval, and the worst-decile flag."""

    rows = (("alpha", "rotation_deg", r"$\alpha$", "rotation"),
            ("alpha", "translation_mm", r"$\alpha$", "translation"),
            ("alpha", "heldout_cmr_mm", r"$\alpha$", "held-out CMR"),
            ("laplace_rotation_deg", "rotation_deg", "Laplace spread", "rotation"),
            ("laplace_translation_mm", "translation_mm", "Laplace spread", "translation"),
            ("laplace_phase_cycles", "phase_map_cycles", "Laplace spread", "phase map"),
            ("cost", "heldout_cmr_mm", "Final objective", "held-out CMR"))
    signals = e95["benchmarks"][bench]["signals"]
    lines = []
    for signal, target, label, error in rows:
        entry = signals[signal][target]
        rho, flag = entry["spearman"], entry["flag"]
        lines.append("%s & %s & %s {\\scriptsize[%s, %s]} & %d/%d\\\\" % (
            label, error, signed(rho["rho"], 2), signed(rho["lower"], 2), signed(rho["upper"], 2),
            flag["hits"], flag["flagged"]))
    return "\n".join(lines)


def relational_table(e90: dict) -> str:
    """E90: median absolute error of each method and Rooted4D minus late fusion, paired at the identity."""

    summary = e90["summary"]
    medians, paired = summary["median_absolute_error"], summary["paired"]

    def row(label, key, digits, late="late_fusion"):
        entry = paired[key]["rooted4d_minus_" + late]
        return "%s & %.*f & %.*f & %s\\\\" % (label, digits, medians[key]["rooted4d"], digits, medians[key][late],
                                            interval(entry, digits))

    relational = (("Free edge to apex, leaflet 1 (mm)", "apex_distance_anterior_mm"),
                  ("Free edge to apex, leaflet 2 (mm)", "apex_distance_posterior_mm"),
                  (r"Leaflet axis to long axis (\si{\degree})", "axis_angle_deg"))
    controls = (("Area (mm$^2$)", "area_mm2", 1), ("Perimeter (mm)", "perimeter_mm", 2),
                ("Intercommissural (mm)", "intercommissural_mm", 2), ("Anteroposterior (mm)", "anteroposterior_mm", 2))
    lines = [r"\multicolumn{4}{@{}l}{\textit{Relational}}\\"]
    lines += [row(label, key, 2) for label, key in relational]
    for late, heading in (("late_fusion", "valve-side"), ("late_fusion_chamber_ring", "chamber-side")):
        lines.append(r"\multicolumn{4}{@{}l}{\textit{Controls, late fusion's %s ring}}\\" % heading)
        lines += [row(label, key, digits, late) for label, key, digits in controls]
    return "\n".join(lines)


#: The measured-error table's columns: (endpoint, digits); the gap is the attachment residual.
MEASURED_COLUMNS = (("rotation_deg", 2), ("translation_mm", 2), ("phase_map_cycles", 4), ("attachment_residual_mm", 2),
                    ("heldout_cmr_mm", 2), ("heldout_tee_mm", 2), ("anatomy_rms_mm", 2))


def measured_table(e89: dict, e101: dict | None = None) -> str:
    """E89: medians on the original and measured-error observations, then the paired differences.

    With E101's report, its commissure-anchored configurations follow: their
    medians, and the degradation each removes (E89's minus its own).
    """

    medians = e89["medians"]
    methods = (("S/joint", "Rooted4D"), ("S/late_fusion_trajectory", "Late fusion"))

    def median_row(label, values):
        cells = [("0" if values[key] == 0.0 else "%.*f" % (digits, values[key])) if key in values else "--"
                 for key, digits in MEASURED_COLUMNS]
        return "%s & %s\\\\" % (label, " & ".join(cells))

    def paired_row(label, values):
        cells = []
        for key, digits in MEASURED_COLUMNS:
            entry = values.get(key)
            if entry is None or (entry["lower"] == entry["upper"] == entry["observed"] == 0.0):
                cells.append("--")
            else:
                cells.append(interval(entry, max(digits - 1, 2) if key != "phase_map_cycles" else 3))
        return "%s & %s\\\\" % (label, " & ".join(cells))

    lines = []
    for instance, heading in (("original", "Original observations, medians"), ("measured", "Measured error, medians")):
        if lines:
            lines.append(r"\addlinespace[2pt]")
        lines.append(r"\multicolumn{8}{@{}l}{\textit{%s}}\\" % heading)
        lines += [median_row(label, medians[instance][method]) for method, label in methods]
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{8}{@{}l}{\textit{Measured minus original, paired at the identity}}\\")
    degradation = e89["degradation_measured_minus_original"]
    lines += [paired_row(label, degradation[method]) for method, label in methods]
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{8}{@{}l}{\textit{Rooted4D minus late fusion under measured error, paired at the identity}}\\")
    lines.append(paired_row("Difference", e89["methods_on_measured"]["rooted4d_minus_late_fusion_trajectory"]))
    if e101 is None:
        return "\n".join(lines)
    m = e101["medians"]
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{8}{@{}l}{\textit{Commissure anchors, measured error, medians}}\\")
    lines.append(median_row("Commissure front end", m["front"]["S/joint"]))
    lines.append(median_row(r"\quad late fusion", m["front"]["S/late_fusion_trajectory"]))
    lines.append(median_row("Ring curve", m["anchors_measured"]["S/joint_curve_ring"]))
    lines.append(median_row("Ring and edge curves", m["anchors_measured"]["S/joint_curves"]))
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{8}{@{}l}{\textit{Degradation removed, paired at the identity}}\\")
    # The pre-registered reading decides on rotation, translation and anatomy only.
    decisive = ("rotation_deg", "translation_mm", "anatomy_rms_mm")
    lines.append(paired_row("Commissure front end",
                            {k: e101["front_end"]["reduction"][k] for k in decisive}))
    for method, label in (("S/joint_curve_ring", "Ring curve"), ("S/joint_curves", "Ring and edge curves")):
        lines.append(paired_row(label, e101["solver"][method]["reduction"]))
    return "\n".join(lines)


def e120_inputs(directory: Path = Path("results/benchmark/E120")):
    """The image-domain inputs of landmarks_table and channel_table, from E120's rerun.

    E120 reran the image studies after the session-gauge fix, and the supplement
    reports the rerun. Each study's own analysis of the rerun is in
    E120_studies.json (E70, E71, E91, E96) and E120_E74.json; E70 and E71 report
    no counts there, so their within-5-degree counts come from E120_comparison.json.
    """
    studies = json.loads((directory / "E120_studies.json").read_text())
    e74 = json.loads((directory / "E120_E74.json").read_text())
    comparison = json.loads((directory / "E120_comparison.json").read_text())["studies"]
    images = {}
    for name in ("E70", "E71"):
        rerun = comparison[f"{name}_v3n96o"]
        assert rerun["missing_from_rerun"] == 0
        images[name] = {**studies[name], "samples": rerun["samples"],
                        "within_5deg": {arm: record["rerun_within_5deg"] for arm, record in rerun["arms"].items()}}
        for arm, medians in studies[name]["medians"].items():
            assert medians["rotation_deg"] == rerun["arms"][arm]["rotation_deg"]["rerun_median"]
    return images, e74, studies["E91"], studies["E96"]


def main(argv) -> int:
    if argv[:1] == ["e120"]:
        # The supplement's image-domain tables, from the session-gauge rerun.
        images, e74, e91, e96 = e120_inputs(*(Path(a) for a in argv[1:2]))
        print("% landmarks table (image-domain fitting), session-gauge rerun (E120)")
        print(landmarks_table(images, e74, e91, e96))
        print("% channel table (supplement), session-gauge rerun (E120)")
        print(channel_table(images))
        return 0
    directory = Path(argv[0]) if argv else Path("results/benchmark/E100")
    main_report, gate, temporal = load(directory, "main"), load(directory, "gate"), load(directory, "temporal_session")
    instances = main_report["instances"] if main_report else {}
    for key, label in PRIMARY + ROBUSTNESS:
        if "medians" not in instances.get(key, {}):
            print(f"% {key}: not finished")
            continue
        print(f"% main table, {key} ({label})")
        print(main_table(instances[key]))
        print(f"% paired table, {key}")
        print(paired_table(instances[key]))
    print("% alignment table: Rooted4D minus trajectory late fusion, primaries then robustness")
    print(alignment_table(instances, [k for k, _ in PRIMARY + ROBUSTNESS]))
    if gate:
        print("% representation table: cross-validation (training) and test")
        print(representation_table(gate, instances, [k for k, _ in PRIMARY + ROBUSTNESS]))
    if temporal:
        print("% temporal table")
        print(temporal_table(temporal))
        print("% scarcity table (supplement)")
        print(scarcity_table(temporal))
    images, e74 = load(directory, "calibration_images"), load(directory, "E74")
    e91, e96 = load(directory, "E91"), load(directory, "E96")
    if images and e74 and e91 and e96 and "E70" in images:
        print("% landmarks table (image-domain fitting)")
        print(landmarks_table(images, e74, e91, e96))
    if images and "E71" in images:
        print("% channel table (supplement)")
        print(channel_table(images))
    if images and "E69" in images:
        print("% conformal table (supplement)")
        print(conformal_table(images))
    # E95 and E90 run on Delta; their reports are fetched into the same directory.
    for name, title, function in (("E95/E95_reliability", "reliability table", reliability_table),
                                  ("E90/E90_test_v4n96o", "relational table", relational_table),
                                  ("E89/E89_analysis", "measured-error table", measured_table)):
        path = directory / f"{name}.json"
        if path.exists():
            print(f"% {title}")
            extra = []
            if name.startswith("E89") and (directory / "E101/E101_analysis.json").exists():
                extra = [json.loads((directory / "E101/E101_analysis.json").read_text())]
            print(function(json.loads(path.read_text()), *extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
