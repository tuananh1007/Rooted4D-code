"""Write the manuscript's main fusion tables from the E62 and E64 reports.

Nothing here is computed afresh: every number is read from
``results/benchmark/E62_stage_b_gate.json`` (the cross-validated gate) or
``results/benchmark/E64_test.json`` (the single test evaluation), and printed
as LaTeX rows. The tables in results.tex are pasted from this output, so a
table and its report cannot drift apart unnoticed: rerun this and diff.

    python e64_manuscript_tables.py > tables.tex        # the evaluation each benchmark reports
    python e64_manuscript_tables.py e64 > e64.tex       # E64 alone, for the record of the first evaluation

E73 selected the session weight on validation afterwards, which changed the
configuration on v4 and v3 at 48 nodes; E75 evaluated those once. The main
tables therefore read E75 for v4 and v3 at 48 nodes and E64 for v3.
"""

import json
from pathlib import Path

GATE = Path("results/benchmark/E62_stage_b_gate.json")
TEST = Path("results/benchmark/E64_test.json")
SECOND = Path("results/benchmark/E75_test.json")
#: E87: late fusion with trajectory registration, the late-fusion row of the
#: tables. Its fits are the feature-registered arm's, so its held-out errors and
#: the held-out comparison of Table 2 are that arm's exactly.
TRAJECTORY = Path("results/benchmark/E87/E87_trajectory_late_fusion.json")
TRAJECTORY_ARM = "S/late_fusion_trajectory"

# Table rows, in reading order: the unimodal fits, the competitors built from
# them, the penalty variants, and the two exact joint models.
ROWS = [("S/cmr_only", "CMR-only"),
        ("S/tee_only", "TEE-only"),
        (TRAJECTORY_ARM, "Late fusion"),
        ("S/exact_repair", "Exact repair"),
        ("S/penalty_repair", r"Penalty repair, $\lambda=10^2$"),
        ("S/soft_matched", r"Soft attachment, $\lambda=1$"),
        ("S/soft_matched_1e4", r"Soft attachment, $\lambda=10^4$"),
        ("S/exact_no_gauges", "Exact, no gauges"),
        ("G/joint", "Rooted4D, global basis"),
        ("S/joint", r"\textbf{Rooted4D}")]
COMPARATORS = [("S/independent", "Late fusion", ("T3_fusion_cost", "comparison")),
               ("S/exact_repair", "Exact repair", ("T4_competitors", "exact_repair")),
               ("S/penalty_repair", r"Penalty repair, $\lambda=10^2$", ("T4_competitors", "penalty_repair")),
               ("S/soft_matched", r"Soft attachment, $\lambda=1$", ("T4_competitors", "soft_matched")),
               ("S/soft_matched_1e4", r"Soft attachment, $\lambda=10^4$", ("T4_competitors", "soft_matched_1e4")),
               ("S/exact_no_gauges", "Exact, no gauges", ("T4_competitors", "exact_no_gauges")),
               ("G/joint", "Rooted4D, global basis", ("T1_representation", "comparison"))]


def number(value, digits=2):
    return "--" if value is None else "%.*f" % (digits, value)


def gap(value):
    if value is None:
        return "--"
    if value == 0.0:
        return "0"
    if value < 0.005:
        mantissa, exponent = ("%.1e" % value).split("e")
        return r"$%s\times10^{%d}$" % (mantissa, int(exponent))
    return "%.2f" % value


def signed(value, digits=3):
    """A signed value, except that one rounding to zero prints unsigned."""
    text = "%+.*f" % (digits, value)
    return "$%s$" % ("%.*f" % (digits, 0.0) if float(text) == 0.0 else text)


def interval(entry, digits=3):
    return r"%s {\scriptsize[%s, %s]}" % (
        signed(entry["observed"], digits), signed(entry["lower"], digits), signed(entry["upper"], digits))


def main_table(test, bench, trajectory=None):
    medians = dict(test["benchmarks"][bench]["medians"])
    if trajectory is not None:
        medians[TRAJECTORY_ARM] = trajectory["benchmarks"][bench]["medians"][TRAJECTORY_ARM]
    lines = []
    for name, label in ROWS:
        m = medians[name]
        fail = 100.0 * m["failures"] / max(m["samples"], 1)
        lines.append("%s & %s & %s & %s & %s & %s & %s & %.0f\\\\" % (
            label, number(m["heldout_cmr_mm"]), number(m["heldout_tee_mm"]),
            number(m["rotation_deg"]), number(m["translation_mm"]),
            number(m["phase_map_cycles"], 3), gap(m["attachment_residual_mm"]), fail))
    return "\n".join(lines)


def paired_table(test, bench):
    block = test["benchmarks"][bench]
    lines = []
    for name, label, path in COMPARATORS:
        entry = block[path[0]][path[1]]
        lines.append("%s & %s & %s & %s\\\\" % (
            label, interval(entry["sum"]), interval(entry["cmr"]), interval(entry["tee"])))
    return "\n".join(lines)


def alignment_table(trajectory):
    """Rooted4D minus trajectory late fusion: rotation, translation, phase map and gap (E87)."""
    lines = []
    for bench, label in (("v4", "Realistic session (primary)"), ("v3", "Small session"),
                         ("v3n48", "Small session, 48 nodes"), ("v3n96", "Small session, 96 nodes")):
        entry = trajectory["benchmarks"][bench]["rooted4d_minus_trajectory"]
        lines.append("%s & %s & %s & %s & %s\\\\" % (
            label, interval(entry["rotation_deg"], 2), interval(entry["translation_mm"], 2),
            interval(entry["phase_map_cycles"], 3), interval(entry["attachment_residual_mm"], 2)))
    return "\n".join(lines)


def registration_table(trajectory):
    """Feature against trajectory registration of the same fits (E87), for the supplement."""
    lines = []
    for bench, label in (("v4", "Realistic session"), ("v3", "Small session"),
                         ("v3n48", "48 nodes"), ("v3n96", "96 nodes")):
        block = trajectory["benchmarks"][bench]
        for name, method in (("S/independent", "feature"), (TRAJECTORY_ARM, "trajectory")):
            m = block["medians"][name]
            lines.append("%s, %s & %s & %s & %s & %s\\\\" % (
                label, method, number(m["rotation_deg"]), number(m["translation_mm"]),
                number(m["phase_map_cycles"], 3), gap(m["attachment_residual_mm"])))
        entry = block["trajectory_minus_feature"]
        lines.append("\\quad trajectory $-$ feature & %s & %s & %s & %s\\\\" % (
            interval(entry["rotation_deg"], 2), interval(entry["translation_mm"], 2),
            interval(entry["phase_map_cycles"], 3), interval(entry["attachment_residual_mm"], 2)))
    return "\n".join(lines)


def representation_table(gate, source):
    lines = []
    for bench, label in (("v4", "v4 (primary)"), ("v3", "v3"), ("v3n48", "v3, 48 nodes")):
        cv = gate["D1"]["comparisons"][bench]
        te = source(bench)["benchmarks"][bench]["T1_representation"]["comparison"]
        lines.append("%s & %s & %s & %s & %s & %s & %s\\\\" % (
            label, interval(cv["sum"]), signed(cv["cmr"]["observed"]), signed(cv["tee"]["observed"]),
            interval(te["sum"]), signed(te["cmr"]["observed"]), signed(te["tee"]["observed"])))
    return "\n".join(lines)


def main(argv) -> int:
    gate = json.loads(GATE.read_text())
    first = json.loads(TEST.read_text())
    second = json.loads(SECOND.read_text())
    if argv[:1] == ["e64"]:
        def source(bench):
            return first
    else:
        def source(bench):
            return second if bench in second["benchmarks"] else first
    trajectory = json.loads(TRAJECTORY.read_text())
    for bench in ("v4", "v3"):
        print("%% main table, %s (%s; late fusion from E87)" % (bench, source(bench).get("experiment", "E64")))
        print(main_table(source(bench), bench, trajectory))
        print("%% paired table, %s" % bench)
        print(paired_table(source(bench), bench))
    print("% alignment table: Rooted4D minus trajectory late fusion (E87)")
    print(alignment_table(trajectory))
    print("% registration table, supplement: feature against trajectory registration (E87)")
    print(registration_table(trajectory))
    print("% representation table (test columns: v4 and v3 at 48 nodes from the evaluation each reports)")
    print(representation_table(gate, source))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
