#!/usr/bin/env python3
"""Render a campaign ``results.json`` into a readable ``RESULTS.md``.

Written as a script rather than inline so every run is summarized the same way
and the summary can be regenerated from the committed JSON at any time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e5:
            return f"{value:.2e}"
        return f"{value:.{digits}g}"
    return str(value)


def _median(block: dict | None) -> str:
    return _fmt(block["median"]) if block else "-"


def render(payload: dict, run_id: str, provenance: dict[str, str]) -> str:
    results = payload["results"]
    environment = payload["environment"]
    out: list[str] = []
    add = out.append

    add(f"# Campaign results — SLURM run {run_id}")
    add("")
    add("| | |")
    add("|---|---|")
    for key in ("commit", "branch", "dirty"):
        if key in provenance:
            add(f"| {key} | `{provenance[key]}` |")
    add(f"| cluster | NCSA Delta, partition `cpu`, account `bhio-delta-cpu` |")
    add(
        f"| environment | rooted4d {environment['rooted4d_version']}, "
        f"numpy {environment['numpy']}, python {environment['python']} |"
    )
    add("")
    add("| experiment | runtime |")
    add("|---|---:|")
    for name, result in results.items():
        add(f"| `{name}` | {result['runtime_seconds']:.1f} s |")
    add("")

    if "t1_annular_resolution" in results:
        add("## T1 — exact attachment across annular densities")
        add("")
        add("| nodes | local | physical | equality rows | rank | null-space | residual |")
        add("|---:|---:|---:|---:|---:|---:|---:|")
        for row in results["t1_annular_resolution"]["rows"]:
            add(
                f"| {row['n_annulus']} | {row['n_local_vertices']} | "
                f"{row['n_physical_vertices']} | {row['n_equality_rows']} | "
                f"{row['constraint_rank']} | {row['nullspace_dimension']} | "
                f"{_fmt(row['attachment_residual_mm_over_cycle'])} mm |"
            )
        add("")

    if "e5_baseline_ladder" in results:
        verdict = results["e5_baseline_ladder"]["verdict"]
        add("## E5 — locked baseline ladder")
        add("")
        add("| arm | max attach (mm) | held-out CMR | held-out TEE | phase (cyc) | alpha | identified | failures |")
        add("|---|---:|---:|---:|---:|---:|---|---:|")
        for arm, stats in verdict["per_arm"].items():
            identified = stats.get("identified")
            label = "-" if not identified else "/".join(_fmt(v) for v in identified)
            add(
                f"| `{arm}` | {_fmt(stats['max_attachment_residual_mm'])} | "
                f"{_median(stats['heldout_cmr_mm'])} | {_median(stats['heldout_tee_mm'])} | "
                f"{_median(stats['phase_error_cycles'])} | {_median(stats.get('alpha'))} | "
                f"{label} | {stats['failures']} |"
            )
        add("")
        add("Medians over all problems. Paired comparisons against the full model:")
        add("")
        add("| comparison | mean difference | 95% CI | p | favours |")
        add("|---|---:|---|---:|---|")
        for name, comparison in verdict["comparisons"].items():
            interval = f"[{comparison['ci_lower']:+.4f}, {comparison['ci_upper']:+.4f}]"
            add(
                f"| `{name}` | {comparison['mean_difference']:+.4f} | {interval} | "
                f"{comparison['p_value']:.4f} | {comparison['favours']} |"
            )
        add("")
        if verdict.get("weakly_identified_arms"):
            add(
                "Weakly identified arms (alpha far below the full model): "
                + ", ".join(f"`{a}`" for a in verdict["weakly_identified_arms"])
            )
            add("")
        if "reproducibility_note" in verdict:
            add(f"> {verdict['reproducibility_note']}")
            add("")
        if "interpretation_of_the_gauge_comparison" in verdict:
            add(f"> {verdict['interpretation_of_the_gauge_comparison']}")
            add("")

    if "e6_image_domain" in results:
        verdict = results["e6_image_domain"]["verdict"]
        add("## E6 — image-domain refinement")
        add("")
        add("| quantity | before | after | improved in |")
        add("|---|---:|---:|---:|")
        for key in (
            "heldout_cmr_planes",
            "heldout_tee_phases",
            "translation_mm",
            "phase_cycles",
            "rotation_deg",
        ):
            block = verdict.get(key)
            if not block:
                continue
            add(
                f"| {key} | {_fmt(block['before']['median'])} | "
                f"{_fmt(block['after']['median'])} | "
                f"{block['improved_fraction'] * 100:.0f}% of seeds |"
            )
        add("")
        add(f"Median {_fmt(verdict['seconds_per_fit']['median'])} s per fit. "
            f"Section 7.6 gates met: {_fmt(verdict['meets_section_7_6_gates'])}.")
        add("")
        for note in ("rotation_is_the_weak_variable", "cost_is_the_limiting_factor"):
            if note in verdict:
                add(f"> {verdict[note]}")
                add("")

    if "t5t6_solver_jacobian" in results:
        verdict = results["t5t6_solver_jacobian"]["verdict"]
        add("## Theorems 5 and 6 on the solver Jacobian")
        add("")
        add(f"- predicted gauge dimension: {verdict['predicted_gauge_dimension']}")
        add(
            "- predicted gauge count matches the measured numerical null space: "
            f"**{_fmt(verdict['gauge_prediction_matches_measurement'])}**"
        )
        add(
            "- unrestricted alpha is zero (the gauges are the whole null space): "
            f"{_fmt(verdict['alpha_unrestricted_is_zero'])}"
        )
        add(
            "- identifiable after gauge removal: "
            f"**{_fmt(verdict['all_identifiable_after_gauge_removal'])}**, "
            f"alpha median {_fmt(verdict['alpha']['median'])}"
        )
        add("")

    if "e10_family_generalization" in results:
        verdict = results["e10_family_generalization"]["verdict"]
        add("## E10 — held-out generator families (anti-inverse-crime)")
        add("")
        add("| family | held-out CMR | held-out TEE | rot (deg) | trans (mm) | phase (cyc) | s7.6 |")
        add("|---|---:|---:|---:|---:|---:|---|")
        for family, stats in verdict["per_family"].items():
            add(
                f"| `{family}` | {_median(stats['heldout_cmr_mm'])} | "
                f"{_median(stats['heldout_tee_mm'])} | {_median(stats['rotation_deg'])} | "
                f"{_median(stats['translation_mm'])} | {_median(stats['phase_error_cycles'])} | "
                f"{'pass' if stats['meets_section_7_6_gates'] else '**FAIL**'} |"
            )
        add("")
        for family, degradation in verdict["degradation_vs_train"].items():
            parts = ", ".join(f"{k} x{v:.2f}" for k, v in degradation.items())
            add(f"- `{family}`: {parts}")
        add("")
        add(f"Attachment exactly zero under every family: "
            f"**{_fmt(verdict['attachment_survives_every_family'])}**.")
        add("")
        add(f"> {verdict['headline']}")
        add("")

    add("## Remaining verdicts")
    add("")
    skip = {"t1_annular_resolution", "e5_baseline_ladder", "t5t6_solver_jacobian"}
    for name, result in results.items():
        if name in skip:
            continue
        add(f"### `{name}`")
        add("")
        add(f"{result['claim']}")
        add("")
        for key, value in result["verdict"].items():
            if isinstance(value, dict) and "median" in value:
                add(f"- {key}: median {_fmt(value['median'])}, range "
                    f"[{_fmt(value['min'])}, {_fmt(value['max'])}] over n={value['n']}")
            elif isinstance(value, dict):
                add(f"- {key}: `{json.dumps(value)[:160]}`")
            else:
                add(f"- {key}: {_fmt(value)}")
        add("")

    add("## Scope")
    add("")
    add(
        "Every number here comes from synthetic geometry with known truth. None of it "
        "is evidence about real CMR or TEE data, clinical accuracy, or the image-domain "
        "likelihood operators, which are not yet wired into the joint solver."
    )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", help="results/<slurm_job_id>")
    arguments = parser.parse_args(argv)

    directory = Path(arguments.run_directory)
    payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    provenance: dict[str, str] = {}
    stamp = directory / "PROVENANCE.txt"
    if stamp.exists():
        for line in stamp.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                provenance[key.strip()] = value.strip()

    target = directory / "RESULTS.md"
    target.write_text(render(payload, directory.name, provenance), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
