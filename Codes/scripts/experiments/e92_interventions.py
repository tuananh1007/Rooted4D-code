"""E92: interventional sensitivity of the decomposition (plan v2, W2.1).

Pre-registered in ``Docs/E92_INTERVENTIONS_PREREGISTRATION_2026-09-28.md``.
The benchmark's generator is a structural model: anatomy, motion, pose, phase
and session are drawn separately. Intervening on one factor at a time, and
leaving every other draw unchanged, tests which estimates stay invariant when
another factor changes.

Each intervention is a new instance of the realistic-session benchmark whose
config differs in one perturbation setting. The sample seeds derive from the
root seed and the donors, so every other draw is identical, and a scaled
setting scales the same underlying variate. Models are learned from the
original benchmark's training split (``model_benchmark``), so only the test
observations differ.

    python e92_interventions.py configs              -> Codes/configs/e92/*.yaml (nine instances)
    python e92_interventions.py specs                -> Codes/configs/e92/<instance>_test.json (ten, with the base)
    python e92_interventions.py analyze <rows root>  -> results/benchmark/E92/E92_interventions.json
    python e92_interventions.py figure <output>      the pre-registered figure, from the report
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E92_INTERVENTIONS_PREREGISTRATION_2026-09-28.md"
SOURCE = Path("Codes/configs/semisynthetic_real_donors_v4.yaml")
ORIGINAL = "data/derived/semisynthetic-real-v4"
FROZEN = Path("Codes/configs/e75/v4_e75_test.json")
OUT = Path("Codes/configs/e92")
REPORT = Path("results/benchmark/E92/E92_interventions.json")
#: Levels per factor; the middle value is the benchmark's own setting.
FACTORS = {
    "session": {"levels": (0.0, 0.5, 1.0, 2.0), "base": 1.0},        # scale c of both session fields
    "translation": {"levels": (3.0, 9.0, 18.0), "base": 9.0},        # pose translation SD, mm
    "rotation": {"levels": (0.07, 0.14, 0.28), "base": 0.14},        # pose axis-angle SD, rad
    "phase": {"levels": (0.1, 0.2, 0.4), "base": 0.2},               # phase-offset half-range, cycles
}
ARMS = ("S/joint", "S/joint_nosession", "S/late_fusion_trajectory")
SCALARS = ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm", "phase_map_cycles")
RESAMPLES, SEED = 20000, 20260910
#: Where the intervention instances are written. E100 points this, SOURCE,
#: ORIGINAL, FROZEN, OUT and REPORT at its rebuilt primary (``e100_components.py``).
INSTANCE_PREFIX = "data/derived/semisynthetic-real-v4-e92"


def instance(factor: str, level: float) -> str:
    if level == FACTORS[factor]["base"]:
        return ORIGINAL
    return f"{INSTANCE_PREFIX}-{factor}-{level:g}"


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def cmd_configs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text()
    for factor, spec in FACTORS.items():
        for level in spec["levels"]:
            if level == spec["base"]:
                continue
            text = source.replace(": ../../", ": ../../../")
            if factor == "session":
                for kind in ("static", "dynamic"):
                    text = _replace_once(text, f"session_{kind}_scale_min: 0.70",
                                         f"session_{kind}_scale_min: {0.70 * level:.2f}")
                    text = _replace_once(text, f"session_{kind}_scale_max: 1.30",
                                         f"session_{kind}_scale_max: {1.30 * level:.2f}")
            elif factor == "translation":
                text = _replace_once(text, "pose_translation_std_mm: 9.0", f"pose_translation_std_mm: {level}")
            elif factor == "rotation":
                text = _replace_once(text, "pose_axis_angle_std_rad: 0.14", f"pose_axis_angle_std_rad: {level}")
            else:
                text = _replace_once(text, "phase_offset_min_cycles: -0.20", f"phase_offset_min_cycles: {-level}")
                text = _replace_once(text, "phase_offset_max_cycles: 0.20", f"phase_offset_max_cycles: {level}")
            header = (f"# E92 ({PREREGISTRATION}): the realistic-session benchmark with one\n"
                      f"# intervention, {factor} = {level:g}. Identical to {SOURCE.name} otherwise.\n"
                      "# Written by e92_interventions.py configs; do not edit by hand.\n")
            path = OUT / f"semisynthetic_real_donors_v4_{factor}_{level:g}.yaml"
            path.write_text(header + text)
            print("wrote", path)


def spec_for(factor: str, level: float) -> dict:
    frozen = json.loads(FROZEN.read_text())
    by_name = {v["name"]: v for v in frozen["variants"]}
    spec = {key: frozen[key] for key in ("start", "models", "base_config")}
    spec.update({"experiment": "E92", "preregistration": PREREGISTRATION, "split": "test",
                 "benchmark": instance(factor, level), "model_benchmark": ORIGINAL,
                 "intervention": {"factor": factor, "level": level}, "frozen_from": str(FROZEN)})
    joint = json.loads(json.dumps(by_name["S/joint"]))
    joint["options"] = {"store_parameters": True}
    without = json.loads(json.dumps(joint))
    without["name"] = "S/joint_nosession"
    without["overrides"] = dict(without["overrides"], use_session=False)
    late = {"name": "S/late_fusion_trajectory", "model": "S", "arm": "late_fusion_trajectory",
            "options": dict(json.loads(json.dumps(by_name["S/independent"]["options"])),
                            store_parameters=True)}
    spec["variants"] = [joint, without, late]
    return spec


def cmd_specs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = set()
    for factor, spec in FACTORS.items():
        for level in spec["levels"]:
            name = "base" if level == spec["base"] else f"{factor}_{level:g}"
            if name in written:
                continue
            written.add(name)
            path = OUT / f"{name}_test.json"
            path.write_text(json.dumps(spec_for(factor, level), indent=2) + "\n")
            print("wrote", path)


# -------------------------------------------------------------- analysis ----

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def geometry_record(row: dict, sample: dict, model, config) -> dict:
    """Session-field recovery and anatomy error, from the stored parameters."""

    from rooted4d.experiments.fitted_geometry import fitted_trajectory, truth_trajectory
    from rooted4d.inference.baselines import parameters_from_record
    from rooted4d.inference.joint import latent_trajectory

    phases = np.linspace(0.0, 1.0, 32, endpoint=False)
    n = len(sample["template_vertices"])
    truth = truth_trajectory(sample)(phases)
    field = sample["session_static"][None] + np.cos(2 * np.pi * phases)[:, None, None] * sample["session_dynamic"][None]
    out = {}
    for arm in ("S/joint", "S/joint_nosession"):
        outcome = row["variants"][arm]
        if not outcome.get("succeeded"):
            continue
        parameters = parameters_from_record(outcome["extras"]["parameters"])
        arm_config = replace(config, use_session=arm == "S/joint")
        shared = fitted_trajectory(parameters, model, arm_config, phases, n)
        record = {"anatomy_rms_mm": float(np.sqrt(((shared - truth) ** 2).sum(axis=2).mean()))}
        if arm == "S/joint" and np.abs(field).max() > 0:
            rows = model.observation_rows(np.arange(n))
            fractions = []
            # The generator adds -field/2 to CMR and +field/2 to TEE.
            for session, sign in (("cmr", -0.5), ("tee", 0.5)):
                latent = latent_trajectory(parameters, phases, session, model, arm_config)
                with_session = model.physical_coordinates(latent)[:, rows, :]
                estimated = with_session - shared
                target = sign * field
                fractions.append(1.0 - ((estimated - target) ** 2).sum() / (target ** 2).sum())
            record["session_recovered_fraction"] = float(np.mean(fractions))
        out[arm] = record
    return out


def _identity_difference(rows, first, second, key):
    identities = sorted({r["identity"] for r in rows})

    def per_identity(name):
        values = []
        for identity in identities:
            vals = [r["values"][name].get(key) for r in rows if r["identity"] == identity]
            vals = [np.nan if v is None else float(v) for v in vals]
            values.append(np.nanmean(vals))
        return np.array(values)

    a, b = per_identity(first), per_identity(second)
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def cmd_analyze(rows_root: str) -> None:
    driver = _load("e59_driver", HERE / "e59_unknown_alignment.py")
    from rooted4d.inference import SolverConfig

    base_spec = spec_for("session", 1.0)
    model, *_ = driver.build_model(Path(ORIGINAL), base_spec["models"]["S"])
    joint = next(v for v in base_spec["variants"] if v["name"] == "S/joint")
    config = replace(SolverConfig(**base_spec["base_config"]), **joint["overrides"])
    report = {"experiment": "E92", "preregistration": PREREGISTRATION, "factors": {}}
    cache = {}
    for factor, spec in FACTORS.items():
        report["factors"][factor] = {}
        for level in spec["levels"]:
            # The benchmark level was run once, as the base spec (``specs`` writes
            # it under the first factor), and it serves every factor.
            level_spec = spec_for("session", 1.0) if level == spec["base"] else spec_for(factor, level)
            directory = Path(rows_root) / _digest(level_spec)
            if not directory.exists():
                report["factors"][factor][f"{level:g}"] = {"status": "not run"}
                continue
            key = str(directory)
            if key not in cache:
                raw = [json.loads(line) for shard in sorted(directory.glob("shard_*.jsonl"))
                       for line in shard.read_text().splitlines() if line.strip()]
                assert len({r["sample"] for r in raw}) == len(raw) == 60, (factor, level, len(raw))
                rows = []
                for row in raw:
                    path = Path(level_spec["benchmark"]) / "samples" / "test" / row["sample"]
                    with np.load(path, allow_pickle=True) as blob:
                        sample = {k: blob[k] for k in blob.files}
                    values = {arm: {k: (row["variants"][arm].get(k) if row["variants"][arm].get("succeeded")
                                        else None) for k in SCALARS} for arm in ARMS}
                    for arm, record in geometry_record(row, sample, model, config).items():
                        values[arm].update(record)
                    rows.append({"identity": row["identity"], "sample": row["sample"], "values": values})
                cache[key] = rows
            rows = cache[key]
            keys = SCALARS + ("anatomy_rms_mm",)
            entry = {"medians": {arm: {k: float(np.nanmedian([np.nan if r["values"][arm].get(k) is None
                                                               else r["values"][arm][k] for r in rows]))
                                       for k in keys + ("session_recovered_fraction",)
                                       if any(r["values"][arm].get(k) is not None for r in rows)}
                                 for arm in ARMS},
                     "with_minus_without_session": {k: _identity_difference(rows, "S/joint", "S/joint_nosession", k)
                                                    for k in keys}}
            report["factors"][factor][f"{level:g}"] = entry
            print(factor, level, {k: round(v["observed"], 3) for k, v in entry["with_minus_without_session"].items()})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", REPORT)


def cmd_figure(output: str) -> None:
    """The pre-registered figure: one panel per factor, with - without the session term, against the level."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "0.25", "axes.linewidth": 0.7, "axes.axisbelow": True,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    factors = json.loads(REPORT.read_text())["factors"]
    series = (("heldout_cmr_mm", "held-out CMR (mm)", "#3b6ea5", "o"),
              ("heldout_tee_mm", "held-out TEE (mm)", "#c05746", "s"),
              ("rotation_deg", "rotation (deg)", "#1a1a1a", "D"),
              ("translation_mm", "translation (mm)", "#d99b3c", "^"),
              ("anatomy_rms_mm", "anatomy RMS (mm)", "#5b8c5a", "v"))
    titles = {"session": ("(a) session-field scale $c$", "scale $c$"),
              "translation": ("(b) pose translation", "SD (mm)"),
              "rotation": ("(c) pose rotation", "SD (rad)"),
              "phase": ("(d) phase offset", "half-range (cycles)")}
    widths = [len(spec["levels"]) for spec in FACTORS.values()]
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(6.8, 2.7), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    for ax, (factor, spec) in zip(axes, FACTORS.items()):
        levels = spec["levels"]
        base = levels.index(spec["base"])
        ax.axvspan(base - 0.5, base + 0.5, color="0.94", lw=0, zorder=0)
        ax.grid(True, axis="y", color="0.9", lw=0.5)
        for offset, (key, label, color, marker) in zip(np.linspace(-0.3, 0.3, len(series)), series):
            points = [(i + offset, factors[factor][f"{level:g}"]["with_minus_without_session"][key])
                      for i, level in enumerate(levels) if "with_minus_without_session" in factors[factor][f"{level:g}"]]
            ax.errorbar([x for x, _ in points], [v["observed"] for _, v in points],
                        yerr=[[v["observed"] - v["lower"] for _, v in points],
                              [v["upper"] - v["observed"] for _, v in points]],
                        fmt=marker, color=color, markersize=3.4, capsize=1.6, linewidth=0.75,
                        markeredgecolor="white", markeredgewidth=0.4, label=label, zorder=3)
        ax.axhline(0.0, color="0.45", linewidth=0.8, zorder=2)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{level:g}" + ("*" if level == spec["base"] else "") for level in levels])
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_xlabel(titles[factor][1])
        ax.set_title(titles[factor][0], loc="left", fontsize=7.5, pad=4)
    axes[0].set_ylabel("with $-$ without the session term")
    axes[0].text(0.03, 0.03, "below 0: the session term helps", transform=axes[0].transAxes,
                 fontsize=6, color="0.4", ha="left", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.31, top=0.9, wspace=0.1)
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False,
               handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.99, 0.11, "* the benchmark's own setting (shaded)", fontsize=6, color="0.4",
             ha="right", va="bottom")
    fig.savefig(output)
    print("wrote", output)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"configs": cmd_configs, "specs": cmd_specs, "analyze": cmd_analyze,
     "figure": cmd_figure}[command](*arguments)
