"""E88: the main protocol replicated over five fresh donor partitions.

Pre-registered in ``Docs/E88_REPLICATION_PREREGISTRATION_2026-09-28.md``. Each
(benchmark, split) pair is a new benchmark instance, identical to the original
except for its split seed, taken through the whole protocol by one unattended
chain with no decision between stages:

1. E62 round 1 (identity weights) and round 2 (soft and repair penalties) on
   validation, with E62's edge extensions; E62's frozen selection is written
   (its cross-validation gate selects nothing and is not run);
2. E73's session-weight selection on validation, with its edge extension;
3. one test evaluation with E75's construction, every E64 test arm, and E87's
   trajectory late fusion at late fusion's selected fits.

Every rule is the original code's (``e62_stage_b.py``, ``e73_session_weight.py``);
this script only points it at the new instances and runs it in order.

    python e88_replication.py configs                 -> Codes/configs/e88/semisynthetic_real_donors_<bench>_split<k>.yaml
    python e88_replication.py chain <bench> <k>        -> the whole protocol for one instance (on Delta)
    python e88_replication.py analyze <chain root> v4=<dirs,...> v3=<dirs,...>
                                                    -> results/benchmark/E88/E88_replication.json
    python e88_replication.py figure <output.pdf>      -> the strip plot of the key differences

``chain`` reads ``R4D_E88_OUT`` (the results root) and ``R4D_WORKERS``, and
resumes: a stage whose rows are complete is not run again, and a last row cut
off mid-write is dropped first. ``analyze`` takes the chain root (a fetched
copy works: each chain records its test directory relative to itself) and the
original split's test rows, E75 or E64 together with E87, for each benchmark.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E88_REPLICATION_PREREGISTRATION_2026-09-28.md"
#: Fixed before any instance is generated. The original split seed is 104729.
SPLIT_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)
ORIGINAL_SPLIT_SEED = 104729
SOURCES = {"v4": "Codes/configs/semisynthetic_real_donors_v4.yaml",
           "v3": "Codes/configs/semisynthetic_real_donors_v3.yaml"}
CONFIGS = Path("Codes/configs/e88")
REPORT = Path("results/benchmark/E88/E88_replication.json")
TRAJECTORY = "S/late_fusion_trajectory"
ENDPOINTS = ("rotation_deg", "translation_mm", "phase_error_cycles", "phase_map_cycles",
             "attachment_residual_mm")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instance(bench: str, k: int) -> str:
    return f"data/derived/semisynthetic-real-{bench}-split{k}"


def cmd_configs() -> None:
    """The original generator configs with only the split seed changed."""

    CONFIGS.mkdir(parents=True, exist_ok=True)
    for bench, source in SOURCES.items():
        text = Path(source).read_text()
        assert text.count(f"split_seed: {ORIGINAL_SPLIT_SEED}") == 1, source
        for k, seed in enumerate(SPLIT_SEEDS, 1):
            body = text.replace(f"split_seed: {ORIGINAL_SPLIT_SEED}", f"split_seed: {seed}")
            # Paths in a config are relative to the config, one level deeper here.
            body = body.replace(": ../../", ": ../../../")
            header = (f"# E88 ({PREREGISTRATION}): split {k} of {bench}. Identical to\n"
                      f"# {source} except split_seed ({ORIGINAL_SPLIT_SEED} -> {seed}) and the\n"
                      f"# relative paths, which point one directory further up. Written by\n"
                      f"# e88_replication.py configs; do not edit by hand.\n")
            path = CONFIGS / f"semisynthetic_real_donors_{bench}_split{k}.yaml"
            path.write_text(header + body)
            print("wrote", path)


# ------------------------------------------------------------------ chain ----

class Chain:
    """One instance taken through selection and a single test evaluation.

    ``Chain(bench, k)`` is an E88 instance. Later experiments that run the same
    protocol on another benchmark (E93, E94) pass ``key``, ``benchmark``,
    ``configs`` and their own labels; with the defaults nothing about E88's
    specs changes. ``store_parameters`` keeps the fitted parameters of Rooted4D
    and of trajectory late fusion in the test stage, for measurements on fits.
    """

    def __init__(self, bench: str, k: int, *, key: str | None = None,
                 benchmark: str | None = None, configs: Path = CONFIGS,
                 experiment: str = "E88", preregistration: str = PREREGISTRATION,
                 store_parameters: bool = False):
        if key is None:
            if bench not in SOURCES or not 1 <= k <= len(SPLIT_SEEDS):
                raise SystemExit(f"unknown instance {bench} split {k}")
            key, benchmark = f"{bench}s{k}", instance(bench, k)
        elif benchmark is None:
            raise SystemExit("a named chain needs its benchmark")
        self.bench, self.key = bench, key
        self.experiment, self.preregistration = experiment, preregistration
        self.store_parameters = store_parameters
        self.root = Path(os.environ["R4D_E88_OUT"]) / self.key
        self.specs = Path(configs) / self.key
        stage = _load("e62_stage_b")
        stage.BENCHMARKS[self.key] = {"benchmark": benchmark,
                                      "session_includes_offset": bench == "v4"}
        stage.OUT = self.specs
        original_write = stage.write

        def write(spec: dict, name: str) -> Path:
            spec["experiment"], spec["preregistration"] = experiment, preregistration
            return original_write(spec, name)

        stage.write = write
        self.stage = stage
        session = _load("e73_session_weight")
        session.OUT, session.PREREGISTRATION = self.specs, preregistration
        self.session = session

    @staticmethod
    def _repair_truncated(directory: Path) -> None:
        """Drop a last shard line cut off mid-write, so the driver resumes cleanly.

        A job killed while writing a row (time limit, OOM) can leave a partial
        JSON line; the driver's own resume parse would then fail on every retry.
        Only an unparsable last line is removed; any other bad line is an error.
        """

        for shard in directory.glob("shard_*.jsonl") if directory.exists() else []:
            lines = shard.read_text().splitlines()
            if not lines:
                continue
            try:
                json.loads(lines[-1])
                intact = shard.read_text().endswith("\n")
            except json.JSONDecodeError:
                intact = False
            if not intact:
                for line in lines[:-1]:
                    json.loads(line)
                shard.write_text("".join(line + "\n" for line in lines[:-1]))
                print(f"removed a truncated last row from {shard}", flush=True)

    def _complete(self, spec: dict, directory: Path) -> bool:
        expected = len(list((Path(spec["benchmark"]) / "samples" / spec["split"]).glob("*.npz")))
        shards = list(directory.glob("shard_*.jsonl")) if directory.exists() else []
        done = {json.loads(line)["sample"] for shard in shards
                for line in shard.read_text().splitlines() if line.strip()}
        return expected > 0 and len(done) == expected

    def run(self, spec_path: Path, label: str, *, allow_test: bool = False) -> str:
        """Run one stage; returns its row directory."""

        spec = json.loads(spec_path.read_text())
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.root / label / digest
        self._repair_truncated(directory)
        if self._complete(spec, directory):
            print(f"[{self.key}] {label}: complete, not re-run", flush=True)
            return str(directory)
        environment = dict(os.environ, R4D_SPEC=str(spec_path), R4D_OUT=str(self.root / label),
                           R4D_SHARD="0/1")
        environment.pop("R4D_ALLOW_TEST", None)
        if allow_test:
            environment["R4D_ALLOW_TEST"] = "1"
        print(f"[{self.key}] {label}: running {spec_path} ({len(spec['variants'])} variants)",
              flush=True)
        subprocess.run([sys.executable, "-u", str(HERE / "e59_unknown_alignment.py")],
                       env=environment, check=True)
        if not self._complete(spec, directory):
            raise RuntimeError(f"{label} finished without every sample")
        return str(directory)

    def _fresh(self, *names: str) -> None:
        for name in names:
            (self.specs / name).unlink(missing_ok=True)

    def selection_rounds(self) -> None:
        key = self.key
        self._fresh(f"{key}_round1.json")
        self.stage.cmd_round1(key)
        first = [self.run(self.specs / f"{key}_round1.json", "round1")]
        while True:
            extension = f"{key}_round1_extension_{len(first)}.json"
            self._fresh(extension, f"{key}_round2.json")
            self.stage.cmd_round2(key, *first)
            if not (self.specs / extension).exists():
                break
            first.append(self.run(self.specs / extension, f"round1_ext{len(first)}"))
        second = [self.run(self.specs / f"{key}_round2.json", "round2")]
        while True:
            extension = f"{key}_round2_extension_{len(second)}.json"
            self._fresh(extension, f"{key}_gate.json")
            self.stage.cmd_gate(key, *second)
            if not (self.specs / extension).exists():
                break
            second.append(self.run(self.specs / extension, f"round2_ext{len(second)}"))
        # The gate spec carries the frozen selections; E64's test template is
        # built from it. Neither is run: the gate selects nothing.
        self._fresh(f"{key}_test.json")
        self.stage.cmd_test(key)

    def _session_spec(self, values_by_arm: dict) -> dict:
        """E73's validation spec, templated on this instance's test template."""

        template = json.loads((self.specs / f"{self.key}_test.json").read_text())
        by_name = {v["name"]: v for v in template["variants"]}
        spec = {key: template[key] for key in ("benchmark", "start", "models", "base_config")}
        spec.update({"experiment": self.experiment, "preregistration": self.preregistration,
                     "split": "val",
                     "frozen_from": str(self.specs / f"{self.key}_test.json"), "variants": []})
        for arm, values in values_by_arm.items():
            for value in values:
                variant = json.loads(json.dumps(by_name[arm]))
                variant["name"] = self.session.name(arm, value)
                variant["overrides"] = dict(by_name[arm].get("overrides") or {},
                                            session_prior_weight=value)
                spec["variants"].append(variant)
        return spec

    def session_weight(self) -> dict:
        session = self.session
        path = session.write(self._session_spec({arm: session.GRID for arm in session.ARMS}),
                             f"{self.key}_session_weight_val.json")
        directories = [self.run(path, "session")]
        while True:
            rows = self.stage.load_rows(directories)
            arms = {arm: session.choose(rows, arm) for arm in session.ARMS}
            pending = {arm: a["extension_needed"] for arm, a in arms.items() if a["extension_needed"]}
            if not pending:
                break
            path = session.write(self._session_spec(pending),
                                 f"{self.key}_session_weight_val_extension{len(directories)}.json")
            directories.append(self.run(path, f"session_ext{len(directories)}"))
        record = {"experiment": self.experiment, "rule": "E73", "grid": session.GRID,
                  "samples": len(rows),
                  "identities": len({r["identity"] for r in rows}), "arms": arms,
                  "directories": directories}
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "session_selection.json").write_text(json.dumps(record, indent=2) + "\n")
        return arms

    def test(self, arms: dict) -> str:
        """E75's construction, plus E87's arm at late fusion's selected fits."""

        template = json.loads((self.specs / f"{self.key}_test.json").read_text())
        changed = {arm: float(a["selected"]) for arm, a in arms.items() if a["changed"]}
        variants = []
        for variant in template["variants"]:
            variant = json.loads(json.dumps(variant))
            weight = changed.get(variant["name"])
            if variant["name"].endswith("/soft_matched_1e4"):
                weight = changed.get(f"{variant['model']}/joint")
            if weight is not None:
                variant["overrides"] = dict(variant.get("overrides") or {},
                                            session_prior_weight=weight)
            variants.append(variant)
        independent = next(v for v in variants if v["name"] == "S/independent")
        variants.append({"name": TRAJECTORY, "model": "S", "arm": "late_fusion_trajectory",
                         "options": json.loads(json.dumps(independent["options"]))})
        if self.store_parameters:
            for variant in variants:
                if variant["name"] in ("S/joint", TRAJECTORY):
                    variant["options"] = dict(variant.get("options") or {}, store_parameters=True)
        spec = dict(template, experiment=self.experiment, preregistration=self.preregistration,
                    round="test", session_weights=changed, variants=variants)
        path = self.specs / f"{self.key}_e88_test.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        return self.run(path, "test", allow_test=True)


def cmd_chain(bench: str, k: str) -> None:
    chain = Chain(bench, int(k))
    chain.selection_rounds()
    arms = chain.session_weight()
    directory = chain.test(arms)
    # Relative to the chain root, so a fetched copy of the root can be analysed.
    (chain.root / "TEST_DONE").write_text(str(Path(directory).relative_to(chain.root)) + "\n")
    print(f"[{chain.key}] test rows in {directory}", flush=True)


# --------------------------------------------------------------- analysis ----

def identity_difference(stage, rows, first, second, key):
    """E62's statistic for any endpoint (same seed and draw order as ``paired``)."""

    ids_a, a = stage.per_identity(rows, first, key)
    ids_b, b = stage.per_identity(rows, second, key)
    assert ids_a == ids_b
    rng = np.random.default_rng(stage.SEED)
    draws = np.empty(stage.RESAMPLES)
    for r in range(stage.RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def test_report(stage, rows) -> dict:
    """E64's T1-T4 and E87's comparisons for one test run."""

    def label(entry):
        return ("joint better" if entry["upper"] < 0 else
                "cost remains" if entry["lower"] > 0 else "no detectable cost")

    names = [f"{m}/{a}" for m in stage.MODELS for a in stage.TEST_ARMS] + [TRAJECTORY]
    # An absent arm would otherwise give NaN intervals that read as "not confirmed".
    assert rows, "no test rows"
    missing = {name for name in names for r in rows if name not in r["variants"]}
    assert not missing, sorted(missing)
    representation = stage.paired(rows, "S/joint", "G/joint")
    fusion = stage.paired(rows, "S/joint", "S/independent")
    return {
        "identities": len({r["identity"] for r in rows}), "samples": len(rows),
        "medians": {name: stage.sample_medians(rows, name) for name in names},
        "T1_representation": {"verdict": stage.verdict(representation["sum"]),
                              "comparison": representation},
        "T2_modalities": {m: stage.verdict(representation[m]) for m in ("cmr", "tee")},
        "T3_fusion_cost": {"cmr": label(fusion["cmr"]), "tee": label(fusion["tee"]),
                           "comparison": fusion},
        # T4 as E64 defined it: S joint against each other S arm, and S against G
        # late fusion; E64/E75/E85 also reported G's own fusion cost.
        "T4_competitors": {arm: stage.paired(rows, "S/joint", f"S/{arm}")
                           for arm in ("exact_no_gauges", "soft_matched", "soft_matched_1e4",
                                       "exact_repair", "penalty_repair")}
                          | {"S_vs_G_independent": stage.paired(rows, "S/independent", "G/independent"),
                             "G_fusion_cost": stage.paired(rows, "G/joint", "G/independent")},
        "rooted4d_minus_trajectory": {k: identity_difference(stage, rows, "S/joint", TRAJECTORY, k)
                                      for k in ENDPOINTS},
    }


#: The paired differences the pre-registration summarizes across test sets:
#: (name, path into an instance entry).
KEY_DIFFERENCES = (
    [(f"late_fusion_{m}", ("T3_fusion_cost", "comparison", m)) for m in ("sum", "cmr", "tee")]
    + [(f"trajectory_{k}", ("rooted4d_minus_trajectory", k))
       for k in ("rotation_deg", "translation_mm", "phase_map_cycles", "attachment_residual_mm")]
    + [("global_basis_sum", ("T1_representation", "comparison", "sum"))])


def _entry(instance: dict, path: tuple) -> dict:
    for key in path:
        instance = instance[key]
    return instance


def summarize(instances: dict) -> dict:
    """Ranges and interval counts over each benchmark's six test sets, and the abstract rule."""

    summary = {}
    for bench in SOURCES:
        members = {name: entry for name, entry in instances.items()
                   if name.startswith(bench + "s") or name == f"{bench}_original"}
        members = {name: entry for name, entry in members.items() if "T1_representation" in entry}
        record = {"test_sets": sorted(members), "differences": {}}
        for name, path in KEY_DIFFERENCES:
            values = {m: _entry(e, path) for m, e in members.items()}
            observed = [v["observed"] for v in values.values()]
            record["differences"][name] = {
                "observed_min": min(observed), "observed_max": max(observed),
                "below_zero": sum(v["upper"] < 0 for v in values.values()),
                "above_zero": sum(v["lower"] > 0 for v in values.values()),
                "includes_zero": sum(v["lower"] <= 0 <= v["upper"] for v in values.values()),
                "per_test_set": values,
            }
        new = {m: e for m, e in members.items() if not m.endswith("_original")}
        original = members.get(f"{bench}_original")
        record["abstract_rule"] = None if original is None or len(new) < len(SPLIT_SEEDS) else {
            "no_cost_remains": all(e["T3_fusion_cost"][m] != "cost remains"
                                   for e in new.values() for m in ("cmr", "tee")),
            "same_sign_as_original": all(
                np.sign(e["T3_fusion_cost"]["comparison"]["sum"]["observed"])
                == np.sign(original["T3_fusion_cost"]["comparison"]["sum"]["observed"])
                for e in new.values()),
        }
        summary[bench] = record
    return summary


def cmd_analyze(chain_root: str, *originals: str) -> None:
    """Every chain's test, and the original split's (``v4=<dirs,...>``, ``v3=<dirs,...>``)."""

    stage = _load("e62_stage_b")
    report = {"experiment": "E88", "preregistration": PREREGISTRATION,
              "split_seeds": {f"split{k}": s for k, s in enumerate(SPLIT_SEEDS, 1)},
              "original_split_seed": ORIGINAL_SPLIT_SEED,
              "statistics": "E62's: identity-averaged, sum of modality medians, differences of "
                            "medians, %d identity resamples, seed %d" % (stage.RESAMPLES, stage.SEED),
              "instances": {}}
    for bench in SOURCES:
        for k in range(1, len(SPLIT_SEEDS) + 1):
            root = Path(chain_root) / f"{bench}s{k}"
            marker = root / "TEST_DONE"
            if not marker.exists():
                report["instances"][f"{bench}s{k}"] = {"status": "not finished"}
                continue
            # TEST_DONE holds the test directory relative to the chain root.
            directory = root / marker.read_text().strip()
            spec = json.loads((directory / "spec.json").read_text())
            assert spec["benchmark"] == instance(bench, k) and spec["split"] == "test", spec["benchmark"]
            rows = stage.load_rows([str(directory)])
            assert len(rows) == 60, (bench, k, len(rows))
            entry = test_report(stage, rows)
            entry["session_weights_changed"] = {
                arm: a["selected"]
                for arm, a in json.loads((root / "session_selection.json").read_text())["arms"].items()
                if a["changed"]}
            report["instances"][f"{bench}s{k}"] = entry
    for argument in originals:
        bench, directories = argument.split("=", 1)
        rows = stage.load_rows(directories.split(","))
        assert len(rows) == 60, (bench, len(rows))
        report["instances"][f"{bench}_original"] = test_report(stage, rows)
    report["summary"] = summarize(report["instances"])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    for name, entry in report["instances"].items():
        if "T1_representation" in entry:
            fusion = entry["T3_fusion_cost"]["comparison"]["sum"]
            print(name, "T1", entry["T1_representation"]["verdict"],
                  "| joint - late fusion (sum) %+.3f [%+.3f, %+.3f]"
                  % (fusion["observed"], fusion["lower"], fusion["upper"]))
        else:
            print(name, entry["status"])
    print("abstract rule:", {b: r["abstract_rule"] for b, r in report["summary"].items()})
    print("wrote", REPORT)


def cmd_figure(output: str) -> None:
    """The pre-registered strip plot: each key difference over each benchmark's test sets."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(REPORT.read_text())["summary"]
    fig, axes = plt.subplots(len(SOURCES), len(KEY_DIFFERENCES),
                             figsize=(2.0 * len(KEY_DIFFERENCES), 2.4 * len(SOURCES)), squeeze=False)
    for row, bench in enumerate(SOURCES):
        for column, (name, _) in enumerate(KEY_DIFFERENCES):
            ax = axes[row, column]
            values = summary[bench]["differences"][name]["per_test_set"]
            for position, (test_set, value) in enumerate(sorted(values.items())):
                original = test_set.endswith("_original")
                ax.errorbar(position, value["observed"],
                            yerr=[[value["observed"] - value["lower"]], [value["upper"] - value["observed"]]],
                            fmt="s" if original else "o", color="#b5553f" if original else "#4f7fb5",
                            capsize=2, markersize=4)
            ax.axhline(0.0, color="0.6", linewidth=0.8)
            ax.set_xticks([])
            ax.set_title(name.replace("_", " "), fontsize=7)
            if column == 0:
                ax.set_ylabel({"v4": "realistic session", "v3": "small session"}[bench], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(output)
    print("wrote", output)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"configs": cmd_configs, "chain": cmd_chain, "analyze": cmd_analyze,
     "figure": cmd_figure}[command](*arguments)
