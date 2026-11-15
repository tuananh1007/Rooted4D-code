"""E120: the image studies rerun with the session gauge applied, against the stored ones.

Pre-registered in ``Docs/E120_IMAGE_GAUGE_RERUN_PREREGISTRATION_2026-11-13.md``, with its
amendment. Both row roots are laid out as ``<root>/<key>/components/<study>_<key>_test``;
the rerun root also holds ``e106/E106_<instance>_test``, ``e30/E30_image_torch.json`` and
``e32/E32_image_landmarks.json`` when those stages have finished.

    python e120_analysis.py <stored rows root> <rerun rows root>
        -> results/benchmark/E120/E120_comparison.json   (stored against rerun, expectations X1-X4)
        -> results/benchmark/E120/E120_studies.json      (each study's own analysis of the rerun)

Each study's own analysis is the code that produced the stored report (``e68_e72_analysis``,
``e74_analysis``, ``e91_e96_analysis``, ``e97_image_late_fusion``, ``e106_image_diagnosis``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E120_IMAGE_GAUGE_RERUN_PREREGISTRATION_2026-11-13.md"
OUT = Path("results/benchmark/E120")
STORED_E30 = Path("results/21900728/E30_image_torch.json")
STORED_E32 = Path("results/21920159/E32_image_landmarks.json")
STORED_E106 = Path("results/benchmark/E106/E106_analysis.json")
METRICS = ("rotation_deg", "translation_mm", "phase_map_cycles")

#: (key, study) -> its directory name; every study E120 refits.
STUDIES = {("v3n96o", "E70"): "E70_v3n96o_test", ("v3n96o", "E71"): "E71_v3n96o_test",
           ("v3n96o", "E74"): "E74_v3n96o_test", ("v3n96o", "E91"): "E91_v3n96o_test",
           ("v3n96o", "E96"): "E96_v3n96o_test", ("v4n96o", "E91"): "E91_v4n96o_test"}
#: Arms copied from E69's point fit, not refitted: they must reproduce exactly (code check).
POINT_ARMS = {"point_fit"}
#: X1: warm-started, landmark and commissure arms (the amendment excludes only the two
#: preliminary warm image-only arms, which are compared below but not under X1).
X1_ARMS = {"E70": ("image_warm", "landmarks_2", "landmarks_4", "landmarks_8"),
           "E71": ("cmr_image_tee_point", "cmr_point_tee_image", "both_image"),
           "E74": ("landmarks_2_cold", "landmarks_4_cold", "landmarks_8_cold",
                   "commissures_cold", "commissures_warm"),
           "E91": ("commissures_search", "commissures_uninformed"),
           "E96": ("noise_0mm", "noise_1mm", "noise_2mm", "noise_4mm", "noise_8mm",
                   "swap", "slide_one", "one_only")}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    spec.loader.exec_module(module)
    return module


def rows_of(directory: Path) -> dict:
    """Sample -> variants, from every shard of a directory."""

    out = {}
    for shard in sorted(directory.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["sample"]] = row["variants"]
    return out


def _value(variants: dict, arm: str, metric: str):
    entry = variants.get(arm) or {}
    return float(entry[metric]) if entry.get("succeeded") and entry.get(metric) is not None else None


def compare_rows(stored: dict, rerun: dict) -> dict:
    """Per arm: medians before and after, counts, and per-sample changes (rerun minus stored)."""

    samples = sorted(set(stored) & set(rerun))
    arms = [a for a in next(iter(stored.values())) if a in next(iter(rerun.values()))]
    out = {"samples": len(samples), "missing_from_rerun": len(set(stored) - set(rerun)), "arms": {}}
    for arm in arms:
        entry = {}
        for metric in METRICS:
            pairs = [(_value(stored[s], arm, metric), _value(rerun[s], arm, metric)) for s in samples]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            before, after = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
            change = after - before
            entry[metric] = {"stored_median": float(np.median(before)), "rerun_median": float(np.median(after)),
                             "median_change": float(np.median(after) - np.median(before)),
                             "per_sample_change_median": float(np.median(change)),
                             "per_sample_change_max_abs": float(np.abs(change).max()),
                             "samples_compared": int(len(pairs))}
        rotation = lambda rows, s: _value(rows[s], arm, "rotation_deg")  # noqa: E731
        for label, rows in (("stored", stored), ("rerun", rerun)):
            values = np.array([v for v in (rotation(rows, s) for s in samples) if v is not None])
            entry[f"{label}_within_5deg"] = int((values < 5).sum())
            entry[f"{label}_beyond_30deg"] = int((values > 30).sum())
            entry[f"{label}_beyond_90deg"] = int((values > 90).sum())
        changes = [abs(rotation(rerun, s) - rotation(stored, s)) for s in samples
                   if rotation(rerun, s) is not None and rotation(stored, s) is not None]
        entry["samples_rotation_changed_over_1deg"] = int(sum(c > 1.0 for c in changes))
        entry["failures"] = {"stored": sum(1 for s in samples if not (stored[s].get(arm) or {}).get("succeeded")),
                             "rerun": sum(1 for s in samples if not (rerun[s].get(arm) or {}).get("succeeded"))}
        out["arms"][arm] = entry
    out["point_arms_identical"] = {
        arm: all(_value(stored[s], arm, m) == _value(rerun[s], arm, m) for s in samples for m in METRICS)
        for arm in arms if arm in POINT_ARMS}
    return out


def _json_rows(path: Path, arm_key: str) -> dict:
    """E30's and E32's flat rows as sample -> arm -> metrics."""

    out: dict = {}
    for row in json.loads(path.read_text())["rows"]:
        arm = str(row[arm_key])
        out.setdefault(row["sample"], {})[arm] = dict(row, succeeded=True)
    return out


def main(stored_root: str, rerun_root: str) -> int:
    stored_root, rerun_root = Path(stored_root), Path(rerun_root)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = {"experiment": "E120", "preregistration": PREREGISTRATION, "studies": {}, "missing": []}
    directories = {}
    for (key, study), name in STUDIES.items():
        before = stored_root / key / "components" / name
        after = rerun_root / key / "components" / name
        rows_before, rows_after = rows_of(before), rows_of(after)
        if len(rows_after) < 60:
            comparison["missing"].append(f"{name}: {len(rows_after)} of 60 rerun rows")
            continue
        comparison["studies"][f"{study}_{key}"] = compare_rows(rows_before, rows_after)
        directories[(key, study)] = str(after)

    # Preliminary configuration (E30, E32) and E106.
    for study, stored, rerun, arm_key in (
            ("E30", STORED_E30, rerun_root / "e30" / "E30_image_torch.json", "arm"),
            ("E32", STORED_E32, rerun_root / "e32" / "E32_image_landmarks.json", "landmarks")):
        if rerun.exists():
            comparison["studies"][study] = compare_rows(_json_rows(stored, arm_key), _json_rows(rerun, arm_key))
        else:
            comparison["missing"].append(f"{study}: no rerun report")
    e106 = _load("e106_image_diagnosis")
    e106_report = {}
    for instance in ("original", "twin"):
        rows = e106._rows(rerun_root / "e106" / f"E106_{instance}_test") \
            if (rerun_root / "e106" / f"E106_{instance}_test").exists() else []
        if len(rows) >= 60:
            e106_report[instance] = {"rerun": e106.summarize(rows),
                                     "stored": json.loads(STORED_E106.read_text())["instances"][instance]}
        else:
            comparison["missing"].append(f"E106 {instance}: {len(rows)} of 60 rerun rows")
    if e106_report:
        comparison["studies"]["E106"] = e106_report

    # Each study's own analysis of the rerun rows.
    studies = {"experiment": "E120", "preregistration": PREREGISTRATION}
    images = _load("e68_e72_analysis")
    for study, arms, reference in (
            ("E70", ("point_fit", "image_cold", "image_warm", "landmarks_2", "landmarks_4", "landmarks_8"),
             "image_warm"),
            ("E71", ("point_fit", "cmr_image_tee_point", "cmr_point_tee_image", "both_image"), "point_fit")):
        if ("v3n96o", study) in directories:
            studies[study] = images.analyze_images(directories[("v3n96o", study)], arms, reference)
    if ("v3n96o", "E71") in directories:
        # X3: the TEE image channel against the CMR image channel, by E71's own statistic.
        studies["E71_tee_channel_against_cmr_channel"] = images.analyze_images(
            directories[("v3n96o", "E71")], ("cmr_image_tee_point", "cmr_point_tee_image"),
            "cmr_image_tee_point")["against_cmr_image_tee_point"]["cmr_point_tee_image"]
    if ("v3n96o", "E74") in directories and ("v3n96o", "E70") in directories:
        _load("e74_analysis").main(directories[("v3n96o", "E74")], directories[("v3n96o", "E70")],
                                   str(OUT / "E120_E74.json"))
    search = _load("e91_e96_analysis")
    e91 = {key: directories[(key, "E91")] for key in ("v3n96o", "v4n96o") if (key, "E91") in directories}
    if e91:
        studies["E91"] = search.e91(e91)
    if ("v3n96o", "E96") in directories:
        studies["E96"] = search.e96(directories[("v3n96o", "E96")])
    e97 = {key: (str(stored_root / key / "components" / f"E97_{key}_test"), e91[key]) for key in e91}
    if e97:
        _load("e97_image_late_fusion").cmd_analyze(e97, str(OUT / "E120_E97.json"))

    # Expectations, as fixed before the fits (with the amendment).
    found = comparison["studies"]
    x = {}
    rows_x1 = []
    for study, arms in X1_ARMS.items():
        for name, record in found.items():
            if name.split("_")[0] == study:
                rows_x1 += [(name, arm, record["arms"][arm]["rotation_deg"]["median_change"]) for arm in arms]
    for name in ("E32",):
        if name in found:
            rows_x1 += [(name, arm, found[name]["arms"][arm]["rotation_deg"]["median_change"])
                        for arm in found[name]["arms"] if arm != "0"]
    if "E106" in found:
        for instance, record in found["E106"].items():
            rows_x1.append((f"E106_{instance}", "image_warm",
                            record["rerun"]["median_rotation_deg"]["image_warm"]
                            - record["stored"]["median_rotation_deg"]["image_warm"]))
    x["X1"] = {"arms": [{"study": s, "arm": a, "median_rotation_change_deg": c} for s, a, c in rows_x1],
               "failing": [f"{s}/{a}" for s, a, c in rows_x1 if abs(c) >= 1.0],
               "holds": bool(rows_x1) and all(abs(c) < 1.0 for _, _, c in rows_x1)}
    cold = {name: found[name]["arms"]["image_cold"]["rotation_deg"]["rerun_median"]
            for name in ("E70_v3n96o", "E74_v3n96o") if name in found}
    x["X2"] = {"rerun_median_cold_rotation_deg": cold, "holds": len(cold) == 2 and all(v > 30 for v in cold.values())}
    if "E71_tee_channel_against_cmr_channel" in studies:
        interval = studies["E71_tee_channel_against_cmr_channel"]["rotation_deg"]
        x["X3"] = {"tee_minus_cmr_channel_rotation_deg": interval, "holds": interval["lower"] > 0}
    pairs = []
    if "E74_v3n96o" in found:
        arms = found["E74_v3n96o"]["arms"]
        pairs.append(("E74_v3n96o", "commissures_cold", "image_cold"))
        pairs.append(("E74_v3n96o", "commissures_warm", None))
    for key in ("v3n96o", "v4n96o"):
        if f"E91_{key}" in found:
            pairs += [(f"E91_{key}", "commissures_search", "image_search"),
                      (f"E91_{key}", "commissures_uninformed", "image_uninformed")]
    checks = []
    for name, arm, image_arm in pairs:
        arms = found[name]["arms"]
        value = arms[arm]["rotation_deg"]["rerun_median"]
        image = arms[image_arm]["rotation_deg"]["rerun_median"] if image_arm else None
        checks.append({"study": name, "arm": arm, "rerun_median_rotation_deg": value,
                       "image_only_arm": image_arm, "image_only_median_rotation_deg": image,
                       "holds": value < 15 and (image is None or value < image)})
    x["X4"] = {"arms": checks, "holds": len(checks) == 6 and all(c["holds"] for c in checks)}
    comparison["expectations"] = x
    comparison["code_checks"] = {name: record.get("point_arms_identical")
                                 for name, record in found.items() if isinstance(record, dict)
                                 and record.get("point_arms_identical")}

    (OUT / "E120_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (OUT / "E120_studies.json").write_text(json.dumps(studies, indent=2) + "\n")
    print("wrote", OUT / "E120_comparison.json", "and", OUT / "E120_studies.json")
    print(json.dumps({k: v.get("holds") for k, v in x.items()}), "missing:", comparison["missing"])
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
