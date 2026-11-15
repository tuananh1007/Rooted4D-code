"""E106: why image fits leave the truth (weak point 3).

Pre-registered in ``Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md``.
E70's image-only fits on E100's small-session benchmark at 96 nodes, from the
zero start and from E69's point fit, with their vectors kept; at the point fit,
the warm end and the cold end, every term of the objective is evaluated
separately (CMR image; TEE image, one-sided and symmetric; prior). The same is
done on a no-shadow twin of the instance, built from the same configuration with
``observation.tee_shadow: false``.

    python e106_image_diagnosis.py config <out yaml>                 (the twin's generator config)
    python e106_image_diagnosis.py check <original root> <twin root> -> E106_twin_check.json
    R4D_E69=<E69 test shards> R4D_INSTANCE=original|twin R4D_ALLOW_TEST=1 \\
        python e106_image_diagnosis.py fit <instance root>           -> <R4D_OUT>/E106_<instance>_test/
    python e106_image_diagnosis.py analyze <rows root>               -> results/benchmark/E106/E106_analysis.json

The model is always the original instance's (``frozen_s("v3n96o")``): the twin's
point data are identical, so is every learned quantity.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

PREREGISTRATION = "Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md"
BENCH = "v3n96o"
SOURCE_CONFIG = Path("Codes/configs/e100/v3n96o.yaml")
OUT = Path("results/benchmark/E106")
WRONG_BASIN_DEG = 30.0
#: Cross-machine rounding allowed in the twin check (the amendment of 4 October).
FLOAT_TOLERANCE = 1e-12


# ------------------------------------------------------------ the twin ----

def cmd_config(out_path: str) -> None:
    text = SOURCE_CONFIG.read_text()
    marker = "  tee_speckle_std: 0.12\n"
    assert text.count(marker) == 1
    text = text.replace(marker, marker + "  # E106: the no-shadow twin; everything else is v3n96o's.\n  tee_shadow: false\n")
    header = ("# E106 (Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md): the no-shadow twin of\n"
              "# E100's v3n96o. Written by e106_image_diagnosis.py config; do not edit by hand.\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(header + text)
    print("wrote", out_path)


def cmd_check(original_root: str, twin_root: str) -> None:
    """Every stored array equal except the TEE image, in every split."""

    record = {"samples": 0, "tee_image_differs": 0, "other_differences": []}
    for path in sorted(Path(original_root).glob("samples/*/*.npz")):
        twin = Path(twin_root) / path.relative_to(original_root)
        with np.load(path, allow_pickle=True) as a, np.load(twin, allow_pickle=True) as b:
            if sorted(a.files) != sorted(b.files):
                record["other_differences"].append([str(path.name), "keys"])
                continue
            for key in a.files:
                x, y = a[key], b[key]
                # Amendment: the twin is built on another machine, so float arrays
                # are compared to cross-machine rounding (1e-12), the rest exactly.
                same = (x.shape == y.shape and bool(np.all(np.abs(x - y) <= FLOAT_TOLERANCE))
                        if x.dtype.kind in "fc" and y.dtype.kind in "fc" else np.array_equal(x, y))
                if key == "tee_surface_likelihood":
                    record["tee_image_differs"] += int(not same)
                elif not same:
                    record["other_differences"].append([str(path.name), key])
        record["samples"] += 1
    record["passes"] = not record["other_differences"] and record["samples"] > 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E106_twin_check.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    if not record["passes"]:
        raise SystemExit("the twin differs beyond the TEE image")


# ----------------------------------------------------------------- fits ----

def initialise(*arguments):
    import e70_image_evidence as e70

    e70.initialise(*arguments)


def _terms(vector, scene, weights, one_sided: dict, symmetric: dict) -> dict:
    import torch
    from rooted4d.differentiable.scene import image_residuals, prior_residuals

    v = torch.as_tensor(np.asarray(vector, dtype=np.float64))
    cmr_only, tee_one, tee_sym = dict(symmetric), dict(one_sided), dict(symmetric)
    cmr_only["tee_values"] = None
    tee_one["cmr_values"] = None
    tee_sym["cmr_values"] = None
    with torch.no_grad():
        out = {"cmr_image": float((image_residuals(v, scene, cmr_only) ** 2).sum()),
               "tee_image_one_sided": float((image_residuals(v, scene, tee_one) ** 2).sum()),
               "tee_image_symmetric": float((image_residuals(v, scene, tee_sym) ** 2).sum()),
               "prior": float((prior_residuals(v, scene, weights) ** 2).sum())}
    out["total"] = out["cmr_image"] + out["tee_image_one_sided"] + out["prior"]
    return out


def fit_sample(path_string: str) -> dict:
    import torch
    import e70_image_evidence as e70
    from rooted4d.differentiable import evidence_tensors
    from rooted4d.differentiable.scene import image_residuals, landmark_residuals, prior_residuals
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.inference.image_evidence import evidence_from_sample
    from rooted4d.inference.joint import _pack, initial_parameters

    state = e70.STATE
    model, point, image, scene, weights = (state[k] for k in ("model", "point", "image", "scene", "weights"))
    path = Path(path_string)
    with np.load(path, allow_pickle=True) as blob:
        identity = str(blob["identity_id"])
        true_phases = np.asarray(blob["tee_true_phases"], dtype=float)
        all_times = np.asarray(blob["tee_times"], dtype=float)
    problem = benchmark_problem(path, model, point)
    record = evidence_from_sample(path, model, budget=200, n_cmr_frames=4, n_tee_frames=4)
    phases = true_phases[np.searchsorted(all_times, record.tee_times)]
    symmetric = evidence_tensors(record, e70.W_CMR, e70.W_TEE)
    one_sided = dict(symmetric)
    one_sided["shadow_tolerance"] = 0.1                     # E70's residual

    def objective(v):
        return torch.cat([image_residuals(v, scene, one_sided), landmark_residuals(v, scene, one_sided),
                          prior_residuals(v, scene, weights)])

    warm = torch.as_tensor(np.asarray(state["warm"][path.name], dtype=np.float64))
    cold = torch.as_tensor(_pack(initial_parameters(model, image), model, image))
    row = {"sample": path.name, "identity": identity, "instance": os.environ.get("R4D_INSTANCE"),
           "variants": {"point_fit": e70._score(warm.numpy(), problem, record.tee_times, phases)}}
    row["variants"]["point_fit"]["terms"] = _terms(warm.numpy(), scene, weights, one_sided, symmetric)
    for name, start in (("image_cold", cold), ("image_warm", warm)):
        started = time.perf_counter()
        try:
            vector = e70._lbfgs(start, objective).numpy()
            outcome = e70._score(vector, problem, record.tee_times, phases)
            outcome["terms"] = _terms(vector, scene, weights, one_sided, symmetric)
            outcome["vector"] = vector.tolist()
        except Exception as error:   # a failed arm is recorded, never fatal
            outcome = {"succeeded": False, "failure_reason": f"{type(error).__name__}: {error}"}
        outcome["seconds"] = round(time.perf_counter() - started, 3)
        row["variants"][name] = outcome
    return row


def cmd_fit(instance_root: str) -> int:
    import e59_unknown_alignment as e59
    import e70_image_evidence as e70
    from uniformity_runner import frozen_s, run

    instance = os.environ["R4D_INSTANCE"]
    recipe, base, weight, root = frozen_s(BENCH)
    # Without the chain's test spec, frozen_s falls back to an older configuration.
    spec = Path("Codes/configs/e100") / BENCH / f"{BENCH}_e88_test.json"
    assert spec.exists(), spec
    warm = e70.load_warm_starts(os.environ["R4D_E69"])
    paths = [p for p in e59.sample_paths(Path(instance_root), "test") if p.name in warm]
    return run(f"E106_{instance}_test", "test", paths, fit_sample, initialise,
               (recipe, base, weight, root, "e70", warm),
               extra={"experiment": "E106", "preregistration": PREREGISTRATION, "instance": instance,
                      "instance_root": instance_root, "model_root": root, "iterations": e70.ITERATIONS,
                      "warm_starts": os.environ["R4D_E69"], "samples": len(paths)})


# ------------------------------------------------------------- analysis ----

def _rows(directory: Path) -> list:
    return [json.loads(line) for shard in sorted(directory.glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()]


def summarize(rows: list) -> dict:
    ok = [r for r in rows if r["variants"].get("image_cold", {}).get("succeeded")
          and r["variants"].get("image_warm", {}).get("succeeded")]
    cold = np.array([r["variants"]["image_cold"]["rotation_deg"] for r in ok])
    warm = np.array([r["variants"]["image_warm"]["rotation_deg"] for r in ok])
    wrong = [r for r in ok if r["variants"]["image_cold"]["rotation_deg"] > WRONG_BASIN_DEG]
    out = {"samples": len(ok),
           "median_rotation_deg": {"point_fit": float(np.median([r["variants"]["point_fit"]["rotation_deg"] for r in ok])),
                                   "image_cold": float(np.median(cold)), "image_warm": float(np.median(warm))},
           "median_translation_mm": {"image_cold": float(np.median([r["variants"]["image_cold"]["translation_mm"] for r in ok])),
                                     "image_warm": float(np.median([r["variants"]["image_warm"]["translation_mm"] for r in ok]))},
           "cold_beyond_30deg": len(wrong), "cold_beyond_90deg": int((cold > 90).sum()),
           "warm_within_5deg": int((warm < 5).sum())}
    for objective in ("total", "tee_image_symmetric_total"):
        def value(r, arm):
            terms = r["variants"][arm]["terms"]
            return terms["total"] if objective == "total" else (
                terms["cmr_image"] + terms["tee_image_symmetric"] + terms["prior"])
        lower = sum(value(r, "image_cold") < value(r, "point_fit") for r in wrong)
        out[f"R1_{objective}"] = {"wrong_basin_samples": len(wrong), "lower_than_point_fit": int(lower),
                                  "share": float(lower / len(wrong)) if wrong else None}
    out["R2_median_change_cold_minus_point_fit"] = {
        term: float(np.median([r["variants"]["image_cold"]["terms"][term] - r["variants"]["point_fit"]["terms"][term]
                               for r in wrong])) if wrong else None
        for term in ("cmr_image", "tee_image_one_sided", "tee_image_symmetric", "prior", "total")}
    out["R2_share_of_samples_where_term_decreases"] = {
        term: float(np.mean([r["variants"]["image_cold"]["terms"][term] < r["variants"]["point_fit"]["terms"][term]
                             for r in wrong])) if wrong else None
        for term in ("cmr_image", "tee_image_one_sided", "tee_image_symmetric", "prior")}
    return out


def cmd_analyze(rows_root: str) -> None:
    report = {"experiment": "E106", "preregistration": PREREGISTRATION, "wrong_basin_deg": WRONG_BASIN_DEG,
              "instances": {}}
    for instance in ("original", "twin"):
        rows = _rows(Path(rows_root) / f"E106_{instance}_test")
        report["instances"][instance] = summarize(rows)
    original, twin = report["instances"]["original"], report["instances"]["twin"]
    report["code_check"] = {"e70_median_cold_rotation_deg": 104.7,
                            "reproduced_deg": original["median_rotation_deg"]["image_cold"],
                            "reproduces": abs(original["median_rotation_deg"]["image_cold"] - 104.7) < 0.1}
    share_o, share_t = original["R1_total"]["share"], twin["R1_total"]["share"]
    report["reading"] = {
        "R1": ("the objective prefers wrong poses" if share_o is not None and share_o >= 0.5
               else "the fit fails to reach the truth's basin"),
        "R3": ("the shadow accounts for it" if share_o and share_t is not None and share_t <= share_o / 2
               else "the shadow does not account for it"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E106_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "config":
        cmd_config(sys.argv[2])
    elif command == "check":
        cmd_check(sys.argv[2], sys.argv[3])
    elif command == "fit":
        sys.exit(cmd_fit(sys.argv[2]))
    elif command == "analyze":
        cmd_analyze(sys.argv[2])
    else:
        sys.exit(f"unknown command {command!r}")
