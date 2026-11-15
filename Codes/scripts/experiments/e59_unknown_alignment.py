r"""E59 driver: shape models and competitors under unknown pose and phase.

The reassessment's §4B asks for the strongest global estimator and the supported
structured candidate (E58: R1) to be taken through unknown pose and phase from
an identical input-derived start, against strengthened competitors. This
driver runs that, and nothing about the comparison is hard-coded: models, base
solver settings and arm variants all come from a JSON spec, so a tuning sweep
and a frozen comparison are the same code reading different specs.

For every sample:

* the problem is built once per model (observations and masks are identical
  across models; only the unscored truth projection differs);
* the start alignment is computed **once, from the observations alone**, and
  handed to every variant, so no two models or arms begin differently and none
  is given the truth; its own error against truth is recorded;
* every variant runs in isolation: an exception becomes a recorded failure with
  its reason, never an aborted run, and every variant is timed.

Results are appended as one JSON line per sample to a per-shard file, and a
restarted job skips samples already present, so a killed job resumes rather
than repeats. Sharding (``R4D_SHARD=k/n``) takes every n-th sample from k, and
``R4D_WORKERS`` runs samples in a spawn-context pool with BLAS pinned to one
thread per worker; the fork-pool deadlock and the 96 GB OOM recorded for the
TetHeart rescore are why it is spawn and why memory is budgeted per worker.

Cross-validation. With ``"cv": {"folds": K, "seed": S}`` and ``"split": "train"``,
the training identities are permuted with seed S and split into K folds. Each
fold's models are learned from the other folds' identities only, and every
sample is scored by the models of its own fold. Every identity is therefore
scored by bases that never saw it, and no selection is made on these scores.
Identities are composites of one chamber donor and one valve donor, and no
donor appears in two training identities, so grouping by identity is grouping
by donor.

Spec format (JSON)::

    {"benchmark": "data/derived/semisynthetic-real-v3", "split": "val",
     "models": {"global": {"builder": "global", "n_modes": 48},
                "structured": {"builder": "structured", "interface_dim": 12,
                               "interior_dim": 12, "coordinates": "covariance"}},
     "base_config": {"fourier_order": 3, "session_order": 1, "max_iterations": 40,
                     "covariance_aware_identity_prior": true,
                     "identity_prior_scale_floor": 0.005},
     "variants": [{"name": "global/joint/w4", "model": "global",
                   "arm": "rooted4d_full",
                   "overrides": {"identity_prior_weight": 4.0},
                   "soft_attachment_weight": 1e4}]}
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

from rooted4d.experiments.benchmark_eval import (  # noqa: E402
    benchmark_problem,
    fit_benchmark_model,
    sample_paths,
)
from rooted4d.experiments.structured_model import fit_structured_benchmark_model  # noqa: E402
from rooted4d.inference import SolverConfig  # noqa: E402
from rooted4d.inference.baselines import run_arm  # noqa: E402
from rooted4d.inference.initialization import input_derived_alignment, start_from  # noqa: E402
from rooted4d.inference.problem import pose_errors  # noqa: E402
from rooted4d.motion import circular_distance  # noqa: E402

_MODELS: dict = {}
_SPEC: dict = {}
_FOLDS: dict = {}


def build_model(root: Path, recipe: dict, exclude: frozenset | None = None):
    builder = recipe["builder"]
    common = dict(n_motion=recipe.get("n_motion", 4), n_session=recipe.get("n_session", 2),
                  n_phase_nodes=recipe.get("n_phase_nodes", 8), exclude_identities=exclude)
    if builder == "global":
        return fit_benchmark_model(root, n_modes=recipe["n_modes"], **common)
    if builder == "structured":
        return fit_structured_benchmark_model(
            root, interface_dim=recipe["interface_dim"], interior_dim=recipe["interior_dim"],
            coordinates=recipe.get("coordinates", "assembly"), **common)
    raise ValueError(f"unknown model builder {builder!r}")


def spec_digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def fold_assignment(spec: dict) -> dict[str, int]:
    """Identity -> fold, from a seeded permutation of the split's identities."""

    cv = spec["cv"]
    identities = set()
    for path in sample_paths(Path(spec["benchmark"]), spec["split"]):
        with np.load(path, allow_pickle=True) as blob:
            identities.add(str(blob["identity_id"]))
    order = sorted(identities)
    permuted = [order[i] for i in np.random.default_rng(int(cv["seed"])).permutation(len(order))]
    return {identity: fold
            for fold, members in enumerate(np.array_split(np.array(permuted), int(cv["folds"])))
            for identity in members.tolist()}


def _initialise(spec: dict) -> None:
    """Build the models once per process; workers call this at start-up.

    Without cross-validation there is one model per key. With it there is one
    per (fold, key), each learned without that fold's identities.
    """

    global _SPEC, _FOLDS
    _SPEC = spec
    # ``model_benchmark`` learns the bases from another instance's training
    # split, so an intervention instance (E92) is fitted with the original
    # benchmark's models; absent, the models come from the benchmark itself.
    root = Path(spec.get("model_benchmark", spec["benchmark"]))
    if "cv" not in spec:
        for key, recipe in spec["models"].items():
            _MODELS[key] = build_model(root, recipe)[0]
        return
    if spec.get("split") != "train":
        raise ValueError("cross-validation is defined on the train split only")
    _FOLDS = fold_assignment(spec)
    for fold in sorted(set(_FOLDS.values())):
        held_out = frozenset(i for i, f in _FOLDS.items() if f == fold)
        for key, recipe in spec["models"].items():
            _MODELS[(fold, key)] = build_model(root, recipe, held_out)[0]


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _start(spec: dict, reference, model, base: SolverConfig):
    """The start alignment every variant shares, and a record of it.

    ``"input_derived"`` (the default) estimates pose and phase from the
    observations alone and records its own error against truth. ``"zero"`` is the
    historical start, kept only to reproduce results produced before the
    input-derived start existed (E51), never for a new comparison. An unusable
    input-derived start is returned as None with its reason, and every variant
    then fails visibly rather than silently starting from zero.
    """

    if spec.get("start", "input_derived") == "zero":
        return None, {"kind": "zero"}
    try:
        alignment = input_derived_alignment(reference.observations)
    except Exception as error:
        return None, {"kind": "input_derived", "failure": f"{type(error).__name__}: {error}"}
    errors = pose_errors(start_from(alignment, model, base), reference.truth)
    return alignment, {
        "kind": "input_derived",
        "rotation_deg": errors["rotation_deg"], "translation_mm": errors["translation_mm"],
        "phase_offset_cycles": float(circular_distance(
            np.array([alignment.phase_offset]), np.array([reference.truth.phase_offset]))[0]),
        "shared_vertices": alignment.shared_vertices,
    }


def run_sample(path_string: str) -> dict:
    spec = _SPEC
    path = Path(path_string)
    base = SolverConfig(**spec.get("base_config", {}))
    with np.load(path, allow_pickle=True) as blob:
        identity = str(blob["identity_id"])
    row = {"sample": path.name, "identity": identity, "variants": {}}

    if _FOLDS:
        row["fold"] = _FOLDS[identity]
        models = {key: _MODELS[(row["fold"], key)] for key in spec["models"]}
    else:
        models = {key: _MODELS[key] for key in spec["models"]}
    # ``observation_thinning`` (E113) keeps every k-th visible frame of a modality;
    # absent, the problem is the full one, as every earlier spec built it.
    problems = {key: benchmark_problem(path, model, base, thinning=spec.get("observation_thinning"))
                for key, model in models.items()}
    reference = next(iter(problems.values()))
    alignment, row["start"] = _start(spec, reference, next(iter(models.values())), base)

    for variant in spec["variants"]:
        name = variant["name"]
        started = time.perf_counter()
        try:
            if (alignment is None and spec.get("start", "input_derived") != "zero"
                    and not variant.get("allow_zero_start", False)):
                raise RuntimeError("no input-derived start for this sample")
            config = replace(base, **variant.get("overrides", {}))
            result = run_arm(
                variant["arm"], problems[variant["model"]], config,
                soft_attachment_weight=variant.get("soft_attachment_weight", 1e4),
                alignment=alignment, options=variant.get("options"))
            outcome = result.as_dict()
        except Exception as error:
            outcome = {"succeeded": False,
                       "failure_reason": f"exception: {type(error).__name__}: {error}",
                       "traceback": traceback.format_exc(limit=4)}
        outcome["seconds"] = round(time.perf_counter() - started, 3)
        row["variants"][name] = outcome
    return _clean(row)


def main() -> int:
    spec_path = Path(os.environ["R4D_SPEC"])
    spec = json.loads(spec_path.read_text())
    digest = spec_digest(spec)
    if spec.get("split") == "test" and os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing to run on test without R4D_ALLOW_TEST=1")
    out = Path(os.environ.get("R4D_OUT", "results/benchmark/e59")) / digest
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")

    k, n = (int(v) for v in os.environ.get("R4D_SHARD", "0/1").split("/"))
    paths = sample_paths(Path(spec["benchmark"]), spec["split"])[k::n]
    shard = out / f"shard_{k:03d}_of_{n:03d}.jsonl"
    done = set()
    if shard.exists():
        for line in shard.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["sample"])
    todo = [str(p) for p in paths if p.name not in done]
    print("spec %s | shard %d/%d | %d samples, %d already done, %d to run"
          % (digest, k, n, len(paths), len(done), len(todo)), flush=True)

    workers = int(os.environ.get("R4D_WORKERS", "1"))
    started = time.time()
    with shard.open("a") as handle:
        if workers > 1:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with context.Pool(workers, initializer=_initialise, initargs=(spec,)) as pool:
                for row in pool.imap_unordered(run_sample, todo):
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print("%s done (%.0fs)" % (row["sample"], time.time() - started), flush=True)
        else:
            _initialise(spec)
            for path in todo:
                row = run_sample(path)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print("%s done (%.0fs)" % (row["sample"], time.time() - started), flush=True)
    print("finished %d samples in %.0fs -> %s" % (len(todo), time.time() - started, shard))
    return 0


if __name__ == "__main__":
    sys.exit(main())
