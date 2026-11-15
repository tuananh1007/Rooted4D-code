"""E107: what the session residual can identify (weak point 5).

Pre-registered in ``Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md``.
At each of E92's base-level S/joint fits on the realistic-session test samples
(96 nodes), the evidence Jacobian (observation rows, central differences, as the
alpha diagnostic computes it) gives the first-order null space. Its
session-coefficient components, taken as the cross-session difference ``d``
(TEE minus CMR coefficients; the common part is the session gauge), span the
unidentifiable subspace U; the identifiable coefficients are its orthogonal
complement. Against the sample's true session field:

* R_rep: the fraction reproduced by the best coefficients d* (least squares);
* R_id:  the fraction reproduced by d* projected onto the identifiable complement;
* measured: E92's recovered fraction, recomputed from the fit with E92's formula.

    python e107_session_identifiability.py <E92 base rows dir> -> results/benchmark/E107/E107_session.json

Runs on Delta: the realistic-session fields are derived from MITEA.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md"
OUT = Path("results/benchmark/E107")
PHASES = np.linspace(0.0, 1.0, 32, endpoint=False)
NULL_TOLERANCE = 1e-6


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(model, config) -> dict:
    from rooted4d.inference.joint import motion_rows, session_rows

    cursor = 0
    blocks = {}
    for name, size in (("identity", model.n_modes), ("motion", motion_rows(config) * model.bases.n_motion)):
        blocks[name] = (cursor, cursor + size)
        cursor += size
    per_session = session_rows(config) * model.bases.n_session
    for session in model.sessions:
        blocks[f"session_{session}"] = (cursor, cursor + per_session)
        cursor += per_session
    return blocks


def _with_difference(parameters, d: np.ndarray, config, model):
    from rooted4d.inference.joint import session_rows

    shape = (session_rows(config), model.bases.n_session)
    return replace(parameters, session={"cmr": (-0.5 * d).reshape(shape), "tee": (0.5 * d).reshape(shape)})


def _session_field(parameters, model, config, n_vertices: int, session: str) -> np.ndarray:
    from rooted4d.inference.joint import latent_trajectory

    rows = model.observation_rows(np.arange(n_vertices))
    with_session = model.physical_coordinates(latent_trajectory(parameters, PHASES, session, model, config))[:, rows]
    shared = model.physical_coordinates(
        latent_trajectory(parameters, PHASES, session, model, replace(config, use_session=False)))[:, rows]
    return with_session - shared


def _nonrigid_ring_share(field: np.ndarray, truth: np.ndarray, ring: np.ndarray) -> float:
    """Share of the field's energy in non-rigid displacement of the ring, phase by phase."""

    residual = 0.0
    for phase in range(len(PHASES)):
        x = truth[phase, ring]
        u = field[phase, ring]
        centre = x.mean(axis=0)
        # Small rigid motion: u ~ omega x (x - c) + t, least squares in (omega, t).
        rows = []
        for point in x - centre:
            skew = np.array([[0.0, point[2], -point[1]], [-point[2], 0.0, point[0]], [point[1], -point[0], 0.0]])
            rows.append(np.hstack([skew, np.eye(3)]))
        design = np.vstack(rows)
        coefficients, *_ = np.linalg.lstsq(design, u.reshape(-1), rcond=None)
        residual += float(((u.reshape(-1) - design @ coefficients) ** 2).sum())
    return residual / float((field ** 2).sum())


def measure(row: dict, sample: dict, model, config) -> dict:
    from rooted4d.experiments.benchmark_eval import benchmark_problem
    from rooted4d.experiments.fitted_geometry import truth_trajectory
    from rooted4d.inference import joint
    from rooted4d.inference.baselines import parameters_from_record

    parameters = parameters_from_record(row["variants"]["S/joint"]["extras"]["parameters"])
    n = len(sample["template_vertices"])
    field = (sample["session_static"][None]
             + np.cos(2 * np.pi * PHASES)[:, None, None] * sample["session_dynamic"][None])
    target = 0.5 * field                                  # the generator adds +field/2 to TEE

    # Measured: E92's formula on the fit itself (both sessions, averaged).
    fractions = []
    for session, sign in (("cmr", -0.5), ("tee", 0.5)):
        estimate = _session_field(parameters, model, config, n, session)
        fractions.append(1.0 - ((estimate - sign * field) ** 2).sum() / ((sign * field) ** 2).sum())
    measured = float(np.mean(fractions))

    # The evidence Jacobian at the fit, observation rows only.
    problem = benchmark_problem(Path(row["_path"]), model, config)
    packed = joint._pack(parameters, model, config)

    def residual(vector):
        return joint._residuals(vector, parameters, problem.observations, model, config)

    residual(packed)
    rows = joint._OBSERVATION_ROWS["n"]
    jacobian = joint._numerical_jacobian(residual, packed, config.finite_difference_step)[:rows]
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=True)
    rank = int((singular > NULL_TOLERANCE * singular[0]).sum())
    null = vt[rank:]                                       # rows span the null space
    layout = _layout(model, config)
    a, b = layout["session_cmr"]
    c, e = layout["session_tee"]
    difference_components = null[:, c:e] - null[:, a:b]   # each null vector's d = tee - cmr
    k = e - c
    if difference_components.size:
        u, s, _ = np.linalg.svd(difference_components.T, full_matrices=False)
        unidentified = u[:, s > NULL_TOLERANCE * max(1.0, s.max())]
    else:
        unidentified = np.zeros((k, 0))
    projector = np.eye(k) - unidentified @ unidentified.T

    # Field produced by each unit difference coefficient (linear in d).
    zero = _session_field(_with_difference(parameters, np.zeros(k), config, model), model, config, n, "tee")
    columns = []
    for j in range(k):
        unit = np.zeros(k)
        unit[j] = 1.0
        columns.append((_session_field(_with_difference(parameters, unit, config, model), model, config, n, "tee")
                        - zero).reshape(-1))
    design = np.stack(columns, axis=1)
    flat = target.reshape(-1)
    best, *_ = np.linalg.lstsq(design, flat, rcond=None)
    energy = float((flat ** 2).sum())
    r_rep = 1.0 - float(((design @ best - flat) ** 2).sum()) / energy
    r_id = 1.0 - float(((design @ (projector @ best) - flat) ** 2).sum()) / energy

    truth = truth_trajectory(sample)(PHASES)
    ring = np.sort(model.groups["annulus"])
    return {"sample": row["sample"], "identity": row["identity"], "measured": measured,
            "R_rep": r_rep, "R_id": r_id, "unidentified_dimension": int(unidentified.shape[1]),
            "session_coefficients": int(k), "jacobian_rank": rank, "parameters": int(len(packed)),
            "ring_share": float((field[:, ring] ** 2).sum() / (field ** 2).sum()),
            "nonrigid_ring_share": _nonrigid_ring_share(field, truth, ring)}


def main(rows_dir: str) -> None:
    from rooted4d.inference import SolverConfig

    spec = json.loads((Path(rows_dir) / "spec.json").read_text())
    driver = _load("e59_driver", HERE / "e59_unknown_alignment.py")
    model, *_ = driver.build_model(Path(spec["benchmark"]), spec["models"]["S"])
    joint_variant = next(v for v in spec["variants"] if v["name"] == "S/joint")
    config = replace(SolverConfig(**spec["base_config"]), **joint_variant["overrides"])
    raw = [json.loads(line) for shard in sorted(Path(rows_dir).glob("shard_*.jsonl"))
           for line in shard.read_text().splitlines() if line.strip()]
    records = []
    for row in raw:
        if not row["variants"].get("S/joint", {}).get("succeeded"):
            continue
        path = Path(spec["benchmark"]) / "samples" / "test" / row["sample"]
        row["_path"] = str(path)
        with np.load(path, allow_pickle=True) as blob:
            sample = {key: blob[key] for key in blob.files}
        record = measure(row, sample, model, config)
        records.append(record)
        print(row["sample"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()
                              if k not in ("sample", "identity")}, flush=True)
    medians = {k: float(np.median([r[k] for r in records]))
               for k in ("measured", "R_rep", "R_id", "ring_share", "nonrigid_ring_share")}
    parts = {"outside_session_span": float(np.median([1 - r["R_rep"] for r in records])),
             "inside_but_unidentifiable": float(np.median([r["R_rep"] - r["R_id"] for r in records])),
             "identifiable_not_recovered": float(np.median([r["R_id"] - r["measured"] for r in records]))}
    largest = max(parts, key=parts.get)
    reading = {"outside_session_span": "representation limits the session term",
               "inside_but_unidentifiable": "identifiability limits the session term",
               "identifiable_not_recovered": "estimation limits the session term"}[largest]

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values))
        out[order] = np.arange(len(values))
        return out

    rho = float(np.corrcoef(ranks(np.array([r["R_id"] for r in records])),
                            ranks(np.array([r["measured"] for r in records])))[0, 1])
    report = {"experiment": "E107", "preregistration": PREREGISTRATION, "rows": rows_dir,
              "benchmark": spec["benchmark"], "samples": len(records), "null_tolerance": NULL_TOLERANCE,
              "medians": medians, "parts_of_unrecovered": parts, "reading": reading,
              "spearman_R_id_measured": rho,
              "unidentified_dimension": sorted({r["unidentified_dimension"] for r in records}),
              "per_sample": records}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E107_session.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
