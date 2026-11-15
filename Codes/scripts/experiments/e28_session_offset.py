"""E28: the session term with the parameter it was missing.

Section 15 found the session residual worth 0.005 mm even at a realistic
magnitude, and attributed it to identity and pose absorbing the discrepancy.
That attribution was incomplete. Chasing it further found a structural
deficiency: the session residual was parameterized as
``periodic_basis(tau, session_order) @ session``, which is pure cosine and sine
and therefore has zero mean over the cycle. The model had **no parameter for a
static offset between two sessions**.

MITEA measures real between-session discrepancy at 2.90 mm rms static against
1.01 mm dynamic. The dominant component of the real effect was exactly the one
the parameterization could not express, so the term was being asked to explain a
3 mm static offset with a basis that integrates to zero.

``session_includes_offset`` adds the constant term. Three arms:

* ``no_session``      - the term switched off entirely;
* ``session_periodic`` - the original parameterization, oscillatory only;
* ``session_offset``   - with the constant term, which is the corrected model.

If the diagnosis is right, the third should separate from the first two and the
first two should remain close together, since a basis that cannot represent the
dominant component is not much better than no basis at all.
"""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rooted4d.experiments.benchmark_eval import (
    _summary,
    benchmark_problem,
    fit_benchmark_model,
    representation_error,
    sample_paths,
)
from rooted4d.inference.baselines import _score
from rooted4d.inference.joint import SolverConfig, fit_joint_scene

ROOT = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v4"))
BASE = SolverConfig(
    fourier_order=2, session_order=1, motion_prior_weight=1.0, max_iterations=40
)

print("fitting shape space...", flush=True)
model, _ = fit_benchmark_model(ROOT, n_modes=32, n_motion=4, n_session=2)
floors = representation_error(model, ROOT, "test")
print("floors: whole %.2f cmr %.2f tee %.2f"
      % (floors["rms_mm"], floors["rms_mm_cmr_visible"], floors["rms_mm_tee_visible"]),
      flush=True)

ARMS = {
    "no_session": replace(BASE, use_session=False),
    "session_periodic": BASE,
    "session_offset": replace(BASE, session_includes_offset=True),
}

rows = []
paths = sample_paths(ROOT, "test")
t0 = time.time()
for index, path in enumerate(paths, 1):
    for name, cfg in ARMS.items():
        problem = benchmark_problem(path, model, cfg)
        fit = fit_joint_scene(problem.observations, model, cfg)
        record = _score(name, fit, problem).as_dict()
        record["sample"] = path.stem
        rows.append(record)
    if index % 15 == 0:
        print("  %d/%d  %.0fs" % (index, len(paths), time.time() - t0), flush=True)

KEYS = ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm",
        "phase_error_cycles")
per_arm = {
    name: {
        "failures": sum(1 for r in rows
                        if r["baseline_id"] == name and not r["succeeded"]),
        **{k: _summary([r[k] for r in rows if r["baseline_id"] == name]) for k in KEYS},
    }
    for name in ARMS
}

by = {}
for row in rows:
    by.setdefault(row["baseline_id"], {})[row["sample"]] = row
samples = sorted(by["no_session"])


def paired(left, right, key):
    x = np.array([by[left][s][key] for s in samples], dtype=float)
    y = np.array([by[right][s][key] for s in samples], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    diff = (x - y)[mask]
    rng = np.random.default_rng(0)
    obs = float(diff.mean())
    null = (rng.choice([-1.0, 1.0], size=(20000, diff.size)) * diff).mean(axis=1)
    boot = diff[rng.integers(0, diff.size, (10000, diff.size))].mean(axis=1)
    return {"difference": obs,
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p": float((np.abs(null) >= abs(obs)).mean())}


COMPARISONS = {
    "periodic_minus_none": ("session_periodic", "no_session"),
    "offset_minus_none": ("session_offset", "no_session"),
    "offset_minus_periodic": ("session_offset", "session_periodic"),
}
comparisons = {
    label: {k: paired(a, b, k) for k in KEYS}
    for label, (a, b) in COMPARISONS.items()
}
json.dump({
    "experiment": "E28_session_offset_term",
    "benchmark": str(ROOT), "split": "test", "n_modes": 32,
    "floors_mm": floors, "per_arm": per_arm, "comparisons": comparisons, "rows": rows,
}, open(os.environ.get("R4D_OUT", "results/benchmark") + "/E28_session_offset.json", "w"), indent=2)

print("\n%-18s %12s %12s %10s %10s %11s"
      % ("arm", "cmr", "tee", "rot", "trans", "phase"), flush=True)
for name in ARMS:
    v = per_arm[name]
    print("%-18s %12.4f %12.4f %10.3f %10.3f %11.4f"
          % (name, v["heldout_cmr_mm"]["median"], v["heldout_tee_mm"]["median"],
             v["rotation_deg"]["median"], v["translation_mm"]["median"],
             v["phase_error_cycles"]["median"]), flush=True)
print("\npaired differences (negative favours the first named arm)", flush=True)
for label, table in comparisons.items():
    for key, entry in table.items():
        print("  %-22s %-20s %9.4f  [%8.4f,%8.4f] p=%.4f%s"
              % (label, key, entry["difference"], entry["ci"][0], entry["ci"][1],
                 entry["p"], " *" if entry["p"] < 0.05 else ""), flush=True)
print("done", flush=True)
