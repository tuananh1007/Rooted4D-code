"""E60: how much of the soft-attachment penalty is the renormalisation?

The published ladder (E51, v3 test) puts soft attachment 0.517 mm (CMR) and
0.192 mm (TEE) behind the full model, and the withdrawn-claims ledger attributes
most of that gap to the detached arm's extra parameters. A different mechanism
was found on 22 September. SceneModel.detached() renormalises its lifted columns
by ||A B_j||, which moves the identity prior, the motion and session subspaces
and the gauges into different coordinates from the coupled model's, so the
historical soft arm never converges to the exact arm however stiff the penalty.
detached(matched=True) keeps the coupled coordinates and the same number of
separation modes. On six training samples the historical arm was worse than the
matched one by a median 0.54 mm (CMR) and 0.35 mm (TEE), while the matched arm
tracked the exact one. Both held attachment to about 1e-4 mm.

This measures it at scale on the development split, in E51's own configuration:
32 global modes, isotropic identity prior at weight 1, unknown pose and phase
from the historical zero start, penalty 1e4. Test is not touched; the published
test numbers are corrected only if and when that is decided separately.

Arms: rooted4d_full, joint_exact_no_gauges, joint_soft_attachment (historical)
and joint_soft_attachment_matched. Differences are paired at the identity, with
variants averaged first, and 20 000 identity resamples, seed 20260910.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from rooted4d.experiments.benchmark_eval import benchmark_problem, fit_benchmark_model, sample_paths
from rooted4d.inference import SolverConfig
from rooted4d.inference.baselines import run_arm

ROOT = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3"))
SPLIT = os.environ.get("R4D_SPLIT", "val")
OUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E60_soft_arm_renormalisation.json"
ARMS = ("rooted4d_full", "joint_exact_no_gauges", "joint_soft_attachment",
        "joint_soft_attachment_matched")
PAIRS = [("joint_soft_attachment", "joint_soft_attachment_matched"),
         ("joint_soft_attachment", "rooted4d_full"),
         ("joint_soft_attachment_matched", "rooted4d_full")]
RESAMPLES, SEED = 20000, 20260910


def main() -> int:
    if SPLIT == "test":
        raise SystemExit("E60 is a development diagnostic; it does not run on test")
    model, provenance = fit_benchmark_model(ROOT, n_modes=32, n_motion=4, n_session=2)
    config = SolverConfig(fourier_order=3, session_order=1, max_iterations=40)
    per_sample = []
    for path in sample_paths(ROOT, SPLIT):
        with np.load(path, allow_pickle=True) as blob:
            identity = str(blob["identity_id"])
        problem = benchmark_problem(path, model, config)
        row = {"sample": path.name, "identity": identity}
        for arm in ARMS:
            started = time.time()
            result = run_arm(arm, problem, config)
            row[arm] = dict(result.as_dict(), seconds=round(time.time() - started, 2))
        per_sample.append(row)
        print("%s  %s" % (path.name[:24], "  ".join(
            "%s %.3f/%.3f" % (arm.replace("joint_", "")[:14], row[arm]["heldout_cmr_mm"],
                              row[arm]["heldout_tee_mm"]) for arm in ARMS)), flush=True)

    identities = sorted({r["identity"] for r in per_sample})

    def by_identity(arm, key):
        return np.array([np.mean([r[arm][key] for r in per_sample if r["identity"] == i])
                         for i in identities])

    rng_master = SEED
    comparisons = {}
    for a, b in PAIRS:
        entry = {}
        for key in ("heldout_cmr_mm", "heldout_tee_mm", "attachment_residual_mm",
                    "rotation_deg", "translation_mm"):
            diff = by_identity(a, key) - by_identity(b, key)
            rng = np.random.default_rng(rng_master)
            boot = np.array([np.median(diff[rng.integers(0, len(diff), len(diff))])
                             for _ in range(RESAMPLES)])
            entry[key] = {"median_paired_difference": float(np.median(diff)),
                          "lower": float(np.percentile(boot, 2.5)),
                          "upper": float(np.percentile(boot, 97.5)),
                          "identities_first_worse": int((diff > 0).sum())}
        comparisons["%s_minus_%s" % (a, b)] = entry
        print("\n%s - %s" % (a, b))
        for key, v in entry.items():
            print("  %-24s median %+.4f  95%% CI [%+.4f, %+.4f]  first worse on %d/%d"
                  % (key, v["median_paired_difference"], v["lower"], v["upper"],
                     v["identities_first_worse"], len(identities)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "E60_soft_arm_renormalisation",
        "question": "how much of the soft-attachment arm's gap to the exact arm is the "
                    "lifted-column renormalisation rather than soft attachment",
        "configuration": "E51: 32 global modes, isotropic identity prior weight 1, "
                         "unknown pose and phase from the historical zero start, "
                         "penalty 1e4, fourier 3, session 1, max_iterations 40",
        "benchmark": str(ROOT), "split": SPLIT, "n_identities": len(identities),
        "statistic": "median over identities of the paired difference (variants "
                     "averaged), percentile bootstrap over identities",
        "resamples": RESAMPLES, "seed": SEED, "provenance": provenance,
        "comparisons": comparisons, "per_sample": per_sample,
    }, indent=2, default=float) + "\n")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
