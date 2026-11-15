"""Paired comparisons for the E13 ladder, Rooted4D against each comparator.

The paired statistics quoted in the manuscript were originally computed on the
v2 donor pool while the reported table summarizes v3. Mixing the two is not
defensible: the pools differ in the two defects the v3 rebuild corrected, and
one comparison changes sign. This recomputes the tests directly from the per-
sample rows of whichever ladder file is named, so the table and the tests always
describe the same run.

Medians of per-sample paired differences are used rather than differences of
medians, because the two are not equal and only the former is a paired
statistic. Significance is a sign-flip permutation test, which is the exact
null for a paired design under exchangeability of the sign of each difference;
2^n is not enumerable at n = 60 so the null is sampled at a fixed seed.

Usage:
    python e33_paired_ladder.py [ladder.json] [output.json]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

IDENTITIES: dict[str, str] = {}
REFERENCE_ARM = "rooted4d_full"
COMPARATORS = ("late_fusion", "joint_soft_attachment", "joint_exact_no_gauges")
ENDPOINTS = (
    "heldout_cmr_mm",
    "heldout_tee_mm",
    "rotation_deg",
    "translation_mm",
    "phase_error_cycles",
    "attachment_residual_mm",
)
PERMUTATIONS = 200_000
BOOTSTRAPS = 20_000
SEED = 20260909


def identity_map(benchmark_root, split="test"):
    """Map each sample to its composite donor identity, from the manifest.

    The benchmark draws two variants from each composite identity, so the 60
    test samples are 30 independent units, and SCIENTIFIC_DECISIONS.md fixes the
    identity as the analysis unit. An earlier version of this script
    sign-flipped and resampled all 60 rows independently, which treats each
    identity as two observations and understates every p-value.

    The ladder rows do not carry the identity, so it is read from the split
    manifest rather than inferred from the sample name; sample identifiers are
    content hashes and carry no recoverable identity structure.
    """

    manifest = Path(benchmark_root) / "manifests" / ("%s.jsonl" % split)
    if not manifest.is_file():
        raise SystemExit(
            "cannot cluster without %s; refusing to fall back to per-sample "
            "tests, which is the defect this replaces" % manifest)
    mapping = {}
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            mapping[record["sample_id"]] = record["identity_id"]
    return mapping


def paired_test(rows_by_arm, arm, comparator, endpoint, rng):
    """Paired difference with the composite identity as the analysis unit.

    Both the permutation null and the bootstrap operate on identities. Sign
    flips are applied per identity so an identity's variants move together,
    because the exchangeable object under the null is the identity, not the
    variant. The reported statistic is still the median over samples.
    """

    shared = sorted(set(rows_by_arm[arm]) & set(rows_by_arm[comparator]))
    left = np.array([rows_by_arm[arm][s][endpoint] for s in shared], dtype=float)
    right = np.array([rows_by_arm[comparator][s][endpoint] for s in shared], dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    difference = left[finite] - right[finite]
    identities = np.array([IDENTITIES[s] for s in shared])[finite]
    n = int(difference.size)
    if n == 0:
        return None

    units = sorted(set(identities.tolist()))
    index_of = {unit: i for i, unit in enumerate(units)}
    unit_index = np.array([index_of[u] for u in identities])
    n_units = len(units)

    observed = float(np.median(difference))
    # One sign per identity, broadcast to that identity's variants.
    unit_signs = rng.choice([-1.0, 1.0], size=(PERMUTATIONS, n_units))
    null = np.median(unit_signs[:, unit_index] * difference, axis=1)
    # +1 in both terms keeps the p-value strictly positive, which is what a
    # sampled permutation null can actually support.
    p_value = float((np.sum(np.abs(null) >= abs(observed) - 1e-15) + 1) / (PERMUTATIONS + 1))

    # Resample identities, carrying each one's variants together.
    members = [np.where(unit_index == i)[0] for i in range(n_units)]
    draws = rng.integers(0, n_units, size=(BOOTSTRAPS, n_units))
    boot = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        picked = np.concatenate([members[j] for j in draws[i]])
        boot[i] = np.median(difference[picked])
    low, high = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return {
        "n": n,
        "n_identities": n_units,
        "analysis_unit": "composite donor identity",
        "difference": observed,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
    }


def main() -> int:
    ladder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "results/benchmark/E13_ladder_test_v3_m32.json"
    )
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "results/benchmark/E33_paired_ladder_v3.json"
    )

    global IDENTITIES
    IDENTITIES = identity_map(
        os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3"))
    payload = json.loads(ladder.read_text())
    rows_by_arm: dict[str, dict] = {}
    for row in payload["rows"]:
        rows_by_arm.setdefault(row["baseline_id"], {})[row["sample"]] = row

    rng = np.random.default_rng(SEED)
    comparisons = {}
    for comparator in COMPARATORS:
        if comparator not in rows_by_arm:
            continue
        comparisons[comparator] = {
            endpoint: paired_test(rows_by_arm, REFERENCE_ARM, comparator, endpoint, rng)
            for endpoint in ENDPOINTS
        }

    medians = {
        arm: {
            endpoint: float(
                np.median(
                    [
                        r[endpoint]
                        for r in rows.values()
                        if r.get(endpoint) is not None and np.isfinite(r[endpoint])
                    ]
                    or [np.nan]
                )
            )
            for endpoint in ENDPOINTS
        }
        for arm, rows in rows_by_arm.items()
    }

    result = {
        "experiment": "E33_paired_ladder",
        "source": str(ladder),
        "reference_arm": REFERENCE_ARM,
        "sign": "difference is reference minus comparator; negative favors the reference",
        "statistic": "median of per-sample paired differences",
        "permutations": PERMUTATIONS,
        "bootstraps": BOOTSTRAPS,
        "seed": SEED,
        "per_arm_median": medians,
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    for comparator, endpoints in comparisons.items():
        print("\n%s minus %s" % (REFERENCE_ARM, comparator))
        for endpoint, stat in endpoints.items():
            if stat is None:
                continue
            print(
                "  %-24s %+9.4f  [%+8.4f, %+8.4f]  p=%.4g"
                % (endpoint, stat["difference"], stat["ci_low"], stat["ci_high"], stat["p_value"])
            )
    print("\nwrote %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
