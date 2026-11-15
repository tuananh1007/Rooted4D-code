"""E61 analysis: correct the published soft-attachment row, reproducing first.

E60 showed on validation that the published soft-attachment gap is the detached
model's coordinate renormalisation, not soft attachment. The decision was to
re-score only the corrected arm on E51's test samples, in E51's exact
configuration. This script turns that re-score into a corrected ladder and
corrected paired statistics, and it will not do so unless everything it does
not change reproduces exactly:

1. **Row reproduction.** The re-run ``rooted4d_full`` and historical
   ``joint_soft_attachment`` rows must reproduce E51's own per-sample rows, by
   the criterion in the amendment below (the original 1e-6 maximum assumed
   bitwise reproduction, which does not hold). Otherwise the harness has
   drifted, and nothing it reports can be read as a correction of E51.
2. **Corrected ladder.** E51's rows, with only the soft-attachment rows
   replaced by the matched arm's. The historical rows are kept under
   ``joint_soft_attachment_historical`` so the change is auditable.
3. **Paired statistics** through E33's own code, unchanged. E33 draws from one
   seeded generator whose draws do not depend on the data, so the late-fusion
   and exact-without-gauges comparisons must come out *identical* to the
   published E33 file. That is checked too.

Amendment, 24 September, made after the first run failed step 1
------------------------------------------------------------------
The 1e-6 tolerance assumed E51 would reproduce bitwise. It does not, and not
because anything drifted:

* the median absolute difference is about 1e-9 on every endpoint, where a code
  or configuration drift would show as systematic differences far above that;
* the differences above 1e-6 sit in a minority of samples, and those are the
  samples where Levenberg-Marquardt took a different number of iterations (15 of
  60, by up to 8-12). A 1e-9 floating-point difference flips an accept/reject or
  stopping decision and the solver converges to a point up to ~4e-4 mm away;
* E51 ran with OpenBLAS left at its default threading. Its sbatch set
  R4D_THREADS, which no code reads, and it ran at an older commit. E61 pins BLAS
  to one thread. Either difference reorders floating-point work.

Step 1 is therefore restated as what it was for: no drift, and the published
numbers reproduced at the precision they are reported. It now requires a
median absolute difference of at most 1e-6 on every endpoint (no systematic
drift), a maximum of at most 1e-3 (every divergent path stays three orders
below the 0.4 mm effect being corrected), and per-arm medians within 5e-4 of
E51's (the table reports two decimals). The corrected paired statistics are
reported two ways: against E51's published rows through E33's own code, and
against E61's own ``rooted4d_full`` rows, from one run with one set of numerics.

Usage:
    python e61_soft_arm_correction.py <e61 shard.jsonl>
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

E51 = Path("results/benchmark/E51_ladder_test_v3_m32_repaired.json")
E33_PUBLISHED = Path("results/benchmark/E33_paired_ladder_v3_repaired.json")
CORRECTED_LADDER = Path("results/benchmark/E61_ladder_test_v3_m32_soft_corrected.json")
CORRECTED_PAIRED = Path("results/benchmark/E61_paired_ladder_v3_soft_corrected.json")
ENDPOINTS = ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm",
             "phase_error_cycles", "attachment_residual_mm")
MEDIAN_TOLERANCE = 1e-6
MAX_TOLERANCE = 1e-3
TABLE_TOLERANCE = 5e-4


def main() -> int:
    shard = Path(sys.argv[1])
    rescored = {}
    for line in shard.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rescored[Path(row["sample"]).stem] = row["variants"]
    ladder = json.loads(E51.read_text())
    published = {(r["baseline_id"], r["sample"]): r for r in ladder["rows"]}
    samples = sorted({r["sample"] for r in ladder["rows"]})
    if sorted(rescored) != samples:
        raise SystemExit("E61 samples differ from E51's: %d vs %d" % (len(rescored), len(samples)))

    # 1. Reproduction of the two arms that were not changed (see the amendment).
    reproduction, passed = {}, True
    for arm in ("rooted4d_full", "joint_soft_attachment"):
        reproduction[arm] = {}
        for endpoint in ENDPOINTS:
            new = np.array([float(rescored[s][arm][endpoint]) for s in samples])
            old = np.array([float(published[(arm, s)][endpoint]) for s in samples])
            gap = np.abs(new - old)
            iterations = sum(int(rescored[s][arm]["iterations"])
                             != int(published[(arm, s)]["iterations"]) for s in samples)
            entry = {"median_abs": float(np.median(gap)), "max_abs": float(gap.max()),
                     "median_change": float(abs(np.median(new) - np.median(old))),
                     "samples_with_different_iteration_count": iterations}
            entry["passed"] = bool(entry["median_abs"] <= MEDIAN_TOLERANCE
                                   and entry["max_abs"] <= MAX_TOLERANCE
                                   and entry["median_change"] <= TABLE_TOLERANCE)
            passed &= entry["passed"]
            reproduction[arm][endpoint] = entry
    print("row reproduction against E51:")
    for arm, endpoints in reproduction.items():
        for endpoint, e in endpoints.items():
            print("  %-24s %-24s median|d| %.1e  max|d| %.1e  table median moves %.1e  %s"
                  % (arm, endpoint, e["median_abs"], e["max_abs"], e["median_change"],
                     "ok" if e["passed"] else "FAIL"))
        print("  %-24s iterations differ on %d of %d samples (solver-path divergence)"
              % (arm, e["samples_with_different_iteration_count"], len(samples)))
    if not passed:
        raise SystemExit("E51 does not reproduce at reporting precision; refusing to correct it")

    # 2. Corrected ladder: only the soft-attachment rows change.
    corrected = copy.deepcopy(ladder)
    rows = []
    for row in ladder["rows"]:
        if row["baseline_id"] != "joint_soft_attachment":
            rows.append(row)
            continue
        historical = dict(row, baseline_id="joint_soft_attachment_historical")
        matched = {k: v for k, v in rescored[row["sample"]]["joint_soft_attachment_matched"].items()
                   if k in row or k in ("heldout_cmr_p95_mm", "heldout_tee_p95_mm", "phase_map_cycles")}
        matched.update(baseline_id="joint_soft_attachment", sample=row["sample"],
                       note="matched detached model (E60/E61); replaces the renormalised arm")
        rows.extend([matched, historical])
    corrected["rows"] = rows
    for arm in ("joint_soft_attachment", "joint_soft_attachment_historical"):
        subset = [r for r in rows if r["baseline_id"] == arm]
        corrected["per_arm"][arm] = {
            "samples": len(subset),
            "failures": sum(1 for r in subset if not r["succeeded"]),
            **{endpoint: {"median": float(np.median([r[endpoint] for r in subset]))}
               for endpoint in ENDPOINTS},
        }
    corrected["note"] = ("E61: E51's rows with the soft-attachment arm re-scored on the "
                         "matched detached model; the renormalised rows are kept as "
                         "joint_soft_attachment_historical")
    CORRECTED_LADDER.write_text(json.dumps(corrected, indent=2) + "\n")

    # 3. Paired statistics through E33's own code.
    subprocess.run([sys.executable, "Codes/scripts/experiments/e33_paired_ladder.py",
                    str(CORRECTED_LADDER), str(CORRECTED_PAIRED)], check=True,
                   capture_output=True, text=True)
    new = json.loads(CORRECTED_PAIRED.read_text())["comparisons"]
    old = json.loads(E33_PUBLISHED.read_text())["comparisons"]
    for comparator in ("late_fusion", "joint_exact_no_gauges"):
        if new[comparator] != old[comparator]:
            raise SystemExit("E33 does not reproduce for %s; refusing to report" % comparator)
    print("E33 reproduction: late_fusion and joint_exact_no_gauges identical to the published file")

    print("\ncorrected soft-attachment row (medians over 60 test samples):")
    for arm in ("rooted4d_full", "joint_soft_attachment", "joint_soft_attachment_historical"):
        stats = corrected["per_arm"][arm]
        print("  %-34s CMR %.4f TEE %.4f rot %.4f trans %.4f phase %.4f gap %.2e fail %d"
              % (arm, *(stats[e]["median"] for e in ENDPOINTS), stats["failures"]))
    print("\nrooted4d_full minus corrected soft attachment (E33 statistic):")
    for endpoint, stat in new["joint_soft_attachment"].items():
        if stat:
            print("  %-24s %+.4f  [%+.4f, %+.4f]  p=%.3g"
                  % (endpoint, stat["difference"], stat["ci_low"], stat["ci_high"], stat["p_value"]))
    # The same comparison inside one run: E61's matched arm against E61's own
    # rooted4d_full, drawn exactly as E33 draws (E33 consumes its generator on
    # late fusion first, and those draws do not depend on the data).
    sys.path.insert(0, "Codes/scripts/experiments")
    import e33_paired_ladder as e33

    e33.IDENTITIES = e33.identity_map("data/derived/semisynthetic-real-v3")
    internal = {"rooted4d_full": {s: rescored[s]["rooted4d_full"] for s in samples},
                "joint_soft_attachment": {s: rescored[s]["joint_soft_attachment_matched"]
                                          for s in samples},
                "late_fusion": {s: published[("late_fusion", s)] for s in samples}}
    rng = np.random.default_rng(e33.SEED)
    for endpoint in e33.ENDPOINTS:
        e33.paired_test(internal, e33.REFERENCE_ARM, "late_fusion", endpoint, rng)
    within = {endpoint: e33.paired_test(internal, e33.REFERENCE_ARM, "joint_soft_attachment",
                                        endpoint, rng) for endpoint in e33.ENDPOINTS}
    print("\nwithin E61 (its own rooted4d_full minus its matched soft arm):")
    for endpoint, stat in within.items():
        if stat:
            print("  %-24s %+.4f  [%+.4f, %+.4f]  p=%.3g"
                  % (endpoint, stat["difference"], stat["ci_low"], stat["ci_high"], stat["p_value"]))
    record = json.loads(CORRECTED_PAIRED.read_text())
    record["e61"] = {"reproduction_against_e51": reproduction,
                     "within_run_soft_comparison": within,
                     "amendment": "see the script docstring: bitwise reproduction does not "
                                  "hold across BLAS threading; the criterion is no drift and "
                                  "reproduction at reporting precision"}
    CORRECTED_PAIRED.write_text(json.dumps(record, indent=2) + "\n")
    print("\nwrote %s and %s" % (CORRECTED_LADDER, CORRECTED_PAIRED))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
