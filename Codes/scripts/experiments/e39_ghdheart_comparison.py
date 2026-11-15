"""E39: GHDHeart against Rooted4D on identical frames, paired.

E36 scored our fitting on the ACDC test frames and E37 scored GHDHeart on the
same frames with the same metric and the same reference boundary points. This
pairs them per frame, which is stronger than comparing two medians: the frames
differ a lot from each other, and a paired statistic removes that variation.

GHDHeart consumes a label volume, so the comparison it belongs in is against
our label-input row. The image-input row is reported too, but it is a harder
problem and is not the like-for-like pairing.

Intervals resample whole patients rather than frames, because a patient's two
frames are not two independent observations.
"""

import json
from pathlib import Path

import numpy as np

GHD = Path("results/21959021_test/E37_test_B64.json")
OURS = Path("results/21931418/E36_image_input_chamber.json")
OUTPUT = Path("results/benchmark/E39_ghdheart_comparison.json")
RESAMPLES = 20_000
PERMUTATIONS = 200_000
SEED = 20260910


def key(row):
    return (row["patient"], row["frame"])


def paired_stats(values_a, values_b, patients, rng):
    """Median paired difference (a - b), with the patient as the analysis unit.

    A patient contributes two frames, end diastole and end systole, and they are
    not independent observations. The bootstrap resampled patients from the
    start, but the permutation null originally flipped each frame's sign
    independently, which treats one patient as two exchangeable units and
    understates the p-value. Both now operate on patients.
    """

    difference = values_a - values_b
    observed = float(np.median(difference))

    units = sorted(set(patients))
    index = {u: np.where(patients == u)[0] for u in units}
    members = [index[u] for u in units]
    n_units = len(units)
    unit_of = np.empty(difference.size, dtype=int)
    for i, u in enumerate(units):
        unit_of[index[u]] = i

    draws = rng.integers(0, n_units, size=(RESAMPLES, n_units))
    boot = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        picked = np.concatenate([members[j] for j in draws[i]])
        boot[i] = np.median(difference[picked])
    low, high = (float(v) for v in np.percentile(boot, [2.5, 97.5]))

    # One sign per patient, broadcast to that patient's frames.
    unit_signs = rng.choice([-1.0, 1.0], size=(PERMUTATIONS, n_units))
    null = np.median(unit_signs[:, unit_of] * difference, axis=1)
    p = float((np.sum(np.abs(null) >= abs(observed) - 1e-15) + 1) / (PERMUTATIONS + 1))
    return {"difference": observed, "ci_low": low, "ci_high": high, "p_value": p,
            "n_frames": int(difference.size), "n_patients": n_units,
            "analysis_unit": "patient"}


def main() -> int:
    ghd_raw = json.loads(GHD.read_text())
    basis = ghd_raw.get("basis_grid", [64])[0]
    ghd_rows = {key(r): r for r in ghd_raw["results"][str(basis)]["rows"] if not r["failed"]}
    our_rows = {key(r): r for r in json.loads(OURS.read_text())["rows"]
                if r.get("label_input") and r.get("image_input")}

    shared = sorted(set(ghd_rows) & set(our_rows))
    print("frames in GHDHeart: %d | in ours: %d | shared: %d"
          % (len(ghd_rows), len(our_rows), len(shared)))
    if not shared:
        print("no shared frames")
        return 1

    patients = np.array([p for p, _ in shared])
    rng = np.random.default_rng(SEED)

    arms = {
        "ghdheart": np.array([ghd_rows[k]["symmetric_mean_mm"] for k in shared]),
        "rooted4d_label_input": np.array([our_rows[k]["label_input"]["symmetric_mean_mm"] for k in shared]),
        "rooted4d_image_input": np.array([our_rows[k]["image_input"]["symmetric_mean_mm"] for k in shared]),
    }
    hausdorff = {
        "ghdheart": np.array([ghd_rows[k]["hausdorff_mm"] for k in shared]),
        "rooted4d_label_input": np.array([our_rows[k]["label_input"]["hausdorff_mm"] for k in shared]),
        "rooted4d_image_input": np.array([our_rows[k]["image_input"]["hausdorff_mm"] for k in shared]),
    }

    result = {"experiment": "E39_ghdheart_comparison", "n_frames": len(shared),
              "n_patients": len(set(patients)), "ghdheart_basis": basis,
              "ghdheart_source": str(GHD), "ours_source": str(OURS),
              "medians": {}, "paired": {}}

    for name, values in arms.items():
        result["medians"][name] = {
            "symmetric_mm": float(np.median(values)),
            "hausdorff_mm": float(np.median(hausdorff[name])),
        }
        print("  %-24s symmetric %.3f   hausdorff %.3f"
              % (name, np.median(values), np.median(hausdorff[name])))

    for ours in ("rooted4d_label_input", "rooted4d_image_input"):
        stat = paired_stats(arms["ghdheart"], arms[ours], patients, rng)
        result["paired"]["ghdheart_minus_" + ours] = stat
        print("\nGHDHeart minus %s: %+.3f mm  [%+.3f, %+.3f]  p=%.4g"
              % (ours, stat["difference"], stat["ci_low"], stat["ci_high"], stat["p_value"]))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print("\nwrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
