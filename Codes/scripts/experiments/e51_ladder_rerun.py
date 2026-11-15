"""E51: regenerate the E13 v3 ladder with the repaired late-fusion baseline.

Late fusion's temporal alignment read the observed point sequences through the
registration vertex set, so a missing annular identifier silently resolved to a
neighbouring vertex's trajectory. Every late-fusion sample in the stored v3
ladder hit this. The repair separates the two vertex sets and asserts
membership, which changes the estimated phase offset and, through it, the pose
and the interface gap.

This reruns the same configuration on the same frozen pool so the new table is
directly comparable to the old one, and writes the result beside it rather than
over it, so the change is auditable rather than asserted.
"""

import json
import os
from pathlib import Path

from rooted4d.experiments.benchmark_eval import experiment_benchmark_ladder

ROOT = os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3")
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / os.environ.get(
    "R4D_OUTNAME", "E51_ladder_test_v3_m32_repaired.json")
# The configuration the stored v3 table was produced with: 32 identity modes
# selected on validation by the one-standard-error rule, four motion modes, two
# session modes.
N_MODES = int(os.environ.get("R4D_MODES", "32"))
N_MOTION = int(os.environ.get("R4D_MOTION", "4"))
N_SESSION = int(os.environ.get("R4D_SESSION", "2"))
MAX_ITERATIONS = int(os.environ.get("R4D_ITERATIONS", "40"))


def main() -> int:
    result = experiment_benchmark_ladder(
        ROOT,
        split="test",
        n_modes=N_MODES,
        n_motion=N_MOTION,
        n_session=N_SESSION,
        max_iterations=MAX_ITERATIONS,
    )
    result["note"] = (
        "regenerated after repairing late fusion's temporal vertex lookup; "
        "compare against E13_ladder_test_v3_m32.json, which used the defective path"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

    print("arms:")
    for arm, stats in sorted(result["per_arm"].items()):
        def median(key):
            entry = stats.get(key)
            return entry["median"] if isinstance(entry, dict) and "median" in entry else float("nan")
        print("  %-24s cmr %7.3f  tee %7.3f  rot %7.3f  trans %7.3f  phase %7.4f  gap %9.4f"
              % (arm, median("heldout_cmr_mm"), median("heldout_tee_mm"),
                 median("rotation_deg"), median("translation_mm"),
                 median("phase_error_cycles"), median("attachment_residual_mm")))
    print("\nwrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
