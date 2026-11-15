"""E38: is GHDHeart converged at the settings we ran it at?

E37 sweeps num_basis at the demo's learning rate and iteration count. If those
defaults under-fit on our data, the resulting comparison flatters us for a
reason that has nothing to do with either method, so this probes the other two
knobs on a subset of validation frames before any number is reported.

A method is only fairly beaten at its best reachable setting. If a longer or
faster schedule helps materially, E37 must be rerun there.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "e37", str(Path(__file__).resolve().parent / "e37_ghdheart_chamber.py"))
e37 = importlib.util.module_from_spec(spec)
sys.modules["e37"] = e37
spec.loader.exec_module(e37)

OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E38_ghdheart_convergence.json"
FRAMES = int(os.environ.get("R4D_PROBE_FRAMES", "10"))
BASIS = int(os.environ.get("R4D_PROBE_BASIS", "64"))
GRID = [(600, 0.01), (2000, 0.01), (600, 0.03), (2000, 0.03), (4000, 0.01)]


def main() -> int:
    subset = []
    for item in e37.frames("val"):
        subset.append(item)
        if len(subset) >= FRAMES:
            break
    print("probing %d validation frames at num_basis=%d" % (len(subset), BASIS), flush=True)

    result = {"experiment": "E38_ghdheart_convergence", "basis": BASIS,
              "frames": len(subset), "grid": [{"iterations": i, "lr": l} for i, l in GRID],
              "settings": {}}

    for iterations, lr in GRID:
        e37.ITERATIONS = iterations
        e37.LR = lr
        scores = []
        started = time.time()
        for patient, frame, labels, zooms in subset:
            try:
                fitted = e37.fit_one(labels, zooms, BASIS)
            except Exception as exc:
                print("  FAILED %s f%02d %s" % (patient, frame, exc), flush=True)
                continue
            if fitted is None:
                continue
            vertices, faces = fitted
            reference = e37.cavity_boundary_points(labels, zooms, cavity_label=e37.CAVITY)
            score = e37.symmetric_surface_distance(reference, vertices, faces)
            scores.append(score["symmetric_mean_mm"])
        key = "iter%d_lr%g" % (iterations, lr)
        if scores:
            result["settings"][key] = {
                "n": len(scores), "median": float(np.median(scores)),
                "mean": float(np.mean(scores)),
                "seconds_total": time.time() - started,
            }
            print("  %-16s median %.3f mm over %d frames (%.0fs)"
                  % (key, np.median(scores), len(scores), time.time() - started), flush=True)
        else:
            result["settings"][key] = {"n": 0}

    usable = {k: v for k, v in result["settings"].items() if v.get("n")}
    if usable:
        best = min(usable, key=lambda k: usable[k]["median"])
        baseline = usable.get("iter600_lr0.01", {}).get("median")
        result["best"] = best
        result["best_median"] = usable[best]["median"]
        if baseline is not None:
            result["improvement_over_default_mm"] = baseline - usable[best]["median"]
            print("\nbest %s at %.3f mm; default iter600_lr0.01 was %.3f, improvement %.3f mm"
                  % (best, usable[best]["median"], baseline,
                     baseline - usable[best]["median"]))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
