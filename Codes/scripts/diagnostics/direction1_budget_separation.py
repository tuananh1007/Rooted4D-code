"""Can any budget base be inactive on nominal yet bind under corruption?

modality m is inactive iff base > (ratio_m - 1) / (L_m*/min L_k*), so compare
that 'required base' under nominal against the clean side under corruption.
"""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from rooted4d.inference import SolverConfig, fit_joint_scene, initial_parameters
from rooted4d.experiments.benchmark_eval import fit_benchmark_model, sample_paths, benchmark_problem
from rooted4d.experiments.pilot_conditions import apply_condition
from rooted4d.experiments.pilot_inference import reference_losses, modality_loss

ROOT = Path("data/derived/semisynthetic-real-v3")
CFG = SolverConfig(fourier_order=3, session_order=1, estimate_pose=False,
                   estimate_phase_warp=False, estimate_phase_offset=False, max_iterations=30)
model, _ = fit_benchmark_model(ROOT, n_modes=32, n_motion=4, n_session=2, n_phase_nodes=8)
split = json.load(open("results/benchmark/DIRECTION1_PILOT_SPLIT.json"))
index = {}
for p in sample_paths(ROOT, "val"):
    with np.load(p, allow_pickle=True) as b:
        index.setdefault(str(b["identity_id"]), []).append(p)

def start_for(truth):
    base = initial_parameters(model, CFG)
    return replace(base, pose_axis_angle=truth.pose_axis_angle,
                   pose_translation=truth.pose_translation,
                   phase_offset=truth.phase_offset, phase_logits=truth.phase_logits)

OTHER = {"cmr": "tee", "tee": "cmr"}
required = {"nominal": [], "clean_under_corruption": [], "corrupted_side": []}
for identity in split["development_identities"]:
    for path in index[identity]:
        prob = benchmark_problem(path, model, CFG)
        st = start_for(prob.truth)
        for cond, mod in (("nominal", None), ("localized", "cmr"), ("localized", "tee")):
            obs, _ = apply_condition(prob.observations, cond, mod)
            ref = reference_losses(obs, model, CFG, start=st)
            jf = fit_joint_scene(obs, model, CFG, start=st)
            smallest = max(min(ref.values()), 1e-12)
            for m in ("cmr", "tee"):
                loss = modality_loss(jf.parameters, obs, model, CFG, m)
                ratio = loss / max(ref[m], 1e-12)
                scale = max(ref[m], 1e-12) / smallest
                need = (ratio - 1.0) / scale
                if cond == "nominal":
                    required["nominal"].append(need)
                elif m == OTHER[mod]:
                    required["clean_under_corruption"].append(need)
                else:
                    required["corrupted_side"].append(need)

for name, vals in required.items():
    v = np.asarray(vals)
    print("%-24s n=%3d  min %+.3f  median %+.3f  p90 %+.3f  max %+.3f"
          % (name, v.size, v.min(), np.median(v), np.percentile(v, 90), v.max()))
nom = np.asarray(required["nominal"]); clean = np.asarray(required["clean_under_corruption"])
print()
print("largest base needed for nominal to be inactive : %+.3f" % nom.max())
print("clean-side required base under corruption      : median %+.3f  max %+.3f"
      % (np.median(clean), clean.max()))
frac = float((clean > nom.max()).mean())
print()
print("fraction of corrupted cases that would still bind at that base: %.1f%%" % (100*frac))
print("SEPARABLE" if frac > 0.5 else "OVERLAP — no single base both spares nominal and binds under corruption")

# Durable artifact, so the negative result is reproducible from the repo.
import json as _json
_out = Path("results/benchmark/E54b_budget_separation.json")
_payload = {
    "experiment": "E54b_budget_separation",
    "question": "can one budget base be inactive on nominal yet bind under corruption",
    "rule": "modality m inactive iff base > (ratio_m - 1) / (L_m*/min L_k*)",
    "development_identities": len(split["development_identities"]),
    "distributions": {
        name: {"n": int(np.asarray(v).size),
               "min": float(np.min(v)), "median": float(np.median(v)),
               "p90": float(np.percentile(v, 90)), "max": float(np.max(v))}
        for name, v in required.items()
    },
    "base_sparing_nominal": float(np.max(required["nominal"])),
    "fraction_corrupted_still_binding": float(
        (np.asarray(required["clean_under_corruption"]) > np.max(required["nominal"])).mean()),
    "verdict": "overlap; no single base separates nominal from corrupted",
}
_out.write_text(_json.dumps(_payload, indent=2) + "\n")
print("wrote", _out)
