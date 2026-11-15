#!/bin/bash
#SBATCH --job-name=rooted4d-sfsens
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
NVME=/work/nvme/bhio/avu2/Rooted4D
cd $NVME/code; mkdir -p logs
export PYTHONPATH=$NVME/code/Codes:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export R4D_PRED_DIR=$NVME/results/22102244_thfinal/preds_test_iter_60000
export R4D_TET_FRAME_REPORT=$NVME/results/22045910_tetframe/E50_tet_frame.json
export R4D_ACDC=/work/hdd/bhio/avu2/Rooted4D/data/raw/acdc
export R4D_OUT=/tmp
$NVME/external/.venv-tetheart/bin/python - <<'PY'
"""Decompose the two scoring-protocol differences between E36 and E49d.

E36 passes the full cavity boundary set with the default 20 samples per face.
E49d strides the reference at max(1, n // 2000) and uses 6. Across the 44 scored
test cases the reference sizes run 280 to 4771, so the stride is 2 on the
largest cases and 1 elsewhere. Both factors are therefore live; an earlier
reading that the stride was always 1 came from inspecting only six files and was
wrong.

Three arms per case isolate each factor:
  A  strided reference,  6 samples/face  -- what E49d reported
  B  strided reference, 20 samples/face  -- face sampling alone
  C  full reference,    20 samples/face  -- E36's convention in full
"""
import json, sys, os
from pathlib import Path
import numpy as np
import nibabel as nib

sys.path.insert(0, "/work/nvme/bhio/avu2/Rooted4D/code/Codes/scripts/experiments")
import e49d_tetheart_rescore as m
from rooted4d.experiments.component_eval import symmetric_surface_distance

RES = Path("/work/nvme/bhio/avu2/Rooted4D/results/22102244_thfinal")
rows = [r for r in json.loads((RES / "E49d_test_iter_60000.json").read_text())["rows"]
        if r.get("status") == "scored"]
# Bias toward the largest meshes, where the stride actually bites.
rows = sorted(rows, key=lambda r: -r.get("verts", 0))[:12]

transforms = {}
for r in json.loads(m.TRANSFORMS.read_text())["rows"]:
    if r.get("status") == "written":
        transforms[(r["case"], r["frame"])] = r
labels = {}
for p in m.ACDC.rglob("*_frame??_gt.nii.gz"):
    labels[(p.parent.name, int(p.name.split("_frame")[1][:2]))] = p

out = []
for rec in rows:
    pat, fr = rec["patient"], rec["frame"]
    handle = nib.load(str(labels[(pat, fr)]))
    lab = np.asarray(handle.dataobj)
    zooms = tuple(float(v) for v in handle.header.get_zooms()[:3])
    mesh = m.load_mesh(m.PRED_DIR / ("%s_2_pred.obj" % rec["name"]))
    verts = m.to_millimetres(mesh.vertices, transforms[(pat, fr)])
    faces = np.asarray(mesh.faces)
    ref = m.boundary_points(lab, zooms, m.CAVITY_LABEL, False)
    stride = max(1, len(ref) // 2000)
    sample = ref[::stride]
    cv, cf, _ = m.clip_to_z(verts, faces, ref[:, 2].min(), ref[:, 2].max())
    a = symmetric_surface_distance(sample, cv, cf, sample_faces=6)["symmetric_mean_mm"]
    b = symmetric_surface_distance(sample, cv, cf)["symmetric_mean_mm"]
    c = symmetric_surface_distance(ref, cv, cf)["symmetric_mean_mm"]
    out.append({"patient": pat, "frame": fr, "ref_points": int(len(ref)),
                "stride": int(stride),
                "A_strided_6": float(a), "B_strided_20": float(b), "C_full_20": float(c),
                "face_sampling_mm": float(b - a), "reference_stride_mm": float(c - b),
                "total_protocol_mm": float(c - a)})
    print("  %-12s f%02d ref %4d stride %d | A %.4f B %.4f C %.4f | face %+.4f stride %+.4f total %+.4f"
          % (pat, fr, len(ref), stride, a, b, c, b - a, c - b, c - a), flush=True)

face = np.array([r["face_sampling_mm"] for r in out])
strd = np.array([r["reference_stride_mm"] for r in out])
tot = np.array([r["total_protocol_mm"] for r in out])
summary = {"cases": len(out),
           "face_sampling_median_abs_mm": float(np.median(np.abs(face))),
           "face_sampling_max_abs_mm": float(np.abs(face).max()),
           "reference_stride_median_abs_mm": float(np.median(np.abs(strd))),
           "reference_stride_max_abs_mm": float(np.abs(strd).max()),
           "total_median_abs_mm": float(np.median(np.abs(tot))),
           "total_max_abs_mm": float(np.abs(tot).max()),
           "total_mean_signed_mm": float(tot.mean())}
(RES / "E49f_protocol_sensitivity.json").write_text(json.dumps(
    {"experiment": "E49f_protocol_sensitivity",
     "question": "decompose face-sampling and reference-stride differences "
                 "between the E36 and E49d scoring paths",
     "summary": summary, "rows": out}, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
echo "finished $(date -u +%H:%M:%SZ)"
