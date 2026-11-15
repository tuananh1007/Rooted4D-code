"""E46: add right-ventricle meshes so TetHeart's loader indexes correctly.

TetHeart's dataset does:

    meshes = load_data(folder, "*_mesh*.obj")
    self.meshes.append([meshes[:3], meshes[3:]])

which hardcodes three structures per frame and splits the sorted list at index
three to separate end diastole from end systole. 4DMR produces only the
endocardial and epicardial LV surfaces, so supplying two meshes per frame would
not merely drop the right ventricle: it would pair end diastole's two meshes
with end systole's first, and hand the remainder over as end systole. Silently,
and with plausible-looking data.

There are two ways out. Modify their loader and the model's class count, which
reaches into the architecture; or supply a third structure so their indexing
holds. This takes the second, deriving the right ventricle from the ACDC label
directly by marching cubes with Taubin smoothing, in the same
index-times-spacing frame as the other two.

That is a real deviation and is reported as one. The right-ventricle target is
label-derived and blockier than the 4DMR surfaces, where in the authors' own
data all three come from the same pipeline. It is acceptable here only because
panel A scores the **left-ventricular cavity**: the right ventricle enters as
supervision on a shared field and never as a measured output. A study reporting
right-ventricular accuracy could not make this trade.
"""

import json
import os
from pathlib import Path

import numpy as np

ACDC = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
GT = Path(os.environ["R4D_TETHEART_GT"])
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E46_rv_meshes.json"
RV_LABEL = 1
SMOOTH_ITERATIONS = int(os.environ.get("R4D_RV_SMOOTH", "20"))


def write_obj(path, vertices, faces):
    with open(path, "w") as handle:
        for v in vertices:
            handle.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:
            handle.write("f %d %d %d\n" % (f[0], f[1], f[2]))


def main() -> int:
    import nibabel as nib
    import pyvista as pv
    from skimage import measure
    from rooted4d.experiments.component_eval import point_to_mesh_mm

    label_paths = {}
    for path in ACDC.rglob("*_frame??_gt.nii.gz"):
        label_paths[(path.parent.name, int(path.name.split("_frame")[1][:2]))] = path

    rows = []
    for case_dir in sorted(GT.glob("ACDC_patient*")):
        patient = case_dir.name.replace("ACDC_", "")
        for lv_path in sorted(case_dir.glob("*_mesh_lv.obj")):
            stem = lv_path.name.replace("_mesh_lv.obj", "")
            frame = int(stem.split("_frame")[1][:2])
            rv_path = case_dir / ("%s_mesh_rv.obj" % stem)
            source = label_paths.get((patient, frame))
            if source is None:
                rows.append({"case": patient, "frame": frame, "status": "no_label"})
                continue

            handle = nib.load(str(source))
            labels = np.asarray(handle.dataobj)
            zooms = np.array([float(z) for z in handle.header.get_zooms()[:3]])
            occupancy = (labels == RV_LABEL).astype(np.float32)
            if occupancy.sum() < 50:
                rows.append({"case": patient, "frame": frame, "status": "too_little_rv"})
                continue
            try:
                verts, faces, _, _ = measure.marching_cubes(occupancy, level=0.5)
            except Exception as exc:
                rows.append({"case": patient, "frame": frame, "status": "marching_cubes_failed",
                             "reason": str(exc)[:80]})
                continue

            # Marching cubes returns index coordinates; scale to the same
            # index-times-spacing frame the 4DMR surfaces already live in.
            verts = verts * zooms
            mesh = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
            mesh = mesh.smooth_taubin(n_iter=SMOOTH_ITERATIONS, pass_band=0.1)
            smoothed = np.asarray(mesh.points, dtype=np.float64)
            smoothed_faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
            write_obj(rv_path, smoothed, smoothed_faces)

            # Smoothing must not pull the surface off the label it came from.
            boundary = np.argwhere(occupancy > 0).astype(float) * zooms
            sample = boundary[:: max(1, len(boundary) // 2000)]
            distance = point_to_mesh_mm(sample, smoothed, smoothed_faces)
            rows.append({"case": patient, "frame": frame, "status": "written",
                         "verts": int(smoothed.shape[0]),
                         "median_label_to_mesh_mm": float(np.median(distance))})
            print("  %-12s frame %02d  rv %5d verts  label agreement %.3f mm"
                  % (patient, frame, smoothed.shape[0], np.median(distance)), flush=True)

    written = [r for r in rows if r["status"] == "written"]
    summary = {"frames": len(rows), "written": len(written),
               "smooth_iterations": SMOOTH_ITERATIONS,
               "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                                 for s in sorted({r["status"] for r in rows})}}
    if written:
        v = np.array([r["median_label_to_mesh_mm"] for r in written])
        summary["label_agreement_mm"] = {"median": float(np.median(v)),
                                         "p95": float(np.percentile(v, 95)),
                                         "max": float(v.max())}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"experiment": "E46_rv_meshes",
                                  "provenance": "right ventricle derived from the ACDC label by "
                                                "marching cubes and Taubin smoothing, not from "
                                                "4DMR; supervision only, never a measured output",
                                  "summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
