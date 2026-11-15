"""E49e: score the cases E49d's in-plane guard withheld, and merge them back in.

E49d verifies the cavity class by comparing in-plane extents and withholds a case
when the winner is not the expected class. That guard is a check against a gross
error, not an arbiter of marginal cases, and applying to it the same test that
exposed the previous guard shows it is wrong in the same way:

  patient127 f01, iter_60000 -- in-plane L1 to the cavity label was 10.86 for the
  myocardium against 16.57 for the cavity, because that frame's predicted cavity
  is under-sized in-plane (56x60 against a 67x66 label). Scoring the cavity mesh
  anyway gives 3.151 mm clipped with end-caps and 2.266 mm without, against
  scored medians of 4.254 and 3.044. The withheld case was *better than median*.

The direction of that bias is what makes it unacceptable. Dropping a
better-than-median case raises the reported median and makes TetHeart look worse,
which favours us. The class identity is not actually in doubt: it follows from
reading their loader, which reorders to ``['myo', 'lv', 'rv']``, so the cavity is
``*_2_pred.obj``. So every case is scored, the in-plane disagreement is kept as a
reported diagnostic rather than an exclusion, and the merge is done here -- in a
separate, auditable step -- rather than by rerunning the whole sweep.

Metric, frame handling and clipping are imported from E49d, so merged rows are
computed by exactly the same code as the rows they join.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SOURCE = Path(os.environ["R4D_E49D_JSON"])
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / os.environ.get(
    "R4D_OUTNAME", "E49e_merged.json")


def main() -> int:
    import nibabel as nib
    import e49d_tetheart_rescore as m
    from rooted4d.experiments.component_eval import symmetric_surface_distance

    payload = json.loads(SOURCE.read_text())
    rows = payload["rows"]

    transforms = {}
    for row in json.loads(m.TRANSFORMS.read_text())["rows"]:
        if row.get("status") == "written":
            transforms[(row["case"], row["frame"])] = row
    labels_by_key = {}
    for path in m.ACDC.rglob("*_frame??_gt.nii.gz"):
        labels_by_key[(path.parent.name, int(path.name.split("_frame")[1][:2]))] = path

    merged = 0
    for record in rows:
        if record.get("status") == "scored" or "patient" not in record:
            continue
        patient, frame = record["patient"], record["frame"]
        source = labels_by_key.get((patient, frame))
        transform = transforms.get((patient, frame))
        mesh_path = m.PRED_DIR / ("%s_%d_pred.obj" % (record["name"], m.EXPECTED_CAVITY_CLASS))
        if source is None or transform is None or not mesh_path.exists():
            print("  %-12s f%02d  cannot merge (missing label, transform or mesh)"
                  % (patient, frame), flush=True)
            continue

        handle = nib.load(str(source))
        labels = np.asarray(handle.dataobj)
        zooms = tuple(float(z) for z in handle.header.get_zooms()[:3])
        mesh = m.load_mesh(mesh_path)
        vertices = m.to_millimetres(mesh.vertices, transform)
        faces = np.asarray(mesh.faces)
        if len(vertices) == 0 or len(faces) == 0:
            print("  %-12s f%02d  empty cavity mesh, left unscored" % (patient, frame))
            continue

        for drop, caps in ((False, "with_endcaps"), (True, "no_endcaps")):
            reference = m.boundary_points(labels, zooms, m.CAVITY_LABEL, drop)
            if reference.shape[0] < 50:
                continue
            sample = reference[:: max(1, len(reference) // 2000)]
            for variant in ("raw", "clipped"):
                if variant == "raw":
                    v, f = vertices, faces
                else:
                    v, f, how = m.clip_to_z(vertices, faces,
                                            reference[:, 2].min(), reference[:, 2].max())
                    record["clip_method"] = how
                    if len(f) == 0:
                        continue
                score = symmetric_surface_distance(sample, v, f, sample_faces=6)
                suffix = "%s_%s" % (variant, caps)
                record["symmetric_mm_" + suffix] = float(score["symmetric_mean_mm"])
                record["hausdorff_mm_" + suffix] = float(score["hausdorff_mm"])
                record["points_to_surface_mm_" + suffix] = float(
                    score["mean_points_to_surface_mm"])
                record["surface_to_points_mm_" + suffix] = float(
                    score["mean_surface_to_points_mm"])

        record["merged_despite_inplane_disagreement"] = True
        record["status"] = "scored"
        record["verts"] = int(vertices.shape[0])
        merged += 1
        print("  %-12s f%02d  MERGED  raw %.3f  clipped %.3f (caps in) | raw %.3f  clipped %.3f (out)"
              % (patient, frame,
                 record.get("symmetric_mm_raw_with_endcaps", float("nan")),
                 record.get("symmetric_mm_clipped_with_endcaps", float("nan")),
                 record.get("symmetric_mm_raw_no_endcaps", float("nan")),
                 record.get("symmetric_mm_clipped_no_endcaps", float("nan"))), flush=True)

    scored = [r for r in rows if r.get("status") == "scored"]
    flagged = [r for r in scored if r.get("merged_despite_inplane_disagreement")]
    summary = {
        "groups": payload["summary"]["groups"],
        "scored": len(scored),
        "patients": len({r["patient"] for r in scored}),
        "merged_despite_inplane_disagreement": len(flagged),
        "status_counts": {s: sum(1 for r in rows if r.get("status") == s)
                          for s in sorted({r.get("status") for r in rows})},
    }
    for metric in ("symmetric_mm", "hausdorff_mm",
                   "points_to_surface_mm", "surface_to_points_mm"):
        for variant in ("raw", "clipped"):
            for caps in ("with_endcaps", "no_endcaps"):
                key = "%s_%s_%s" % (metric, variant, caps)
                values = np.array([r[key] for r in scored if key in r])
                if values.size:
                    summary[key] = {"median": float(np.median(values)),
                                    "mean": float(values.mean()),
                                    "p95": float(np.percentile(values, 95)),
                                    "max": float(values.max())}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "experiment": "E49e_merge_withheld",
        "source": str(SOURCE),
        "predictions": str(m.PRED_DIR),
        "note": "every case scored; in-plane disagreement is reported as a "
                "diagnostic, not used to exclude, because excluding it dropped "
                "better-than-median cases and biased the median in our favour",
        "summary": summary, "rows": rows}, indent=2) + "\n")
    print("merged %d case(s); %d scored over %d patients"
          % (merged, summary["scored"], summary["patients"]))
    print(json.dumps({k: v for k, v in summary.items()
                      if k.startswith("symmetric_mm") or k in
                      ("scored", "patients", "merged_despite_inplane_disagreement")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
