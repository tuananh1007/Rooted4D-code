"""E42: our leaflet fitting behind nnU-Net, on the MVSeg2023 test cases.

What this measures, stated precisely because the obvious reading is wrong.
nnU-Net produces segmentations, not meshes, so this is not a mesh-reconstruction
comparison. It is our fitting fed by a trained segmenter against our fitting fed
by the provider's reference labels, scored against those same reference labels
in both cases. That is a front-end comparison, and the paper labels it as one.

Restricted to the 40 MVSeg2023 test cases, which are the only ones nnU-Net may
be scored on since it trained on the challenge's 105. E40 restricts our own
reference-label figures to the same 40 so the two rows are comparable.

Label convention comes from the dataset card: 1 posterior, 2 anterior.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL, fit_valve
from rooted4d.experiments.component_eval import leaflet_faces, symmetric_surface_distance

REFERENCE = Path(os.environ.get("R4D_MVSEG_TEST", "data/raw/mvseg2023/test"))
PREDICTED = Path(os.environ["R4D_NNUNET_PRED"])
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E42_nnunet_valve.json"
N_ANNULUS, N_DEPTH = 32, 6
RESAMPLES, SEED = 20_000, 20260911


def score_one(fit_labels, reference_labels, zooms, case):
    """Fit to fit_labels, score against reference_labels."""
    fit = fit_valve(fit_labels, zooms, n_annulus=N_ANNULUS, n_depth=N_DEPTH, case=case)
    if fit is None:
        return None
    record: dict[str, Any] = {}
    for name, sheet, bins, label in (
        ("anterior", fit.anterior, fit.anterior_root_indices, ANTERIOR_LABEL),
        ("posterior", fit.posterior, fit.posterior_root_indices, POSTERIOR_LABEL),
    ):
        faces = leaflet_faces(bins, N_DEPTH, N_ANNULUS)
        voxels = np.argwhere(reference_labels == label)
        if len(faces) == 0 or len(voxels) < 50:
            return None
        points = voxels * np.asarray(zooms)
        record[name] = symmetric_surface_distance(
            points, sheet.reshape(-1, 3), faces, sample_faces=6)
    return record


def boot_median(values, rng):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    point = float(np.median(v))
    idx = rng.integers(0, v.size, size=(RESAMPLES, v.size))
    b = np.median(v[idx], axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"median": point, "ci_low": float(lo), "ci_high": float(hi), "n": int(v.size)}


def main() -> int:
    import nibabel as nib

    rows = []
    for ref_path in sorted(REFERENCE.glob("*-label.nii.gz")):
        case = ref_path.name.replace("-label.nii.gz", "")
        pred_path = PREDICTED / ("%s.nii.gz" % case)
        if not pred_path.is_file():
            print("  %s: no prediction, skipped" % case, flush=True)
            continue
        handle = nib.load(str(ref_path))
        reference = np.asarray(handle.dataobj)
        zooms = tuple(float(z) for z in handle.header.get_zooms()[:3])
        predicted = np.asarray(nib.load(str(pred_path)).dataobj)
        if predicted.shape != reference.shape:
            print("  %s: shape mismatch %s vs %s, skipped"
                  % (case, predicted.shape, reference.shape), flush=True)
            continue

        entry = {"case": case}
        for tag, source in (("reference_input", reference), ("nnunet_input", predicted)):
            entry[tag] = score_one(source, reference, zooms, case)
        rows.append(entry)
        def show(tag, leaflet):
            v = entry[tag]
            return float("nan") if v is None else v[leaflet]["symmetric_mean_mm"]
        print("  %-10s ref A/P %.3f/%.3f   nnunet A/P %.3f/%.3f"
              % (case, show("reference_input", "anterior"), show("reference_input", "posterior"),
                 show("nnunet_input", "anterior"), show("nnunet_input", "posterior")), flush=True)

    rng = np.random.default_rng(SEED)
    summary = {}
    for tag in ("reference_input", "nnunet_input"):
        usable = [r for r in rows if r[tag] is not None]
        summary[tag] = {"n_cases": len(usable), "n_failed": len(rows) - len(usable)}
        for leaflet in ("anterior", "posterior"):
            for key, name in (("symmetric_mean_mm", "symmetric"), ("hausdorff_mm", "hausdorff")):
                summary[tag]["%s_%s_mm" % (leaflet, name)] = boot_median(
                    [r[tag][leaflet][key] for r in usable], rng)

    # Paired, over the cases where both fits succeeded: the two arms differ only
    # in which labels the fit consumed.
    paired = {}
    both = [r for r in rows if r["reference_input"] and r["nnunet_input"]]
    for leaflet in ("anterior", "posterior"):
        diff = np.array([r["nnunet_input"][leaflet]["symmetric_mean_mm"]
                         - r["reference_input"][leaflet]["symmetric_mean_mm"] for r in both])
        idx = rng.integers(0, diff.size, size=(RESAMPLES, diff.size))
        boot = np.median(diff[idx], axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        signs = rng.choice([-1.0, 1.0], size=(200_000, diff.size))
        null = np.median(signs * diff, axis=1)
        observed = float(np.median(diff))
        p = float((np.sum(np.abs(null) >= abs(observed) - 1e-15) + 1) / 200_001)
        paired[leaflet] = {"difference": observed, "ci_low": float(lo),
                           "ci_high": float(hi), "p_value": p, "n": int(diff.size)}

    result = {"experiment": "E42_nnunet_valve",
              "what_this_measures": ("our leaflet fitting fed by nnU-Net segmentations against "
                                     "the same fitting fed by reference labels, both scored "
                                     "against the reference labels; a front-end comparison, "
                                     "not a mesh-reconstruction one"),
              "split": "MVSeg2023 challenge test, 40 cases",
              "predictions": str(PREDICTED), "n_rows": len(rows),
              "summary": summary, "paired_nnunet_minus_reference": paired, "rows": rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

    print("\n%s" % json.dumps(summary, indent=2))
    print("\npaired, nnU-Net minus reference labels:")
    for leaflet, stat in paired.items():
        print("  %-10s %+.3f mm  [%+.3f, %+.3f]  p=%.4g  (n=%d)"
              % (leaflet, stat["difference"], stat["ci_low"], stat["ci_high"],
                 stat["p_value"], stat["n"]))
    print("\nwrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
