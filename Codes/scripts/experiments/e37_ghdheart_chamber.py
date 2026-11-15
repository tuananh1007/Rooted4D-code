"""E37: GHDHeart on our held-out ACDC patients, scored with our metric.

GHDHeart is the first external comparison in the campaign because it needs no
training and no released weights: it is an optimization method that fits a
canonical mesh to a label volume. That makes it directly comparable to our
label-input row, with no argument about training budgets.

Three things had to be got right before the number means anything.

*The target surface.* The shipped demo fits ``label_value[2]``, which in ACDC is
the myocardium, and puts the cavity into the negative set. Our panel A scores
the cavity. Both are called "LV", so running the demo as shipped would produce
a confidently wrong comparison. The whole-heart canonicals include ``3.obj``,
which is the cavity (``canonical_shapes/wholeheart_canonical/readme.txt`` gives
``lv: 1, la: 2, cav: 3``), and that is what we fit.

*The frame.* GHDHeart works in a normalized box where its canonicals are scaled
so that [-1, 1] spans 200 mm, and it reaches physical coordinates through an
affine. Our reference points are index times spacing. Fitting in a per-axis
normalized box would distort the aspect ratio, so we fit in a frame that is our
millimetre frame divided by a scalar, which preserves shape, and multiply back
afterwards.

*Proof that the frame is right.* A frame error produces plausible numbers
rather than an exception, so the script first runs a control: take the cavity
label, march it to a mesh through the same transform, and score it against the
reference points. That has to come out at roughly half a voxel. If it does not,
the mapping is wrong and every GHDHeart number below it is meaningless, so the
run aborts rather than reporting.

Sweeps ``num_basis``, GHD's capacity parameter, because only one cavity
canonical is shipped and so canonical resolution cannot be varied. Reports
GHDHeart's best setting, selected on validation patients.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

GHD_ROOT = Path(os.environ.get("GHD_ROOT", "/work/nvme/bhio/avu2/Rooted4D/external/GHDHeart"))
sys.path.insert(0, str(GHD_ROOT))

import nibabel as nib
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes

from GHD.core import GHD_config
from GHD.GHD_cardiac import GHD_Cardiac

from rooted4d.experiments.component_eval import (
    cavity_boundary_points,
    symmetric_surface_distance,
)
from rooted4d.segmentation import load_patient_splits

ACDC = os.environ.get("R4D_ACDC", "data/raw/acdc")
SPLITS = load_patient_splits(os.environ["R4D_SPLITS"])
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / os.environ.get(
    "R4D_OUTNAME", "E37_ghdheart_chamber.json")
CANONICAL = GHD_ROOT / "canonical_shapes" / "wholeheart_canonical" / "3.obj"
CAVITY = 3
# The canonical is normalized so that [-1, 1] spans 200 mm, so dividing our
# millimetre frame by 100 puts the target at the canonical's own scale.
RESCALAR = 100.0
BASIS_GRID = [int(b) for b in os.environ.get("R4D_BASIS", "16,36,64,100").split(",")]
# "validate" scores one basis on the validation patients; "test" scores the
# already-selected basis on the test patients. Splitting the two keeps a single
# array task inside a walltime and keeps selection off the test split.
MODE = os.environ.get("R4D_MODE", "validate")
SKIP_CONTROL = os.environ.get("R4D_SKIP_CONTROL", "0") == "1"
ITERATIONS = int(os.environ.get("R4D_ITER", "600"))
LR = float(os.environ.get("R4D_LR", "0.01"))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# The control compares a marching-cubes surface against voxel-centre points, so
# it can never reach zero: it is limited by discretization along the coarsest
# axis, which for ACDC is the 10 mm slice thickness. Half that is the natural
# bound, and it still catches a gross frame error by an order of magnitude,
# since a swapped axis displaces the surface by the size of the heart.
CONTROL_TOLERANCE_VOXELS = float(os.environ.get("R4D_CONTROL_TOL_VOXELS", "0.5"))


def normalized_to_mm(vertices: torch.Tensor, shape, zooms) -> np.ndarray:
    """Map GHDHeart's normalized coordinates into our index-times-spacing frame.

    GHDHeart stacks its coordinate components reversed relative to array axes,
    so component 0 varies along axis 2 and component 2 along axis 0. Getting
    this backwards is the frame error the control below exists to catch.
    """

    v = vertices.detach().cpu().numpy().astype(np.float64)
    n0, n1, n2 = shape
    out = np.empty_like(v)
    out[:, 0] = (v[:, 2] + 1.0) * 0.5 * (n0 - 1) * zooms[0]
    out[:, 1] = (v[:, 1] + 1.0) * 0.5 * (n1 - 1) * zooms[1]
    out[:, 2] = (v[:, 0] + 1.0) * 0.5 * (n2 - 1) * zooms[2]
    return out


def normalized_coordinate_grid(shape, device):
    """The same grid GHDHeart builds: linspace(-1, 1) per axis, components reversed."""

    axes = [torch.linspace(-1.0, 1.0, int(n), device=device) for n in shape]
    grid = torch.meshgrid(axes, indexing="ij")
    return torch.stack(grid[::-1], dim=-1)


def aspect_preserving_scale(shape, zooms):
    """Per-axis factors taking the normalized box to millimetres over RESCALAR."""

    n0, n1, n2 = shape
    return torch.tensor(
        [(n2 - 1) * zooms[2] * 0.5 / RESCALAR,
         (n1 - 1) * zooms[1] * 0.5 / RESCALAR,
         (n0 - 1) * zooms[0] * 0.5 / RESCALAR],
        dtype=torch.float32,
    )


def frames(split):
    for gt_path in sorted(Path(ACDC).rglob("*_gt.nii.gz")):
        patient = gt_path.parent.name
        if SPLITS.get(patient) != split:
            continue
        handle = nib.load(str(gt_path))
        labels = np.asarray(handle.dataobj)
        zooms = tuple(float(z) for z in handle.header.get_zooms()[:3])
        frame = int(gt_path.name.split("_frame")[-1].split("_")[0])
        yield patient, frame, labels, zooms


def control_check():
    """March the cavity label into a mesh and score it against the reference.

    This passes through exactly the transform the fitted mesh will, so it tests
    the mapping and nothing else. The answer must be about half a voxel.
    """

    worst_ratio = 0.0
    checked = 0
    for patient, frame, labels, zooms in frames("test"):
        occupancy = torch.as_tensor((labels == CAVITY).astype(np.float32))[None]
        mesh = cubify(occupancy, 0.5)
        if mesh.isempty() or mesh.verts_packed().shape[0] == 0:
            continue
        vertices = normalized_to_mm(mesh.verts_packed(), labels.shape, zooms)
        faces = mesh.faces_packed().cpu().numpy()
        reference = cavity_boundary_points(labels, zooms, cavity_label=CAVITY)
        score = symmetric_surface_distance(reference, vertices, faces)
        bound = CONTROL_TOLERANCE_VOXELS * max(zooms)
        ratio = score["symmetric_mean_mm"] / bound
        worst_ratio = max(worst_ratio, ratio)
        checked += 1
        print("  control %s f%02d  %.3f mm  (bound %.2f, ratio %.2f)"
              % (patient, frame, score["symmetric_mean_mm"], bound, ratio), flush=True)
        if ratio > 1.0:
            break
    return worst_ratio, checked


def fit_one(labels, zooms, num_basis):
    device = torch.device(DEVICE)
    grid = normalized_coordinate_grid(labels.shape, device)
    scale = aspect_preserving_scale(labels.shape, zooms).to(device)
    grid = grid * scale  # aspect-preserving, canonical-scaled

    label_tensor = torch.as_tensor(np.ascontiguousarray(labels), device=device)
    positive_mask = label_tensor == CAVITY
    if int(positive_mask.sum()) < 200:
        return None
    positives = grid[positive_mask]
    negatives = grid[~positive_mask]

    low = positives.min(dim=0)[0] - 0.05
    high = positives.max(dim=0)[0] + 0.05
    inside = ((negatives > low) & (negatives < high)).all(dim=1)
    negatives = negatives[inside]

    cfg = GHD_config(
        base_shape_path=str(CANONICAL),
        num_basis=num_basis,
        mix_laplacian_tradeoff={"cotlap": 1.0, "dislap": 0.1, "stdlap": 0.1},
        device=device,
        if_nomalize=True,
        if_return_scipy=True,
    )
    heart = GHD_Cardiac(cfg)

    sample_count = min(2000, positives.shape[0])
    choice = np.random.choice(positives.shape[0], sample_count, replace=False)
    heart.global_registration(positives[choice].detach().cpu().numpy())
    heart.fitting2target(
        positives, negatives,
        loss_dict={"Loss_occupancy": 1, "Loss_Laplacian": 0.001, "Loss_thickness": 0.001},
        lr_start=LR, num_iter=ITERATIONS, if_reset=True,
        if_fit_R=False, if_fit_s=True, if_fit_T=True, Distance_weighted=1.,
    )
    mesh: Meshes = heart.rendering()
    vertices = mesh.verts_packed() / scale  # back to the normalized box
    return normalized_to_mm(vertices, labels.shape, zooms), mesh.faces_packed().cpu().numpy()


class SystematicFailure(RuntimeError):
    """Raised when every fit fails the same way, which is a bug and not a result."""


def evaluate(split, num_basis):
    rows = []
    # A missing dependency makes every fit raise identically, which the
    # per-frame handler below would otherwise record as sixty independent
    # failures and carry on for hours. Fail on the pattern instead.
    consecutive_failures = 0
    first_reason = None
    for patient, frame, labels, zooms in frames(split):
        started = time.time()
        try:
            fitted = fit_one(labels, zooms, num_basis)
        except Exception as exc:  # a failed fit is a result, not a crash
            reason = "%s: %s" % (type(exc).__name__, str(exc)[:120])
            rows.append({"patient": patient, "frame": frame, "failed": True,
                         "reason": reason})
            print("  B%-3d %s f%02d  FAILED %s" % (num_basis, patient, frame, exc), flush=True)
            if first_reason is None or reason == first_reason:
                first_reason = reason
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    raise SystematicFailure(
                        "three consecutive fits failed identically (%s); this is a "
                        "configuration or dependency problem, not a per-case result"
                        % reason) from exc
            else:
                consecutive_failures = 0
            continue
        if fitted is None:
            rows.append({"patient": patient, "frame": frame, "failed": True,
                         "reason": "too few cavity voxels"})
            continue
        vertices, faces = fitted
        reference = cavity_boundary_points(labels, zooms, cavity_label=CAVITY)
        score = symmetric_surface_distance(reference, vertices, faces)
        consecutive_failures = 0
        rows.append({"patient": patient, "frame": frame, "failed": False,
                     "seconds": time.time() - started, **score})
        print("  B%-3d %s f%02d  symmetric %.3f  hausdorff %.3f  (%.1fs)"
              % (num_basis, patient, frame, score["symmetric_mean_mm"],
                 score["hausdorff_mm"], time.time() - started), flush=True)
    return rows


def summarize(rows):
    good = [r for r in rows if not r["failed"]]
    if not good:
        return {"n": 0}
    def column(key):
        v = np.array([r[key] for r in good], dtype=float)
        return {"median": float(np.median(v)), "mean": float(v.mean()),
                "p95": float(np.percentile(v, 95)), "max": float(v.max())}
    return {"n": len(good), "n_failed": len(rows) - len(good),
            "symmetric_mm": column("symmetric_mean_mm"),
            "hausdorff_mm": column("hausdorff_mm")}


def main() -> int:
    print("device %s | mode %s | canonical %s" % (DEVICE, MODE, CANONICAL), flush=True)
    if not CANONICAL.is_file():
        print("cavity canonical not found", file=sys.stderr)
        return 1

    result = {"experiment": "E37_ghdheart_chamber", "mode": MODE,
              "canonical": str(CANONICAL), "iterations": ITERATIONS, "lr": LR,
              "device": DEVICE, "basis_grid": BASIS_GRID}

    if not SKIP_CONTROL:
        print("### control: does the frame mapping hold?", flush=True)
        worst_ratio, checked = control_check()
        print("worst control ratio to the half-voxel bound: %.2f over %d frames"
              % (worst_ratio, checked), flush=True)
        if worst_ratio > 1.0 or checked == 0:
            print("frame mapping is wrong; refusing to report GHDHeart numbers",
                  file=sys.stderr)
            return 1
        result["control_worst_ratio"] = worst_ratio
        result["control_frames"] = checked

    split = "val" if MODE == "validate" else "test"
    result["split"] = split
    result["results"] = {}
    for basis in BASIS_GRID:
        print("\n### %s, num_basis=%d" % (split, basis), flush=True)
        try:
            rows = evaluate(split, basis)
        except SystematicFailure as exc:
            print("ABORT: %s" % exc, file=sys.stderr)
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            result["aborted"] = str(exc)
            OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
            return 1
        result["results"][str(basis)] = {"summary": summarize(rows), "rows": rows}
        print("  -> B%d %s: %s" % (basis, split,
                                   json.dumps(result["results"][str(basis)]["summary"])),
              flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
