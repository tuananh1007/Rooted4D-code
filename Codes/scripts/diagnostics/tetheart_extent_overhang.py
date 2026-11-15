"""Diagnose the 44 mm Hausdorff: is the prediction spanning a different extent
than the target? Prints predicted vs GT bounding boxes in mm and splits the
distance by direction."""
import json, os, glob
from pathlib import Path
import numpy as np, nibabel as nib, trimesh

PRED = Path("/work/nvme/bhio/avu2/Rooted4D/results/22102244_thfinal/preds_test_iter_60000")
TRANSFORMS = Path("/work/nvme/bhio/avu2/Rooted4D/results/22045910_tetframe/E50_tet_frame.json")
ACDC = Path("/work/hdd/bhio/avu2/Rooted4D/data/raw/acdc")
PATCH = np.array([64.0, 64.0, 64.0])

tf = {}
for row in json.loads(TRANSFORMS.read_text())["rows"]:
    if row.get("status") == "written":
        tf[(row["case"], row["frame"])] = row

def to_mm(v, r):
    v = (np.asarray(v, float) / 2.0 + 0.5) * PATCH - 0.5
    perm = np.asarray(r["permutation"]); flips = np.asarray(r["flips"], float)
    scale = np.asarray(r["scale"], float); offset = np.asarray(r["offset"], float)
    p = (v - offset) / scale / flips
    out = np.empty_like(p); out[:, perm] = p
    return out

def boundary(labels, zooms, label):
    occ = labels == label
    pad = np.pad(occ, 1)
    inter = (pad[1:-1,1:-1,1:-1] & pad[:-2,1:-1,1:-1] & pad[2:,1:-1,1:-1]
             & pad[1:-1,:-2,1:-1] & pad[1:-1,2:,1:-1]
             & pad[1:-1,1:-1,:-2] & pad[1:-1,1:-1,2:])
    r,c,s = np.nonzero(occ & ~inter)
    return np.stack([r*zooms[0], c*zooms[1], s*zooms[2]], 1)

def bb(p):
    return "x[%.0f,%.0f] y[%.0f,%.0f] z[%.0f,%.0f]  span(%.0f,%.0f,%.0f)" % (
        p[:,0].min(),p[:,0].max(),p[:,1].min(),p[:,1].max(),p[:,2].min(),p[:,2].max(),
        np.ptp(p[:,0]),np.ptp(p[:,1]),np.ptp(p[:,2]))

for f in sorted(PRED.glob("*_2_pred.obj"))[:5]:
    name = f.name
    case = name.split("_")[1]           # patient007
    frame = int(name.split("frame")[1][:2])
    rec = tf.get((case, frame)) or tf.get(("ACDC_"+case, frame))
    if rec is None:
        print(name, "no transform record"); continue
    m = trimesh.load(str(f), process=False, force="mesh")
    pred = to_mm(m.vertices, rec)
    g = sorted(ACDC.glob("**/%s_frame%02d_gt.nii.gz" % (case, frame)))
    if not g:
        print(name, "no gt found"); continue
    img = nib.load(str(g[0])); lab = np.asarray(img.dataobj); z = img.header.get_zooms()[:3]
    cav = boundary(lab, z, 3)
    print("==", name, "zooms", tuple(round(float(x),2) for x in z))
    print("   pred ", bb(pred), " nv=%d" % len(pred))
    print("   gt(3)", bb(cav), " n=%d" % len(cav))
    from scipy.spatial import cKDTree
    d_pg = cKDTree(cav).query(pred)[0]
    d_gp = cKDTree(pred).query(cav)[0]
    print("   pred->gt mean %.2f max %.2f | gt->pred mean %.2f max %.2f"
          % (d_pg.mean(), d_pg.max(), d_gp.mean(), d_gp.max()))
