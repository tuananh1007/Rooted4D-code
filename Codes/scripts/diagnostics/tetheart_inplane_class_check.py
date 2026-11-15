"""Is an in-plane extent test a valid class check where the full-extent one is not?

The full-extent test compares |pred extent - cavity extent| over all three axes.
Every predicted class overshoots the label along the stack axis by ~35 mm, so
that sum is dominated by a term common to all classes and the winner is noise.
The in-plane extent (x, y) is not affected by the overhang. This checks whether
the in-plane test recovers the code-derived answer (cavity = *_2_pred.obj) on
both the cases the full-extent test kept and the ones it withheld."""
import json
from pathlib import Path
import numpy as np, nibabel as nib, trimesh

RES = Path("/work/nvme/bhio/avu2/Rooted4D/results/22102244_thfinal")
PRED = RES / "preds_test_iter_60000"
TRANSFORMS = Path("/work/nvme/bhio/avu2/Rooted4D/results/22045910_tetframe/E50_tet_frame.json")
ACDC = Path("/work/hdd/bhio/avu2/Rooted4D/data/raw/acdc")
PATCH = np.array([64.0]*3)

tf = {}
for row in json.loads(TRANSFORMS.read_text())["rows"]:
    if row.get("status") == "written":
        tf[(row["case"], row["frame"])] = row

def to_mm(v, r):
    v = (np.asarray(v, float)/2.0 + 0.5)*PATCH - 0.5
    perm=np.asarray(r["permutation"]); flips=np.asarray(r["flips"],float)
    scale=np.asarray(r["scale"],float); offset=np.asarray(r["offset"],float)
    p=(v-offset)/scale/flips; out=np.empty_like(p); out[:,perm]=p; return out

def boundary(lab, z, label):
    occ=lab==label; pad=np.pad(occ,1)
    inter=(pad[1:-1,1:-1,1:-1]&pad[:-2,1:-1,1:-1]&pad[2:,1:-1,1:-1]
           &pad[1:-1,:-2,1:-1]&pad[1:-1,2:,1:-1]&pad[1:-1,1:-1,:-2]&pad[1:-1,1:-1,2:])
    r,c,s=np.nonzero(occ&~inter)
    return np.stack([r*z[0],c*z[1],s*z[2]],1)

lk={}
for p in ACDC.rglob("*_frame??_gt.nii.gz"):
    lk[(p.parent.name,int(p.name.split("_frame")[1][:2]))]=p

d=json.loads((RES/"E49_test_iter_60000.json").read_text())
rows=[r for r in d["rows"] if r.get("status") in ("scored","withheld_extent_disagreement")]
agree_full=agree_plane=0; n=0
for rec in rows:
    pat,fr,name=rec["patient"],rec["frame"],rec["name"]
    src=lk.get((pat,fr)); t=tf.get((pat,fr))
    if src is None or t is None: continue
    h=nib.load(str(src)); lab=np.asarray(h.dataobj); z=tuple(float(x) for x in h.header.get_zooms()[:3])
    cav=boundary(lab,z,3); cav_ext=np.ptp(cav,axis=0)
    full={}; plane={}
    for idx in (1,2,3):
        f=PRED/("%s_%d_pred.obj"%(name,idx))
        if not f.exists(): continue
        m=trimesh.load(str(f),process=False,force="mesh")
        ext=np.ptp(to_mm(m.vertices,t),axis=0)
        full[idx]=float(np.abs(ext-cav_ext).sum())
        plane[idx]=float(np.abs(ext[:2]-cav_ext[:2]).sum())
    if not plane: continue
    n+=1
    bf=min(full,key=full.get); bp=min(plane,key=plane.get)
    agree_full+= (bf==2); agree_plane+= (bp==2)
    print("  %-12s f%02d %-9s full->c%d %s | inplane->c%d %s"%(
        pat,fr,rec["status"][:9],bf," ".join("c%d %.1f"%(k,v) for k,v in sorted(full.items())),
        bp," ".join("c%d %.1f"%(k,v) for k,v in sorted(plane.items()))))
print("cases %d | full-extent picks c2: %d | in-plane picks c2: %d"%(n,agree_full,agree_plane))
