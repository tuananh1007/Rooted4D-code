"""Are the 7 withheld cases hiding failures? E49 withholds a case when the
predicted class whose extent best matches the cavity is not the expected one.
If the expected mesh is merely a near-tie it is heuristic noise; if it is wildly
off, withholding is removing genuine failures and flattering the method."""
import json, os
from pathlib import Path
import numpy as np, nibabel as nib, trimesh

RES = Path("/work/nvme/bhio/avu2/Rooted4D/results/22102244_thfinal")
PRED = RES / "preds_test_iter_60000"
TRANSFORMS = Path("/work/nvme/bhio/avu2/Rooted4D/results/22045910_tetframe/E50_tet_frame.json")
ACDC = Path("/work/hdd/bhio/avu2/Rooted4D/data/raw/acdc")
PATCH = np.array([64.0]*3)
from rooted4d.experiments.component_eval import symmetric_surface_distance

tf = {}
for row in json.loads(TRANSFORMS.read_text())["rows"]:
    if row.get("status") == "written":
        tf[(row["case"], row["frame"])] = row

def to_mm(v, r):
    v = (np.asarray(v, float)/2.0 + 0.5)*PATCH - 0.5
    perm=np.asarray(r["permutation"]); flips=np.asarray(r["flips"],float)
    scale=np.asarray(r["scale"],float); offset=np.asarray(r["offset"],float)
    p=(v-offset)/scale/flips; out=np.empty_like(p); out[:,perm]=p; return out

def boundary(lab, z, label, drop):
    occ=lab==label; pad=np.pad(occ,1)
    inter=(pad[1:-1,1:-1,1:-1]&pad[:-2,1:-1,1:-1]&pad[2:,1:-1,1:-1]
           &pad[1:-1,:-2,1:-1]&pad[1:-1,2:,1:-1]&pad[1:-1,1:-1,:-2]&pad[1:-1,1:-1,2:])
    r,c,s=np.nonzero(occ&~inter)
    if drop:
        pres=np.unique(np.argwhere(occ)[:,2]); k=(s>pres.min())&(s<pres.max())
        r,c,s=r[k],c[k],s[k]
    return np.stack([r*z[0],c*z[1],s*z[2]],1)

lk={}
for p in ACDC.rglob("*_frame??_gt.nii.gz"):
    lk[(p.parent.name,int(p.name.split("_frame")[1][:2]))]=p

d=json.loads((RES/"E49_test_iter_60000.json").read_text())
wh=[r for r in d["rows"] if r.get("status")=="withheld_extent_disagreement"]
print("withheld cases:",len(wh))
for rec in wh:
    pat,fr,name=rec["patient"],rec["frame"],rec["name"]
    src=lk.get((pat,fr)); t=tf.get((pat,fr))
    h=nib.load(str(src)); lab=np.asarray(h.dataobj); z=tuple(float(x) for x in h.header.get_zooms()[:3])
    cav=boundary(lab,z,3,False); cav_ext=np.ptp(cav,axis=0)
    wall=np.isin(lab,[3,2]); wall_ext=np.ptp(np.argwhere(wall)*np.asarray(z),axis=0)
    line=[]
    for idx in (1,2,3):
        f=PRED/("%s_%d_pred.obj"%(name,idx))
        if not f.exists(): continue
        m=trimesh.load(str(f),process=False,force="mesh")
        ext=np.ptp(to_mm(m.vertices,t),axis=0)
        line.append("c%d L1_to_cav %.1f"%(idx,np.abs(ext-cav_ext).sum()))
    # score the expected class anyway, clipped to the label's z-extent
    m=trimesh.load(str(PRED/("%s_2_pred.obj"%name)),process=False,force="mesh")
    verts=to_mm(m.vertices,t); faces=np.asarray(m.faces)
    ref=boundary(lab,z,3,True)
    mm=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    for nrm,org in (([0,0,1.0],[0,0,ref[:,2].min()]),([0,0,-1.0],[0,0,ref[:,2].max()])):
        if len(mm.faces): mm=trimesh.intersections.slice_mesh_plane(mm,plane_normal=np.array(nrm),plane_origin=np.array(org,float),cap=False)
    s=None
    if mm is not None and len(mm.faces):
        sample=ref[::max(1,len(ref)//2000)]
        s=symmetric_surface_distance(sample,np.asarray(mm.vertices),np.asarray(mm.faces),sample_faces=6)
    print("  %-12s f%02d picked c%d | %s | clipped sym %s" % (
        pat,fr,rec["extent_pick"]," ".join(line),
        ("%.3f"%s["symmetric_mean_mm"]) if s else "n/a"))
