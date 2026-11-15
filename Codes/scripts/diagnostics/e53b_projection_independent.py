"""E53b: independent check of the E53 projection asymmetry.

Derives the structured span from E53's OWN assembly operator rather than
rebuilding the harmonic matrix by hand: each coefficient direction is pushed
through the same propagate-average-add reconstruction E53 uses, so the columns
span exactly the set of fields E53 can produce. Then:

  step 1 (harness check) reproduce E53's published blockwise number
  step 2 solve the SAME span jointly by least squares

If step 1 matches the published value, the harness is E53's own arithmetic and
step 2 isolates the projection change alone.
"""
import json, os
import numpy as np
from rooted4d.geometry import make_toy_complex
from rooted4d.geometry.interface_modes import interface_conditioned_basis

BENCH = os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3")
from pathlib import Path
BENCH = Path(BENCH)

def principal_basis(rows, n):
    if n <= 0 or rows.size == 0:
        return np.zeros((rows.shape[1], 0))
    _, _, right = np.linalg.svd(rows, full_matrices=False)
    return right[: min(n, right.shape[0])].T.copy()

c = make_toy_complex(n_annulus=24)
vertices = c.global_vertices
interface = np.sort(c.groups["annulus"])
free = np.sort(c.groups["leaflet_free"])
components = {
    "chamber": np.sort(np.union1d(c.groups["chamber_unique"], interface)),
    "anterior_leaflet": np.sort(np.union1d(free[: free.size // 2], interface)),
    "posterior_leaflet": np.sort(np.union1d(free[free.size // 2:], interface)),
}
faces = {n: c.faces[n] for n in components}
n_phys = vertices.shape[0]
capacity = {n: int(np.setdiff1d(i, interface).size) for n, i in components.items()}
structure = interface_conditioned_basis(vertices, faces, components, interface, capacity)

def load(split):
    paths = sorted((BENCH / "samples" / split).glob("*.npz"))
    return np.stack([np.load(p)["template_vertices"] for p in paths])

train, test = load("train"), load("test")
mean = train.mean(axis=0)
train_u, test_u = train - mean, test - mean
position = {int(v): i for i, v in enumerate(interface)}

def split_fields(fields):
    a = fields[:, interface, :].reshape(len(fields), -1)
    res = {}
    for name, b in structure.blocks.items():
        boundary, interior = b["boundary"], b["interior"]
        take = np.array([position[int(v)] for v in boundary])
        prop = np.einsum("ij,njk->nik", b["constraint_modes"],
                         fields[:, interface, :][:, take, :])
        actual = fields[:, np.concatenate([boundary, interior]), :]
        res[name] = (actual - prop)[:, boundary.size:, :].reshape(len(fields), -1)
    return a, res

train_a, train_z = split_fields(train_u)
test_a, test_z = split_fields(test_u)

def assemble(a_fields, z_fields):
    """E53's reconstruction: propagate ring, average overlaps, add interiors."""
    n = len(a_fields)
    total = np.zeros((n, n_phys, 3))
    mult = np.zeros(n_phys)
    for name, b in structure.blocks.items():
        boundary, interior = b["boundary"], b["interior"]
        take = np.array([position[int(v)] for v in boundary])
        prop = np.einsum("ij,njk->nik", b["constraint_modes"], a_fields[:, take, :])
        rows = np.concatenate([boundary, interior])
        contrib = prop.copy()
        if z_fields[name] is not None:
            contrib[:, boundary.size:, :] += z_fields[name].reshape(n, interior.size, 3)
        total[:, rows, :] += contrib
        mult[rows] += 1.0
    t = mult > 0
    total[:, t, :] /= mult[t, None]
    return total

def med_rms(res):
    return float(np.median(np.sqrt((res ** 2).sum(-1).mean(1))))

published = json.loads(Path("results/benchmark/E53_structured_vs_learned.json").read_text())
print("%-8s %-12s %-14s %-14s %s" % ("budget", "published", "my blockwise", "my joint", "verdict"))
for row in published["rows"]:
    budget = row["budget"]; k = row["interface_dim"]; m = row["interior_dim_per_component"]
    a_basis = principal_basis(train_a, k)
    z_bases = {n: principal_basis(train_z[n], m) for n in components}

    # step 1: E53's blockwise rule
    a_hat = (test_a @ a_basis) @ a_basis.T if a_basis.shape[1] else np.zeros_like(test_a)
    z_hat = {n: ((test_z[n] @ z_bases[n]) @ z_bases[n].T if z_bases[n].shape[1]
                 else np.zeros_like(test_z[n])) for n in components}
    blockwise = med_rms(test_u - assemble(a_hat.reshape(-1, interface.size, 3), z_hat))

    # step 2: columns of the SAME span, from the same assembly operator
    cols = []
    zero_z = {n: None for n in components}
    for j in range(a_basis.shape[1]):
        af = a_basis[:, j].reshape(1, interface.size, 3)
        cols.append(assemble(af, zero_z).ravel())
    for name in components:
        for j in range(z_bases[name].shape[1]):
            zf = {n: None for n in components}
            zf[name] = z_bases[name][:, j][None, :]
            cols.append(assemble(np.zeros((1, interface.size, 3)), zf).ravel())
    G = np.column_stack(cols)
    flat = test_u.reshape(len(test_u), -1)
    coef, *_ = np.linalg.lstsq(G, flat.T, rcond=None)
    joint = med_rms((flat - (G @ coef).T).reshape(test_u.shape))
    delta = abs(blockwise - row["learned_structured"])
    print("%-8d %-12.5f %-14.5f %-14.5f harness dev %.2e | cols %d rank %d"
          % (budget, row["learned_structured"], blockwise, joint, delta,
             G.shape[1], np.linalg.matrix_rank(G)))

# Durable artifact: recorded so the correction is reproducible from the repo.
import json as _json
_out = Path("results/benchmark/E53b_projection_independent.json")
_rows = []
for row in published["rows"]:
    budget = row["budget"]; k = row["interface_dim"]; m = row["interior_dim_per_component"]
    a_basis = principal_basis(train_a, k)
    z_bases = {n: principal_basis(train_z[n], m) for n in components}
    a_hat = (test_a @ a_basis) @ a_basis.T if a_basis.shape[1] else np.zeros_like(test_a)
    z_hat = {n: ((test_z[n] @ z_bases[n]) @ z_bases[n].T if z_bases[n].shape[1]
                 else np.zeros_like(test_z[n])) for n in components}
    blockwise = med_rms(test_u - assemble(a_hat.reshape(-1, interface.size, 3), z_hat))
    cols = []
    zero_z = {n: None for n in components}
    for j in range(a_basis.shape[1]):
        cols.append(assemble(a_basis[:, j].reshape(1, interface.size, 3), zero_z).ravel())
    for name in components:
        for j in range(z_bases[name].shape[1]):
            zf = {n: None for n in components}
            zf[name] = z_bases[name][:, j][None, :]
            cols.append(assemble(np.zeros((1, interface.size, 3)), zf).ravel())
    G = np.column_stack(cols)
    flat = test_u.reshape(len(test_u), -1)
    coef, *_ = np.linalg.lstsq(G, flat.T, rcond=None)
    _rows.append({
        "budget": budget,
        "published_blockwise_mm": row["learned_structured"],
        "reproduced_blockwise_mm": blockwise,
        "harness_deviation_mm": abs(blockwise - row["learned_structured"]),
        "joint_projection_mm": med_rms((flat - (G @ coef).T).reshape(test_u.shape)),
        "published_global_mm": row["learned_global"],
        "columns": int(G.shape[1]), "rank": int(np.linalg.matrix_rank(G)),
    })
_out.write_text(_json.dumps({
    "experiment": "E53b_projection_independent",
    "purpose": "Independent confirmation that E53's structured arm was scored by a "
               "blockwise rule while the global arm received an optimal projection.",
    "construction": "span derived by pushing unit coefficient vectors through E53's own "
                    "assemble() operator, not by rebuilding the harmonic matrix",
    "limitation": "allocations were selected on test by E53; this is a corrected diagnostic, "
                  "not a validated held-out result",
    "train_samples": int(len(train)), "test_samples": int(len(test)),
    "rows": _rows}, indent=2) + "\n")
print("wrote", _out)
