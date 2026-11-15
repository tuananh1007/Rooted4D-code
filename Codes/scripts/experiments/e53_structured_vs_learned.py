r"""E53: structured learning against unstructured learning, at matched size.

E52 compared the interface-conditioned basis against ``smooth_shape_basis`` and
found a large win, and that comparison used the wrong incumbent. The real-donor
benchmark does not use the trigonometric basis at all: ``fit_benchmark_model``
learns the shape space by singular value decomposition of the training
identities, and the trigonometric basis appears only in the procedural test
fixture. A data-adaptive basis is a far stronger baseline than an analytic one,
so the earlier margin said little.

The question that actually decides the direction is therefore:

    Does learning *within* the interface-conditioned structure beat learning
    without it, at the same number of coefficients, on held-out identities?

Both arms learn from the training identities alone.

*   ``learned_global`` -- the incumbent. One principal subspace over the whole
    displacement field, exactly what ``fit_benchmark_model`` builds.
*   ``learned_structured`` -- the same amount of learning, allocated inside the
    Craig--Bampton split: a principal subspace for the interface displacement
    and a separate one for each component's interior, where interior modes
    cannot move the interface.

If the structured arm does not win here, the direction has no representational
argument and would have to earn its place purely through inference. Reporting
that outcome is the point of running this before building the inference.
"""

import json
import os
from pathlib import Path

import numpy as np

from rooted4d.geometry import make_toy_complex
from rooted4d.geometry.interface_modes import interface_conditioned_basis

BENCHMARK = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3"))
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E53_structured_vs_learned.json"
BUDGETS = [int(v) for v in os.environ.get("R4D_BUDGETS", "12,24,32,48,64").split(",")]


def principal_basis(rows: np.ndarray, n: int) -> np.ndarray:
    """Leading right singular vectors of mean-removed rows, as columns."""
    if n <= 0 or rows.size == 0:
        return np.zeros((rows.shape[1], 0))
    _, _, right = np.linalg.svd(rows, full_matrices=False)
    return right[: min(n, right.shape[0])].T.copy()


def main() -> int:
    complex_ = make_toy_complex(n_annulus=24)
    vertices = complex_.global_vertices
    interface = np.sort(complex_.groups["annulus"])
    free = np.sort(complex_.groups["leaflet_free"])
    components = {
        "chamber": np.sort(np.union1d(complex_.groups["chamber_unique"], interface)),
        "anterior_leaflet": np.sort(np.union1d(free[: free.size // 2], interface)),
        "posterior_leaflet": np.sort(np.union1d(free[free.size // 2 :], interface)),
    }
    faces = {name: complex_.faces[name] for name in components}
    n_physical = vertices.shape[0]

    # Full-capacity structured parameterization, used only to define the split.
    capacity = {name: int(np.setdiff1d(idx, interface).size)
                for name, idx in components.items()}
    structure = interface_conditioned_basis(
        vertices, faces, components, interface, capacity)

    def load(split):
        paths = sorted((BENCHMARK / "samples" / split).glob("*.npz"))
        if not paths:
            raise SystemExit("no samples under %s" % (BENCHMARK / "samples" / split))
        return np.stack([np.load(p)["template_vertices"] for p in paths])

    train, test = load("train"), load("test")
    mean_shape = train.mean(axis=0)
    train_u = train - mean_shape
    test_u = test - mean_shape
    print("train %d | test %d identities" % (len(train), len(test)))

    def split_fields(fields):
        """Interface displacement, and each component's interior residual."""
        a = fields[:, interface, :].reshape(len(fields), -1)
        residuals = {}
        for name, block in structure.blocks.items():
            boundary, interior = block["boundary"], block["interior"]
            position = {int(v): i for i, v in enumerate(interface)}
            take = np.array([position[int(v)] for v in boundary])
            propagated = np.einsum(
                "ij,njk->nik", block["constraint_modes"],
                fields[:, interface, :][:, take, :])
            actual = fields[:, np.concatenate([boundary, interior]), :]
            residuals[name] = (actual - propagated)[:, boundary.size:, :].reshape(
                len(fields), -1)
        return a, residuals

    train_a, train_z = split_fields(train_u)
    test_a, test_z = split_fields(test_u)

    def structured_error(interface_dim, interior_dims):
        """Reconstruction error of the structured arm at a given allocation."""
        a_basis = principal_basis(train_a, interface_dim)
        a_hat = (test_a @ a_basis) @ a_basis.T if a_basis.shape[1] else np.zeros_like(test_a)
        total = np.zeros((len(test_u), n_physical, 3))
        a_fields = a_hat.reshape(len(test_u), interface.size, 3)
        position = {int(v): i for i, v in enumerate(interface)}
        multiplicity = np.zeros(n_physical)
        for name, block in structure.blocks.items():
            boundary, interior = block["boundary"], block["interior"]
            take = np.array([position[int(v)] for v in boundary])
            propagated = np.einsum("ij,njk->nik", block["constraint_modes"],
                                   a_fields[:, take, :])
            z_basis = principal_basis(train_z[name], interior_dims[name])
            if z_basis.shape[1]:
                z_hat = (test_z[name] @ z_basis) @ z_basis.T
            else:
                z_hat = np.zeros_like(test_z[name])
            interior_field = z_hat.reshape(len(test_u), interior.size, 3)
            rows = np.concatenate([boundary, interior])
            contribution = propagated.copy()
            contribution[:, boundary.size:, :] += interior_field
            total[:, rows, :] += contribution
            multiplicity[rows] += 1.0
        touched = multiplicity > 0
        total[:, touched, :] /= multiplicity[touched, None]
        residual = test_u - total
        return float(np.median(np.sqrt((residual ** 2).sum(-1).mean(1))))

    flat_train = train_u.reshape(len(train_u), -1)
    flat_test = test_u.reshape(len(test_u), -1)

    rows = []
    for budget in BUDGETS:
        basis = principal_basis(flat_train, budget)
        approx = (flat_test @ basis) @ basis.T
        residual = (flat_test - approx).reshape(len(test_u), -1, 3)
        global_error = float(np.median(np.sqrt((residual ** 2).sum(-1).mean(1))))

        # Search allocations of the same budget between interface and interiors.
        best = None
        for interface_dim in range(1, budget):
            remaining = budget - interface_dim
            per = remaining // len(components)
            if per < 1:
                continue
            dims = {name: min(per, 3 * capacity[name]) for name in components}
            used = interface_dim + sum(dims.values())
            if used > budget:
                continue
            value = structured_error(interface_dim, dims)
            if best is None or value < best[0]:
                best = (value, interface_dim, per, used)
        row = {"budget": budget, "learned_global": global_error}
        if best:
            row.update({"learned_structured": best[0],
                        "interface_dim": best[1],
                        "interior_dim_per_component": best[2],
                        "dimensions_used": best[3]})
        rows.append(row)
        print("  budget %3d  learned_global %7.4f | learned_structured %s"
              % (budget, global_error,
                 ("%7.4f  (interface %d, interior %d/comp, used %d)"
                  % (best[0], best[1], best[2], best[3])) if best else "--"), flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(
        {"experiment": "E53_structured_vs_learned",
         "train_identities": int(len(train)), "test_identities": int(len(test)),
         "incumbent": "principal subspace of the whole displacement field, as "
                      "fit_benchmark_model builds it",
         "rows": rows}, indent=2) + "\n")
    print("\nwrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
