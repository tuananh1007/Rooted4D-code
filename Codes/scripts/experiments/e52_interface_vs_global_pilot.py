r"""E52: does interface-conditioning represent real anatomy better, at matched size?

This is the gate for the interface-conditioned direction, and it is designed to
be able to fail. The hypothesis is that some of the accuracy cost currently
attributed to attachment is really compression: a single global basis makes the
chamber and the leaflets move together, and the v3 representation floors sit
close enough to the fusion errors that this is plausible.

The test is a pure representation question, deliberately separated from
inference. Given real donor geometry, how much of it can each basis express?

*   ``global_trig`` -- the current ``smooth_shape_basis``, one trigonometric
    basis over every physical vertex.
*   ``global_laplacian`` -- eigenmodes of the *whole-template* cotangent
    Laplacian, with no interface conditioning. This control is the one that
    matters. The interface-conditioned basis is built from the mesh
    connectivity, so without it a win could simply mean that Laplacian
    eigenmodes beat trigonometric functions of Cartesian position, which would
    be a useful engineering change and no argument for interface conditioning.
*   ``interface_*`` -- the Craig--Bampton split, with a shared interface block
    and per-component Dirichlet interiors.

Both are given the **same number of free coefficients**, because a basis with
more parameters fitting better is not a finding. Two variants of the second are
reported: one with the interface free, three coefficients per annular vertex,
and one with the interface itself reduced to a low-frequency ring basis, which
is what a usable method would do since a free interface spends most of the
budget on it.

Reported on the benchmark's per-sample composite identity geometry, with no
pose, phase or session variation, so nothing here depends on the solver.

An earlier version pooled the raw heart and valve donors and produced a
twenty-four-fold improvement, which was an artifact and not a result. Each donor
type carries real geometry only for its own structure and placeholder
coordinates elsewhere: heart donors place the free leaflet vertices at about
(-105, -122, -21), nowhere near their own annulus, and valve donors do the same
to the chamber. Pooling them creates variance that a component-wise basis
absorbs by construction, so the comparison measured the pooling convention.
The composite identities used here are single coherent hearts.

Both bases are analytic functions of the template and its connectivity, fitted
to no data, so projecting held-out samples onto them leaks nothing. The mean
shape is the one quantity estimated from data and is taken from the training
split alone.
"""

import json
import os
from pathlib import Path

import numpy as np

from rooted4d.geometry import make_toy_complex
from rooted4d.geometry.interface_modes import (
    cotangent_laplacian,
    interface_conditioned_basis,
)
from rooted4d.inference.scene import smooth_shape_basis

BENCHMARK = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3"))
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E52_interface_vs_global.json"
BUDGETS = [int(v) for v in os.environ.get("R4D_BUDGETS", "36,60,90,120,150").split(",")]


def ring_basis(vertices: np.ndarray, interface: np.ndarray, n_modes: int) -> np.ndarray:
    """Low-frequency basis on the interface ring, by arc length.

    The annulus is a closed curve, so its natural reduced basis is the circular
    Fourier series in arc length rather than the trigonometric functions of
    Cartesian position the global basis uses.
    """

    points = vertices[interface]
    steps = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    total = arc[-1] + steps[-1]
    phase = 2.0 * np.pi * arc / max(total, 1e-12)
    columns = [np.ones_like(phase)]
    order = 1
    while len(columns) < n_modes:
        columns.append(np.cos(order * phase))
        if len(columns) < n_modes:
            columns.append(np.sin(order * phase))
        order += 1
    scalar = np.stack(columns[:n_modes], axis=1)
    orthonormal, _ = np.linalg.qr(scalar)
    return orthonormal[:, :n_modes]


def residual_rms(basis_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-donor RMS residual of the least-squares projection onto a basis."""

    coefficients, *_ = np.linalg.lstsq(basis_matrix, targets.T, rcond=None)
    residual = targets.T - basis_matrix @ coefficients
    return np.sqrt((residual ** 2).reshape(-1, targets.shape[0]).mean(axis=0))


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

    def load(split):
        paths = sorted((BENCHMARK / "samples" / split).glob("*.npz"))
        if not paths:
            raise SystemExit("no samples under %s" % (BENCHMARK / "samples" / split))
        return np.stack([np.load(p)["template_vertices"].reshape(-1) for p in paths])

    train_shapes = load("train")
    test_shapes = load("test")
    mean_shape = train_shapes.mean(axis=0)     # the only quantity taken from data
    targets = test_shapes - mean_shape
    n_physical = vertices.shape[0]
    print("train %d | test %d composite identities | physical vertices %d | ambient %d"
          % (train_shapes.shape[0], test_shapes.shape[0], n_physical, targets.shape[1]))

    interior_capacity = {
        name: int(np.setdiff1d(idx, interface).size) for name, idx in components.items()
    }
    print("interior capacity per component: %s" % interior_capacity)

    def interface_matrix(interface_modes_count, interior_split):
        """Dense (3N, d) matrix for the interface-conditioned basis."""
        # A component cannot carry more interior modes than it has interior
        # vertices; requesting more silently mismatched the assembler.
        counts = {name: min(interior_split, interior_capacity[name])
                  for name in components}
        basis = interface_conditioned_basis(
            vertices, faces, components, interface, counts)
        ring = (ring_basis(vertices, interface, interface_modes_count)
                if interface_modes_count else np.eye(interface.size))
        columns = []
        # Interface coefficients: three Cartesian components per ring mode.
        for mode in range(ring.shape[1]):
            for axis in range(3):
                a = np.zeros((interface.size, 3))
                a[:, axis] = ring[:, mode]
                columns.append(basis.assemble(a, None).reshape(-1))
        # Interior coefficients, per component and axis.
        for name in components:
            k = basis.blocks[name]["interior_modes"].shape[1]
            for mode in range(k):
                for axis in range(3):
                    z = {n: np.zeros((counts[n], 3)) for n in counts}
                    z[name][mode, axis] = 1.0
                    columns.append(
                        basis.assemble(np.zeros((interface.size, 3)), z).reshape(-1))
        matrix = np.stack(columns, axis=1)
        q, _ = np.linalg.qr(matrix)
        return q

    # The control: eigenmodes of the whole template's Laplacian, no interface
    # conditioning, three Cartesian components per scalar mode.
    all_faces = np.vstack([faces[name] for name in components])
    whole = cotangent_laplacian(vertices, all_faces)
    whole = 0.5 * (whole + whole.T)
    _, whole_vectors = np.linalg.eigh(whole)

    def global_laplacian_matrix(dimension):
        scalar_modes = max(dimension // 3, 1)
        columns = []
        for mode in range(min(scalar_modes, whole_vectors.shape[1])):
            for axis in range(3):
                field = np.zeros((n_physical, 3))
                field[:, axis] = whole_vectors[:, mode]
                columns.append(field.reshape(-1))
        matrix = np.stack(columns, axis=1)
        q, _ = np.linalg.qr(matrix)
        return q

    rows = []
    for budget in BUDGETS:
        result = {"budget": budget}
        g = smooth_shape_basis(vertices, min(budget, targets.shape[1]))
        result["global_trig"] = float(np.median(residual_rms(g, targets)))
        gl = global_laplacian_matrix(budget)
        result["global_laplacian"] = float(np.median(residual_rms(gl, targets)))
        result["global_laplacian_dim"] = int(gl.shape[1])

        # Free interface: 3 per annular vertex, remainder split across interiors.
        spent = 3 * interface.size
        if budget > spent:
            per = max((budget - spent) // (3 * len(components)), 0)
            m = interface_matrix(0, per)
            result["interface_free"] = float(np.median(residual_rms(m, targets)))
            result["interface_free_dim"] = int(m.shape[1])

        # Reduced interface: split the budget between ring modes and interiors.
        best = None
        for ring_modes in range(1, 13):
            spent = 3 * ring_modes
            if spent >= budget:
                break
            per = max((budget - spent) // (3 * len(components)), 0)
            if per == 0:
                continue
            m = interface_matrix(ring_modes, per)
            if m.shape[1] > budget:
                continue
            value = float(np.median(residual_rms(m, targets)))
            if best is None or value < best[0]:
                best = (value, ring_modes, per, int(m.shape[1]))
        if best is not None:
            result["interface_reduced"] = best[0]
            result["interface_reduced_ring_modes"] = best[1]
            result["interface_reduced_interior_per_component"] = best[2]
            result["interface_reduced_dim"] = best[3]
        rows.append(result)
        print("  budget %3d  trig %7.4f  laplacian %7.4f | free %s | reduced %s"
              % (budget, result["global_trig"], result["global_laplacian"],
                 ("%7.4f" % result["interface_free"]) if "interface_free" in result else "   --  ",
                 ("%7.4f (ring %d, interior %d/comp)"
                  % (result["interface_reduced"], result["interface_reduced_ring_modes"],
                     result["interface_reduced_interior_per_component"]))
                 if "interface_reduced" in result else "--"), flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"experiment": "E52_interface_vs_global",
                                  "train_identities": int(train_shapes.shape[0]),
                                  "test_identities": int(test_shapes.shape[0]),
                                  "rows": rows}, indent=2) + "\n")
    print("\nwrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
