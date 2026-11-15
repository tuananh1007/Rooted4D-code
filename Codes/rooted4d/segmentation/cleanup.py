"""Connected-component cleanup of predicted label volumes, in plain NumPy.

The chamber fit (``rooted4d.active.acdc.fit_lv_surface``) resamples a predicted
cavity label slice by slice, taking the outermost boundary pixel at each angle.
A stray pixel anywhere on a slice that also holds cavity therefore pulls the
surface out to it. E84 found one such label set a whole ejection-fraction
interval. Keeping only the largest connected component of the cavity removes
such strays before the fit.

This is written without SciPy, which the cluster environment does not provide.
The cavity masks are small (thousands of voxels), so a breadth-first search over
the foreground is fast enough.
"""

from __future__ import annotations

from collections import deque
from itertools import product

import numpy as np


def _offsets(ndim: int, full: bool) -> list[tuple[int, ...]]:
    """Neighbour offsets: every one of the 3**ndim - 1 (full) or only the 2*ndim face neighbours."""

    return [o for o in product((-1, 0, 1), repeat=ndim)
            if any(o) and (full or sum(abs(v) for v in o) == 1)]


def component_labels(mask: np.ndarray, full_connectivity: bool = True) -> tuple[np.ndarray, list[int]]:
    """Connected components of a boolean array.

    Returns an integer array (0 background, components 1..n numbered in the
    order of their first voxel in C order) and the list of component sizes.
    """

    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    points = [tuple(int(v) for v in p) for p in np.argwhere(mask)]
    foreground = set(points)
    offsets = _offsets(mask.ndim, full_connectivity)
    sizes: list[int] = []
    for start in points:
        if labels[start]:
            continue
        current = len(sizes) + 1
        labels[start] = current
        queue, count = deque([start]), 0
        while queue:
            point = queue.popleft()
            count += 1
            for offset in offsets:
                neighbour = tuple(p + o for p, o in zip(point, offset))
                if neighbour in foreground and not labels[neighbour]:
                    labels[neighbour] = current
                    queue.append(neighbour)
        sizes.append(count)
    return labels, sizes


def largest_component(mask: np.ndarray, full_connectivity: bool = True) -> np.ndarray:
    """The largest connected component of a boolean array (the first of equal ones)."""

    labels, sizes = component_labels(mask, full_connectivity)
    if len(sizes) <= 1:
        return np.asarray(mask, dtype=bool).copy()
    return labels == 1 + int(np.argmax(sizes))


def keep_largest_cavity_component(labels: np.ndarray, cavity_label: int = 3, per_slice: bool = False) -> np.ndarray:
    """A copy of ``labels`` with every cavity voxel outside the largest 3D component set to background.

    Connectivity is full (26 neighbours in 3D). With ``per_slice``, each slice of
    the stack (last axis) then keeps only its largest in-plane component
    (8 neighbours) of what remains. Other labels are untouched.
    """

    out = np.array(labels, copy=True)
    cavity = out == cavity_label
    kept = largest_component(cavity, full_connectivity=True)
    if per_slice:
        for index in range(out.shape[2]):
            if kept[:, :, index].any():
                kept[:, :, index] = largest_component(kept[:, :, index], full_connectivity=True)
    out[cavity & ~kept] = 0
    return out
