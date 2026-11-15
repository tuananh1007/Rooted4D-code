"""E115: our BCPD port against the authors' binary.

    R4D_BCPD_BIN=<path to bcpd> python e115_bcpd_reference.py <out.json>

Rings of 96 points shaped like the annulus, each with a smooth displacement,
a small rotation and noise, are registered by ``recent_fusion.bcpd_nonrigid``
and by the authors' program with the same options (``-Tn -ux -g0.1 -b -l``).
The deformed sources must agree to 1e-4 mm. A case where our loop stopped at
zero width before the authors' minimum of 30 iterations is reported and
excluded, since there the two codes differ by design.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from rooted4d.inference.recent_fusion import bcpd_nonrigid

TOLERANCE_MM = 1e-4
CONFIGS = [(0.25, 0.1), (1.0, 1.0), (4.0, 100.0), (0.0625, 0.001)]


def case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    target = np.stack([26 * np.cos(angles), 21 * np.sin(angles), 3 * np.cos(2 * angles)], axis=1)
    k = rng.integers(1, 4)
    smooth = np.stack([rng.normal() * np.cos(k * angles + rng.uniform(0, 6)) for _ in range(3)], axis=1)
    angle = np.deg2rad(rng.normal(scale=2.0))
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    source = target @ rotation.T + smooth + rng.normal(scale=0.05, size=target.shape) + rng.normal(scale=0.5, size=3)
    return source, target


def binary(source, target, beta, lam) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savetxt(tmp / "X.txt", target, fmt="%.17g")
        np.savetxt(tmp / "Y.txt", source, fmt="%.17g")
        subprocess.run([os.environ["R4D_BCPD_BIN"], "-x", str(tmp / "X.txt"), "-y", str(tmp / "Y.txt"),
                        "-Tn", "-ux", "-g0.1", f"-b{beta:g}", f"-l{lam:g}", "-q", "-o", str(tmp / "out_")],
                       check=True, capture_output=True)
        return np.loadtxt(tmp / "out_y.txt")


def main(out: str) -> None:
    rows, worst = [], 0.0
    for seed in range(6):
        source, target = case(seed)
        for beta, lam in CONFIGS:
            ours = bcpd_nonrigid(source, target, beta=beta, lam=lam)
            theirs = binary(source, target, beta, lam)
            difference = float(np.abs(ours["deformed"] - theirs).max())
            excluded = ours["stopped_at_zero_width"]
            rows.append({"seed": seed, "beta": beta, "lambda": lam, "max_abs_difference_mm": difference,
                         "iterations": ours["iterations"], "excluded_zero_width": excluded,
                         "displacement_mm": float(np.abs(ours["deformed"] - source).max())})
            if not excluded:
                worst = max(worst, difference)
            print(rows[-1], flush=True)
    compared = sum(not r["excluded_zero_width"] for r in rows)
    report = {"binary": os.environ["R4D_BCPD_BIN"], "options": "-Tn -ux -g0.1 -b<beta> -l<lambda>",
              "tolerance_mm": TOLERANCE_MM, "cases": rows, "compared": compared,
              "max_abs_difference_mm": worst, "passed": bool(compared > 0 and worst <= TOLERANCE_MM)}
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print("compared", compared, "of", len(rows), "| largest difference", worst, "mm | passed", report["passed"])


if __name__ == "__main__":
    main(*sys.argv[1:])
