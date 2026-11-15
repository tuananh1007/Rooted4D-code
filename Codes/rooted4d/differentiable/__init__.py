r"""Differentiable reference implementations, for exact Jacobians.

Numerical Jacobians are the wall this project kept hitting. A central-difference
Jacobian over sixty parameters costs a hundred and twenty-six residual
evaluations, each of which renders a likelihood volume, which is why the
image-domain experiments in sections 12 and 16 ran on six samples at twelve
modes and the existing refinement path restricts itself to four shape modes.

This package reimplements the pieces that dominate that cost in PyTorch so the
Jacobian can be taken exactly, in one batched forward-mode pass, instead of by
differencing. It is a *reference*, checked against the NumPy implementations it
mirrors rather than replacing them: the NumPy versions remain the definition and
the tests assert the two agree.

**This package never uses the GPU.** Everything is pinned to CPU at import, on
the machine owner's instruction: the available card is small and the display
runs on it, and a solver that fills its memory freezes the desktop. Any tensor
created here is on CPU and no code path calls ``.cuda()`` or ``.to(device)``.
"""

from __future__ import annotations

import os

# Set before torch initializes its device list, so the process never enumerates
# the card at all rather than merely declining to use it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

from .geometry import (  # noqa: E402
    point_to_mesh_distance,
    surface_likelihood,
)
from .scene import (  # noqa: E402
    TorchScene,
    evidence_tensors,
    fit_image_lbfgs,
    image_residuals,
    jacobian,
    landmark_residuals,
    point_residuals,
    prior_residuals,
    prior_weights,
)

__all__ = [
    "TorchScene",
    "evidence_tensors",
    "fit_image_lbfgs",
    "prior_weights",
    "image_residuals",
    "prior_residuals",
    "jacobian",
    "landmark_residuals",
    "point_residuals",
    "point_to_mesh_distance",
    "surface_likelihood",
]


def assert_cpu_only() -> None:
    """Fail loudly if anything has made a CUDA device reachable here.

    Called by the tests. The environment variable is set at import, but an
    earlier import elsewhere in the process could have initialized CUDA first,
    and silently running on the card is exactly the outcome this package is
    supposed to make impossible.
    """

    if torch.cuda.is_available():
        raise RuntimeError(
            "rooted4d.differentiable must not see a CUDA device; "
            "CUDA_VISIBLE_DEVICES was not empty when torch initialized"
        )
