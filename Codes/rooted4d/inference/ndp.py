"""Neural deformation pyramid (Li and Harada, NeurIPS 2022), landmark mode (E115).

Adapted from https://github.com/rabbityl/DeformationPyramid (commit 63e9e27):
``model/nets.py``, ``model/rigid_body.py`` and the NDP loop of
``model/registration.py`` (``optimize_deformation_pyramid``, landmark branch
with no Chamfer term). Kept as in the authors' code: the pyramid of SE(3)
layers with axis-angle rotations, the positional encoding ``2 ** (level + k0)``,
the 0.001 output scale, Xavier initialization, level-by-level Adam with the
per-level hand-off of the warped landmarks, and the stopping rules (loss below
1e-4, or 15 iterations whose loss changed by less than 0.1%, the counter never
reset). Left out: the non-rigidity branch (unused at ``w_reg = 0``, the
authors' setting), the Chamfer, ARAP and point-sampling code, and everything
that needs pytorch3d. Defaults are the authors' ``config/NDP.yaml``.

The original code carries this notice:

    MIT License

    Copyright (c) 2023 Yang Li

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import numpy as np

#: The authors' ``config/NDP.yaml``.
DEFAULTS = {"iterations": 500, "learning_rate": 0.01, "max_break_count": 15,
            "break_threshold_ratio": 0.001, "k0": -8, "depth": 3, "width": 128}


def _modules():
    import torch
    from torch import nn

    def skew(w):
        zero = torch.zeros_like(w[..., 0])
        return torch.stack([zero, -w[..., 2], w[..., 1],
                            w[..., 2], zero, -w[..., 0],
                            -w[..., 1], w[..., 0], zero], dim=-1).reshape((-1, 3, 3))

    def exp_so3(w, theta):
        theta = theta[..., None]
        W = skew(w)
        identity = torch.eye(3)[None].to(theta)
        return identity + torch.sin(theta) * W + (1 - torch.cos(theta)) * W @ W

    class MLP(nn.Module):
        def __init__(self, depth, width):
            super().__init__()
            self.pts_linears = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])

        def forward(self, x):
            for layer in self.pts_linears:
                x = torch.relu(layer(x))
            return x

    class NDPLayer(nn.Module):
        """One SE(3) level, axis-angle rotation, no non-rigidity branch."""

        def __init__(self, depth, width, k0, m):
            super().__init__()
            self.k0, self.m = k0, m
            self.input = nn.Sequential(nn.Linear(6, width), nn.ReLU())
            self.mlp = MLP(depth=depth, width=width)
            self.rot_brach = nn.Linear(width, 3)
            self.trn_branch = nn.Linear(width, 3)
            self.mlp_scale = 0.001
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def posenc(self, pos):
            mul_term = 2 ** (self.m + self.k0)
            x, y, z = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
            return torch.cat([torch.sin(x * mul_term), torch.cos(x * mul_term),
                              torch.sin(y * mul_term), torch.cos(y * mul_term),
                              torch.sin(z * mul_term), torch.cos(z * mul_term)], dim=-1)

        def forward(self, x):
            fea = self.mlp(self.input(self.posenc(x)))
            t = self.mlp_scale * self.trn_branch(fea)
            rotation = self.mlp_scale * self.rot_brach(fea)
            theta = torch.norm(rotation, dim=-1, keepdim=True)
            R = exp_so3(rotation / theta, theta)
            return (R @ x[..., None]).squeeze() + t

    return torch, NDPLayer


def landmark_pyramid(source: np.ndarray, target: np.ndarray, queries: list[np.ndarray], *,
                     levels: int, seed: int = 0, **settings) -> list[list[np.ndarray]]:
    """Fit NDP to landmark pairs ``source[i] -> target[i]``, in float32.

    The authors' code, and every stored NDP result, ran with torch's default
    float32. ``rooted4d.differentiable`` sets the process default to float64 at
    import, so a process that imports both (the test suite) built float64 layers
    for float32 inputs and failed. The default is held at float32 here and
    restored afterwards; in a float32 process nothing changes.
    """

    torch, _ = _modules()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        return _landmark_pyramid(source, target, queries, levels=levels, seed=seed, **settings)
    finally:
        torch.set_default_dtype(previous)


def _landmark_pyramid(source: np.ndarray, target: np.ndarray, queries: list[np.ndarray], *,
                      levels: int, seed: int = 0, **settings) -> list[list[np.ndarray]]:
    """Fit NDP to landmark pairs ``source[i] -> target[i]`` and warp ``queries``.

    Coordinates are given in any unit; as in the authors' code each cloud is
    centered on its own mean, and both are divided by the target's root-mean-
    square distance from its mean so that the encoding acts at the unit scale
    it was designed for. Returns, for every level count ``1..levels``, the
    warped queries in the input unit; a pyramid of ``m`` levels is the first
    ``m`` levels of a longer one, since each level is fitted on the output of
    the levels before it and initialized independently of the levels after it.
    """

    torch, NDPLayer = _modules()
    options = dict(DEFAULTS, **settings)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    scale = float(np.sqrt(((target - target_mean) ** 2).sum(axis=1).mean()))

    def tensor(points):
        return torch.from_numpy(((np.asarray(points, dtype=np.float64) - source_mean) / scale).astype(np.float32))

    torch.manual_seed(seed)
    pyramid = [NDPLayer(options["depth"], options["width"], options["k0"], level + 1) for level in range(levels)]
    src_ldmk = tensor(source)
    tgt_ldmk = torch.from_numpy(((target - target_mean) / scale).astype(np.float32))
    for level, layer in enumerate(pyramid):
        for other in pyramid:
            for parameter in other.parameters():
                parameter.requires_grad = other is layer
        optimizer = torch.optim.Adam(layer.parameters(), lr=options["learning_rate"])
        break_counter, loss_prev = 0, 1e6
        warped_ldmk = src_ldmk
        for _ in range(options["iterations"]):
            warped_ldmk = layer(src_ldmk)
            loss = torch.mean(torch.sum((warped_ldmk - tgt_ldmk) ** 2, dim=-1))
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * options["break_threshold_ratio"]:
                break_counter += 1
            if break_counter >= options["max_break_count"]:
                break
            loss_prev = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        src_ldmk = warped_ldmk.detach()

    out = []
    with torch.no_grad():
        for points in queries:
            warped, per_level = tensor(points), []
            for layer in pyramid:
                warped = layer(warped)
                per_level.append(warped.double().numpy() * scale + target_mean)
            out.append(per_level)
    return [[out[q][m] for q in range(len(queries))] for m in range(levels)]
