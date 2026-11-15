r"""A compact U-Net, sized for CPU training.

Deliberately small. The claim being supported is not that this is a
state-of-the-art segmenter -- it is not, and the paper should not imply
otherwise -- but that the downstream pipeline can be measured against a
realistic segmentation front end rather than against ground-truth labels. A
larger network would change the Dice and would not change what that measurement
is for.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

__all__ = ["CompactUNet", "dice_per_class"]


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CompactUNet(nn.Module):
    """Three-level U-Net with 4-class output."""

    def __init__(self, n_classes: int = 4, width: int = 16):
        super().__init__()
        self.down1 = _block(1, width)
        self.down2 = _block(width, width * 2)
        self.down3 = _block(width * 2, width * 4)
        self.bottom = _block(width * 4, width * 8)
        self.up3 = _block(width * 8 + width * 4, width * 4)
        self.up2 = _block(width * 4 + width * 2, width * 2)
        self.up1 = _block(width * 2 + width, width)
        self.head = nn.Conv2d(width, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(functional.max_pool2d(d1, 2))
        d3 = self.down3(functional.max_pool2d(d2, 2))
        b = self.bottom(functional.max_pool2d(d3, 2))

        def up(source: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            resized = functional.interpolate(
                source, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            return torch.cat([resized, skip], dim=1)

        u3 = self.up3(up(b, d3))
        u2 = self.up2(up(u3, d2))
        u1 = self.up1(up(u2, d1))
        return self.head(u1)


def dice_per_class(
    prediction: torch.Tensor, target: torch.Tensor, n_classes: int = 4
) -> torch.Tensor:
    """Dice for each class, over a whole batch.

    Classes absent from both prediction and target return NaN rather than 1.0.
    Scoring an absent structure as perfect inflates the mean over slices where
    the right ventricle or the cavity simply is not present, which is most apical
    slices, and the inflation is large enough to change the headline.
    """

    scores = torch.full((n_classes,), float("nan"), dtype=torch.float64)
    for label in range(n_classes):
        p = prediction == label
        t = target == label
        denominator = p.sum() + t.sum()
        if denominator == 0:
            continue
        scores[label] = (2.0 * (p & t).sum() / denominator).double()
    return scores
