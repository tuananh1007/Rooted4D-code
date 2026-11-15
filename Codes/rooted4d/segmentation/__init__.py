r"""A CMR segmentation baseline, and the validation the pipeline actually needs.

Everything downstream in this project consumes segmentations. Until now they have
been the provider's ground-truth labels, which quietly assumes away the first
stage of any real pipeline. This package trains a compact network on ACDC and
measures what its errors do *to the fit*, not only how they score as
segmentations.

Two validations, and the second is the point. Dice against held-out labels is
the standard number and is reported. But a segmentation is only an input here,
so the question that matters is how far the surface fitted from a predicted
label lands from the surface fitted from the true one, and how much ejection
fraction moves. A network can lose Dice on a thin structure and barely move a
volume, or win Dice and shift a boundary systematically; only the downstream
number distinguishes those.

Training respects the frozen split lock. The ACDC people assigned to validation
and test in ``GLOBAL_SPLIT_LOCK.json`` are the same people held out here, so a
network trained on this split can be used on the benchmark without leaking a
patient across a partition.

**CPU only.** Same reason as ``rooted4d.differentiable``: the available card
drives the display and filling it freezes the machine.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

__all__ = ["ACDCSlices", "CompactUNet", "dice_per_class", "load_patient_splits"]

# The loader and the network need PyTorch, which the core environment does not
# install. They load on first use, so the NumPy-only submodule ``cleanup`` (and
# its test) imports without PyTorch.
_LAZY = {"ACDCSlices": "data", "load_patient_splits": "data",
         "CompactUNet": "unet", "dice_per_class": "unet"}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
