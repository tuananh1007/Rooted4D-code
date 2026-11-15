"""A stand-in for NVIDIA apex, providing only what TetHeart imports.

THIS IS NOT APEX. It exists because apex cannot be built on this cluster and
TetHeart imports it unconditionally.

apex's setup refuses to compile when the CUDA toolkit's version differs from the
one PyTorch was built with, and it rejects a minor mismatch, not just a major
one: torch here is cu121 and the cluster provides CUDA 11.8, 12.9 and 13.2, with
no 12.1. apex's own error notes the check can be commented out "at your own
risk", which is not a risk worth taking for an optimizer.

`tetheart/core/optimizers/builder.py` does `import apex`, then iterates
`dir(apex.optimizers)` and registers every torch.optim.Optimizer subclass it
finds under that class's name. The configs then ask for `type='FusedAdam'`. So
exposing a FusedAdam here is enough for their configuration to work unmodified.

The substitution is numerically faithful. apex's FusedAdam defaults to
`adam_w_mode=True`, making it AdamW with a fused CUDA kernel; fusion is a
throughput optimization and does not change the update rule. This maps it to
`torch.optim.AdamW` and accepts, then ignores, the apex-specific keyword
arguments that only concern kernel behaviour.

Deliberately absent: `apex.normalization`. MONAI reaches for
`InstanceNorm3dNVFuser` when a config asks for INSTANCE_NVFUSER and falls back to
standard instance normalization when the import fails. Providing a fake would
risk MONAI binding to something that is not the fused kernel it expects; letting
the import fail gets the documented fallback instead.
"""

from . import optimizers  # noqa: F401

__all__ = ["optimizers"]
__apex_shim__ = True
__version__ = "0.0.0+rooted4d-shim"
