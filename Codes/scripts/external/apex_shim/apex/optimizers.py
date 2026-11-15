"""FusedAdam mapped onto torch.optim.AdamW. See the package docstring."""

import warnings

import torch

__all__ = ["FusedAdam"]


class FusedAdam(torch.optim.AdamW):
    """AdamW under apex's FusedAdam name.

    apex's FusedAdam with its default ``adam_w_mode=True`` performs the AdamW
    update in a fused kernel. The update rule is the same; only the kernel
    differs. The apex-only arguments are accepted so a configuration written for
    apex loads unchanged, and are ignored because they describe kernel
    behaviour rather than optimization.
    """

    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999),
                 eps=1e-8, adam_w_mode=True, weight_decay=0.0, amsgrad=False,
                 set_grad_none=True, capturable=False, master_weights=False,
                 **ignored):
        if not adam_w_mode:
            # Plain Adam applies weight decay inside the gradient; AdamW
            # decouples it. Silently swapping one for the other would change
            # the objective, so refuse instead.
            raise ValueError(
                "the apex shim implements FusedAdam only in adam_w_mode; "
                "adam_w_mode=False is a different update rule")
        if not bias_correction:
            raise ValueError(
                "the apex shim does not implement bias_correction=False")
        if ignored:
            warnings.warn("apex shim ignoring apex-only arguments: %s"
                          % sorted(ignored), RuntimeWarning, stacklevel=2)
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
        self._apex_shim = True
