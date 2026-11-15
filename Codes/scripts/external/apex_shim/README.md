# apex shim

Not apex. A three-file stand-in providing `apex.optimizers.FusedAdam` as
`torch.optim.AdamW`, because apex will not build against this cluster's CUDA
toolkits and TetHeart imports apex unconditionally.

Install into the TetHeart environment with:

    cp -r apex <venv>/lib/python3.10/site-packages/

Every substitution is recorded in `Docs/EXTERNAL_COMPARISON_AUDIT.md`. Any run
that uses this shim must report the optimizer as AdamW, not FusedAdam.
