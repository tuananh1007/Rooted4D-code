# Third-party resources

No medical data are included in this package. One file includes adapted third-party code; see
"Included third-party code" below.

The links in `baselines/catalog.yaml` and `rooted4d/data/catalog.yaml` are discovery metadata only. Before cloning, downloading, modifying, or redistributing any resource:

1. inspect its current license and provider terms;
2. accept any required data-use agreement yourself;
3. cite the original authors and datasets;
4. record the exact repository commit or dataset release;
5. keep restricted data outside version control.

An absent repository license does not mean public-domain or unrestricted use.

## Reviewed user-supplied prototype

`Cardiac_temp.zip` was reviewed on 29 August 2026 as design input. It did not contain a
license, is not bundled in this release, and none of its source files were copied into the
package. Rooted4D 0.5 independently implements the selected standard geometry operations
under its own fixed-topology contracts and tests. See
`docs/CARDIAC_TEMP_INTEGRATION_AUDIT.md` for the exact adopt/reject record and archive hash.

## Included third-party code

`Codes/rooted4d/inference/ndp.py` adapts the network, rotation and level loop of the neural
deformation pyramid (Li and Harada, NeurIPS 2022) from https://github.com/rabbityl/DeformationPyramid,
commit 63e9e27, MIT License, Copyright (c) 2023 Yang Li. The license notice is reproduced in that
file. It is used only by the E115 comparator.

`Codes/rooted4d/inference/recent_fusion.py` re-implements Bayesian coherent point drift (Hirose,
TPAMI 2021) in numpy, following https://github.com/ohirose/bcpd (MIT); no source file is copied.
E115 checks it against the authors' program built from commit 078ca6b, which is not bundled.
