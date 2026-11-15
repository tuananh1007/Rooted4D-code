# Session-variation subspaces (derived from MITEA)

Three arrays here are not covered by the repository's MIT license:

| File | Topology | Used by |
|---|---|---|
| `E25b_session_subspace_split.npz` | 24 ring nodes | the 24-node realistic-session configurations, e.g. `Codes/configs/semisynthetic_real_donors_v4_oriented.yaml` |
| `E93_session_subspace_split_n96.npz` | 96 ring nodes | the realistic-session benchmark `Codes/configs/e100/v4n96o.yaml`, its replications, interventions (`Codes/configs/e100/e92/`) and clocked composites (`Codes/configs/e108/v4n96a.yaml`) |
| `E100_session_subspace_split_n48.npz` | 48 ring nodes | `Codes/configs/e100/v4n48o.yaml` |

- **What they are.** Each holds four static and four dynamic modes of
  between-session shape change, with the spread of each mode's coefficient.
  The modes were measured on MITEA scan–rescan pairs of 134 subjects. The 48-
  and 96-node arrays are the 24-node one interpolated around the ring. They
  contain no per-subject data. Summaries are in the `.json` file of the same
  name.
- **Why they are here.** The realistic-session benchmarks draw their session
  change from these modes and cannot be regenerated without them.

They are derived from MITEA (Cardiac Atlas Project) and are shared under the
same terms, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
attribution, non-commercial use only, and derivatives under the same license.
Cite the dataset:

> D. Zhao, E. Ferdian, G. D. Maso Talou, et al. MITEA: A dataset for machine
> learning segmentation of the left ventricle in 3D echocardiography using
> subject-specific labels from cardiac magnetic resonance imaging. *Frontiers
> in Cardiovascular Medicine* 9:1016703, 2023. doi:10.3389/fcvm.2022.1016703
