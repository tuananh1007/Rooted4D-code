# Results

Per-sample scores and analysis reports behind the paper's tables.
`Codes/scripts/experiments/main_table.py` recomputes Table 1 from them.

| Path | Contents |
|---|---|
| `benchmark/ENN/` | one folder per study: per-sample rows, the study's report, job logs and provenance |
| `benchmark/ENN_*.json` | reports of earlier studies that wrote a single file |
| `benchmark/stage_c/`, `benchmark/uniformity/` | the real-shape studies and the component studies |
| `benchmark/*_session_subspace_*.npz` | session-variation subspaces measured on MITEA (CC BY-NC-SA; see `benchmark/SESSION_SUBSPACES.md`) |
| `<Slurm job id>[_name]/` | early studies (preliminary configuration, segmenters, external methods), filed by the job that produced them |

## File types

- **Rows** (`*.jsonl`): one JSON object per test sample. It holds each
  method's held-out CMR and TEE errors (median and 95th percentile),
  attachment gap, rotation, translation and phase-map error.
- **Reports** (`ENN_*.json`): a study's analysis of its rows: medians, paired
  differences with bootstrap intervals, and the outcome of each decision rule
  fixed before the run.
- **Provenance** (`PROVENANCE_<job>.txt`, `ENVIRONMENT_*.txt`): the code
  commit, the operating system and CPU, the Python and NumPy versions, and the
  BLAS build of each job.

The commits named in provenance files belong to the authors' development
repository. This release is a single-commit snapshot of it, and its commit
message names the development commit it was made from.
