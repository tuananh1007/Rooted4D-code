# Reproducing the experiments

`README.md` has the quick start: install, the tests, Table 1 recomputed from
the stored rows, and a rerun of Rooted4D on Table 1's benchmark. This file
covers the rest. Every command runs from the repository root, with the
environment from `README.md`:

```bash
export PYTHONPATH=Codes OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The image-evidence studies, the segmenters and the neural comparison methods
also need PyTorch. Use the CPU build: every fit in the paper ran on CPU.

## 1. Data

| Source | Used for | Place under |
|---|---|---|
| ACDC | chamber donors, CMR segmenter | `data/raw/acdc/` |
| MVSeg2023 | valve donors, valve segmenter | `data/raw/mvseg2023/` |
| MITEA | the real same-patient study | `data/raw/mitea/` |

- **Checking a download.** `data/manifests/` records the content hash of the
  ACDC and MVSeg2023 archives as filed, and
  `python -m rooted4d.data.cli validate <key> --root data/raw/<key>` checks a
  source's file layout.
- **Access routes.** `datasets/DATASETS.md` lists where each source is
  obtained.

## 2. Donor pool and benchmarks

The donor pool fits real chambers (ACDC) and valves (MVSeg2023) to the shared
topology. Every benchmark is then generated from it by one configuration.
Generation refuses an existing output directory and writes an audit of the
splits:

```bash
python Codes/scripts/build_donor_pool.py --acdc-root data/raw/acdc --mvseg-root data/raw/mvseg2023 \
  --output data/derived/donors_v3_n96 --n-annulus 96 --n-angles 96 --valve-annulus 96
python -m rooted4d.synthetic.dataset_cli generate --config Codes/configs/e100/<key>.yaml --output data/derived/e100-<key>
python -m rooted4d.synthetic.dataset_cli audit --root data/derived/e100-<key>
```

| Benchmark | `<key>` |
|---|---|
| realistic-session (primary, Table 1) | `v4n96o` |
| small-session | `v3n96o` |
| five further donor partitions of each | `v4n96os1` … `v4n96os5`, `v3n96os1` … `v3n96os5` |
| 48 ring nodes | `v4n48o`, `v3n48o` |
| generator interventions | the files in `Codes/configs/e100/e92/` |
| richer procedural instance, clocked composites | `Codes/configs/e102/v3or.yaml`, `Codes/configs/e108/v4n96a.yaml`, `v3n96a.yaml` |

- **Session change.** The realistic-session benchmarks draw their session
  change from subspaces measured on MITEA, which are included
  (`results/benchmark/SESSION_SUBSPACES.md`).
- **Determinism.** A regeneration on a second machine differed from the
  original by at most 5.7 × 10⁻¹⁴ in any stored value.
- **Checking a regenerated benchmark.** `reproduce_rooted4d_main.py
  check-data` reports whether a benchmark is byte-identical to the one in the
  paper. A regeneration that differs only in the last digits fails that
  check. In that case `compare`, whose tolerance is 0.005 mm per value,
  decides.
- **Exact jobs.** `Codes/scripts/delta/submit_build_v3_n96.sbatch` and
  `submit_e100_build.sbatch` built and checked the pool and the benchmarks.

## 3. The main configuration

`Codes/configs/rooted4d_main.json` holds the method as reported, per
benchmark:
- the solver settings and model recipe;
- the residual field's setting;
- the dataset fingerprints;
- the per-sample reference rows of the reported test run.

```bash
python Codes/scripts/experiments/reproduce_rooted4d_main.py check-data
R4D_WORKERS=30 python Codes/scripts/experiments/reproduce_rooted4d_main.py run out
python Codes/scripts/experiments/reproduce_rooted4d_main.py compare out out/report.json
```

- **What it covers.** This reruns Rooted4D on all six reported test sets.
- **Pass rule.** Every value within 0.005 mm of the reference, a median
  difference of at most 10⁻⁶ mm, and medians equal at reporting precision.
- **The recorded run.** On the recorded environment
  (`results/benchmark/E116/ENVIRONMENT_delta.txt`), one BLAS thread reproduced
  every value exactly: `results/benchmark/E116/REPRODUCTION_23148536.json`.
- **In code.** `rooted4d.inference.residual_field.fit_rooted4d_with_field`
  returns the joint fit and its residual field.

## 4. The other studies

Each study has three parts:
- **Driver:** `Codes/scripts/experiments/eNN_*.py`, whose docstring lists its
  commands (usually `run`, then `analyze`, which writes the report).
- **Slurm job:** `Codes/scripts/delta/submit_eNN*.sbatch`, the exact job that
  ran it.
- **Results:** the rows and report in `results/benchmark/ENN/`, or for early
  studies in `results/<Slurm job id>/`.

The studies behind the main-text tables:

| Table | Study | Start from |
|---|---|---|
| 1, fusion | E100 (joint fit, late fusion, the other comparisons), E115 (BCPD, GPMM), E116 (the residual field) | `submit_chain.sbatch` with `R4D_CHAIN_SCRIPT=Codes/scripts/experiments/e100_rebuild.py` and `R4D_KEY=<key>`, then `submit_e115.sbatch` and `submit_e116.sbatch` |
| 2, real ACDC chamber | E36 (Rooted4D), E37–E39 (GHDHeart), E43–E49 (TetHeart) | `e36_image_input_chamber.py`; `submit_ghdheart*.sbatch`, then `e39_ghdheart_comparison.py`; `submit_tetheart_*.sbatch`, then `e49c_tetheart_comparison.py` |
| 3, measurements | E90 | `e90_relational.py` |
| 4, MITEA | E99, E117 | `e99_mitea_segmenter.py`, `e99_mitea_fusion.py`, `e117_mitea_field.py` |
| 5, residual-field variants | E119, E116 | `submit_e119.sbatch` |

- **Order.** Later studies read the fits of earlier ones from a results root
  (`R4D_RESULTS` or `R4D_E88_OUT`). Run the main chain on a benchmark before
  the studies that use its fits.
- **Selection and test.** The chain selects settings on validation and then
  evaluates once on test.

## 5. Running on a cluster

The large studies ran on NCSA Delta, on AMD EPYC 7763 CPU nodes.
- **Paths.** The Slurm scripts use that cluster's paths (`/work/nvme/...` for
  code and results, `/work/hdd/...` for data); change them for another
  cluster.
- **Resuming.** The benchmark runner skips samples already in its output, so
  an interrupted job can be resubmitted.
- **Split runs.** The image-study rerun ran as separate parts. Each part
  records the digest of its starting fit, and the merge refuses parts that
  started differently.

Comments, run specifications and stored reports mention notes under `Docs/`.
Those are the authors' internal protocol notes and are not part of this
repository.
