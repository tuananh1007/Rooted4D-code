# Rooted4D

Code and per-sample results for

> **Rooted4D: Attachment-Exact Chamber–Valve Reconstruction from Asynchronous
> CMR and Transesophageal Echocardiography.** Submitted to *Information
> Fusion*.

Rooted4D reconstructs the cardiac chambers from CMR and the mitral valve from
TEE around one shared annulus. The chamber boundary and the leaflet roots are
the same vertices, so they stay attached at every phase. Anatomy, motion,
between-session change, pose and timing are fitted jointly, and a residual
field then adds shape detail without changing attachment, pose or timing.

## Install

```bash
conda env create -f Codes/environment.yml && conda activate rooted4d-starter   # or: pip install -r Codes/requirements-core.lock
pip install --no-deps -e Codes
export PYTHONPATH=Codes OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Run every command from the repository root. BLAS must use one thread for the
fits to reproduce the reported values exactly.

## Quick start

**1. Check the installation** (a few minutes, no data needed).

```bash
python -m unittest discover -s Codes/tests -q
```

**2. Recompute Table 1 from the stored results** (seconds, no data needed).

```bash
python Codes/scripts/experiments/main_table.py            # Table 1, realistic-session benchmark
python Codes/scripts/experiments/main_table.py v3n96o     # the same methods on the small-session benchmark
```

- **What it does.** Recomputes each method's medians from the per-sample test
  rows in `results/benchmark/`.
- **What it checks.** The medians against the study's report and against the
  values printed in the paper, ending with `passed`.

**3. Rerun Rooted4D on Table 1's benchmark** (needs ACDC and MVSeg2023, see
[Data](#data)).

```bash
# a. the donor pool: real chambers and valves fitted to the shared topology
python Codes/scripts/build_donor_pool.py --acdc-root data/raw/acdc --mvseg-root data/raw/mvseg2023 \
  --output data/derived/donors_v3_n96 --n-annulus 96 --n-angles 96 --valve-annulus 96
# b. the benchmark (repeat with v3n96o for the small-session benchmark)
python -m rooted4d.synthetic.dataset_cli generate --config Codes/configs/e100/v4n96o.yaml --output data/derived/e100-v4n96o
# c. rerun the first 5 test samples, then compare them with the reported rows
python Codes/scripts/experiments/reproduce_rooted4d_main.py run out --keys v4n96o --levels full --samples 5
python Codes/scripts/experiments/reproduce_rooted4d_main.py compare out out/report.json
```

- **Runtime.** One test sample takes about 30 s on one core.
- **Configuration.** The method's frozen settings are in
  `Codes/configs/rooted4d_main.json`.
- **Pass rule.** `compare` passes when every value is within 0.005 mm of the
  reported sample and the median difference is at most 10⁻⁶ mm.
- **Full run.** Drop `--samples` to rerun all 60 test samples, using
  `R4D_WORKERS` processes. Drop `--keys` and `--levels` to rerun all six test
  sets of the main configuration: both benchmarks, and Table 1's benchmark
  with thinned evidence.

**4. Everything else.** `REPRODUCING.md` covers the other benchmarks,
the comparison methods and every study in the paper.

## Repository layout

| Path | Contents |
|---|---|
| `Codes/rooted4d/` | the method: shared-annulus template, shape space, joint fit, residual field, comparison methods, observation models, benchmark generator |
| `Codes/configs/` | the frozen main configuration and the benchmark configurations |
| `Codes/scripts/experiments/` | one driver per study (`eNN_*.py`; the docstring lists its commands) |
| `Codes/scripts/delta/` | the Slurm jobs that ran each study on NCSA Delta |
| `Codes/tests/` | unit tests |
| `results/` | per-sample scores and analysis reports; see `results/README.md` |
| `GLOBAL_SPLIT_LOCK.json` | the patient-level split lock |

File names use internal names. A study is `ENN`. In benchmark keys, `v4` is the
realistic-session benchmark, `v3` the small-session benchmark, and `n96` the
number of ring nodes.

## Data

No imaging data are included. Obtain the sources from their providers:

| Source | Content | License | Place under |
|---|---|---|---|
| ACDC | cardiac MR | CC BY-NC-SA 4.0 | `data/raw/acdc/` |
| MVSeg2023 | 3D TEE, mitral leaflets | CC BY-NC-ND 4.0 (gated) | `data/raw/mvseg2023/` |
| MITEA | 3D echo with CMR-derived labels | CC BY-NC-SA, on approval | `data/raw/mitea/` (only for the real-data study) |

The donor pool and the benchmarks are derived from MVSeg2023, whose
NoDerivatives term prevents releasing them. They regenerate from the
configurations, as in step 3. The session-variation subspaces measured on
MITEA, which the realistic-session benchmark needs, are included under CC
BY-NC-SA (`results/benchmark/SESSION_SUBSPACES.md`).

## License

The code is under MIT (see `LICENSE`). The three MITEA-derived arrays are under
CC BY-NC-SA 4.0. Third-party components are listed in `THIRD_PARTY.md`. A
citation will be added on publication.
