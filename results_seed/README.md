# Seed inputs that must travel with the code

`results/` is excluded from `sync_to_delta.sh`, because it is output. A few small
files are *inputs* that happen to have been produced by an earlier run, and they
have to reach the cluster or the jobs that need them fail at startup rather than
at the end.

* `acdc_patient_splits.json` — ACDC patient to split map, derived from
  `GLOBAL_SPLIT_LOCK.json`. Segmentation training reads it to hold out the same
  people the benchmark holds out. Regenerable, but a job that regenerates it
  would need the lock and the donor manifest, and getting the split wrong is
  silent, so it ships.
