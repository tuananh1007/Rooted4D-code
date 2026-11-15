"""A per-sample runner for the E68-E72 scripts, mirroring the E59 driver.

One JSON line per sample in a per-shard file, resumable (samples already
present are skipped), sharded with ``R4D_SHARD=k/n``, and parallel over a
spawn-context pool of ``R4D_WORKERS`` processes with BLAS pinned to one thread,
for the reasons the E59 driver records. Test is refused without
``R4D_ALLOW_TEST=1``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def frozen_s(bench: str) -> tuple[dict, dict, float, str]:
    """The frozen S recipe, base settings, identity weight and benchmark root (E62's gate).

    For an E100 instance (a key with a chain under ``Codes/configs/e100/``) it is
    that instance's final configuration: S/joint in its chain's test spec, with
    the session weight selected on validation (E73) carried in the base settings.
    """

    rebuilt = Path("Codes/configs/e100") / bench / f"{bench}_e88_test.json"
    if rebuilt.exists():
        spec = json.loads(rebuilt.read_text())
        overrides = dict(next(v for v in spec["variants"] if v["name"] == "S/joint").get("overrides") or {})
        weight = float(overrides.pop("identity_prior_weight"))
        return spec["models"]["S"], {**spec["base_config"], **overrides}, weight, spec["benchmark"]
    gate = json.loads((Path("Codes/configs/e62") / f"{bench}_gate.json").read_text())
    return (gate["models"]["S"], dict(gate["base_config"]),
            float(gate["frozen"]["round1"]["S/joint"]), gate["benchmark"])


def run(name: str, split: str, paths, sample_function, initializer, initargs, extra=None) -> int:
    if split == "test" and os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing to run on test without R4D_ALLOW_TEST=1")
    out = Path(os.environ.get("R4D_OUT", "results/benchmark/uniformity")) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(clean(extra or {}), indent=2, sort_keys=True) + "\n")
    k, n = (int(v) for v in os.environ.get("R4D_SHARD", "0/1").split("/"))
    paths = list(paths)[k::n]
    shard = out / f"shard_{k:03d}_of_{n:03d}.jsonl"
    done = set()
    if shard.exists():
        done = {json.loads(line)["sample"] for line in shard.read_text().splitlines() if line.strip()}
    todo = [str(p) for p in paths if Path(p).name not in done]
    print("%s | shard %d/%d | %d samples, %d done, %d to run" % (name, k, n, len(paths), len(done), len(todo)),
          flush=True)
    workers = int(os.environ.get("R4D_WORKERS", "1"))
    started = time.time()
    with shard.open("a") as handle:
        if workers > 1:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(workers, initializer=initializer, initargs=initargs) as pool:
                for row in pool.imap_unordered(sample_function, todo):
                    handle.write(json.dumps(clean(row)) + "\n")
                    handle.flush()
                    print("%s done (%.0fs)" % (row["sample"], time.time() - started), flush=True)
        else:
            initializer(*initargs)
            for path in todo:
                row = sample_function(path)
                handle.write(json.dumps(clean(row)) + "\n")
                handle.flush()
                print("%s done (%.0fs)" % (row["sample"], time.time() - started), flush=True)
    print("finished %d samples in %.0fs -> %s" % (len(todo), time.time() - started, shard))
    return 0
