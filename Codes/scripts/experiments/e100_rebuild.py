"""E100: every benchmark rebuilt with oriented valves, main tables at 96 nodes.

Pre-registered in ``Docs/E100_ORIENTED_REBUILD_PREREGISTRATION_2026-09-29.md``.
The instances are the originals' configs with ``donors.orient_valves: true`` (and,
for the replication, E88's split seeds); each goes through E88's unattended
chain once.

    python e100_rebuild.py configs          -> Codes/configs/e100/<key>.yaml
    python e100_rebuild.py transfer48       -> results/benchmark/E100_session_subspace_split_n48.npz
    R4D_KEY=<key> python e100_rebuild.py chain     (on Delta; reads R4D_E88_OUT)

The realistic-session robustness instance at 24 nodes is E94's, reused.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E100_ORIENTED_REBUILD_PREREGISTRATION_2026-09-29.md"
CONFIGS = Path("Codes/configs/e100")
ORIGINAL_SPLIT_SEED = 104729
SPLIT_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)
TRANSFER48 = Path("results/benchmark/E100_session_subspace_split_n48.npz")
#: key -> (protocol benchmark, source config, non-oriented counterpart or None)
BASES = {
    "v4n96o": ("v4", "Codes/configs/semisynthetic_real_donors_v4_n96.yaml", "data/derived/semisynthetic-real-v4-n96"),
    "v3n96o": ("v3", "Codes/configs/semisynthetic_real_donors_v3_n96.yaml", "data/derived/semisynthetic-real-v3-n96"),
    "v3o": ("v3", "Codes/configs/semisynthetic_real_donors_v3.yaml", "data/derived/semisynthetic-real-v3"),
    "v3n48o": ("v3", "Codes/configs/semisynthetic_real_donors_v3_n48.yaml", "data/derived/semisynthetic-real-v3-n48"),
    "v4n48o": ("v4", "Codes/configs/semisynthetic_real_donors_v3_n48.yaml", None),
}
PRIMARY = ("v4n96o", "v3n96o")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keys() -> list[str]:
    return list(BASES) + [f"{base}s{k}" for base in PRIMARY for k in range(1, len(SPLIT_SEEDS) + 1)]


def resolve(key: str) -> tuple[str, str, int, str | None]:
    """(protocol benchmark, base key, split index or 0, counterpart) for a key."""

    if key in BASES:
        return BASES[key][0], key, 0, BASES[key][2]
    base, k = key.rsplit("s", 1)
    if base not in PRIMARY or not 1 <= int(k) <= len(SPLIT_SEEDS):
        raise SystemExit(f"unknown instance {key}")
    return BASES[base][0], base, int(k), None


def benchmark(key: str) -> str:
    return f"data/derived/e100-{key}"


def config_path(key: str) -> Path:
    return CONFIGS / f"{key}.yaml"


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def cmd_configs() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for key in keys():
        bench, base, k, _ = resolve(key)
        source = BASES[base][1]
        text = Path(source).read_text()
        text = text[text.index("schema_version:"):]            # the source's own header is replaced
        if base == "v4n48o":
            # E93's construction at 48 nodes: v3's config at this density plus the
            # MITEA subspace transferred to it.
            text = _replace_once(text, "  session_dynamic_scale_max: 1.30\n",
                                 "  session_dynamic_scale_max: 1.30\n"
                                 "  # E100: the MITEA scan-rescan subspace (E25b), transferred to 48 nodes\n"
                                 "  # by E93's band-limited interpolation around each ring.\n"
                                 f"  session_subspace_path: ../../{TRANSFER48}\n")
        text = _replace_once(text, "  split_lock: null\n",
                             "  split_lock: null\n"
                             "  # E100: each valve re-indexed to its chamber's ring handedness (E94).\n"
                             "  orient_valves: true\n")
        if k:
            text = _replace_once(text, f"split_seed: {ORIGINAL_SPLIT_SEED}", f"split_seed: {SPLIT_SEEDS[k - 1]}")
        # Paths in a config are relative to it, one directory deeper here.
        text = text.replace(": ../../", ": ../../../")
        header = (f"# E100 ({PREREGISTRATION}): {key}.\n"
                  f"# {Path(source).name} with donors.orient_valves"
                  + (f", split seed {SPLIT_SEEDS[k - 1]}" if k else "")
                  + (" and the MITEA subspace transferred to 48 nodes" if base == "v4n48o" else "")
                  + ".\n# Written by e100_rebuild.py configs; do not edit by hand.\n")
        config_path(key).write_text(header + text)
        print("wrote", config_path(key))


def cmd_transfer48() -> None:
    e93 = _load("e93_realistic_session_96")
    e93.FINE = 48
    blob = np.load(e93.SOURCE)
    arrays, record = {}, {"experiment": "E100", "preregistration": PREREGISTRATION,
                          "source": str(e93.SOURCE), "method": "E93's transfer, FINE = 48", "exact_max_mm": {}}
    coarse, fine = e93.groups(e93.COARSE), e93.groups(48)
    for kind in ("static", "dynamic"):
        components = blob[f"{kind}_components"]
        transferred = e93.transfer_components(components)
        before = components.reshape(len(components), -1, 3)
        after = transferred.reshape(len(components), -1, 3)
        exact = max(float(np.abs(after[:, fine[ring][::2]] - before[:, coarse[ring]]).max())
                    for ring in ("annulus", "lower"))
        assert exact < 1e-9, exact
        record["exact_max_mm"][kind] = exact
        arrays[f"{kind}_components"] = transferred
        arrays[f"{kind}_coefficient_std"] = blob[f"{kind}_coefficient_std"]
    np.savez_compressed(TRANSFER48, **arrays)
    TRANSFER48.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print("wrote", TRANSFER48, record["exact_max_mm"])


def cmd_chain() -> None:
    key = os.environ["R4D_KEY"]
    bench, _, k, _ = resolve(key)
    e88 = _load("e88_replication")
    chain = e88.Chain(bench, k, key=key, benchmark=benchmark(key), configs=CONFIGS, experiment="E100",
                      preregistration=PREREGISTRATION, store_parameters=k == 0)
    chain.selection_rounds()
    arms = chain.session_weight()
    directory = chain.test(arms)
    (chain.root / "TEST_DONE").write_text(str(Path(directory).relative_to(chain.root)) + "\n")
    print(f"[{key}] test rows in {directory}", flush=True)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"configs": cmd_configs, "transfer48": cmd_transfer48, "chain": cmd_chain}[command](*arguments)
