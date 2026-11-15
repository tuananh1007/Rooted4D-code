"""E93: the realistic-session benchmark at 96 ring nodes (plan v2, W2.2).

Pre-registered in ``Docs/E93_REALISTIC_SESSION_96_PREREGISTRATION_2026-09-28.md``.

The realistic-session benchmark draws its session fields from the MITEA
scan-rescan subspace (E25b), which is defined on the 73 vertices of the 24-node
topology. This transfers that subspace to the 289 vertices of the 96-node
topology and writes the benchmark's generator config:

* each ring's field (annulus, lower chamber ring) is interpolated around the
  ring by band-limited periodic (Fourier) interpolation, which reproduces the
  24 measured samples exactly at every fourth vertex; the apex is copied and the
  leaflet free edge stays zero, as measured;
* the coefficient spreads are kept, so the per-vertex displacement the generator
  draws has the same magnitude at both densities.

The transfer adds no information beyond the 24 measured angular samples: it
tests the model at the denser ring, not a denser measurement of session change.

    python e93_realistic_session_96.py transfer   -> results/benchmark/E93_session_subspace_split_n96.{npz,json}
    python e93_realistic_session_96.py config     -> Codes/configs/semisynthetic_real_donors_v4_n96.yaml
    python e93_realistic_session_96.py chain      -> E88's unattended protocol on it (on Delta; reads R4D_E88_OUT)
    python e93_realistic_session_96.py analyze <test dir>   -> results/benchmark/E93/E93_realistic_session_96.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PREREGISTRATION = "Docs/E93_REALISTIC_SESSION_96_PREREGISTRATION_2026-09-28.md"
SOURCE = Path("results/benchmark/E25b_session_subspace_split.npz")
TARGET = Path("results/benchmark/E93_session_subspace_split_n96.npz")
RECORD = TARGET.with_suffix(".json")
CONFIG_SOURCE = Path("Codes/configs/semisynthetic_real_donors_v3_n96.yaml")
CONFIG = Path("Codes/configs/semisynthetic_real_donors_v4_n96.yaml")
COARSE, FINE = 24, 96


def groups(n: int) -> dict[str, np.ndarray]:
    """Vertex ranges of the reduced topology at ``n`` ring nodes (``make_toy_complex``)."""

    return {"annulus": np.arange(0, n), "lower": np.arange(n, 2 * n), "apex": np.array([2 * n]),
            "free": np.arange(2 * n + 1, 3 * n + 1)}


def fourier_upsample(values: np.ndarray, factor: int) -> np.ndarray:
    """Band-limited periodic interpolation along axis 0, exact at the original samples."""

    n = values.shape[0]
    spectrum = np.fft.rfft(values, axis=0)
    padded = np.zeros((n * factor // 2 + 1,) + values.shape[1:], dtype=complex)
    padded[: spectrum.shape[0]] = spectrum
    if n % 2 == 0:
        # The coarse Nyquist term is shared between +n/2 and -n/2 in the finer
        # spectrum; halving it keeps the interpolant real and through the samples.
        padded[n // 2] = spectrum[n // 2] / 2.0
    return np.fft.irfft(padded, n=n * factor, axis=0) * factor


def transfer_components(components: np.ndarray) -> np.ndarray:
    coarse, fine = groups(COARSE), groups(FINE)
    fields = components.reshape(len(components), 3 * COARSE + 1, 3)
    out = np.zeros((len(components), 3 * FINE + 1, 3))
    for index, field in enumerate(fields):
        for ring in ("annulus", "lower", "free"):
            out[index, fine[ring]] = fourier_upsample(field[coarse[ring]], FINE // COARSE)
        out[index, fine["apex"]] = field[coarse["apex"]]
    return out.reshape(len(components), -1)


def cmd_transfer() -> None:
    blob = np.load(SOURCE)
    coarse, fine = groups(COARSE), groups(FINE)
    record = {"experiment": "E93", "preregistration": PREREGISTRATION, "source": str(SOURCE),
              "method": "band-limited periodic interpolation per ring; apex copied; free edge zero",
              "kinds": {}}
    arrays = {}
    for kind in ("static", "dynamic"):
        components = blob[f"{kind}_components"]
        deviations = blob[f"{kind}_coefficient_std"]
        transferred = transfer_components(components)
        before = components.reshape(len(components), -1, 3)
        after = transferred.reshape(len(components), -1, 3)
        exact = max(float(np.abs(after[:, fine[ring][::FINE // COARSE]] - before[:, coarse[ring]]).max())
                    for ring in ("annulus", "lower"))
        assert exact < 1e-9, exact
        assert np.abs(after[:, fine["free"]]).max() == 0.0

        def rms(fields, ring_indices):
            # Expected per-vertex displacement rms of the drawn field, in mm.
            variance = (deviations[:, None] ** 2 * (fields[:, ring_indices] ** 2).sum(axis=2)).sum(axis=0)
            return float(np.sqrt(variance.mean()))

        record["kinds"][kind] = {
            "exact_at_measured_vertices_max_mm": exact,
            "per_vertex_rms_mm": {
                ring: {"n24": rms(before, coarse[ring]), "n96": rms(after, fine[ring])}
                for ring in ("annulus", "lower", "apex")},
            "coefficient_std": deviations.tolist(),
        }
        arrays[f"{kind}_components"] = transferred
        arrays[f"{kind}_coefficient_std"] = deviations
    np.savez_compressed(TARGET, **arrays)
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["kinds"], indent=2))
    print("wrote", TARGET, "and", RECORD)


def cmd_config() -> None:
    text = CONFIG_SOURCE.read_text()
    # The source's leading comment describes v3 at 96 nodes (and says there is no
    # v4 there); it is replaced by this file's own header.
    text = text[text.index("schema_version:"):]
    anchor = "  session_dynamic_scale_max: 1.30\n"
    assert text.count(anchor) == 1
    body = text.replace(anchor, anchor + (
        "  # E93: the MITEA scan-rescan subspace (E25b), transferred from the 24-node\n"
        "  # topology by band-limited interpolation around each ring.\n"
        "  session_subspace_path: ../../results/benchmark/E93_session_subspace_split_n96.npz\n"))
    header = (f"# E93 ({PREREGISTRATION}): the realistic-session benchmark at 96 ring nodes.\n"
              f"# Identical to {CONFIG_SOURCE.name} (v3 at 96 nodes) except the session\n"
              "# subspace, which is v4's transferred to this topology. Written by\n"
              "# e93_realistic_session_96.py config; do not edit by hand.\n")
    CONFIG.write_text(header + body)
    print("wrote", CONFIG)


BENCHMARK = "data/derived/semisynthetic-real-v4-n96"
KEY = "v4n96"


def _load(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_chain() -> None:
    e88 = _load("e88_replication")
    chain = e88.Chain("v4", 0, key=KEY, benchmark=BENCHMARK, configs=Path("Codes/configs/e93"),
                      experiment="E93", preregistration=PREREGISTRATION)
    chain.selection_rounds()
    arms = chain.session_weight()
    directory = chain.test(arms)
    (chain.root / "TEST_DONE").write_text(str(Path(directory).relative_to(chain.root)) + "\n")
    print(f"[{KEY}] test rows in {directory}", flush=True)


def cmd_analyze(test_dir: str) -> None:
    e88 = _load("e88_replication")
    stage = e88._load("e62_stage_b")
    spec = json.loads((Path(test_dir) / "spec.json").read_text())
    assert spec["benchmark"] == BENCHMARK and spec["split"] == "test", spec["benchmark"]
    rows = stage.load_rows([test_dir])
    assert len(rows) == 60, len(rows)
    report = {"experiment": "E93", "preregistration": PREREGISTRATION,
              "test": e88.test_report(stage, rows)}
    t = report["test"]
    report["ordering_holds"] = (t["T1_representation"]["verdict"] == "confirmed"
                                and all(t["T3_fusion_cost"][m] != "cost remains" for m in ("cmr", "tee")))
    out = Path("results/benchmark/E93/E93_realistic_session_96.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("T1", t["T1_representation"]["verdict"], "| T3", {m: t["T3_fusion_cost"][m] for m in ("cmr", "tee")},
          "| ordering holds:", report["ordering_holds"])
    print("wrote", out)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"transfer": cmd_transfer, "config": cmd_config, "chain": cmd_chain,
     "analyze": cmd_analyze}[command](*arguments)
