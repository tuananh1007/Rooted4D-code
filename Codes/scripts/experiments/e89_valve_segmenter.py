"""E89, part 1: a valve segmenter trained only on the benchmark's training valve donors.

Pre-registered in ``Docs/E89_VALVE_SEGMENTER_PREREGISTRATION_2026-09-28.md``.

The measured-front-end-error benchmark (plan v2, W1.3) needs valve predictions
for the benchmark's test donors from a network that never trained on them. The
existing valve nnU-Net (E41/E42, Dataset501_MVSeg) was trained on the MVSeg2023
challenge's 105 training cases, and 25 of the benchmark's 35 test-partition
valve donors are among them. This network is trained on the benchmark's own
valve partition instead:

* ``imagesTr``/``labelsTr``: the 105 training-partition and 35
  validation-partition valve donors; fold 0 (``splits_final.json``) trains on
  the first and validates on the second, as E84 did for the chamber;
* ``imagesTs``: the 35 test-partition valve donors, images only.

The chamber side needs no retrain: the benchmark's heart partitions are exactly
the ACDC split lock that E84's chamber network was trained under.

    R4D_MODE=convert python e89_valve_segmenter.py   MVSeg2023 -> nnU-Net raw (Dataset503_MVSegBench)
    R4D_MODE=splits  python e89_valve_segmenter.py   fold 0 = the benchmark's train/val valve donors
    R4D_MODE=check   python e89_valve_segmenter.py   donor index -> MVSeg2023 case, re-fitted on four cases

``convert`` and ``splits`` need only the standard library, so they run in the
nnU-Net environment; ``check`` needs the project environment.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

PREREGISTRATION = "Docs/E89_VALVE_SEGMENTER_PREREGISTRATION_2026-09-28.md"
DATASET = "Dataset503_MVSegBench"
LABELS = {"background": 0, "posterior_leaflet": 1, "anterior_leaflet": 2}
SOURCE = Path(os.environ.get("R4D_MVSEG", "data/raw/mvseg2023"))
#: Both benchmarks were generated with the same split seed, so they must agree.
BENCHMARKS = ("data/derived/semisynthetic-real-v4", "data/derived/semisynthetic-real-v3")
DONORS = Path("data/derived/donors_v3")


def label_paths() -> list[Path]:
    """The donor pool's valve order.

    ``rooted4d.donors.valve.load_valve_fits`` fits ``sorted(rglob("*-label.nii.gz"))``
    and keeps the fits that succeed; all 175 did (the pool has 175 valves), so
    donor ``mvseg-valve-<i>`` is path ``i``. ``check`` re-fits cases to confirm it.
    """

    paths = sorted(SOURCE.rglob("*-label.nii.gz"))
    assert len(paths) == 175, len(paths)
    return paths


def partitions() -> dict[str, list[str]]:
    records = [json.loads((Path(b) / "dataset.json").read_text())["donor_partitions"]
               for b in BENCHMARKS]
    parts = {split: sorted(records[0][split]["valve"]) for split in ("train", "val", "test")}
    for other in records[1:]:
        assert {s: sorted(other[s]["valve"]) for s in parts} == parts, "valve partitions differ"
    assert not (set(parts["train"]) & set(parts["test"]) or set(parts["val"]) & set(parts["test"]))
    return parts


def index_of(donor: str) -> int:
    return int(donor.rsplit("-", 1)[-1])


def case_of(donor: str) -> str:
    return f"valve_{index_of(donor):04d}"


def convert() -> None:
    root = Path(os.environ["nnUNet_raw"])
    raw = root / DATASET
    if raw.exists():
        raise SystemExit(f"{raw} exists; refusing to overwrite")
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (raw / sub).mkdir(parents=True)
    paths, parts = label_paths(), partitions()
    mapping, counts = {}, {}
    for split in ("train", "val", "test"):
        for donor in parts[split]:
            label = paths[index_of(donor)]
            image = Path(str(label).replace("-label.nii.gz", "-US.nii.gz"))
            case = case_of(donor)
            mapping[case] = {"donor": donor, "partition": split,
                             "source": str(label.relative_to(SOURCE))}
            if split == "test":
                # Test donors are images only: the network never sees their labels.
                shutil.copy(image, raw / "imagesTs" / f"{case}_0000.nii.gz")
            else:
                shutil.copy(image, raw / "imagesTr" / f"{case}_0000.nii.gz")
                shutil.copy(label, raw / "labelsTr" / f"{case}.nii.gz")
        counts[split] = len(parts[split])
    dataset = {
        "channel_names": {"0": "US"},
        "labels": LABELS,
        "numTraining": counts["train"] + counts["val"],
        "file_ending": ".nii.gz",
        "name": "MVSeg2023, benchmark valve partition",
        "description": ("MVSeg2023 valves under the fusion benchmark's donor partition (split seed "
                        "104729). imagesTr holds the training- and validation-partition donors; fold 0 "
                        "trains on the first and validates on the second. Test-partition donors are "
                        "imagesTs only."),
        "reference": "https://huggingface.co/datasets/pcarnahan/MVSeg2023",
        "licence": "CC BY-NC-ND 4.0",
    }
    (raw / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    # Kept beside, not inside, the dataset folder so nnU-Net's checks see only its own files.
    (root / f"{DATASET}_case_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps({"donors": counts, "raw": str(raw)}, indent=2))


def write_splits() -> None:
    folder = Path(os.environ["nnUNet_preprocessed"]) / DATASET
    parts = partitions()
    fold0 = {"train": [case_of(d) for d in parts["train"]], "val": [case_of(d) for d in parts["val"]]}
    (folder / "splits_final.json").write_text(json.dumps([fold0], indent=2) + "\n")
    print("fold 0: %d training donors, %d validation donors -> %s"
          % (len(fold0["train"]), len(fold0["val"]), folder / "splits_final.json"))


def donor_label_convention(labels):
    """The leaflet labels as the donor pool read them: 1 and 2 exchanged.

    The pool (``donors_v3``, built 4 September) predates the 9 September fix of
    the MVSeg2023 label convention (commit 5ec4e68: 1 is the posterior leaflet, 2
    the anterior), so it was fitted with the two leaflets exchanged. Fitting a
    case with 1 and 2 exchanged reproduces its stored donor exactly (checked on
    cases 0, 40, 145 and 174: 0.0 mm), which is what confirms the donor order.
    """
    import numpy as np

    volume = np.asarray(labels)
    exchanged = volume.copy()
    exchanged[volume == 1] = 2
    exchanged[volume == 2] = 1
    return exchanged


def check() -> None:
    """Re-fit four cases, one per MVSeg2023 folder and the last, against the stored donors."""

    import nibabel as nib
    import numpy as np

    from rooted4d.donors.export import _canonicalize, valve_donor_vertices
    from rooted4d.donors.valve import fit_valve

    manifest = json.loads((DONORS / "donors.json").read_text())
    n_annulus = int(manifest["n_annulus"])
    paths = label_paths()
    worst = 0.0
    for index in (0, 40, 145, 174):
        image = nib.load(str(paths[index]))
        # build_donor_pool.py's defaults for this pool: a 32-point valve ring, depth 6,
        # and the pool's leaflet labels (``donor_label_convention``).
        fit = fit_valve(donor_label_convention(np.asarray(image.dataobj)), image.header.get_zooms()[:3],
                        n_annulus=32, n_depth=6, case=paths[index].name.replace("-label.nii.gz", ""))
        refitted = _canonicalize(valve_donor_vertices(fit, n_annulus=n_annulus))
        stored = np.load(DONORS / "fixed_topology" / f"valve-{index:04d}.npz")["vertices"]
        difference = float(np.abs(refitted - stored).max())
        worst = max(worst, difference)
        print(f"valve-{index:04d} <- {paths[index].relative_to(SOURCE)}: max |difference| {difference:.3g} mm")
    if worst > 1e-6:
        raise SystemExit(f"the donor order does not match: max difference {worst:.3g} mm")
    print("donor order confirmed")


if __name__ == "__main__":
    mode = os.environ.get("R4D_MODE")
    {"convert": convert, "splits": write_splits, "check": check}.get(
        mode, lambda: sys.exit(f"unknown R4D_MODE {mode!r}"))()
