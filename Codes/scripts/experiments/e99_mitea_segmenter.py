"""E99, part 1: an echo segmenter for real two-modality chamber data (plan v2, W3.1).

Pre-registered in ``Docs/E99_MITEA_SEGMENTER_PREREGISTRATION_2026-09-29.md``.
MITEA pairs 3D echocardiography of 134 subjects with LV labels derived from CMR
of the same subject and registered into the echo frame, at end diastole and end
systole, with a scan and a rescan. W3.1 fits the CMR-derived labels on one side
and a segmenter's prediction on the echo image on the other, so it needs a
segmenter that never saw the held-out subjects:

* ``split``: a subject-level split, stratified by diagnostic category, with a
  fixed seed -> ``Codes/configs/e99/mitea_split.json`` (committed with the
  record, before any conversion);
* ``convert``: MITEA -> nnU-Net raw (``Dataset504_MITEA``). ``imagesTr`` and
  ``labelsTr`` hold every volume of the training and validation subjects;
  ``imagesTs`` holds the test subjects' images only;
* ``splits``: fold 0 trains on the training subjects and validates on the
  validation subjects.

    R4D_MODE=split   python e99_mitea_segmenter.py
    R4D_MODE=convert python e99_mitea_segmenter.py    (nnU-Net environment)
    R4D_MODE=splits  python e99_mitea_segmenter.py

Only the standard library and NumPy are needed.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

PREREGISTRATION = "Docs/E99_MITEA_SEGMENTER_PREREGISTRATION_2026-09-29.md"
DATASET = "Dataset504_MITEA"
LABELS = {"background": 0, "myocardium": 1, "cavity": 2}
SOURCE = Path(os.environ.get("R4D_MITEA", "data/raw/mitea"))
SPLIT = Path("Codes/configs/e99/mitea_split.json")
SEED = 20261007
FRACTIONS = {"val": 0.15, "test": 0.15}
VOLUMES = ("scan1_ED", "scan1_ES", "scan2_ED", "scan2_ES")


def write_split() -> None:
    import numpy as np

    rows = list(csv.DictReader((SOURCE / "demographics.csv").open()))
    by_category: dict[str, list[str]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["mitea_id"])
    rng = np.random.default_rng(SEED)
    split = {"train": [], "val": [], "test": []}
    for category in sorted(by_category):
        subjects = sorted(by_category[category])
        order = [subjects[i] for i in rng.permutation(len(subjects))]
        n_test = round(FRACTIONS["test"] * len(order))
        n_val = round(FRACTIONS["val"] * len(order))
        split["test"] += order[:n_test]
        split["val"] += order[n_test:n_test + n_val]
        split["train"] += order[n_test + n_val:]
    split = {k: sorted(v) for k, v in split.items()}
    assert not (set(split["train"]) & set(split["val"]) or set(split["train"]) & set(split["test"])
                or set(split["val"]) & set(split["test"]))
    assert sum(len(v) for v in split.values()) == len(rows) == 134
    record = {"experiment": "E99", "preregistration": PREREGISTRATION, "seed": SEED,
              "stratified_by": "category", "fractions": FRACTIONS,
              "counts": {k: len(v) for k, v in split.items()}, **split}
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["counts"]), "->", SPLIT)


def _split() -> dict:
    return json.loads(SPLIT.read_text())


def convert() -> None:
    root = Path(os.environ["nnUNet_raw"])
    raw = root / DATASET
    if raw.exists():
        raise SystemExit(f"{raw} exists; refusing to overwrite")
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (raw / sub).mkdir(parents=True)
    split = _split()
    counts = {}
    for part in ("train", "val", "test"):
        for subject in split[part]:
            for volume in VOLUMES:
                name = f"{subject}_{volume}.nii.gz"
                case = f"{subject}_{volume}"
                if part == "test":
                    # Test subjects are images only: the network never sees their labels.
                    shutil.copy(SOURCE / "images" / name, raw / "imagesTs" / f"{case}_0000.nii.gz")
                else:
                    shutil.copy(SOURCE / "images" / name, raw / "imagesTr" / f"{case}_0000.nii.gz")
                    shutil.copy(SOURCE / "labels" / name, raw / "labelsTr" / f"{case}.nii.gz")
        counts[part] = len(split[part]) * len(VOLUMES)
    dataset = {
        "channel_names": {"0": "US"},
        "labels": LABELS,
        "numTraining": counts["train"] + counts["val"],
        "file_ending": ".nii.gz",
        "name": "MITEA, subject-level split",
        "description": ("MITEA 3D echocardiography with CMR-derived LV labels, split by subject "
                        "(stratified by category, seed %d). imagesTr holds the training and validation "
                        "subjects' volumes; fold 0 trains on the first and validates on the second. Test "
                        "subjects are imagesTs only." % SEED),
    }
    (raw / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    print(json.dumps({"volumes": counts, "raw": str(raw)}, indent=2))


def write_splits() -> None:
    folder = Path(os.environ["nnUNet_preprocessed"]) / DATASET
    split = _split()
    fold0 = {part: [f"{s}_{v}" for s in split[key] for v in VOLUMES]
             for part, key in (("train", "train"), ("val", "val"))}
    (folder / "splits_final.json").write_text(json.dumps([fold0], indent=2) + "\n")
    print("fold 0: %d training volumes, %d validation volumes -> %s"
          % (len(fold0["train"]), len(fold0["val"]), folder / "splits_final.json"))


if __name__ == "__main__":
    mode = os.environ.get("R4D_MODE")
    {"split": write_split, "convert": convert, "splits": write_splits}.get(
        mode, lambda: sys.exit(f"unknown R4D_MODE {mode!r}"))()
