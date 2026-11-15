"""E41: convert MVSeg2023 into nnU-Net v2 raw format, respecting the split.

MVSeg2023 ships its own 105/30/40 split. nnU-Net must train on the 105 and be
scored on the 40, and it must never see the 40 during training or model
selection. nnU-Net does its own five-fold cross-validation over whatever lands
in imagesTr, so putting only the 105 training cases there is what enforces the
boundary; the 30 validation cases are kept out of training entirely and are
available if any selection is ever needed, which for a self-configuring method
it is not.

Our own leaflet figures are restricted to the same 40 cases by E40, because a
175-case number beside a 40-case number would not be a comparison.

Labels follow the MVSeg2023 card: 0 background, 1 posterior leaflet, 2 anterior
leaflet. That convention is taken from the dataset rather than assumed; getting
it backwards is a mistake this project has already made once.
"""

import json
import os
import shutil
from pathlib import Path

SOURCE = Path(os.environ.get("R4D_MVSEG", "data/raw/mvseg2023"))
RAW = Path(os.environ["nnUNet_raw"]) / "Dataset501_MVSeg"
LABELS = {"background": 0, "posterior_leaflet": 1, "anterior_leaflet": 2}


def cases(split):
    return sorted(p.name.replace("-US.nii.gz", "")
                  for p in (SOURCE / split).glob("*-US.nii.gz"))


def main() -> int:
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (RAW / sub).mkdir(parents=True, exist_ok=True)

    counts = {}
    for split, image_dir, label_dir in (("train", "imagesTr", "labelsTr"),
                                        ("test", "imagesTs", "labelsTs")):
        names = cases(split)
        counts[split] = len(names)
        for name in names:
            # nnU-Net wants <case>_<modality index>.nii.gz for images and
            # <case>.nii.gz for labels; one modality here.
            shutil.copy(SOURCE / split / ("%s-US.nii.gz" % name),
                        RAW / image_dir / ("%s_0000.nii.gz" % name))
            shutil.copy(SOURCE / split / ("%s-label.nii.gz" % name),
                        RAW / label_dir / ("%s.nii.gz" % name))
        print("  %-5s %d cases -> %s" % (split, len(names), image_dir), flush=True)

    held_out = cases("val")
    counts["val_held_out_unused"] = len(held_out)
    print("  val   %d cases deliberately not converted (kept out of training)"
          % len(held_out))

    dataset = {
        "channel_names": {"0": "US"},
        "labels": LABELS,
        "numTraining": counts["train"],
        "file_ending": ".nii.gz",
        "name": "MVSeg2023",
        "description": ("Mitral valve leaflets from 3D TEE. Trained on the challenge's own "
                        "105 training cases; the 40 challenge test cases are inference only "
                        "and the 30 validation cases are excluded entirely."),
        "reference": "https://huggingface.co/datasets/pcarnahan/MVSeg2023",
        "licence": "CC BY-NC-ND 4.0",
    }
    (RAW / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    print("\nwrote %s" % (RAW / "dataset.json"))
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
