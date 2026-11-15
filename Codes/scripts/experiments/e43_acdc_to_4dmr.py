"""E43: stage ACDC label frames as 4DMR inference inputs.

TetHeart is supervised on ground-truth meshes that its repository does not
generate, and its README points at 4DMR to produce them. 4DMR's release takes
mask volumes and ships inference-only ACDC checkpoints, so this is the
documented recipe rather than a reconstruction of one.

4DMR expects `0=background, 1=RV cavity, 2=LV myocardium, 3=LV cavity`, which
is ACDC's own convention, so no relabelling is needed. It expects sequentially
numbered frames per case with a nominated reference phase; ACDC labels exactly
two frames per patient, end diastole and end systole, and the lower frame
number is end diastole. TetHeart's own sample carries meshes at those same two
frames, so two-frame input matches what it was trained on rather than reducing
it.

Writes only the patients a given split assigns, so ground truth is never
generated for a patient the evaluation will not use.
"""

import json
import os
import shutil
from pathlib import Path

ACDC = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
DEST = Path(os.environ["R4D_4DMR_INPUTS"])
SPLITS_FILE = os.environ.get("R4D_SPLITS")
WANTED = set(os.environ.get("R4D_SPLIT_NAMES", "train,val,test").split(","))
LIMIT = int(os.environ.get("R4D_LIMIT", "0")) or None


def main() -> int:
    splits = {}
    if SPLITS_FILE:
        splits = json.loads(Path(SPLITS_FILE).read_text())

    by_patient: dict[str, list[Path]] = {}
    for path in sorted(ACDC.rglob("*_frame??_gt.nii.gz")):
        by_patient.setdefault(path.parent.name, []).append(path)

    staged = {}
    for patient, paths in sorted(by_patient.items()):
        if splits and splits.get(patient) not in WANTED:
            continue
        frames = sorted(paths, key=lambda p: int(p.name.split("_frame")[1][:2]))
        if len(frames) < 2:
            print("  %s: only %d labeled frame(s), skipped" % (patient, len(frames)))
            continue
        case_dir = DEST / patient / "masks"
        case_dir.mkdir(parents=True, exist_ok=True)
        mapping = []
        for index, source in enumerate(frames):
            target = case_dir / ("%02d.nii.gz" % index)
            shutil.copy(source, target)
            mapping.append({"index": index, "source": source.name,
                            "acdc_frame": int(source.name.split("_frame")[1][:2])})
        staged[patient] = mapping
        if LIMIT and len(staged) >= LIMIT:
            break

    # The reference phase is index 0 for every case, because ACDC's lower frame
    # number is end diastole; recorded explicitly so the assumption is visible.
    cases = {"default": {"reference_phase": 0, "mirror": "auto", "slice_order": "auto",
                         "num_iterations": int(os.environ.get("R4D_4DMR_ITER", "2000")),
                         "phase_batch_size": 2, "resolution": 64,
                         "retain_artifacts": False,
                         "stages": {"preprocess": True, "inference": True,
                                    "postprocess": True, "video": False}},
             "cases": {}}
    (DEST / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (DEST / "frame_mapping.json").write_text(json.dumps(staged, indent=2) + "\n")

    print("staged %d cases under %s" % (len(staged), DEST))
    print("frames per case: %s" % sorted({len(v) for v in staged.values()}))
    print("wrote cases.json (video stage disabled) and frame_mapping.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
