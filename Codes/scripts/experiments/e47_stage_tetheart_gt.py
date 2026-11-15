"""E47: merge validated ground truth into TetHeart's preprocessed folders.

TetHeart's loader reads `*_mesh*.obj` from a patient folder and splits the
sorted list at index three, so a patient is only usable when it has exactly six
meshes: three structures at end diastole and three at end systole. A patient
with one accepted frame would present three meshes, and the loader would take
all three as end diastole and hand an empty list over as end systole.

So completeness is enforced here rather than discovered during training: a
patient contributes only if both of its frames cleared E44's gate and all three
structures were written. Everything else is excluded and counted.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

GT = Path(os.environ["R4D_TETHEART_GT"])
DEST = Path(os.environ["R4D_TETHEART_DATA"]) / "914_3d"
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E47_gt_staging.json"
SPLITS = json.loads(Path(os.environ["R4D_SPLITS"]).read_text())
STRUCTURES = ("lv", "myo", "rv")


def main() -> int:
    by_patient = defaultdict(lambda: defaultdict(set))
    for case_dir in sorted(GT.glob("ACDC_patient*")):
        for mesh in case_dir.glob("*_mesh_*.obj"):
            stem = mesh.name.replace(".obj", "")
            structure = stem.rsplit("_mesh_", 1)[1]
            frame = int(stem.split("_frame")[1][:2])
            by_patient[case_dir.name][frame].add(structure)

    staged, rejected = [], []
    for case, frames in sorted(by_patient.items()):
        complete = {f for f, s in frames.items() if set(STRUCTURES) <= s}
        if len(complete) != 2:
            rejected.append({"case": case, "reason": "needs two complete frames",
                             "complete_frames": sorted(complete),
                             "frames_present": {str(f): sorted(s) for f, s in frames.items()}})
            continue
        target = DEST / case
        if not target.is_dir():
            rejected.append({"case": case, "reason": "no preprocessed folder"})
            continue
        # Remove any meshes from an earlier staging so the folder cannot end up
        # with a stale third frame and a silently wrong split at index three.
        for old in target.glob("*_mesh_*.obj"):
            old.unlink()
        copied = 0
        for frame in sorted(complete):
            for structure in STRUCTURES:
                name = "%s_frame%02d_mesh_%s.obj" % (case, frame, structure)
                shutil.copy(GT / case / name, target / name)
                copied += 1
        patient = case.replace("ACDC_", "")
        staged.append({"case": case, "split": SPLITS.get(patient, "unknown"),
                       "frames": sorted(complete), "meshes": copied})

    counts = defaultdict(int)
    for row in staged:
        counts[row["split"]] += 1
    summary = {"patients_with_any_gt": len(by_patient),
               "staged": len(staged), "rejected": len(rejected),
               "staged_by_split": dict(counts),
               "meshes_per_patient": sorted({r["meshes"] for r in staged})}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"experiment": "E47_gt_staging", "summary": summary,
                                  "staged": staged, "rejected": rejected}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for row in rejected[:10]:
        print("  excluded %-22s %s" % (row["case"], row["reason"]))
    if len(rejected) > 10:
        print("  ... and %d more" % (len(rejected) - 10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
