"""E45: strip numerical debris from 4DMR raw surfaces so postprocess succeeds.

25 of 150 ACDC patients failed 4DMR postprocessing with a GeometryError, either
"the retained zero-level components did not reconnect into one surface" or
"clipped surface has no boundary loop". Inspecting the retained artifacts shows
the cause is not anatomy: the raw endocardial surface already carries tiny
disconnected fragments before any clipping. On the three cases examined the main
body has an area of 47190, 23532 and 15549 while the extra components are 1.6,
28.7 and 3.4 -- between 3e-5 and 1.2e-3 of the surface.

Every one of those is below `MAX_ARTIFACT_AREA_RATIO = 0.005`, which is 4DMR's
own threshold for a negligible artifact component. Their older code path,
`_select_primary_parts`, tolerates components under that ratio explicitly. The
current algorithm, `independent_base_zero_level_loops`, does not: it requires
the retained components to merge into exactly one surface, so a speck three
hundredths of a percent of the surface aborts the case.

Rather than modify their algorithm, this removes the sub-threshold components
from the raw meshes that postprocessing reads, using their constant and their
own component definition. The originals are kept beside the cleaned files, and
every removal is reported, so the intervention is auditable and is bounded to
debris their code already calls negligible.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(os.environ["R4D_4DMR_DIR"])
sys.path.insert(0, str(SOURCE))
CASES = [c for c in os.environ.get("R4D_CLEAN_CASES", "").split() if c]
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E45_debris_removal.json"


def main() -> int:
    import pyvista as pv
    import postprocess_4dmm_bridge as bridge

    ratio_limit = bridge.MAX_ARTIFACT_AREA_RATIO
    print("using 4DMR's own MAX_ARTIFACT_AREA_RATIO = %g" % ratio_limit, flush=True)

    rows = []
    for case in CASES:
        raw_root = SOURCE / "outputs" / case / "artifacts" / "raw"
        if not raw_root.is_dir():
            rows.append({"case": case, "status": "no_raw_artifacts"})
            print("  %-12s no retained raw artifacts" % case, flush=True)
            continue
        for surface_path in sorted(raw_root.rglob("*.obj")):
            mesh = pv.read(str(surface_path))
            try:
                components = bridge._surface_components(mesh)
            except Exception as exc:
                rows.append({"case": case, "file": surface_path.name,
                             "status": "components_failed",
                             "reason": "%s: %s" % (type(exc).__name__, str(exc)[:80])})
                continue
            if len(components) <= 1:
                rows.append({"case": case, "file": surface_path.name,
                             "status": "single_component"})
                continue

            areas = np.array([float(c.area) for c in components], dtype=float)
            order = np.argsort(areas)[::-1]
            main_area = float(areas[order[0]])
            dropped, kept_indices = [], []
            for index in order:
                ratio = float(areas[index] / main_area) if main_area > 0 else 1.0
                if index == order[0] or ratio > ratio_limit:
                    kept_indices.append(int(index))
                else:
                    dropped.append({"area": float(areas[index]), "ratio": ratio})

            if not dropped:
                # Extra components are all above the negligible threshold, so
                # this is a genuine geometric problem and must not be silently
                # papered over.
                rows.append({"case": case, "file": surface_path.name,
                             "status": "components_above_threshold",
                             "areas": areas.tolist()})
                print("  %-12s %-28s %d components, none negligible -- left alone"
                      % (case, surface_path.name, len(components)), flush=True)
                continue

            backup = surface_path.with_suffix(".obj.orig")
            if not backup.exists():
                shutil.copy(surface_path, backup)
            merged = components[kept_indices[0]]
            for index in kept_indices[1:]:
                merged = merged.merge(components[index]).extract_surface().triangulate()
            merged.save(str(surface_path))

            rows.append({"case": case, "file": surface_path.name, "status": "cleaned",
                         "components_before": len(components),
                         "components_kept": len(kept_indices),
                         "dropped": dropped,
                         "largest_dropped_ratio": max(d["ratio"] for d in dropped)})
            print("  %-12s %-28s dropped %d of %d components, largest %.2e of the surface"
                  % (case, surface_path.name, len(dropped), len(components),
                     max(d["ratio"] for d in dropped)), flush=True)

    cleaned = [r for r in rows if r["status"] == "cleaned"]
    summary = {"cases": len(CASES), "files": len(rows), "cleaned": len(cleaned),
               "ratio_limit": ratio_limit,
               "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                                 for s in sorted({r["status"] for r in rows})}}
    if cleaned:
        worst = max(r["largest_dropped_ratio"] for r in cleaned)
        summary["largest_dropped_ratio_overall"] = worst
        print("\nlargest component removed anywhere: %.2e of its surface, against a "
              "threshold of %g" % (worst, ratio_limit))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"experiment": "E45_debris_removal",
                                  "summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
