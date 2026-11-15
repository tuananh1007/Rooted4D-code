"""E49c: TetHeart against Rooted4D on identical frames, paired.

E49d scored TetHeart's predicted cavity on the ACDC test frames, in two variants:
the prediction as it comes, and the prediction clipped to the reference label's
own extent. This pairs either against our fitting on exactly the frames TetHeart
produced a scorable mesh for, with the patient as the analysis unit, reusing
E39's statistics so the two external comparisons are computed the same way.

The default input is the harmonized E49g artifact, rescored with E36's reference
and surface sampling convention. R4D_TETHEART_JSON can select another artifact
for an explicit historical or sensitivity comparison.

Why the clipped variant is the one to pair. TetHeart reconstructs complete
anatomy; the ACDC cavity label stops at the basal and apical slices. On the test
split the prediction spans about 126 mm along the stack axis where the label
spans 90, so the unclipped surface-to-points direction is charged for anatomy
the reference never annotated: the two directions differ by roughly a factor of
six in the end-cap-free diagnostic. That is an extent mismatch, not an accuracy
difference, and a symmetric mean taken over it is not a like-for-like quantity.
Both are reported, neither is
dropped, and the raw one stays on the record.

The primary metric is the end-cap-inclusive one, because that is what our panel A
row computes: it scores against the full cavity boundary without dropping the end
slices. The end-cap-free variant is reported alongside it.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e39_ghdheart_comparison import PERMUTATIONS, RESAMPLES, SEED, paired_stats  # noqa: E402

TETHEART = Path(os.environ.get(
    "R4D_TETHEART_JSON", "results/22102244_thfinal/E49g_test_iter_60000_matched.json"))
OURS = Path("results/21931418/E36_image_input_chamber.json")
OUTPUT = Path(os.environ.get(
    "R4D_OUT_JSON", "results/benchmark/E49c_tetheart_comparison.json"))


def main() -> int:
    payload = json.loads(TETHEART.read_text())
    tet_rows = [r for r in payload["rows"] if r.get("status") == "scored"]
    our_rows = {(r["patient"], r["frame"]): r
                for r in json.loads(OURS.read_text())["rows"]
                if r.get("label_input") and r.get("image_input")}

    report = {"experiment": "E49c_tetheart_comparison",
              "tetheart_source": str(TETHEART),
              "tetheart_summary_status": payload["summary"]["status_counts"],
              "class_check": payload.get("class_check"),
              "resamples": RESAMPLES, "permutations": PERMUTATIONS, "seed": SEED,
              "variants": {}, "directional": {}}

    for variant in ("clipped", "raw"):
        for caps in ("with_endcaps", "no_endcaps"):
            key = "symmetric_mm_%s_%s" % (variant, caps)
            tet = {(r["patient"], r["frame"]): r[key] for r in tet_rows if key in r}
            shared = sorted(set(tet) & set(our_rows))
            if not shared:
                continue
            patients = np.array([p for p, _ in shared])
            rng = np.random.default_rng(SEED)
            arms = {
                "tetheart": np.array([tet[k] for k in shared]),
                "rooted4d_label_input": np.array(
                    [our_rows[k]["label_input"]["symmetric_mean_mm"] for k in shared]),
                "rooted4d_image_input": np.array(
                    [our_rows[k]["image_input"]["symmetric_mean_mm"] for k in shared]),
            }
            # Our per-case scores (E36) are end-cap inclusive and have no
            # end-cap-free counterpart, so only the with_endcaps rows pair like
            # with like. The no_endcaps rows are kept as a property of
            # TetHeart's own reconstruction and must not be read as a paired
            # comparison against us.
            matched = caps == "with_endcaps"
            entry = {"n_frames": len(shared),
                     "n_patients": int(len(set(patients))),
                     "matched_to_our_scoring": matched,
                     "medians": {k: float(np.median(v)) for k, v in arms.items()},
                     "paired": {}}
            if not matched:
                entry["caveat"] = ("TetHeart scored without end caps against our "
                                   "end-cap-inclusive scores; not a like-for-like pairing")
            for other in ("rooted4d_label_input", "rooted4d_image_input"):
                entry["paired"]["tetheart_minus_" + other] = paired_stats(
                    arms["tetheart"], arms[other], patients, rng)
            report["variants"]["%s_%s" % (variant, caps)] = entry

            d = entry["paired"]["tetheart_minus_rooted4d_label_input"]
            print("%-22s n=%2d/%2d pat | TetHeart %.3f  ours %.3f | paired %+.3f "
                  "CI [%.3f, %.3f] p=%.2g"
                  % (variant + " " + caps, entry["n_frames"], entry["n_patients"],
                     entry["medians"]["tetheart"],
                     entry["medians"]["rooted4d_label_input"],
                     d["difference"], d["ci_low"], d["ci_high"], d["p_value"]))

            # The directional split is the evidence that the raw symmetric mean is
            # dominated by extent rather than by accuracy.
            p2s = [r["points_to_surface_mm_%s_%s" % (variant, caps)] for r in tet_rows
                   if "points_to_surface_mm_%s_%s" % (variant, caps) in r]
            s2p = [r["surface_to_points_mm_%s_%s" % (variant, caps)] for r in tet_rows
                   if "surface_to_points_mm_%s_%s" % (variant, caps) in r]
            if p2s:
                report["directional"]["%s_%s" % (variant, caps)] = {
                    "label_to_prediction_mm": float(np.median(p2s)),
                    "prediction_to_label_mm": float(np.median(s2p)),
                    "ours_label_to_prediction_mm": float(np.median(
                        [our_rows[k]["label_input"]["mean_points_to_surface_mm"]
                         for k in shared])),
                    "ours_prediction_to_label_mm": float(np.median(
                        [our_rows[k]["label_input"]["mean_surface_to_points_mm"]
                         for k in shared])),
                }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
