"""E55: score the Direction 1 stage-1 pilot against its pre-declared gates.

Reads ``E54_stage1.json`` and applies the endpoints and gates frozen in
``Docs/DIRECTION1_PILOT_RULE_FROZEN_2026-09-17.md`` before the run. Nothing here
chooses a threshold after seeing the numbers.

The primary endpoint is **contamination**: how much a modality's held-out error
grows when the *other* modality's evidence is corrupted. It is measured per
identity, against that identity's own nominal error, so a method that is simply
better everywhere does not score as protective.

    contamination(arm, identity, corrupted side) =
        heldout(clean side | corrupted) - heldout(clean side | nominal)

Variants are averaged within identity first, because two variants of one donor
are not two independent observations. Tail error is then the **mean of the worst
three identity values out of twelve**, which is what the contract specifies for a
sample this small -- not a percentile, which twelve points cannot estimate.

Gates, all required, each corruption direction inspected separately:

1. attachment exact to numerical tolerance;
2. the budgeted rule's tail improves by >= 20% against *both* baseline 1 (ring
   projection and refitting) and baseline 3 (robust exact joint fitting);
3. nominal accuracy within the tolerance frozen from development;
4. no modality funds its gain through an unreported loss in the other.
"""

import json
import os
from pathlib import Path

import numpy as np

STAGE1 = Path(os.environ.get("R4D_STAGE1", "results/benchmark/E54_stage1.json"))
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E55_stage1_report.json"
TAIL_WORST = int(os.environ.get("R4D_TAIL_WORST", "3"))
GATE_IMPROVEMENT = float(os.environ.get("R4D_GATE_IMPROVEMENT", "0.20"))
NOMINAL_TOLERANCE = float(os.environ.get("R4D_NOMINAL_TOLERANCE", "0.02"))
#: Below this, an absolute difference is numerical noise rather than a worse
#: arm. The budgeted rule coincides with the joint fit to about 1e-7 mm when its
#: constraint never binds, and a 1e-9 threshold flagged that coincidence as a
#: regression, which is a broken guard rather than a finding.
ABSOLUTE_TOLERANCE_MM = float(os.environ.get("R4D_ABSOLUTE_TOLERANCE_MM", "1e-6"))

ARMS = ("ring_refit", "joint", "robust", "budgeted")
OTHER = {"cmr": "tee", "tee": "cmr"}


def heldout(row, arm, side):
    block = row["arms"].get(arm)
    if not block:
        return None
    value = block.get("heldout", {}).get("heldout_%s_mm" % side)
    return None if value is None else float(value)


def main() -> int:
    payload = json.loads(STAGE1.read_text())
    results = payload["results"]

    # identity -> condition key -> arm -> [values across variants]
    gathered: dict[str, dict[str, dict[str, list[float]]]] = {}
    attachment: list[float] = []
    for entry in results:
        identity = entry["identity"]
        for row in entry["rows"]:
            key = row["condition"] if row["modality"] is None else "%s_%s" % (
                row["condition"], row["modality"])
            slot = gathered.setdefault(identity, {}).setdefault(key, {})
            for arm in ARMS:
                for side in ("cmr", "tee"):
                    value = heldout(row, arm, side)
                    if value is not None:
                        slot.setdefault("%s|%s" % (arm, side), []).append(value)
            joint = row["arms"].get("joint", {})
            if "attachment_residual_mm" in joint:
                attachment.append(float(joint["attachment_residual_mm"]))

    identities = sorted(gathered)

    def mean_of(identity, key, arm, side):
        values = gathered.get(identity, {}).get(key, {}).get("%s|%s" % (arm, side))
        return float(np.mean(values)) if values else None

    report = {"experiment": "E55_direction1_pilot_report",
              "identities": len(identities),
              "tail_definition": "mean of the worst %d identity values" % TAIL_WORST,
              "gate_improvement": GATE_IMPROVEMENT,
              "attachment_residual_max_mm": (float(np.max(attachment)) if attachment else None),
              "contamination": {}, "nominal": {}, "gates": {}}

    # Nominal cost of the rule against the unconstrained joint fit.
    for side in ("cmr", "tee"):
        joint = [mean_of(i, "nominal", "joint", side) for i in identities]
        budgeted = [mean_of(i, "nominal", "budgeted", side) for i in identities]
        pairs = [(a, b) for a, b in zip(joint, budgeted) if a and b]
        ratio = [b / a - 1.0 for a, b in pairs]
        report["nominal"][side] = {
            "joint_median_mm": float(np.median([a for a, _ in pairs])) if pairs else None,
            "budgeted_median_mm": float(np.median([b for _, b in pairs])) if pairs else None,
            "worst_relative_cost": float(np.max(ratio)) if ratio else None,
        }

    # Contamination, per corrupted side, per arm.
    for corrupted in ("cmr", "tee"):
        clean = OTHER[corrupted]
        key = "localized_%s" % corrupted
        block = {}
        for arm in ARMS:
            values = []
            for identity in identities:
                base = mean_of(identity, "nominal", arm, clean)
                hurt = mean_of(identity, key, arm, clean)
                if base is not None and hurt is not None:
                    values.append(hurt - base)
            if not values:
                continue
            ordered = np.sort(np.asarray(values))[::-1]
            block[arm] = {
                "n_identities": len(values),
                "median_mm": float(np.median(values)),
                "tail_mm": float(np.mean(ordered[:TAIL_WORST])),
            }
        report["contamination"]["corrupted_%s_clean_%s" % (corrupted, clean)] = block

        if {"budgeted", "ring_refit", "robust"} <= set(block):
            rule = block["budgeted"]["tail_mm"]
            gate = {}
            for rival in ("ring_refit", "robust"):
                other = block[rival]["tail_mm"]
                improvement = (other - rule) / other if other else float("nan")
                gate[rival] = {"rival_tail_mm": other, "rule_tail_mm": rule,
                               "improvement": float(improvement),
                               "passes": bool(improvement >= GATE_IMPROVEMENT)}
            report["gates"]["corrupted_%s" % corrupted] = gate

    # Absolute accuracy, because contamination is a *within-arm* delta: an arm
    # that is uniformly worse has little room to worsen and scores well on it.
    # In the first stage-1 run the robust arm had the lowest contamination in
    # both directions while being worse than the joint fit in all ten
    # condition/side cells, which would have read as a win.
    conditions = sorted({k for i in identities for k in gathered.get(i, {})})
    for key in conditions:
        block = {}
        for arm in ARMS:
            for side in ("cmr", "tee"):
                values = [mean_of(i, key, arm, side) for i in identities]
                values = [v for v in values if v is not None]
                if values:
                    block["%s|%s" % (arm, side)] = float(np.median(values))
        report.setdefault("absolute_median_mm", {})[key] = block

    confounded = []
    for key, block in report.get("absolute_median_mm", {}).items():
        for arm in ARMS:
            if arm == "joint":
                continue
            worse = [
                side for side in ("cmr", "tee")
                if block.get("%s|%s" % (arm, side)) is not None
                and block.get("joint|%s" % side) is not None
                and block["%s|%s" % (arm, side)] > block["joint|%s" % side] + ABSOLUTE_TOLERANCE_MM
            ]
            if worse:
                confounded.append({"condition": key, "arm": arm, "worse_on": worse})
    report["absolute_worse_than_joint"] = confounded

    attachment_ok = (report["attachment_residual_max_mm"] or 0.0) < 1e-6
    nominal_ok = all(
        (v["worst_relative_cost"] is None) or (v["worst_relative_cost"] <= NOMINAL_TOLERANCE)
        for v in report["nominal"].values())
    improvement_ok = bool(report["gates"]) and all(
        entry["passes"] for gate in report["gates"].values() for entry in gate.values())

    # A contamination advantage is credited only if the arm claiming it is not
    # simply worse everywhere. Contamination is a within-arm delta, so a flat,
    # mediocre arm scores well on it without protecting anything. The check is
    # on the *rule*, because that is where a contamination-only false positive
    # would arise: a rival being flattered makes the gate harder, not easier.
    #
    # ``absolute_worse_than_joint`` above is a broader diagnostic -- it lists any
    # arm worse than the joint fit on any condition, whether or not that arm won
    # anything -- so it is reported but is deliberately not the gate.
    rule_worse = [e for e in report["absolute_worse_than_joint"] if e["arm"] == "budgeted"]
    rule_not_confounded = not rule_worse
    report["rule_absolutely_worse_than_joint"] = rule_worse

    report["verdict"] = {
        "attachment_exact": attachment_ok,
        "nominal_within_tolerance": nominal_ok,
        "improvement_gate_met_in_both_directions": improvement_ok,
        "rule_not_absolutely_worse_than_joint": rule_not_confounded,
        "advance": bool(
            attachment_ok and nominal_ok and improvement_ok and rule_not_confounded),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    print("identities %d | attachment max %.2e mm"
          % (report["identities"], report["attachment_residual_max_mm"] or 0.0))
    for name, block in report["contamination"].items():
        print("\n%s" % name)
        for arm, stats in block.items():
            print("  %-11s median %+.4f  tail %+.4f  (n=%d)"
                  % (arm, stats["median_mm"], stats["tail_mm"], stats["n_identities"]))
    for name, gate in report["gates"].items():
        print("\ngate %s" % name)
        for rival, entry in gate.items():
            print("  vs %-11s rule %.4f against %.4f -> %+.1f%%  %s"
                  % (rival, entry["rule_tail_mm"], entry["rival_tail_mm"],
                     100.0 * entry["improvement"], "PASS" if entry["passes"] else "FAIL"))
    print("\nverdict:", json.dumps(report["verdict"]))
    print("wrote", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
