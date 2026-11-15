"""E117: the residual field on the other benchmark sets and on MITEA.

What the comparison depends on: the available-evidence placement of MITEA
chambers does not depend on the end-diastolic label chamber (E99's does), and
re-placing stored chambers gives the result of placing the raw ones, so both
conventions can be computed from E99's stored geometry; each benchmark set's
validation reference is its joint fit at the frozen session weight; the frozen
settings scored on test are the main configuration's; and the predictions are
read per set, A4 on the realistic-session sets only.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rooted4d.paths import workspace_root

ROOT = workspace_root()
EXPERIMENTS = ROOT / "Codes/scripts/experiments"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SETS = _load("e117_field_replication")
SETS.MAIN_CONFIGURATION = ROOT / "Codes/configs/rooted4d_main.json"
MITEA = _load("e117_mitea_field")


def _chamber(rng, scale=1.0):
    """A reduced chamber: annulus (24), lower ring (24), apex, ring numbered counterclockwise."""

    angles = 2 * np.pi * np.arange(24) / 24
    annulus = np.stack([20 * np.cos(angles), 20 * np.sin(angles), np.zeros(24)], axis=1)
    lower = np.stack([14 * np.cos(angles), 14 * np.sin(angles), np.full(24, -25.0)], axis=1)
    apex = np.array([[0.0, 0.0, -50.0]])
    return scale * np.concatenate([annulus, lower, apex]) + rng.normal(0, 0.5, size=(49, 3))


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PlacementTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.template = _chamber(rng)
        self.raw = {(scan, phase, source): _chamber(rng, 0.9 if phase == "ES" else 1.1)
                    for scan in ("scan1", "scan2") for phase in ("ES", "ED") for source in ("label", "prediction")}
        self.raw[("scan2", "ED", "prediction")] = None          # a front-end failure is carried through
        rotation, translation = _rotation(rng), rng.normal(0, 30, size=3)
        self.moved = {k: (None if v is None else MITEA.E99._reverse(v) @ rotation.T + translation)
                      for k, v in self.raw.items()}

    def _close(self, first, second, tolerance=1e-9):
        for key in first:
            if first[key] is None:
                self.assertIsNone(second[key])
            else:
                self.assertLess(np.abs(first[key] - second[key]).max(), tolerance, key)

    def test_stored_chambers_give_the_raw_result(self):
        for rule in ("es", "es_ed_mean"):
            self._close(MITEA.place(self.raw, self.template, rule)[0],
                        MITEA.place(self.moved, self.template, rule)[0])

    def test_e99_rule_leaves_its_own_result_unchanged(self):
        placed, _ = MITEA.place(self.raw, self.template, "es_ed_mean")
        again, moved = MITEA.place(placed, self.template, "es_ed_mean")
        self._close(placed, again)
        self.assertLess(moved["rotation_deg"], 1e-6)
        self.assertLess(moved["translation_mm"], 1e-9)

    def test_available_evidence_placement_ignores_the_end_diastolic_label(self):
        changed = dict(self.raw)
        changed[("scan1", "ED", "label")] = self.raw[("scan1", "ED", "label")] + np.array([0.0, 6.0, 3.0])
        key = ("scan1", "ES", "label")
        es_before = MITEA.place(self.raw, self.template, "es")[0]
        es_after = MITEA.place(changed, self.template, "es")[0]
        self.assertLess(np.abs(es_before[key] - es_after[key]).max(), 1e-12)
        mean_before = MITEA.place(self.raw, self.template, "es_ed_mean")[0]
        mean_after = MITEA.place(changed, self.template, "es_ed_mean")[0]
        self.assertGreater(np.abs(mean_before[key] - mean_after[key]).max(), 0.5)


class BenchmarkSetsTest(unittest.TestCase):
    def test_frozen_settings_are_the_main_configuration(self):
        self.assertEqual(SETS.frozen_settings(), {"R4D+GP-frozen": ["shared", 4.0, 0.5, 5.0],
                                                  "R4D+GP-session-frozen": ["session", 4.0, 0.5, 5.0]})

    def test_fourteen_sets_seven_per_session_level(self):
        self.assertEqual(len(SETS.SETS), 14)
        self.assertEqual(sum(1 for v in SETS.SETS.values() if v[1] == "realistic"), 7)
        self.assertNotIn("v4n96o", SETS.SETS)
        self.assertNotIn("v3n96o", SETS.SETS)

    def _chain(self, root, identity, overrides):
        chain = root / "e100" / "v3o"
        (chain / "test" / "t1").mkdir(parents=True)
        (chain / "TEST_DONE").write_text("test/t1\n")
        frozen = {"base_config": {"fourier_order": 3},             # the session weight left at the default
                  "variants": [{"name": "S/joint", "overrides": overrides}]}
        (chain / "test" / "t1" / "spec.json").write_text(json.dumps(frozen))
        for round_name, weights in (("session", (0.5, 12.0)), ("session_ext1", (0.125,))):
            (chain / round_name / "r").mkdir(parents=True)
            (chain / round_name / "r" / "spec.json").write_text(json.dumps({"split": "val", "variants": [
                {"name": f"S/joint/bs={w:g}", "overrides": {"identity_prior_weight": identity,
                                                            "session_prior_weight": w}} for w in weights]}))
        return chain

    def test_validation_reference_is_the_frozen_session_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["R4D_RESULTS"] = directory
            chain = self._chain(Path(directory), 0.5, {"identity_prior_weight": 0.5})
            frozen, _ = SETS._frozen("v3o")
            self.assertEqual(SETS.session_weight(frozen), 12.0)       # the solver's default (E73's)
            path, name = SETS.validation_reference("v3o", frozen)
            self.assertEqual((path, name), (chain / "session" / "r", "S/joint/bs=12"))

    def test_validation_reference_refuses_another_identity_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["R4D_RESULTS"] = directory
            self._chain(Path(directory), 2.0, {"identity_prior_weight": 0.5, "session_prior_weight": 0.125})
            frozen, _ = SETS._frozen("v3o")
            with self.assertRaises(ValueError):
                SETS.validation_reference("v3o", frozen)

    def test_predictions_per_set(self):
        def entry(upper, lower=None):
            return {"observed": 0.0, "upper": upper, "lower": -1.0 if lower is None else lower}

        def comparisons(below, above):
            out = {f"{a} - {b}": {"sum": entry(-0.1 if below else 0.1), "cmr": entry(-0.1), "tee": entry(-0.1)}
                   for a, b in SETS.COMPARISONS}
            out["R4D+GP - R4D+GP-session"]["sum"] = entry(1.0, 0.1 if above else -0.1)
            return out

        sets = {"v4n48o": {"minus": comparisons(True, True)}, "v3o": {"minus": comparisons(False, False)}}
        result = SETS.predictions(sets)
        self.assertEqual(result["A1_sum_lower_than_joint_fit"]["holds_on"], 1)
        self.assertFalse(result["A1_sum_lower_than_joint_fit"]["holds"])
        self.assertEqual(result["A4_per_modality_sum_lower_on_realistic_session"]["sets"], 1)
        self.assertTrue(result["A4_per_modality_sum_lower_on_realistic_session"]["holds"])


if __name__ == "__main__":
    unittest.main()
