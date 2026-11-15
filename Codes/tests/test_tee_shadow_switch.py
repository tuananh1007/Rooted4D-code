"""E106's no-shadow switch: off, it changes the TEE image and nothing else."""

import unittest

import numpy as np

from rooted4d.synthetic.benchmark import generate_case
from rooted4d.synthetic.config import ObservationConfig


SMALL = dict(seed=11, observation_domain="image_likelihood", n_cmr_frames=4, n_tee_frames=4)


class TeeShadowSwitchTests(unittest.TestCase):
    def test_default_is_the_shadowed_benchmark(self):
        default = generate_case(**SMALL).arrays()
        explicit = generate_case(**SMALL, tee_shadow=True).arrays()
        self.assertEqual(sorted(default), sorted(explicit))
        for key in default:
            np.testing.assert_array_equal(default[key], explicit[key], err_msg=key)

    def test_switching_off_changes_only_the_tee_image(self):
        changed_any = False
        for seed in range(1, 6):
            settings = dict(SMALL, seed=seed)
            shadowed = generate_case(**settings).arrays()
            clear = generate_case(**settings, tee_shadow=False).arrays()
            self.assertEqual(sorted(shadowed), sorted(clear))
            changed = [key for key in shadowed if not np.array_equal(shadowed[key], clear[key])]
            self.assertIn(changed, ([], ["tee_surface_likelihood"]))
            if changed:
                changed_any = True
                # The shadow only removes signal.
                self.assertGreaterEqual(float(clear["tee_surface_likelihood"].sum()),
                                        float(shadowed["tee_surface_likelihood"].sum()))
        self.assertTrue(changed_any, "the shadow touched no TEE voxel in any seed")

    def test_config_record_unchanged_for_the_default(self):
        from rooted4d.synthetic.config import SemisyntheticConfig

        self.assertTrue(ObservationConfig().tee_shadow)
        record = SemisyntheticConfig().as_dict()["observation"]
        self.assertNotIn("tee_shadow", record)


if __name__ == "__main__":
    unittest.main()
