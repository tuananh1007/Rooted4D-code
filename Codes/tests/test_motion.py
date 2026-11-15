import unittest

import numpy as np

from rooted4d.motion import FourierMotion, MonotoneCircularPhaseWarp, circular_distance


class MotionTests(unittest.TestCase):
    def test_fourier_motion_is_periodic(self) -> None:
        rng = np.random.default_rng(3)
        base = rng.normal(size=(10, 3))
        coefficients = rng.normal(size=(6, 10, 3))
        motion = FourierMotion(base, coefficients)
        self.assertLess(motion.seam_error(), 1e-12)

    def test_phase_warp_is_strictly_monotone_unwrapped(self) -> None:
        warp = MonotoneCircularPhaseWarp(np.array([-2.0, 0.0, 1.0, -0.5]), offset=0.73)
        time = np.linspace(0.0, 1.0, 1001)
        phase = warp.evaluate_unwrapped(time)
        self.assertTrue(np.all(np.diff(phase) > 0.0))
        self.assertAlmostEqual(phase[-1] - phase[0], 1.0)

    def test_circular_distance_wraps(self) -> None:
        distance = circular_distance(np.array([0.99]), np.array([0.01]))
        self.assertAlmostEqual(float(distance[0]), 0.02)


if __name__ == "__main__":
    unittest.main()

