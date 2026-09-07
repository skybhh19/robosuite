import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.scripts.collect_tool_hang_wrench_joint import release_support_rotation


class ReleaseSupportTests(unittest.TestCase):
    def test_adequate_cant_no_rotation(self):
        for projection in (-1, -.8, -.5, .5, .8, 1):
            v, _ = release_support_rotation([projection, 0, np.sqrt(1-projection**2)], [1,0,0])
            np.testing.assert_array_equal(v, np.zeros(3))

    def test_shallow_cant_reaches_target_both_signs(self):
        for sign in (-1, 1):
            n = np.array([sign*.433, 0, np.sqrt(1-.433**2)])
            v, audit = release_support_rotation(n, [1,0,0])
            r = Rotation.from_rotvec(v).as_matrix()
            self.assertAlmostEqual((r@n)[0], sign*.5)
            self.assertLess(audit['rotation_deg'], 10)
            np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-14)

    def test_large_request_is_capped(self):
        v, audit = release_support_rotation([0,0,1], [1,0,0])
        self.assertAlmostEqual(np.linalg.norm(v), np.deg2rad(10))
        self.assertEqual(audit['max_rotation_deg'], 10)

    def test_rigid_pivot_preserves_hole_center(self):
        eef = np.array([.1,.2,.3]); hole = np.array([.15,.1,.4])
        original = Rotation.from_euler('xyz', [.1,.2,.3]).as_matrix()
        local_hole = original.T@(hole-eef)
        v, _ = release_support_rotation([-.433,0,.9], [1,0,0])
        r = Rotation.from_rotvec(v).as_matrix()
        new_eef = hole+r@(eef-hole)
        np.testing.assert_allclose(new_eef+(r@original)@local_hole, hole, atol=1e-14)

    def test_invalid_vectors(self):
        for normal in ([0,0,0], [np.nan,0,1], [1,2], [np.inf,0,0]):
            with self.assertRaises(ValueError):
                release_support_rotation(normal, [1,0,0])
