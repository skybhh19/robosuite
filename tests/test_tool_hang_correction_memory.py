import unittest
import numpy as np
from robosuite.scripts.collect_tool_hang_wrench_joint import bounded_correction_goal, correction_memory_required, GeometricJointPolicy


class CorrectionMemoryTests(unittest.TestCase):
    def test_seated_gate_uses_existing_geometry(self):
        good = dict(hole_frame_contact=True, hole_straddles_hook=True,
                    line_distance_m=.005, normalized_insertion=.07324)
        self.assertFalse(correction_memory_required(good, True))
        for key,value in [('hole_frame_contact',False),('hole_straddles_hook',False),
                          ('line_distance_m',.005001),('normalized_insertion',.065),
                          ('normalized_insertion',1.)]:
            self.assertTrue(correction_memory_required(dict(good,**{key:value}), True))
        self.assertTrue(correction_memory_required(dict(good,normalized_insertion=.05518), True))
        self.assertFalse(correction_memory_required(dict(good,normalized_insertion=.05518), False))

    def test_gate_requires_memory_option(self):
        with self.assertRaises(ValueError):
            GeometricJointPolicy(controller_backend='joint_position', line_correction_memory_unseated_only=True)
        policy=GeometricJointPolicy(controller_backend='joint_position', line_correction_memory=True,
                                    line_correction_memory_unseated_only=True)
        self.assertTrue(policy.line_correction_memory_unseated_only)

    def test_first_target_is_original(self):
        actual = np.array([.2, -.1, 1.])
        delta = np.array([.001, .002, -.003])
        target, audit = bounded_correction_goal(actual, None, delta)
        np.testing.assert_array_equal(target, actual + delta)
        self.assertFalse(audit['lead_limited'])

    def test_stalled_actual_retains_bounded_goal(self):
        actual = np.zeros(3); target = None
        for expected in (.004, .008, .008, .008):
            target, _ = bounded_correction_goal(actual, target, [0, 0, -.004])
            self.assertAlmostEqual(target[2], -expected)

    def test_tracking_and_direction_reversal_unwind(self):
        target, _ = bounded_correction_goal([0,0,0], [0,0,-.008], [0,0,.004])
        np.testing.assert_allclose(target, [0,0,-.004])
        target, _ = bounded_correction_goal(target, target, [0,0,.004])
        np.testing.assert_allclose(target, [0,0,0])

    def test_lead_is_vector_bound_not_component_bound(self):
        target, audit = bounded_correction_goal([0,0,0], [.008,0,0], [0,.004,0])
        self.assertAlmostEqual(np.linalg.norm(target), .008)
        self.assertTrue(audit['lead_limited'])
        np.testing.assert_allclose(target / np.linalg.norm(target), np.array([2,1,0])/np.sqrt(5))

    def test_no_input_mutation(self):
        inputs = [np.array([0.,0.,0.]), np.array([0.,.008,0.]), np.array([0.,.004,0.])]
        copies = [x.copy() for x in inputs]
        bounded_correction_goal(*inputs)
        for a,b in zip(inputs,copies): np.testing.assert_array_equal(a,b)

    def test_invalid_inputs_and_backend(self):
        for x in ([np.nan,0,0], [0,0], [np.inf,0,0]):
            with self.assertRaises(ValueError): bounded_correction_goal(x,None,[0,0,0])
        for bound in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError): bounded_correction_goal([0,0,0],None,[0,0,0],bound)
        with self.assertRaises(ValueError): GeometricJointPolicy(line_correction_memory=True)
        self.assertFalse(GeometricJointPolicy().line_correction_memory)


if __name__ == '__main__': unittest.main()
