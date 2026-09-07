"""Actual nested move math with mock FK; no MuJoCo success claims."""
import ast
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest
import numpy as np


class LiftRetimeTests(unittest.TestCase):
    def execute(self, target, frames, retime=True, cartesian=False, prior=None, tolerance=None):
        source = (Path(__file__).resolve().parents[1] / "robosuite/scripts/collect_tool_hang_wrench_joint.py").read_text()
        tree = ast.parse(source)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "move")
        progress = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_hermite_progress")
        scope = {"np": np}
        exec(textwrap.dedent(ast.get_source_segment(source, progress)), scope)
        def fk(env, q):
            return np.array([q[0] + .2*q[0]**2, q[1], q[2]]), np.eye(3)
        env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(7))))
        policy = SimpleNamespace(controller_backend="joint_position", _joint_pose=fk,
                                 _hermite_progress=scope["_hermite_progress"])
        commands = []
        scope.update(self=policy, env=env, indexes=np.arange(7), previous_joint_target=np.zeros(7),
                     previous_joint_delta=np.zeros(7) if prior is None else prior.copy(),
                     variation_params={}, get_eef_pose=lambda e: fk(e, e.sim.data.qpos))
        def step(q, gripper):
            scope["previous_joint_delta"] = q - scope["previous_joint_target"]
            scope["previous_joint_target"] = q.copy()
            env.sim.data.qpos[:] = q
            commands.append(q.copy())
            return True
        scope["step"] = step
        exec(textwrap.dedent(ast.get_source_segment(source, node)), scope)
        passed = scope["move"](np.asarray(target), 1., frames, cartesian_parameterization=cartesian,
                               start_slope=.1, end_slope=.15, retime_name="lift" if retime else None,
                               retime_endpoint_tolerance_m=tolerance)
        return passed, np.asarray(commands), scope["variation_params"]

    def test_reaches_same_endpoint_with_bounds(self):
        target=np.array([.6,-.3,.2,-.4,.5,.1,-.2])
        for cartesian in (False, True):
            passed,q,params=self.execute(target,6,cartesian=cartesian)
            self.assertTrue(passed)
            np.testing.assert_array_equal(q[-1],target)
            delta=np.diff(np.vstack([np.zeros(7),q]),axis=0)
            second=np.diff(np.vstack([np.zeros(7),delta]),axis=0)
            self.assertLessEqual(np.abs(delta).max(),(.015 if cartesian else .030)+1e-12)
            self.assertLessEqual(np.linalg.norm(delta,axis=1).max(),.055+1e-12)
            self.assertLessEqual(np.linalg.norm(second,axis=1).max(),.045+1e-12)
            self.assertGreater(params['linear_retiming']['lift']['actual_frames'],6)
            self.assertGreater(params['linear_retiming']['lift']['legacy_clipped_endpoint_error_rad'],.1)

    def test_safe_original_curve_is_exactly_unchanged(self):
        target=np.array([.04,-.02,.01,-.03,.01,.02,0.])
        for cartesian in (False,True):
            _,old,_=self.execute(target,18,False,cartesian)
            passed,new,params=self.execute(target,18,True,cartesian)
            self.assertTrue(passed)
            np.testing.assert_array_equal(old,new)
            self.assertEqual(params['linear_retiming']['lift']['actual_frames'],18)

    def test_legacy_clipping_shortfall_reproduced(self):
        target=np.ones(7)*.6
        _,old,_=self.execute(target,6,False)
        self.assertGreater(np.linalg.norm(old[-1]-target),.5)
        passed,new,_=self.execute(target,6,True)
        self.assertTrue(passed)
        np.testing.assert_array_equal(new[-1],target)

    def test_infeasible_entry_fails_without_executing_or_dropping_state(self):
        passed,q,params=self.execute(np.ones(7)*.1,6,True,prior=np.ones(7)*10.)
        self.assertFalse(passed)
        self.assertEqual(len(q),0)
        self.assertFalse(params['linear_retiming']['lift']['passed'])

    def test_endpoint_gate_preserves_subtolerance_legacy_commands(self):
        target=np.ones(7)*.04
        _,old,_=self.execute(target,18,False)
        passed,new,params=self.execute(target,18,True,tolerance=.006)
        self.assertTrue(passed)
        np.testing.assert_array_equal(old,new)
        self.assertFalse(params['endpoint_retime_gate']['lift']['triggered'])
        self.assertNotIn('linear_retiming',params)

    def test_endpoint_gate_retimes_large_truncation(self):
        target=np.array([.6,-.3,.2,-.4,.5,.1,-.2])
        passed,q,params=self.execute(target,6,True,tolerance=.006)
        self.assertTrue(passed)
        self.assertTrue(params['endpoint_retime_gate']['lift']['triggered'])
        self.assertGreater(params['endpoint_retime_gate']['lift']['legacy_command_endpoint_error_m'],.006)
        np.testing.assert_array_equal(q[-1],target)

    def test_endpoint_gate_ignores_tighter_bounds_if_endpoint_already_reached(self):
        # A late nonzero entry delta exceeds the tighter second-difference
        # planner limit, but the legacy endpoint itself is not truncated.
        target=np.ones(7)*.001
        prior=np.ones(7)*.03
        _,old,_=self.execute(target,18,False,prior=prior)
        passed,new,params=self.execute(target,18,True,prior=prior,tolerance=.006)
        self.assertTrue(passed)
        np.testing.assert_array_equal(old,new)
        self.assertFalse(params['endpoint_retime_gate']['lift']['triggered'])


if __name__ == '__main__':
    unittest.main()
