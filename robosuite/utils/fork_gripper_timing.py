"""Opt-in Panda command slew per second, shared across planning and execution."""
from types import MethodType
import numpy as np


def validate_gripper_position_gain(gain):
    gain = float(gain)
    if not np.isfinite(gain) or gain != 0. and not 1000. <= gain <= 2000.:
        raise ValueError('Panda position gain must be default0 or within1000..2000 N/m')
    return gain


def configure_gripper_position_gain(model, gain):
    """Change two position gains only; preserve force caps and all arm actuators."""
    gain = validate_gripper_position_gain(gain)
    if not gain:
        return
    ids = [i for i, name in enumerate(model.actuator_names)
           if name and ('gripper' in name or 'finger' in name)]
    m = getattr(model, '_model', model)
    if len(ids) != 2 or any(m.actuator_gainprm[i, 0] <= 0 or
            not np.isclose(m.actuator_gainprm[i, 0], -m.actuator_biasprm[i, 1]) for i in ids):
        raise ValueError('Require two existing Panda position actuators')
    for i in ids:
        m.actuator_gainprm[i, 0] = gain
        m.actuator_biasprm[i, 1] = -gain


def _format_time_scaled_action(gripper, action):
    action = np.asarray(action)
    if action.shape != (1,) or not np.isfinite(action).all():
        raise ValueError('Expected one finite Panda gripper command')
    increment = gripper._fork_command_rate_per_s / gripper._fork_control_hz
    gripper.current_action = np.clip(gripper.current_action +
        np.array([-1., 1.]) * increment * np.sign(action), -1., 1.)
    return gripper.current_action


def configure_gripper_timing(gripper, control_hz, command_rate_per_s):
    from robosuite.models.grippers.panda_gripper import PandaGripper
    if command_rate_per_s == 0:
        return
    if (not isinstance(gripper, PandaGripper) or not np.isfinite([control_hz, command_rate_per_s]).all()
            or control_hz <= 0 or not 2.8 <= command_rate_per_s <= 12.):
        raise ValueError('Require Panda and bounded 2.8..12 command units/second')
    gripper._fork_command_rate_per_s = float(command_rate_per_s)
    gripper._fork_control_hz = float(control_hz)
    # Instance-local: other Panda tasks and the default Fork stay unchanged.
    gripper.format_action = MethodType(_format_time_scaled_action, gripper)
