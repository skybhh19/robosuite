"""The published environment matches the frozen V17 dataset metadata."""
import json
from pathlib import Path


def test_fork_v17_environment_contract():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'robosuite/configs/fork_v17/environment.json').read_text())
    kwargs = config['env_kwargs']
    assert config['env_name'] == 'YCBForkInRack'
    assert kwargs['control_freq'] == 20
    assert kwargs['controller_configs']['body_parts']['right']['input_type'] == 'absolute'
    assert kwargs['fork_geometry_profile'] == 'legacy_precision_guided'
    assert kwargs['fork_pad_contact_dynamics'] == 'low_slip_stiff'
    assert kwargs['fork_wrist_camera_fovy_deg'] == 50
    assert kwargs['camera_names'] == ['agentview', 'robot0_eye_in_hand']
    assert kwargs['camera_heights'] == kwargs['camera_widths'] == 256
    assert kwargs['fork_handle_extension_m'] == kwargs['fork_mass_redistribution_kg'] == 0


def test_environment_config_is_packaged():
    root = Path(__file__).resolve().parents[1]
    assert 'recursive-include robosuite/configs/' in (root / 'MANIFEST.in').read_text()
