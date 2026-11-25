"""Environments used by the 14 mm no-slip Threading v14 and v15 datasets."""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.threading import (
    Threading_D09_Harder_WristUp_GripperFriction1p5,
)


# Preserve the compiled inertial of the original 12 mm tripod when widening the ring.
_TRIPOD_INERTIAL = {
    "pos": (-0.005100504493155495, 0.0, -0.028264127399973835),
    "quat": (0.0, 0.4089928305180391, 0.0, 0.9125375962582818),
    "mass": 0.01806750383789755,
    "diaginertia": (3.84967363375098e-05, 3.847115163859766e-05, 3.225164386491392e-05),
}


class Threading_D09_Aperture14_NoSlip(Threading_D09_Harder_WristUp_GripperFriction1p5):
    """Threading with a 14 mm aperture and five MuJoCo no-slip iterations."""

    tripod_ring_inner_size = 0.014
    tripod_ring_outer_size = 0.020

    def _load_model(self):
        super()._load_model()
        root = self.model.root
        options = root.findall("option")
        if len(options) != 1:
            raise RuntimeError("Expected exactly one <option> element")
        options[0].set("noslip_iterations", "5")

        bodies = root.findall(".//body[@name='tripod_obj_root']")
        if len(bodies) != 1:
            raise RuntimeError("Expected exactly one tripod root body")
        body = bodies[0]
        for inertial in body.findall("inertial"):
            body.remove(inertial)
        attrs = {
            key: " ".join(repr(float(value)) for value in np.atleast_1d(values))
            for key, values in _TRIPOD_INERTIAL.items()
        }
        body.insert(0, ET.Element("inertial", attrs))


class Threading_D09_Aperture14_NoSlip_WristD(Threading_D09_Aperture14_NoSlip):
    """Threading v15, calibrated to look along the needle from the wrist."""

    # These hooks also set the camera pose in replay XML through the base class.
    @classmethod
    def _get_wrist_camera_pos(cls):
        return np.array([-0.03, 0.0, 0.05])

    @classmethod
    def _get_wrist_camera_quat(cls):
        return np.array(
            [0.5756666036989545, -0.41061899783827693, 0.41061899783827693, -0.5756666036989546]
        )
