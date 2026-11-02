"""One fixed, low side camera for Fork; no label or scene tracking."""
import numpy as np
from scipy.spatial.transform import Rotation

SIDE_AGENTVIEW_POSITION = np.array((.65, 0., .88))
SIDE_AGENTVIEW_TARGET = np.array((.12, -.02, .94))
SIDE_AGENTVIEW_FOVY = 45.


def side_agentview_camera():
    backward = SIDE_AGENTVIEW_POSITION - SIDE_AGENTVIEW_TARGET
    backward /= np.linalg.norm(backward)
    right = np.cross((0., 0., 1.), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    xyzw = Rotation.from_matrix(np.column_stack((right, up, backward))).as_quat()
    return SIDE_AGENTVIEW_POSITION.copy(), xyzw[[3, 0, 1, 2]], SIDE_AGENTVIEW_FOVY
