import numpy as np

import robosuite as suite
from robosuite.environments.manipulation.threading import Threading, Threading_D05


def test_threading_d05_registered():
    assert "Threading_D05" in suite.ALL_ENVIRONMENTS


def test_threading_d05_placement_bounds_are_between_d0_and_d1():
    env = object.__new__(Threading_D05)
    env.table_offset = np.array((0.0, 0.0, 0.8))
    bounds = env._get_initial_placement_bounds()

    assert bounds["needle"]["x"] == (-0.2, -0.05)
    assert bounds["needle"]["y"] == (0.15, 0.25)
    assert bounds["tripod"]["x"] == (-0.02, 0.02)
    assert bounds["tripod"]["y"] == (-0.17, -0.13)
    np.testing.assert_allclose(bounds["tripod"]["z_rot"], (np.pi / 3.0, 2.0 * np.pi / 3.0))


def test_aperture_intersection_accepts_inside_and_rejects_outside():
    ring_pos = np.zeros(3)
    ring_mat = np.eye(3)
    ring_normal = np.array([1.0, 0.0, 0.0])
    # Needle local y is aligned with the ring normal.
    needle_mat = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    inside = Threading._aperture_intersection_metrics(
        needle_pos=np.array([0.0, 0.004, 0.0]),
        needle_mat=needle_mat,
        ring_pos=ring_pos,
        ring_mat=ring_mat,
        ring_normal=ring_normal,
    )
    outside = Threading._aperture_intersection_metrics(
        needle_pos=np.array([0.0, 0.009, 0.0]),
        needle_mat=needle_mat,
        ring_pos=ring_pos,
        ring_mat=ring_mat,
        ring_normal=ring_normal,
    )

    assert inside["finite_shaft_crosses_ring_plane"]
    assert inside["clean_aperture"]
    assert inside["clean_aperture_margin"] == 0.004
    assert outside["finite_shaft_crosses_ring_plane"]
    assert not outside["clean_aperture"]


def test_aperture_intersection_rejects_infinite_line_only_crossing():
    ring_mat = np.eye(3)
    needle_mat = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = Threading._aperture_intersection_metrics(
        needle_pos=np.array([0.07, 0.0, 0.0]),
        needle_mat=needle_mat,
        ring_pos=np.zeros(3),
        ring_mat=ring_mat,
        ring_normal=np.array([1.0, 0.0, 0.0]),
    )

    assert not result["finite_shaft_crosses_ring_plane"]
    assert not result["clean_aperture"]
