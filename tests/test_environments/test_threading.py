import numpy as np

import robosuite as suite
from robosuite.environments.manipulation.threading import (
    Threading,
    Threading_D0,
    Threading_D05,
    Threading_D06,
    Threading_D06_Hard,
    Threading_D07,
    Threading_D08,
    Threading_D1,
)
from robosuite.models.objects.composite.needle import (
    NEEDLE_SHAFT_HALF_LENGTH,
    SHORT_NEEDLE_SHAFT_HALF_LENGTH,
    NeedleObject,
)
from robosuite.models.objects.composite.ring_tripod import RingTripodObject


def test_threading_d05_registered():
    assert "Threading_D05" in suite.ALL_ENVIRONMENTS


def test_threading_d07_registered():
    assert "Threading_D07" in suite.ALL_ENVIRONMENTS


def test_threading_d06_registered():
    assert "Threading_D06" in suite.ALL_ENVIRONMENTS


def test_threading_d06_hard_registered():
    assert "Threading_D06_Hard" in suite.ALL_ENVIRONMENTS


def test_threading_variants_select_expected_needle_lengths():
    assert Threading_D0.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D05.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D06.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D07.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D08.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D06_Hard.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D1.needle_shaft_half_length == NEEDLE_SHAFT_HALF_LENGTH


def test_threading_d06_hard_uses_exact_smaller_ring_geometry():
    legacy = RingTripodObject(name="legacy_tripod")
    hard = RingTripodObject(
        name="hard_tripod",
        ring_outer_size=Threading_D06_Hard.tripod_ring_outer_size,
        ring_inner_size=Threading_D06_Hard.tripod_ring_inner_size,
    )

    assert legacy.num_ring_geoms == 20
    assert legacy.ring_outer_size == 0.024
    assert legacy.ring_inner_size == 0.016
    assert hard.num_ring_geoms == 4
    assert hard.ring_outer_size == 0.022
    assert hard.ring_inner_size == 0.014
    assert hard.aperture_half_extent == 0.007

    ring_geoms = [
        geom
        for geom in hard.get_obj().iter("geom")
        if "ring_" in geom.get("name", "") and geom.get("group") == "0"
    ]
    positions = np.array([np.fromstring(geom.get("pos"), sep=" ") for geom in ring_geoms])
    sizes = np.array([np.fromstring(geom.get("size"), sep=" ") for geom in ring_geoms])
    outer_low = np.min(positions - sizes, axis=0)
    outer_high = np.max(positions + sizes, axis=0)
    np.testing.assert_allclose(outer_high - outer_low, (0.010, 0.022, 0.022))

    horizontal_bars = sizes[np.isclose(sizes[:, 1], 0.011)]
    vertical_bars = sizes[np.isclose(sizes[:, 2], 0.007)]
    assert len(horizontal_bars) == 2
    assert len(vertical_bars) == 2
    np.testing.assert_allclose(horizontal_bars, np.tile((0.005, 0.011, 0.002), (2, 1)))
    np.testing.assert_allclose(vertical_bars, np.tile((0.005, 0.002, 0.007), (2, 1)))


def test_needle_object_scales_shaft_and_bounding_box_together():
    needle = NeedleObject(name="short_needle", shaft_half_length=SHORT_NEEDLE_SHAFT_HALF_LENGTH)

    assert needle.shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    np.testing.assert_allclose(needle.get_bounding_box_half_size(), (0.02, 0.074, 0.02))


def test_threading_d05_uses_d0_needle_bounds_and_modest_tripod_variation():
    env = object.__new__(Threading_D05)
    env.table_offset = np.array((0.0, 0.0, 0.8))
    bounds = env._get_initial_placement_bounds()

    assert bounds["needle"]["x"] == (-0.1, -0.0)
    assert bounds["needle"]["y"] == (0.15, 0.25)
    assert bounds["tripod"]["x"] == (-0.01, 0.01)
    assert bounds["tripod"]["y"] == (-0.16, -0.14)
    np.testing.assert_allclose(
        bounds["tripod"]["z_rot"],
        (np.pi / 2.0 - np.pi / 20.0, np.pi / 2.0 + np.pi / 20.0),
    )


def test_threading_d06_explicitly_defines_all_placement_bounds():
    d06 = object.__new__(Threading_D06)
    d06.table_offset = np.array((0.0, 0.0, 0.8))
    bounds = d06._get_initial_placement_bounds()

    assert bounds["needle"]["x"] == (-0.10, 0.00)
    assert bounds["needle"]["y"] == (0.15, 0.25)
    np.testing.assert_allclose(bounds["needle"]["z_rot"], np.deg2rad((80.0, 100.0)))
    np.testing.assert_array_equal(bounds["needle"]["reference"], d06.table_offset)
    assert bounds["tripod"]["x"] == (-0.07, 0.07)
    assert bounds["tripod"]["y"] == (-0.22, -0.12)
    np.testing.assert_allclose(bounds["tripod"]["z_rot"], np.deg2rad((75.0, 135.0)))
    np.testing.assert_array_equal(bounds["tripod"]["reference"], d06.table_offset)


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


def test_hard_aperture_uses_6p5_mm_half_extent():
    ring_mat = np.eye(3)
    needle_mat = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    common = {
        "needle_mat": needle_mat,
        "ring_pos": np.zeros(3),
        "ring_mat": ring_mat,
        "ring_normal": np.array([1.0, 0.0, 0.0]),
        "aperture_half_extent": 0.0065,
    }

    inside = Threading._aperture_intersection_metrics(
        needle_pos=np.array([0.0, 0.0064, 0.0]),
        **common,
    )
    outside = Threading._aperture_intersection_metrics(
        needle_pos=np.array([0.0, 0.0066, 0.0]),
        **common,
    )

    assert inside["clean_aperture"]
    np.testing.assert_allclose(inside["clean_aperture_margin"], 0.0001)
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
