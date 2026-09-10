import xml.etree.ElementTree as ET

import numpy as np

import robosuite as suite
from robosuite.environments.manipulation.threading import (
    Threading,
    Threading_D0,
    Threading_D05,
    Threading_D06,
    Threading_D06_Hard,
    Threading_D06_Hard_WristUp,
    Threading_D06_Harder,
    Threading_D06_Harder_WristUp,
    Threading_D06_WristUp,
    Threading_D09_Harder_WristUp,
    Threading_D09_Harder_WristUp_GripperFriction1p5,
    Threading_D09_Harder_WristUp_GripperFriction3,
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
from robosuite.scripts.collect_threading_scripted_grasp_angle import (
    INSERT_TIME_WARP_SKEW_RANGE,
    delayed_smooth_progress,
    insertion_correction_envelope,
    insertion_spatial_bump_envelope,
    insertion_time_warp,
)


def test_threading_d05_registered():
    assert "Threading_D05" in suite.ALL_ENVIRONMENTS


def test_threading_d07_registered():
    assert "Threading_D07" in suite.ALL_ENVIRONMENTS


def test_threading_d06_registered():
    assert "Threading_D06" in suite.ALL_ENVIRONMENTS


def test_threading_d06_wrist_up_registered():
    assert "Threading_D06_WristUp" in suite.ALL_ENVIRONMENTS


def test_threading_d06_hard_registered():
    assert "Threading_D06_Hard" in suite.ALL_ENVIRONMENTS


def test_threading_d06_hard_wrist_up_registered():
    assert "Threading_D06_Hard_WristUp" in suite.ALL_ENVIRONMENTS


def test_threading_d06_harder_registered():
    assert "Threading_D06_Harder" in suite.ALL_ENVIRONMENTS


def test_threading_d06_harder_wrist_up_registered():
    assert "Threading_D06_Harder_WristUp" in suite.ALL_ENVIRONMENTS


def test_threading_d09_harder_wrist_up_registered():
    assert "Threading_D09_Harder_WristUp" in suite.ALL_ENVIRONMENTS


def test_threading_d09_gripper_friction3_registered():
    assert "Threading_D09_Harder_WristUp_GripperFriction3" in suite.ALL_ENVIRONMENTS


def test_threading_d09_gripper_friction1p5_registered():
    assert "Threading_D09_Harder_WristUp_GripperFriction1p5" in suite.ALL_ENVIRONMENTS


def test_threading_variants_select_expected_needle_lengths():
    assert Threading_D0.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D05.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D06.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D07.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D08.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D06_Hard.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D06_Hard_WristUp.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D06_Harder.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D06_Harder_WristUp.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert Threading_D09_Harder_WristUp.needle_shaft_half_length == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    assert (
        Threading_D09_Harder_WristUp_GripperFriction3.needle_shaft_half_length
        == SHORT_NEEDLE_SHAFT_HALF_LENGTH
    )
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


def test_threading_d06_hard_wrist_up_combines_geometry_and_camera_changes():
    assert Threading_D06_Hard_WristUp.tripod_ring_outer_size == 0.022
    assert Threading_D06_Hard_WristUp.tripod_ring_inner_size == 0.014
    assert Threading_D06_Hard_WristUp.wrist_camera_pitch_deg == 5.0
    assert (
        Threading_D06_Hard_WristUp._get_initial_placement_bounds
        is Threading_D06._get_initial_placement_bounds
    )


def test_threading_d06_harder_uses_exact_smallest_ring_geometry():
    harder = RingTripodObject(
        name="harder_tripod",
        ring_outer_size=Threading_D06_Harder.tripod_ring_outer_size,
        ring_inner_size=Threading_D06_Harder.tripod_ring_inner_size,
    )

    assert harder.num_ring_geoms == 4
    assert harder.ring_outer_size == 0.020
    assert harder.ring_inner_size == 0.012
    assert harder.aperture_half_extent == 0.006

    ring_geoms = [
        geom
        for geom in harder.get_obj().iter("geom")
        if "ring_" in geom.get("name", "") and geom.get("group") == "0"
    ]
    positions = np.array([np.fromstring(geom.get("pos"), sep=" ") for geom in ring_geoms])
    sizes = np.array([np.fromstring(geom.get("size"), sep=" ") for geom in ring_geoms])
    outer_low = np.min(positions - sizes, axis=0)
    outer_high = np.max(positions + sizes, axis=0)
    np.testing.assert_allclose(outer_high - outer_low, (0.010, 0.020, 0.020))

    horizontal_bars = sizes[np.isclose(sizes[:, 1], 0.010)]
    vertical_bars = sizes[np.isclose(sizes[:, 2], 0.006)]
    assert len(horizontal_bars) == 2
    assert len(vertical_bars) == 2
    np.testing.assert_allclose(horizontal_bars, np.tile((0.005, 0.010, 0.002), (2, 1)))
    np.testing.assert_allclose(vertical_bars, np.tile((0.005, 0.002, 0.006), (2, 1)))


def test_threading_d06_harder_wrist_up_combines_geometry_and_camera_changes():
    assert Threading_D06_Harder_WristUp.tripod_ring_outer_size == 0.020
    assert Threading_D06_Harder_WristUp.tripod_ring_inner_size == 0.012
    assert (
        Threading_D06_Harder_WristUp.wrist_camera_pitch_deg
        == Threading_D06_Hard_WristUp.wrist_camera_pitch_deg
    )
    assert Threading_D06_Harder_WristUp._get_wrist_camera_quat is not None
    np.testing.assert_allclose(
        Threading_D06_Harder_WristUp._get_wrist_camera_quat(),
        Threading_D06_Hard_WristUp._get_wrist_camera_quat(),
    )
    assert (
        Threading_D06_Harder_WristUp._get_initial_placement_bounds
        is Threading_D06._get_initial_placement_bounds
    )


def test_threading_d09_harder_wrist_up_placement_bounds():
    env = object.__new__(Threading_D09_Harder_WristUp)
    env.table_offset = np.array((0.0, 0.0, 0.8))
    bounds = env._get_initial_placement_bounds()

    assert Threading_D09_Harder_WristUp.tripod_ring_outer_size == 0.020
    assert Threading_D09_Harder_WristUp.tripod_ring_inner_size == 0.012
    assert (
        Threading_D09_Harder_WristUp.wrist_camera_pitch_deg
        == Threading_D06_Hard_WristUp.wrist_camera_pitch_deg
    )
    np.testing.assert_allclose(
        Threading_D09_Harder_WristUp._get_wrist_camera_quat(),
        Threading_D06_Hard_WristUp._get_wrist_camera_quat(),
    )
    assert bounds["needle"]["x"] == (-0.10, 0.00)
    assert bounds["needle"]["y"] == (0.17, 0.27)
    np.testing.assert_allclose(bounds["needle"]["z_rot"], np.deg2rad((80.0, 100.0)))
    assert bounds["tripod"]["x"] == (-0.07, 0.07)
    assert bounds["tripod"]["y"] == (-0.22, -0.12)
    np.testing.assert_allclose(bounds["tripod"]["z_rot"], np.deg2rad((75.0, 115.0)))


def test_threading_d09_gripper_friction3_changes_only_gripper_contact_geoms():
    root = ET.fromstring(
        """
        <worldbody>
          <geom name="needle_obj_handle" friction="1 0.005 0.0001"/>
          <geom name="gripper0_right_finger1_collision" friction="1 0.005 0.0001"/>
          <geom name="gripper0_right_finger1_pad_collision" friction="2 0.05 0.0001"/>
          <geom name="gripper0_right_finger2_collision" friction="1 0.005 0.0001"/>
          <geom name="gripper0_right_finger2_pad_collision" friction="2 0.05 0.0001"/>
        </worldbody>
        """
    )
    changed = Threading_D09_Harder_WristUp_GripperFriction3._apply_gripper_contact_friction(root)

    assert changed == 4
    handle = root.find("./geom[@name='needle_obj_handle']")
    np.testing.assert_allclose(np.fromstring(handle.get("friction"), sep=" "), (1.0, 0.005, 0.0001))
    for geom in root.findall("./geom"):
        if geom.get("name", "").startswith("gripper0_"):
            np.testing.assert_allclose(
                np.fromstring(geom.get("friction"), sep=" "),
                Threading_D09_Harder_WristUp_GripperFriction3.gripper_contact_friction,
            )


def test_threading_d09_gripper_friction1p5_has_expected_fixed_friction():
    assert Threading_D09_Harder_WristUp_GripperFriction1p5.gripper_contact_friction == (
        1.5,
        0.0375,
        0.0001,
    )


def test_threading_d09_gripper_friction1p5_reverses_needle_yaw_only_for_variant():
    env = object.__new__(Threading_D09_Harder_WristUp_GripperFriction1p5)
    env.table_offset = np.array((0.0, 0.0, 0.8))
    bounds = env._get_initial_placement_bounds()

    np.testing.assert_allclose(bounds["needle"]["z_rot"], np.deg2rad((260.0, 280.0)))
    assert bounds["needle"]["x"] == (-0.10, 0.00)
    assert bounds["needle"]["y"] == (0.17, 0.27)
    np.testing.assert_allclose(bounds["tripod"]["z_rot"], np.deg2rad((75.0, 115.0)))


def test_threading_d09_gripper_friction1p5_flips_wrist_camera_about_gripper_axis():
    # This variant uses its own +3-degree wrist pitch before applying the
    # 180-degree symmetric camera flip.
    half_pitch = np.deg2rad(3.0) / 2.0
    inv_sqrt_two = 1.0 / np.sqrt(2.0)
    base_quat = np.array(
        (
            -inv_sqrt_two * np.sin(half_pitch),
            inv_sqrt_two * np.cos(half_pitch),
            inv_sqrt_two * np.cos(half_pitch),
            -inv_sqrt_two * np.sin(half_pitch),
        )
    )
    flipped_quat = Threading_D09_Harder_WristUp_GripperFriction1p5._get_wrist_camera_quat()

    assert Threading_D09_Harder_WristUp_GripperFriction1p5.wrist_camera_pitch_deg == 3.0
    np.testing.assert_allclose(np.linalg.norm(flipped_quat), 1.0)
    np.testing.assert_allclose(flipped_quat, (-base_quat[3], -base_quat[2], base_quat[1], base_quat[0]))
    # Unit-quaternion dot zero means the relative camera rotation is 180 degrees.
    np.testing.assert_allclose(np.dot(base_quat, flipped_quat), 0.0, atol=1e-8)


def test_threading_d09_gripper_friction1p5_moves_wrist_camera_to_opposite_side():
    base_pos = Threading_D09_Harder_WristUp._get_wrist_camera_pos()
    mirrored_pos = Threading_D09_Harder_WristUp_GripperFriction1p5._get_wrist_camera_pos()

    np.testing.assert_allclose(base_pos, (0.05, 0.0, 0.0))
    np.testing.assert_allclose(mirrored_pos, (-0.05, 0.0, 0.0))

    env = object.__new__(Threading_D09_Harder_WristUp_GripperFriction1p5)
    xml = (
        '<mujoco><asset/><worldbody>'
        '<camera name="robot0_eye_in_hand" pos="0.05 0 0" quat="0 0.707107 0.707107 0"/>'
        '<geom name="gripper0_finger1_collision"/>'
        '<geom name="gripper0_finger2_collision"/>'
        '<geom name="gripper0_finger1_pad_collision"/>'
        '<geom name="gripper0_finger2_pad_collision"/>'
        '</worldbody></mujoco>'
    )
    root = ET.fromstring(env.edit_model_xml(xml))
    camera = root.find(".//camera[@name='robot0_eye_in_hand']")

    np.testing.assert_allclose(np.fromstring(camera.get("pos"), sep=" "), mirrored_pos)


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


def test_threading_d06_wrist_up_only_changes_camera_orientation():
    assert Threading_D06_WristUp._get_initial_placement_bounds is Threading_D06._get_initial_placement_bounds

    quat = Threading_D06_WristUp._get_wrist_camera_quat()
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0)
    np.testing.assert_allclose(
        quat,
        (-0.06162842, 0.70441603, 0.70441603, -0.06162842),
        atol=1e-8,
    )


def test_threading_d06_wrist_up_persists_camera_orientation_in_replay_xml():
    env = object.__new__(Threading_D06_WristUp)
    xml = (
        '<mujoco><asset/><worldbody><camera name="robot0_eye_in_hand" '
        'quat="0 0.707107 0.707107 0"/></worldbody></mujoco>'
    )

    root = ET.fromstring(env.edit_model_xml(xml))
    camera = root.find(".//camera[@name='robot0_eye_in_hand']")
    persisted_quat = np.fromstring(camera.get("quat"), sep=" ")

    np.testing.assert_allclose(
        persisted_quat,
        Threading_D06_WristUp._get_wrist_camera_quat(),
        atol=1e-8,
    )


def test_humanized_insertion_time_warp_is_monotonic_and_endpoint_preserving():
    progress = np.linspace(0.0, 1.0, 1001)

    for skew in INSERT_TIME_WARP_SKEW_RANGE:
        warped = np.array([insertion_time_warp(value, skew) for value in progress])
        assert warped[0] == 0.0
        assert warped[-1] == 1.0
        assert np.all(np.diff(warped) > 0.0)


def test_humanized_insertion_corrections_vanish_at_endpoints_and_ring_plane():
    for progress, normal_offset in ((0.0, -0.025), (1.0, 0.045), (0.4, 0.0)):
        assert insertion_spatial_bump_envelope(progress, normal_offset) == 0.0
        assert insertion_correction_envelope(progress, normal_offset) == 0.0


def test_delayed_alignment_progress_always_finishes_at_endpoint():
    assert delayed_smooth_progress(0.0, delay=0.25, span=0.85) == 0.0
    assert delayed_smooth_progress(1.0, delay=0.25, span=0.85) == 1.0


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
