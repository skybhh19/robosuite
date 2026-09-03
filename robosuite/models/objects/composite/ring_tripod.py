"""Ring tripod object used by the Threading task."""

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict


class RingTripodObject(CompositeObject):
    """Procedural tripod with a small ring that the needle must pass through."""

    def __init__(self, name, ring_outer_size=None, ring_inner_size=None):
        if (ring_outer_size is None) != (ring_inner_size is None):
            raise ValueError("ring_outer_size and ring_inner_size must be specified together")
        if ring_outer_size is not None:
            ring_outer_size = float(ring_outer_size)
            ring_inner_size = float(ring_inner_size)
            if ring_outer_size <= 0.0 or ring_inner_size <= 0.0:
                raise ValueError("Ring dimensions must be positive")
            if ring_inner_size >= ring_outer_size:
                raise ValueError("ring_inner_size must be smaller than ring_outer_size")

        self._name = name
        self.tripod_mat_name = "lightwood_mat"
        self._important_sites = {}
        self._use_legacy_ring_geometry = ring_outer_size is None
        self.ring_outer_size = 0.024 if ring_outer_size is None else ring_outer_size
        self.ring_inner_size = 0.016 if ring_inner_size is None else ring_inner_size
        self.ring_depth = 0.010
        self.aperture_half_extent = 0.5 * self.ring_inner_size

        super().__init__(**self._get_geom_attrs())

        tripod_mat = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="lightwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        self.append_material(tripod_mat)

    def _get_geom_attrs(self):
        total_size = (0.05, 0.05, 0.1)
        base_args = {
            "total_size": total_size,
            "name": self.name,
            "locations_relative_to_center": False,
            "obj_types": "all",
            "density": 100.0,
            "solref": (0.02, 1.0),
            "solimp": (0.9, 0.95, 0.001),
        }
        obj_args = {}

        legacy_unit_size = np.array([0.005, 0.002, 0.002])
        legacy_pattern = np.ones((6, 1, 6))
        for i in range(1, 5):
            legacy_pattern[i][0][1:5] = np.zeros(4)
        legacy_ring_size = legacy_unit_size * np.array(
            [legacy_pattern.shape[1], legacy_pattern.shape[2], legacy_pattern.shape[0]]
        )
        legacy_ring_offset = np.array(
            [
                total_size[0] - legacy_ring_size[0],
                total_size[1] - legacy_ring_size[1],
                2.0 * (total_size[2] - legacy_ring_size[2]),
            ]
        )

        self.num_ring_geoms = 0
        if self._use_legacy_ring_geometry:
            # Preserve the original 20-box ring exactly for all existing tasks.
            self.ring_size = legacy_ring_size
            nz, nx, ny = legacy_pattern.shape
            for k in range(nz):
                for i in range(nx):
                    for j in range(ny):
                        if legacy_pattern[k, i, j] <= 0:
                            continue
                        add_to_dict(
                            dic=obj_args,
                            geom_types="box",
                            geom_locations=tuple(
                                np.array(
                                    [
                                        i * 2.0 * legacy_unit_size[0],
                                        j * 2.0 * legacy_unit_size[1],
                                        k * 2.0 * legacy_unit_size[2],
                                    ]
                                )
                                + legacy_ring_offset
                            ),
                            geom_quats=(1, 0, 0, 0),
                            geom_sizes=tuple(legacy_unit_size),
                            geom_names=f"ring_{self.num_ring_geoms}",
                            geom_rgbas=None,
                            geom_materials=self.tripod_mat_name,
                            geom_frictions=(0.3, 5e-3, 1e-4),
                        )
                        self.num_ring_geoms += 1
        else:
            # Keep the new ring centered at the original ring center while
            # constructing exact outer and inner dimensions from four bars.
            ring_center = (
                -np.array(total_size)
                + legacy_unit_size
                + legacy_ring_offset
                + np.array(
                    [
                        0.0,
                        (legacy_pattern.shape[2] - 1) * legacy_unit_size[1],
                        (legacy_pattern.shape[0] - 1) * legacy_unit_size[2],
                    ]
                )
            )
            outer_half = 0.5 * self.ring_outer_size
            inner_half = 0.5 * self.ring_inner_size
            border_half = 0.5 * (outer_half - inner_half)
            bar_offset = inner_half + border_half
            depth_half = 0.5 * self.ring_depth
            self.ring_size = np.array([depth_half, outer_half, outer_half])
            ring_bars = (
                (ring_center + np.array([0.0, 0.0, -bar_offset]), (depth_half, outer_half, border_half)),
                (ring_center + np.array([0.0, 0.0, bar_offset]), (depth_half, outer_half, border_half)),
                (ring_center + np.array([0.0, -bar_offset, 0.0]), (depth_half, border_half, inner_half)),
                (ring_center + np.array([0.0, bar_offset, 0.0]), (depth_half, border_half, inner_half)),
            )
            for center, size in ring_bars:
                # CompositeObject expects locations from the lower corner of
                # its bounding box, rather than center-relative positions.
                location = center + np.array(total_size) - np.array(size)
                add_to_dict(
                    dic=obj_args,
                    geom_types="box",
                    geom_locations=tuple(location),
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=size,
                    geom_names=f"ring_{self.num_ring_geoms}",
                    geom_rgbas=None,
                    geom_materials=self.tripod_mat_name,
                    geom_frictions=(0.3, 5e-3, 1e-4),
                )
                self.num_ring_geoms += 1

        tripod_capsule_r = 0.01
        tripod_capsule_h = 0.03
        tripod_geom_locations = [
            (0.0, 0.0, 0.0),
            (0.0, 2.0 * total_size[1] - 2.0 * tripod_capsule_r, 0.0),
            (2.0 * total_size[0] - 2.0 * tripod_capsule_r, total_size[1] - tripod_capsule_r, 0.0),
        ]
        tripod_center = np.array([total_size[0], total_size[1], 0.0])
        xy_offset = np.array([tripod_capsule_r, tripod_capsule_r, 0.0])
        rotation_angle = -np.pi / 6.0
        tripod_geom_quats = []
        for loc in tripod_geom_locations:
            capsule_loc = np.array(loc) + xy_offset
            capsule_loc[2] = 0.0
            vec_to_center = tripod_center - capsule_loc
            vec_to_center = vec_to_center / np.linalg.norm(vec_to_center)
            rot_vec = np.cross(vec_to_center, np.array([0.0, 0.0, 1.0]))
            rot_quat = T.mat2quat(T.rotation_matrix(angle=rotation_angle, direction=rot_vec))
            tripod_geom_quats.append(T.convert_quat(rot_quat, to="wxyz"))

        for i, loc in enumerate(tripod_geom_locations):
            add_to_dict(
                dic=obj_args,
                geom_types="capsule",
                geom_locations=loc,
                geom_quats=tripod_geom_quats[i],
                geom_sizes=(tripod_capsule_r, tripod_capsule_h),
                geom_names=f"tripod_{i}",
                geom_rgbas=None,
                geom_materials=self.tripod_mat_name,
                geom_frictions=None,
            )

        base_thickness = 0.005
        post_size = 0.005
        post_geom_sizes = [
            (total_size[0], total_size[1], base_thickness),
            (
                post_size,
                post_size,
                total_size[2]
                - self.ring_size[2]
                - base_thickness
                - tripod_capsule_r
                - tripod_capsule_h,
            ),
        ]
        post_geom_locations = [
            (0.0, 0.0, 2.0 * (tripod_capsule_r + tripod_capsule_h)),
            (total_size[0] - post_size, total_size[1] - post_size, 2.0 * (tripod_capsule_r + tripod_capsule_h + base_thickness)),
        ]
        for i, loc in enumerate(post_geom_locations):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=loc,
                geom_quats=(1, 0, 0, 0),
                geom_sizes=post_geom_sizes[i],
                geom_names=f"post_{i}",
                geom_rgbas=None,
                geom_materials=self.tripod_mat_name,
                geom_frictions=None,
            )

        obj_args.update(base_args)
        return obj_args
