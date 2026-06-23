"""Ring tripod object used by the Threading task."""

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict


class RingTripodObject(CompositeObject):
    """Procedural tripod with a small ring that the needle must pass through."""

    def __init__(self, name):
        self._name = name
        self.tripod_mat_name = "lightwood_mat"
        self._important_sites = {}

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

        unit_size = [0.005, 0.002, 0.002]
        pattern = np.ones((6, 1, 6))
        for i in range(1, 5):
            pattern[i][0][1:5] = np.zeros(4)
        ring_size = [
            unit_size[0] * pattern.shape[1],
            unit_size[1] * pattern.shape[2],
            unit_size[2] * pattern.shape[0],
        ]
        self.ring_size = np.array(ring_size)

        ring_offset = [
            total_size[0] - ring_size[0],
            total_size[1] - ring_size[1],
            2.0 * (total_size[2] - ring_size[2]),
        ]

        nz, nx, ny = pattern.shape
        self.num_ring_geoms = 0
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    if pattern[k, i, j] <= 0:
                        continue
                    add_to_dict(
                        dic=obj_args,
                        geom_types="box",
                        geom_locations=(
                            (i * 2.0 * unit_size[0]) + ring_offset[0],
                            (j * 2.0 * unit_size[1]) + ring_offset[1],
                            (k * 2.0 * unit_size[2]) + ring_offset[2],
                        ),
                        geom_quats=(1, 0, 0, 0),
                        geom_sizes=tuple(unit_size),
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
            (post_size, post_size, total_size[2] - ring_size[2] - base_thickness - tripod_capsule_r - tripod_capsule_h),
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
