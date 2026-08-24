"""Needle object used by the Threading task."""

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict


NEEDLE_SHAFT_HALF_SIZE = (0.005, 0.054, 0.005)
NEEDLE_SHAFT_HALF_LENGTH = NEEDLE_SHAFT_HALF_SIZE[1]
NEEDLE_HANDLE_HALF_SIZE = (0.02, 0.02, 0.02)
NEEDLE_TOTAL_HALF_SIZE = (
    max(NEEDLE_SHAFT_HALF_SIZE[0], NEEDLE_HANDLE_HALF_SIZE[0]),
    NEEDLE_SHAFT_HALF_LENGTH + NEEDLE_HANDLE_HALF_SIZE[1],
    max(NEEDLE_SHAFT_HALF_SIZE[2], NEEDLE_HANDLE_HALF_SIZE[2]),
)


class NeedleObject(CompositeObject):
    """Procedural needle with a graspable handle."""

    shaft_half_length = NEEDLE_SHAFT_HALF_LENGTH

    def __init__(self, name):
        self._name = name
        self.needle_mat_name = "darkwood_mat"
        self._important_sites = {}

        super().__init__(**self._get_geom_attrs())

        needle_mat = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="darkwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        self.append_material(needle_mat)

    def _get_geom_attrs(self):
        base_args = {
            "total_size": NEEDLE_TOTAL_HALF_SIZE,
            "name": self.name,
            "locations_relative_to_center": False,
            "obj_types": "all",
            "density": 100.0,
        }
        obj_args = {}

        needle_size = NEEDLE_SHAFT_HALF_SIZE
        handle_size = NEEDLE_HANDLE_HALF_SIZE

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=((handle_size[0] - needle_size[0]), 0.0, (handle_size[2] - needle_size[2])),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=tuple(needle_size),
            geom_names="needle",
            geom_rgbas=None,
            geom_materials=self.needle_mat_name,
            geom_frictions=(0.3, 5e-3, 1e-4),
        )

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0.0, 2.0 * needle_size[1], 0.0),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=tuple(handle_size),
            geom_names="handle",
            geom_rgbas=None,
            geom_materials=self.needle_mat_name,
            geom_frictions=None,
        )

        obj_args.update(base_args)
        return obj_args
