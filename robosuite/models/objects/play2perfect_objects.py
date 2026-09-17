"""Robot-independent object wrappers for the Play2Perfect MuJoCo assets."""

from robosuite.models.objects.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion


class _Play2PerfectXMLObject(MujocoXMLObject):
    asset_name = None
    dynamic = True
    duplicate_collision_geoms = False

    def __init__(self, name):
        joints = [dict(type="free", damping="0.0005")] if self.dynamic else None
        super().__init__(
            xml_path_completion(f"objects/play2perfect/{self.asset_name}.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=self.duplicate_collision_geoms,
        )


class Play2PerfectForkObject(_Play2PerfectXMLObject):
    asset_name = "fork"


class Play2PerfectForkRackObject(_Play2PerfectXMLObject):
    asset_name = "fork_rack"
    dynamic = False


class Play2PerfectBeamPart0Object(_Play2PerfectXMLObject):
    asset_name = "beam_part0"
    duplicate_collision_geoms = True


class Play2PerfectBeamPart2Object(_Play2PerfectXMLObject):
    asset_name = "beam_part2"
    duplicate_collision_geoms = True


class Play2PerfectBeamFixtureObject(_Play2PerfectXMLObject):
    asset_name = "beam_fixture"
    dynamic = False

