from robosuite_model_zoo.utils.mjcf_obj import MJCFObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion


class FishObject(MJCFObject):
    """
    Blender object (used in BlenderPlay)
    """

    def __init__(self, name):
        super().__init__(
            mjcf_path=xml_path_completion("objects/fish/model.xml"),
            name=name,
            scale=0.8,
        )

class CarrotObject(MJCFObject):
    """
    Blender object (used in BlenderPlay)
    """

    def __init__(self, name):
        super().__init__(
            mjcf_path=xml_path_completion("objects/carrot/model.xml"),
            name=name,
            scale=[0.9, 0.9, 0.8],
        )