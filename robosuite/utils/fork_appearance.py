"""Opt-in fork styling with exactly the existing easy-grasp contact envelope.

This module intentionally has no simulator imports. The same transformation
can be applied to live task MJCF or to a saved demonstration's model XML.
"""

import xml.etree.ElementTree as ET


APPEARANCE_PROFILES = ("default", "legacy_style")
RED = "0.78 0.12 0.12 1"
STEEL = "0.83 0.85 0.87 1"
RACK = "0.86 0.88 0.91 1"
VISUAL_NAMES = ("fork_legacy_red_handle_visual", "fork_legacy_steel_neck_visual")


def apply_fork_appearance(fork_root, rack_root, profile="default"):
    """Restyle, never resize, the compiled 125 x 24 x 24 mm grasp rail.

    The two visual boxes exactly tile its collision box: red [-99,-27] mm,
    silver [-27,+26] mm. There is no invisible larger grasp geometry, new
    collision surface, change in inertia, or label-dependent styling.
    """
    if profile not in APPEARANCE_PROFILES:
        raise ValueError(f"Unknown fork appearance profile: {profile}")
    if profile == "default":
        return
    handle = fork_root.find(".//geom[@name='fork_handle_collision']")
    if handle is None:
        raise ValueError("Missing fork_handle_collision")
    size = tuple(float(v) for v in handle.get("size", "").split())
    position = tuple(float(v) for v in handle.get("pos", "").split())
    if size != (.0625, .012, .012) or position != (-.0365, 0., 0.):
        raise ValueError("legacy_style requires the unchanged easy-grasp rail")
    parent = next((body for body in fork_root.iter() if handle in list(body)), None)
    if parent is None:
        raise ValueError("Missing fork body")
    # The original contact geom remains intact; its exact visible envelope is
    # supplied by two massless non-contact boxes with the old color split.
    handle.set("rgba", "0.78 0.12 0.12 0")
    for child in list(parent):
        if child.get("name") in VISUAL_NAMES:
            parent.remove(child)
    for name, pos, size, color in (
        (VISUAL_NAMES[0], "-0.063 0 0", "0.036 0.012 0.012", RED),
        (VISUAL_NAMES[1], "-0.0005 0 0", "0.0265 0.012 0.012", STEEL),
    ):
        ET.SubElement(parent, "geom", name=name, type="box", pos=pos,
                      size=size, rgba=color, group="1", contype="0",
                      conaffinity="0", mass="0")
    # Keep the larger actual mouth, the dark rim/floor, and all internal guides.
    # Restoring the old solid rack mesh would incorrectly cover the new mouth.
    for name in ("rack_left", "rack_right", "rack_front", "rack_back"):
        wall = rack_root.find(f".//geom[@name='{name}']")
        if wall is None:
            raise ValueError(f"Missing {name}")
        wall.set("rgba", RACK)


def fork_appearance_xml(model_xml, profile="default"):
    """Return a transformed saved model; default returns the identical string."""
    if profile == "default":
        return model_xml
    root = ET.fromstring(model_xml)
    apply_fork_appearance(root, root, profile)
    return ET.tostring(root, encoding="unicode")
