"""Opt-in legacy silhouettes with honest, compiled easy-grasp geometry.

This is a physical geometry profile, not an appearance-only transformation.
The source scan supplies the red handle and perforated holder surfaces. Both
grasp intervals use a mildly thickened rounded shank; the original steel head,
grasp friction and (without a tail extension) inertia and goal pose are left
alone. An optional physical tail extension updates both inertia and target
height in the environment. The precision profile
adds a smaller honest aperture and one shared wrist mount for both labels. The internal
guide uses a lower sliding friction so the rounded shank seats under gravity.
"""

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PROFILE = "legacy_shape_guided"
PRECISION_PROFILE = "legacy_precision_guided"
LEGACY_PROFILES = (PROFILE, PRECISION_PROFILE)
EASY_GEOMETRY_PROFILES = ("easy_grasp", "easy_grasp_guided", *LEGACY_PROFILES)
PRECISION_BORE_HALF = (.010, .011)
PRECISION_CHAMFER_M = .00075
PRECISION_WRIST_X_M = .067
PRECISION_WRIST_TILT_DEG = -2.6
HALF_WIDTH = .010
HALF_THICKNESS = .009
BEVEL_RADIUS = .003
GUIDE_SLIDE_FRICTION = .2
ASSET_ROOT = Path(__file__).resolve().parents[1] / "models/assets/objects/play2perfect"


def shared_wrist_quaternion(tilt_deg, pan_deg=0., roll_deg=0.):
    """Shared physical mount: old local-X tilt, optional local-Y pan (wxyz)."""
    tilt, pan = np.deg2rad((tilt_deg, pan_deg)) / 2
    w, x, y, z = np.array((-np.sin(tilt), np.cos(tilt), np.cos(tilt), -np.sin(tilt))) / np.sqrt(2)
    c, s = np.cos(pan), np.sin(pan)
    quat = np.array((w*c-y*s, x*c-z*s, y*c+w*s, z*c+x*s))
    if not roll_deg:
        return quat
    w, x, y, z = quat
    c, s = np.cos(np.deg2rad(roll_deg / 2)), np.sin(np.deg2rad(roll_deg / 2))
    return np.array((w*c-z*s, x*c+y*s, y*c-x*s, z*c+w*s))


def shared_wrist_angles_for_forward(forward):
    """Exact shared-mount aim; pan stays within +/-90, without numerical fitting.

    The camera looks along its negative local Z. Under the existing mount
    convention its parent-frame forward axis is
    (sin(tilt)*cos(pan), -sin(pan), cos(tilt)*cos(pan)). Optical-axis roll
    changes the image plane, not this direction. This does not change any
    camera/default/profile and does not establish image visibility.
    """
    axis = np.asarray(forward, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError('Camera forward axis must be three finite values')
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError('Camera forward axis must be nonzero')
    axis = axis / length
    pan = np.arcsin(np.clip(-axis[1], -1., 1.))
    tilt = np.arctan2(axis[0], axis[2]) if np.hypot(axis[0], axis[2]) > 1e-12 else 0.
    return tuple(np.rad2deg((tilt, pan)))


def redistribute_fork_mass(body, moved_mass_kg):
    """Opt-in shared density change, not a silhouette or total-mass change.

    Move a small internal mass from the red rear (-72 mm) toward the metal
    neck (+22 mm). Preserve the inertia about the body origin before applying
    the two point-mass changes, then recenter by the parallel-axis theorem.
    A zero setting leaves the original MJCF untouched.
    """
    if not np.isfinite(moved_mass_kg) or not 0 <= moved_mass_kg <= .015:
        raise ValueError('Redistributed fork mass must be between zero and 15 grams')
    if not moved_mass_kg:
        return
    inertial = body.find('.//inertial')
    mass = float(inertial.get('mass'))
    if moved_mass_kg >= mass / 2:
        raise ValueError('Redistribution cannot move half or more of the total fork mass')
    center = np.fromstring(inertial.get('pos'), sep=' ')
    if inertial.get('fullinertia'):
        xx, yy, zz, xy, xz, yz = np.fromstring(inertial.get('fullinertia'), sep=' ')
        tensor = np.array(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))
    else:
        from scipy.spatial.transform import Rotation
        quaternion = np.fromstring(inertial.get('quat', '1 0 0 0'), sep=' ')
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        tensor = rotation @ np.diag(np.fromstring(inertial.get('diaginertia'), sep=' ')) @ rotation.T
    rear, neck = np.array((-.072, 0., 0.)), np.array((.022, 0., 0.))

    def parallel(point):
        return np.dot(point, point) * np.eye(3) - np.outer(point, point)

    new_center = center + moved_mass_kg / mass * (neck - rear)
    new_tensor = (tensor + mass * parallel(center)
                  + moved_mass_kg * (parallel(neck) - parallel(rear))
                  - mass * parallel(new_center))
    principal = np.linalg.eigvalsh(new_tensor)
    if np.any(principal <= 0) or principal[-1] > principal[:2].sum():
        raise ValueError('Requested density redistribution produces an invalid inertia tensor')
    inertial.set('pos', ' '.join(map(str, new_center)))
    inertial.attrib.pop('quat', None)
    inertial.attrib.pop('diaginertia', None)
    inertial.set('fullinertia', ' '.join(map(str, (new_tensor[0, 0], new_tensor[1, 1],
                  new_tensor[2, 2], new_tensor[0, 1], new_tensor[0, 2], new_tensor[1, 2]))))


@lru_cache(maxsize=8)
def read_obj(path):
    vertices, faces = [], []
    with Path(path).open() as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == "f":
                indices = [int(x.split("/")[0]) for x in fields[1:]]
                indices = [i - 1 if i > 0 else len(vertices) + i for i in indices]
                faces.extend((indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1))
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def _source_mesh(asset, name):
    mesh = asset.find(f"mesh[@name='{name}']")
    if mesh is None or not mesh.get("file"):
        raise ValueError(f"Missing source scan: {name}")
    path = Path(mesh.get("file"))
    if not path.is_absolute():
        path = ASSET_ROOT / path
    return read_obj(str(path))


def _put_mesh(asset, name, vertices, faces=None):
    existing = asset.find(f"mesh[@name='{name}']")
    if existing is not None:
        asset.remove(existing)
    attrs = dict(name=name, vertex=" ".join(f"{v:.7g}" for v in np.asarray(vertices).ravel()))
    if faces is not None:
        attrs["face"] = " ".join(str(int(v)) for v in np.asarray(faces).ravel())
    return ET.SubElement(asset, "mesh", **attrs)


def rounded_cross_section(angles):
    """Ray intersection with a 20 x 18 mm rounded rectangle (3 mm corners)."""
    direction = np.c_[np.cos(angles), np.sin(angles)]
    absolute = np.abs(direction)
    half = np.array((HALF_WIDTH, HALF_THICKNESS))
    distance = np.min(half / np.maximum(absolute, 1e-12), axis=1)
    point = distance[:, None] * absolute
    corner = np.all(point > half - BEVEL_RADIUS, axis=1)
    center = half - BEVEL_RADIUS
    dot = absolute @ center
    disc = dot * dot - (center @ center - BEVEL_RADIUS**2)
    distance[corner] = dot[corner] + np.sqrt(np.maximum(disc[corner], 0))
    return distance[:, None] * direction


def handle_vertices(original, extension_m=0.0, front_extension_m=0.0):
    """Keep scan tessellation/end silhouette and gently flatten grasp sides.

    A broad rounded rectangular section is shared by both grasp intervals.
    Sub-0.06 mm scan relief is retained, rather than placing a much larger
    invisible box around the old fork. The contact geom uses this same mesh.
    """
    result = original.copy()
    x = original[:, 0]
    knots = np.linspace(x.min(), x.max(), 91)
    centers, halves = [], []
    for knot in knots:
        neighborhood = original[np.abs(x - knot) < .0016, 1:]
        lo, hi = neighborhood.min(0), neighborhood.max(0)
        centers.append((lo + hi) / 2)
        halves.append(np.maximum((hi - lo) / 2, .0003))
    centers, halves = np.asarray(centers), np.asarray(halves)
    normalized = np.empty_like(original[:, 1:])
    for axis in range(2):
        normalized[:, axis] = (original[:, axis + 1] - np.interp(x, knots, centers[:, axis])) / np.interp(x, knots, halves[:, axis])
    angles = np.arctan2(normalized[:, 1], normalized[:, 0])
    section = rounded_cross_section(angles)
    scale = np.ones(len(x))
    tip = x < -.088
    scale[tip] = np.sqrt(np.maximum(.04, 1 - ((x[tip] + .088) / .011)**2))
    front = x > -.014
    scale[front] = np.interp(x[front], (-.014, -.009573), (1., .92))
    relief = np.clip(np.linalg.norm(normalized, axis=1) - 1, -.12, .12) * .0005
    result[:, 1:] = section * scale[:, None] + relief[:, None] * np.c_[np.cos(angles), np.sin(angles)]
    # Extend only the ungrasped red tail. Keep both grasp bands, shoulder,
    # steel head and the 18 x 20 mm contact section byte-identical to v11.
    if not 0 <= extension_m <= .08:
        raise ValueError("Handle extension must be between 0 and 80 mm")
    weight = np.clip((-.060 - x) / .028, 0., 1.)
    result[:, 0] -= extension_m * weight
    if not np.isfinite(front_extension_m) or not 0 <= front_extension_m <= .025:
        raise ValueError("Front grasp extension must be between 0 and 25 mm")
    # Lengthen the red shoulder, not the steel head. The rear grasp band
    # remains unchanged; the silver transition translates onto the scan stem.
    if front_extension_m:
        result[:, 0] += front_extension_m * np.clip((x + .035) / (.035 - .009573), 0., 1.)
    return result


def neck_vertices(precision=False, front_extension_m=0.0):
    if not np.isfinite(front_extension_m) or not 0 <= front_extension_m <= .025:
        raise ValueError("Front grasp extension must be between 0 and 25 mm")
    if front_extension_m and not precision:
        raise ValueError("Front grasp extension requires the precision profile")
    section = rounded_cross_section(np.linspace(0, 2 * np.pi, 48, endpoint=False))
    vertices, faces = [], []
    uniform_end = .017 if precision else .014
    for x, scales in ((-.0105, (1, 1)), (uniform_end, (1, 1)), (.018, (.85, .27))):
        vertices.extend((x + front_extension_m, y * scales[0], z * scales[1]) for y, z in section)
    count = len(section)
    for ring in range(2):
        for i in range(count):
            j = (i + 1) % count
            a, b, c, d = ring * count + i, ring * count + j, (ring + 1) * count + j, (ring + 1) * count + i
            faces.extend(((a, b, c), (a, c, d)))
    for i in range(1, count - 1):
        faces.extend(((0, i + 1, i), (2 * count, 2 * count + i, 2 * count + i + 1)))
    return np.asarray(vertices), np.asarray(faces)


def widened_coordinate(value, original_inner, outer, precision=False, axis=0):
    magnitude = np.abs(value)
    if precision:
        inner = PRECISION_BORE_HALF[axis]
        # Preserve the scan's bevel vertices, with only a 0.75 mm lead-in.
        mapped = np.interp(magnitude, (0, original_inner, original_inner + .0029, outer),
                           (0, inner, inner + PRECISION_CHAMFER_M, outer))
        return np.sign(value) * mapped
    mapped = np.where(magnitude <= original_inner, magnitude * .014 / original_inner,
                      .014 + (magnitude - original_inner) * (outer - .014) / (outer - original_inner))
    return np.sign(value) * mapped


def rack_vertices(original, precision=False):
    result = original.copy()
    result[:, 0] = widened_coordinate(original[:, 0], .008717, .045, precision, 0)
    result[:, 1] = widened_coordinate(original[:, 1], .0103465, .035, precision, 1)
    return result


def wall_vertices(axis, sign, precision=False):
    inner, outer = ((.008717, .045), (.0103465, .035))[axis]
    top = float(widened_coordinate(inner + .0029, inner, outer, precision, axis))
    bore = PRECISION_BORE_HALF[axis] if precision else .014
    cross_outer = (.035, .045)[axis]
    vertices = []
    for z, depth in ((-.0375, bore), (.0396, bore), (.0425, top)):
        for along in (depth, outer):
            for cross in (-cross_outer, cross_outer):
                point = [0., 0., z]
                point[axis], point[1 - axis] = sign * along, cross
                vertices.append(point)
    return vertices, top


def apply_legacy_shape(fork, rack, fork_asset, rack_asset, precision=False, extension_m=0.0,
                       front_extension_m=0.0, guide_slide_friction=GUIDE_SLIDE_FRICTION):
    """Compile matching surfaces and contacts; no label-dependent geometry."""
    if not np.isfinite(guide_slide_friction) or not .1 <= guide_slide_friction <= .6:
        raise ValueError('Passive guide sliding friction must be between 0.1 and 0.6')
    if not precision and guide_slide_friction != GUIDE_SLIDE_FRICTION:
        raise ValueError('Adjusted passive guide friction requires the precision profile')
    for label in ("left", "right", "front", "back"):
        guide = rack.find(f".//geom[@name='rack_guide_{label}']")
        if guide is None:
            raise ValueError("Legacy shape requires the compiled passive guide")
        guide.set("friction", f"{guide_slide_friction} 0.001 0.00001")
    if fork.find(".//geom[@name='fork_handle_collision']").get("mesh") == "fork_legacy_handle_mesh":
        # Upgrade saved prototypes too, without repainting old states or
        # retaining a longer invisible grasp sleeve.
        vertices, faces = neck_vertices(precision, front_extension_m)
        mesh = fork_asset.find("mesh[@name='fork_legacy_neck_mesh']")
        saved = np.fromstring(mesh.get('vertex', ''), sep=' ').reshape(-1, 3)
        if saved.shape != vertices.shape or not np.allclose(saved, vertices, rtol=1e-5, atol=1e-7):
            _put_mesh(fork_asset, 'fork_legacy_neck_mesh', vertices, faces)
        if not precision:
            return
    source_handle = fork.find(".//geom[@name='fork_handle_visual']")
    source_name = source_handle.get("mesh")
    if source_name == "fork_legacy_handle_mesh":
        source_name = next(m.get('name') for m in fork_asset.findall('mesh') if m.get('file', '').endswith('e501e6f78c2a50498394.obj'))
    original, faces = _source_mesh(fork_asset, source_name)
    _put_mesh(fork_asset, "fork_legacy_handle_mesh", handle_vertices(original, extension_m, front_extension_m), faces)
    vertices, faces = neck_vertices(precision, front_extension_m)
    _put_mesh(fork_asset, "fork_legacy_neck_mesh", vertices, faces)
    for name, mesh, group, color in (
        ("fork_handle_collision", "fork_legacy_handle_mesh", "0", "0.78 0.12 0.12 1"),
        ("fork_neck_collision", "fork_legacy_neck_mesh", "1", "0.83 0.85 0.87 1"),
    ):
        geom = fork.find(f".//geom[@name='{name}']")
        geom.attrib.pop("size", None)
        geom.set("pos", "0 0 0")
        geom.set("type", "mesh")
        geom.set("mesh", mesh)
        geom.set("group", group)
        geom.set("rgba", color)
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
    source_handle.set("mesh", "fork_legacy_handle_mesh")
    source_handle.set("rgba", "0.78 0.12 0.12 1")
    for child in list(fork):
        if child.get("name", "").startswith("fork_legacy_") and child.tag == "geom":
            fork.remove(child)
    visual = rack.find(".//geom[@name='rack_visual']")
    source_name = visual.get("mesh")
    if source_name == 'rack_legacy_shell_mesh':
        source_name = next(m.get('name') for m in rack_asset.findall('mesh') if m.get('file', '').endswith('c532f465cd9ec1678d9b.obj'))
    original, faces = _source_mesh(rack_asset, source_name)
    _put_mesh(rack_asset, "rack_legacy_shell_mesh", rack_vertices(original, precision), faces)
    visual.set("mesh", "rack_legacy_shell_mesh")
    visual.set("rgba", "0.86 0.88 0.91 1")
    for label, axis, sign in (("left", 0, -1), ("right", 0, 1), ("front", 1, 1), ("back", 1, -1)):
        vertices, top = wall_vertices(axis, sign, precision)
        mesh = f"rack_legacy_wall_{label}_mesh"
        _put_mesh(rack_asset, mesh, vertices)
        geom = rack.find(f".//geom[@name='rack_{label}']")
        geom.attrib.pop("size", None)
        geom.set("pos", "0 0 0")
        geom.set("type", "mesh")
        geom.set("mesh", mesh)
        geom.set("group", "0")
        rim = rack.find(f".//geom[@name='rack_slot_rim_{label}']")
        point = [0., 0., .0435]
        point[axis] = sign * (top + .001)
        rim.set("pos", " ".join(str(v) for v in point))
        cross_half = (PRECISION_BORE_HALF[1 - axis] + PRECISION_CHAMFER_M + .002) if precision else .0175
        rim.set("size", f"0.001 {cross_half} 0.00045" if axis == 0 else f"{cross_half} 0.001 0.00045")
        guide = rack_asset.find(f"mesh[@name='rack_guide_{label}_mesh']")
        if guide is None:
            raise ValueError("Legacy shape requires the compiled passive guide")
        sleeve = (.0092, .0102)[axis]
        vertices = []
        bore = PRECISION_BORE_HALF[axis] if precision else .014
        cross_half = PRECISION_BORE_HALF[1 - axis] if precision else .014
        for z, depths in ((-.0375, (bore, sleeve)), (-.0125, (bore, sleeve)), (-.0025, (bore,))):
            for depth in depths:
                for cross in (-cross_half, cross_half):
                    point = [0., 0., z]
                    point[axis], point[1 - axis] = sign * depth, cross
                    vertices.append(point)
        guide.set("vertex", " ".join(str(v) for p in vertices for v in p))


def legacy_shape_xml(xml, precision=False):
    root = ET.fromstring(xml)
    asset = root.find("asset")
    apply_legacy_shape(root, root, asset, asset, precision)
    return ET.tostring(root, encoding="unicode")


def verify_legacy_shape_xml(xml, precision=False, wrist_tilt_deg=None, wrist_x_m=None,
                            extension_m=0.0, wrist_z_m=0.0, wrist_y_m=0.0,
                            wrist_pan_deg=0.0,
                            continuous_grasp=False, wrist_roll_deg=0., front_extension_m=0.,
                            guide_slide_friction=GUIDE_SLIDE_FRICTION, wrist_fovy_deg=40.):
    """Reject a profile tag paired with old geometry or camera during export."""
    root = ET.fromstring(xml)
    asset = root.find("asset")
    if precision:
        if not np.isfinite(wrist_fovy_deg) or not 25 <= wrist_fovy_deg <= 90:
            raise ValueError('Declared precision wrist FOV must be finite and within25..90degrees')
        camera = root.find(".//camera[@name='robot0_eye_in_hand']")
        if camera is not None:
            tilt = PRECISION_WRIST_TILT_DEG if wrist_tilt_deg is None else wrist_tilt_deg
            expected_quat = shared_wrist_quaternion(tilt, wrist_pan_deg, wrist_roll_deg)
            camera_x = PRECISION_WRIST_X_M if wrist_x_m is None else wrist_x_m
            if (not np.allclose(np.fromstring(camera.get('pos', ''), sep=' '), (camera_x, wrist_y_m, wrist_z_m), atol=1e-7) or
                    not np.allclose(np.fromstring(camera.get('quat', ''), sep=' '), expected_quat, atol=1e-6) or
                    float(camera.get('fovy', 0)) not in (wrist_fovy_deg, float(f'{wrist_fovy_deg:.6g}'))):
                raise RuntimeError('Precision wrist mount does not match the shared validated camera')
    for geom_name, mesh_name in (("fork_handle_collision", "fork_legacy_handle_mesh"),
                                 ("fork_neck_collision", "fork_legacy_neck_mesh")):
        if continuous_grasp and geom_name == 'fork_handle_collision':
            mesh_name = 'fork_continuous_grasp_mesh'
        geom = root.find(f".//geom[@name='{geom_name}']")
        if geom is None or geom.get("type") != "mesh" or geom.get("mesh") != mesh_name:
            raise RuntimeError(f"Saved model does not match legacy shape: {geom_name}")
        expected_contact = '0' if continuous_grasp and geom_name == 'fork_neck_collision' else '1'
        if geom.get("contype", "1") != expected_contact or geom.get("conaffinity", "1") != expected_contact:
            raise RuntimeError(f"Missing legacy grasp contacts: {geom_name}")
    if root.find(".//geom[@name='fork_handle_visual']").get("mesh") != "fork_legacy_handle_mesh":
        raise RuntimeError("Legacy handle visible and contact meshes must match")
    handle = np.fromstring(asset.find("mesh[@name='fork_legacy_handle_mesh']").get("vertex", ""), sep=" ").reshape(-1, 3)
    expected_handle = handle_vertices(read_obj(str(ASSET_ROOT / 'meshes/e501e6f78c2a50498394.obj'))[0], extension_m, front_extension_m)
    if handle.shape != expected_handle.shape or not np.allclose(handle, expected_handle, rtol=1e-5, atol=1e-7):
        raise RuntimeError("Legacy handle differs from the declared extension")
    if continuous_grasp:
        rail = np.fromstring(asset.find("mesh[@name='fork_continuous_grasp_mesh']").get('vertex'), sep=' ').reshape(-1, 3)
        expected_rail = np.r_[expected_handle, neck_vertices(precision, front_extension_m)[0]]
        if rail.shape != expected_rail.shape or not np.allclose(rail, expected_rail, rtol=1e-5, atol=1e-7):
            raise RuntimeError("Continuous rail does not match the visible shank envelope")
    if root.find(".//geom[@name='rack_visual']").get("mesh") != "rack_legacy_shell_mesh":
        raise RuntimeError("Missing perforated legacy holder")
    shell = asset.find("mesh[@name='rack_legacy_shell_mesh']")
    actual_shell = np.fromstring(shell.get('vertex', ''), sep=' ').reshape(-1, 3)
    source_shell = read_obj(str(ASSET_ROOT / 'meshes/c532f465cd9ec1678d9b.obj'))[0]
    expected_shell = rack_vertices(source_shell, precision)
    if actual_shell.shape != expected_shell.shape or not np.allclose(actual_shell, expected_shell, rtol=1e-5, atol=1e-7):
        raise RuntimeError('Legacy visible holder aperture differs from its physical profile')
    for name, expected in (("fork_legacy_neck_mesh", neck_vertices(precision, front_extension_m)[0]),):
        actual = np.fromstring(asset.find(f"mesh[@name='{name}']").get("vertex", ""), sep=" ").reshape(-1, 3)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"Legacy geometry dimensions changed: {name}")
    for label, axis, sign in (("left", 0, -1), ("right", 0, 1), ("front", 1, 1), ("back", 1, -1)):
        geom = root.find(f".//geom[@name='rack_{label}']")
        name = f"rack_legacy_wall_{label}_mesh"
        if geom is None or geom.get("mesh") != name:
            raise RuntimeError("Missing enlarged physical legacy holder wall")
        actual = np.fromstring(asset.find(f"mesh[@name='{name}']").get("vertex", ""), sep=" ").reshape(-1, 3)
        if not np.allclose(actual, wall_vertices(axis, sign, precision)[0], rtol=1e-5, atol=1e-7):
            raise RuntimeError("Legacy holder contact envelope changed")
        mesh = asset.find(f"mesh[@name='rack_guide_{label}_mesh']")
        if mesh is None:
            raise RuntimeError("Missing legacy passive guide")
        vertices = np.fromstring(mesh.get("vertex", ""), sep=" ").reshape(-1, 3)
        if vertices.shape != (10, 3) or not np.isclose(np.min(np.abs(vertices[:, axis])), (.0092, .0102)[axis]):
            raise RuntimeError("Legacy guide does not match the thinner fork")
        if not np.isclose(np.max(np.abs(vertices[:, axis])), PRECISION_BORE_HALF[axis] if precision else .014):
            raise RuntimeError("Legacy guide entrance does not match the bore")
        guide = root.find(f".//geom[@name='rack_guide_{label}']")
        if guide is None or not np.allclose(np.fromstring(guide.get('friction', ''), sep=' '), (guide_slide_friction, .001, .00001)):
            raise RuntimeError("Legacy passive guide friction differs from the tested profile")


def wrist_holder_pixel_count(model, state, size=256):
    """Read-only rendering of one stored state, using the actual RGB groups."""
    import mujoco
    if len(state) != 1 + model.nq + model.nv or model.na:
        raise ValueError("Unexpected native state layout")
    data = mujoco.MjData(model)
    data.time, data.qpos[:], data.qvel[:] = state[0], state[1:1 + model.nq], state[1 + model.nq:]
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[0], option.geomgroup[1] = 0, 1
    rack_ids = [i for i in range(model.ngeom) if str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)).startswith("rack_")]
    with mujoco.Renderer(model, height=size, width=size) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="robot0_eye_in_hand", scene_option=option)
        segmentation = renderer.render()
    return int(np.sum(np.isin(segmentation[:, :, 0], rack_ids) & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)))


def wrist_pixel_contract(label, count, size=256):
    if label not in ('full', 'partial') or size <= 0:
        raise ValueError("Invalid wrist pixel audit label or size")
    return count == 0 if label == 'partial' else count >= 750 * (size / 128)**2
