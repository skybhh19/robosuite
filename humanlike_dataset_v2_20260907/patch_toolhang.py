"""Apply the v2 free-space transfer hook to a frozen ToolHang source copy."""

from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
if "HUMAN_V2_HOOK" in source:
    raise RuntimeError(f"v2 hook already present in {path}")
start = source.index("        def move_through(")
end = source.index("        def move_cartesian_delta(", start)
part = source[start:end]
needle = "            start = ("
insert = '''            # HUMAN_V2_HOOK: human-calibrated continuous free-space transfer.
            from human_motion import profile, smooth_progress
            hp = profile(self.seed, "transfer") if getattr(self, "human_enabled", False) else None
            if hp is not None:
                frames = int(np.ceil(frames * hp["duration_scale"]))
                variation_params["human_transfer_v2"] = hp
            def transfer_progress(u, start_slope, end_slope):
                progress = smooth_progress(u, hp) if hp else u
                return self._hermite_progress(progress, start_slope, end_slope)
            def human_joint_target(desired):
                if hp is None:
                    return desired
                current = np.asarray(env.sim.data.qpos[indexes], dtype=float)
                lower, upper = env.sim.model.jnt_range[indexes].T
                lead = np.clip(current + hp["residual_gain"] * (desired - current), lower, upper)
                if previous_joint_target is None:
                    return lead
                alpha = hp["command_alpha"]
                return alpha * lead + (1.0 - alpha) * previous_joint_target
'''
if needle not in part:
    raise RuntimeError("move_through start marker missing")
part = part.replace("self._hermite_progress(", "transfer_progress(")
part = part.replace(
    "if not step(desired, gripper):",
    "if not step(human_joint_target(desired), gripper):",
)
part = part.replace(needle, insert + needle, 1)
source = source[:start] + part + source[end:]

needle = '            if style != "vertical_first":\n'
insert = '''            if getattr(self, "human_enabled", False):
                from human_motion import profile
                hp_geometry = profile(self.seed, "transfer_geometry")
                shift1 = np.asarray(hp_geometry["offset_first"])
                shift2 = np.asarray(hp_geometry["offset_second"])
                shift1[2] = abs(shift1[2])
                weights = np.linspace(0.35, 1.0, len(control_holes))
                control_holes = [
                    point + hook_basis.dot(weight * shift1 + np.sin(np.pi * weight) * shift2)
                    for point, weight in zip(control_holes, weights)
                ]
                variation_params["human_transfer_geometry_v2"] = hp_geometry
'''
if source.count(needle) != 1:
    raise RuntimeError("transfer geometry marker count changed")
source = source.replace(needle, insert + needle)
path.write_text(source)
