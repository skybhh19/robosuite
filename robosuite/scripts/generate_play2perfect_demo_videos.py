"""Generate oracle pose-demo videos for the Play2Perfect robosuite ports.

These videos validate geometry, goal poses, and assembly order. They are not
robot-action demonstrations: object poses are interpolated directly because the
original Play2Perfect policies control a KUKA iiwa with a dexterous Sharpa hand,
while this port uses a Panda two-finger gripper.

Example:
    MUJOCO_GL=glfw python robosuite/scripts/generate_play2perfect_demo_videos.py
"""

import argparse
import json
from pathlib import Path
import sys

# Prefer this checkout when another robosuite version is installed globally.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import robosuite as suite
import robosuite.utils.transform_utils as T


TASKS = ("YCBForkInRack", "MultiPartAssembly")
DISPLAY_NAMES = {
    "YCBForkInRack": "YCB fork in rack",
    "MultiPartAssembly": "Multi-part assembly",
}


def _smoothstep(value):
    """Quintic interpolation with zero velocity and acceleration at each end."""
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _set_pose(env, obj, pos, quat_xyzw):
    env._set_free_object_pose(obj, pos, np.asarray(quat_xyzw)[[3, 0, 1, 2]])
    env.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
    env.sim.forward()


def _render(env, size, title, phase):
    camera_name = "assemblyview" if title == DISPLAY_NAMES["MultiPartAssembly"] else "agentview_full"
    pixels = env.sim.render(height=size, width=size, camera_name=camera_name)
    image = Image.fromarray(np.flipud(pixels))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, size, 58), fill=(13, 18, 25, 218))
    draw.text((12, 8), title, fill=(255, 255, 255, 255))
    draw.text((12, 31), f"Oracle pose preview | {phase}", fill=(255, 210, 80, 255))
    return np.asarray(image)


def _hold(env, frames, size, title, phase):
    frame = _render(env, size, title, phase)
    return [frame.copy() for _ in range(frames)]


def _move(env, obj, target_pos, target_quat, frames, size, title, phase):
    start_pos, start_quat = env._body_pose(obj)
    output = []
    for index in range(1, frames + 1):
        alpha = _smoothstep(index / frames)
        pos = (1.0 - alpha) * start_pos + alpha * np.asarray(target_pos)
        quat = T.quat_slerp(start_quat, np.asarray(target_quat), alpha)
        _set_pose(env, obj, pos, quat)
        output.append(_render(env, size, title, phase))
    return output


def _settle(env, frames, render_every, size, title, phase):
    output = []
    for index in range(frames * render_every):
        env.sim.step()
        if (index + 1) % render_every == 0:
            output.append(_render(env, size, title, phase))
    return output


def _insert_sequence(env, obj, goal_pos, goal_quat, frames_per_move, size, title, label):
    start_pos, start_quat = env._body_pose(obj)
    lift_pos = start_pos.copy()
    lift_pos[2] = max(start_pos[2] + 0.16, goal_pos[2] + 0.18)
    above_pos = np.asarray(goal_pos) + np.array((0.0, 0.0, 0.18))
    preinsert_pos = np.asarray(goal_pos) + np.array((0.0, 0.0, 0.055))

    frames = []
    frames += _move(
        env, obj, lift_pos, start_quat, frames_per_move, size, title, f"{label}: lift"
    )
    frames += _move(
        env,
        obj,
        above_pos,
        goal_quat,
        frames_per_move + 10,
        size,
        title,
        f"{label}: transport and align",
    )
    frames += _move(
        env,
        obj,
        preinsert_pos,
        goal_quat,
        frames_per_move,
        size,
        title,
        f"{label}: pre-insert",
    )
    frames += _move(
        env,
        obj,
        goal_pos,
        goal_quat,
        frames_per_move,
        size,
        title,
        f"{label}: insert",
    )
    return frames


def _task_frames(env, task, fps, size):
    title = DISPLAY_NAMES[task]
    frames = _hold(env, fps, size, title, "reset")

    if task == "YCBForkInRack":
        goal_pos, goal_quat = env._goal_pose(env.FORK_GOAL_POS, env.FORK_GOAL_QUAT)
        frames += _insert_sequence(
            env, env.fork, goal_pos, goal_quat, fps, size, title, "fork"
        )
    else:
        goal_pos, goal_quat = env._goal_pose(env.PART2_GOAL_POS, env.PART2_GOAL_QUAT)
        frames += _insert_sequence(
            env, env.part2, goal_pos, goal_quat, fps, size, title, "part 2"
        )
        frames += _hold(env, fps // 2, size, title, "part 2 complete")
        goal_pos, goal_quat = env._goal_pose(env.PART0_GOAL_POS, env.PART0_GOAL_QUAT)
        frames += _insert_sequence(
            env, env.part0, goal_pos, goal_quat, fps, size, title, "part 0"
        )

    frames += _settle(env, fps, 4, size, title, "complete")
    frames += _hold(env, fps, size, title, f"complete | success={bool(env._check_success())}")
    return frames


def _write_contact_sheet(frames, output_path, task):
    indices = (0, len(frames) // 2, len(frames) - 1)
    labels = ("reset", "mid-sequence", "complete")
    panels = []
    for index, label in zip(indices, labels):
        panel = Image.fromarray(frames[index]).copy()
        draw = ImageDraw.Draw(panel, "RGBA")
        draw.rectangle((0, panel.height - 30, panel.width, panel.height), fill=(0, 0, 0, 190))
        draw.text((12, panel.height - 23), label, fill=(255, 255, 255, 255))
        panels.append(panel)
    sheet = Image.new("RGB", (panels[0].width * 3, panels[0].height), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * panel.width, 0))
    sheet.save(output_path)
    print(f"{task}: contact_sheet={output_path}")


def generate(output_dir, tasks=TASKS, robot="Panda", seed=7, fps=30, size=512):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for task in tasks:
        env = suite.make(
            task,
            robots=robot,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            hard_reset=False,
            reward_shaping=True,
            seed=seed,
        )
        try:
            env.reset()
            frames = _task_frames(env, task, fps, size)
            success = bool(env._check_success())
            video_path = output_dir / f"{task}_oracle_demo.mp4"
            with imageio.get_writer(
                video_path,
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
                pixelformat="yuv420p",
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)

            sheet_path = output_dir / f"{task}_contact_sheet.png"
            _write_contact_sheet(frames, sheet_path, task)
            result = {
                "task": task,
                "robot": robot,
                "kind": "oracle_pose_preview",
                "is_robot_action_demo": False,
                "seed": seed,
                "fps": fps,
                "frame_count": len(frames),
                "duration_seconds": len(frames) / fps,
                "success": success,
                "video": str(video_path),
                "contact_sheet": str(sheet_path),
            }
            results.append(result)
            print(
                f"{task}: video={video_path}, duration={result['duration_seconds']:.1f}s, "
                f"success={success}"
            )
        finally:
            env.close()

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"metadata={metadata_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/play2perfect_demo_videos"))
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    generate(args.output_dir.resolve(), tasks, args.robot, args.seed, args.fps, args.size)


if __name__ == "__main__":
    main()
