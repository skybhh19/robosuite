"""Render reset and completed examples for the Play2Perfect task ports.

Example:
    MUJOCO_GL=glfw python robosuite/scripts/render_play2perfect_examples.py
"""

import argparse
from pathlib import Path
import sys

# Prefer this checkout when the script is executed by path and another
# robosuite version is also installed in site-packages.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw

import robosuite as suite


TASKS = ("YCBForkInRack", "MultiPartAssembly")


def _render(env, size, task):
    camera_name = "assemblyview" if task == "MultiPartAssembly" else "agentview_full"
    image = env.sim.render(height=size, width=size, camera_name=camera_name)
    return Image.fromarray(np.flipud(image))


def _goal_specs(env, task):
    if task == "YCBForkInRack":
        return ((env.fork, env.FORK_GOAL_POS, env.FORK_GOAL_QUAT),)
    return (
        (env.part2, env.PART2_GOAL_POS, env.PART2_GOAL_QUAT),
        (env.part0, env.PART0_GOAL_POS, env.PART0_GOAL_QUAT),
    )


def _place_at_goal(env, task):
    for obj, local_pos, local_quat in _goal_specs(env, task):
        goal_pos, goal_quat_xyzw = env._goal_pose(local_pos, local_quat)
        goal_quat_wxyz = goal_quat_xyzw[[3, 0, 1, 2]]
        env._set_free_object_pose(obj, goal_pos, goal_quat_wxyz)
    env.sim.forward()
    # Let contacts settle so the completed image is a physically stable state.
    for _ in range(120):
        env.sim.step()


def _label(image, text):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 34), fill=(25, 25, 25))
    draw.text((12, 9), text, fill=(255, 255, 255))
    return image


def render_examples(output_dir, robot="Panda", seed=7, size=512):
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for task in TASKS:
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
            reset_image = _render(env, size, task)
            reset_path = output_dir / f"{task}_reset.png"
            reset_image.save(reset_path)

            _place_at_goal(env, task)
            complete_image = _render(env, size, task)
            complete_path = output_dir / f"{task}_complete.png"
            complete_image.save(complete_path)

            success = bool(env._check_success())
            panels.extend(
                (
                    _label(reset_image, f"{task} - reset"),
                    _label(complete_image, f"{task} - complete (success={success})"),
                )
            )
            print(f"{task}: reset={reset_path}, complete={complete_path}, success={success}")
        finally:
            env.close()

    overview = Image.new("RGB", (2 * size, 2 * size), color=(255, 255, 255))
    for index, panel in enumerate(panels):
        overview.paste(panel, ((index % 2) * size, (index // 2) * size))
    overview_path = output_dir / "overview.png"
    overview.save(overview_path)
    print(f"overview={overview_path}")
    return overview_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/play2perfect_examples"))
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    render_examples(args.output_dir.resolve(), robot=args.robot, seed=args.seed, size=args.size)


if __name__ == "__main__":
    main()
