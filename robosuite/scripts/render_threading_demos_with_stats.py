"""Render saved Threading demonstration states to mp4 videos.

Example:
    $ python robosuite/scripts/render_threading_demos.py --limit 3
    $ python robosuite/scripts/render_threading_demos.py --separate
"""

import argparse
import html
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robosuite as suite
import robosuite.macros as macros


macros.IMAGE_CONVENTION = "opencv"


def episode_dirs(dataset_dir):
    return sorted(path for path in glob(os.path.join(dataset_dir, "ep_*")) if os.path.isdir(path))


def render_episode(env, ep_dir, output_path, cameras, width, height, fps, skip_frame, separate=False, flip_vertical=True):
    xml_path = os.path.join(ep_dir, "model.xml")
    state_paths = sorted(glob(os.path.join(ep_dir, "state_*.npz")))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Missing model.xml in {ep_dir}")
    if not state_paths:
        raise FileNotFoundError(f"Missing state_*.npz in {ep_dir}")

    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writers = {}
    if separate:
        base, ext = os.path.splitext(output_path)
        for camera in cameras:
            writers[camera] = imageio.get_writer(f"{base}_{camera}{ext}", fps=fps)
    else:
        writers["side_by_side"] = imageio.get_writer(output_path, fps=fps)
    frame_count = 0
    try:
        state_index = 0
        for state_path in state_paths:
            data = np.load(state_path, allow_pickle=True)
            for state in data["states"]:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                if state_index % skip_frame == 0:
                    frames = [
                        env.sim.render(
                            camera_name=camera,
                            width=width,
                            height=height,
                            depth=False,
                        )
                        for camera in cameras
                    ]
                    if flip_vertical:
                        frames = [np.flipud(frame) for frame in frames]
                    if separate:
                        for camera, frame in zip(cameras, frames):
                            writers[camera].append_data(frame)
                    else:
                        writers["side_by_side"].append_data(np.concatenate(frames, axis=1))
                    frame_count += 1
                state_index += 1
    finally:
        for writer in writers.values():
            writer.close()
    return frame_count


def render_compilation(
    env,
    episode_paths,
    output_path,
    cameras,
    width,
    height,
    fps,
    skip_frame,
    flip_vertical=True,
    separator_frames=3,
):
    """Render all episode state sequences into one trajectory review video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    frame_count = 0
    separator = np.zeros((height, width * len(cameras), 3), dtype=np.uint8)
    try:
        for episode_index, ep_dir in enumerate(episode_paths, start=1):
            xml_path = os.path.join(ep_dir, "model.xml")
            state_paths = sorted(glob(os.path.join(ep_dir, "state_*.npz")))
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Missing model.xml in {ep_dir}")
            if not state_paths:
                raise FileNotFoundError(f"Missing state_*.npz in {ep_dir}")

            with open(xml_path, "r") as xml_file:
                env.reset_from_xml_string(xml_file.read())

            state_index = 0
            for state_path in state_paths:
                data = np.load(state_path, allow_pickle=True)
                for state in data["states"]:
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    if state_index % skip_frame == 0:
                        frames = [
                            env.sim.render(
                                camera_name=camera,
                                width=width,
                                height=height,
                                depth=False,
                            )
                            for camera in cameras
                        ]
                        if flip_vertical:
                            frames = [np.flipud(frame) for frame in frames]
                        writer.append_data(np.concatenate(frames, axis=1))
                        frame_count += 1
                    state_index += 1

            if episode_index < len(episode_paths):
                for _ in range(separator_frames):
                    writer.append_data(separator)
                    frame_count += 1
            if episode_index % 10 == 0 or episode_index == len(episode_paths):
                print(
                    f"rendered compilation episodes={episode_index}/{len(episode_paths)} frames={frame_count}",
                    flush=True,
                )
    finally:
        writer.close()
    return frame_count


def load_policy_stats(ep_dir):
    stats_path = os.path.join(ep_dir, "policy_stats.json")
    if not os.path.exists(stats_path):
        return {}
    with open(stats_path, "r") as f:
        return json.load(f)


def write_html_index(output_dir, rendered_videos):
    index_path = os.path.join(output_dir, "index.html")
    cards = []
    for video_path, frame_count, stats in rendered_videos:
        name = os.path.basename(video_path)
        rel_path = os.path.relpath(video_path, output_dir)
        # ``collection_success`` can mean "kept for analysis" when collection
        # runs use --allow-failures. Prefer the actual task-policy outcome so
        # retained failures are labeled correctly in review pages.
        success = bool(stats.get("policy_success", stats.get("collection_success", False)))
        outcome = "SUCCESS" if success else "FAILED"
        failure_reason = stats.get("failure_reason", "unknown")
        angle = stats.get("grasp_approach_angle_deg")
        target_angle = stats.get("target_grasp_approach_angle_deg")
        close_angle = stats.get("actual_close_angle_deg")
        lift_angle = stats.get("actual_lift_angle_deg")
        insert_angle = stats.get("actual_insert_angle_deg")
        lift_error = stats.get("lift_angle_error_deg")
        style = stats.get("motion_style", "unknown")
        variant = stats.get("style_variant", "unknown")
        angle_text = "visual grasp angle: n/a" if angle is None else f"visual grasp angle: {angle:.1f} deg"
        if target_angle is not None:
            angle_text += f" (target {target_angle:.1f})"
        if close_angle is not None or lift_angle is not None or insert_angle is not None:
            parts = []
            if close_angle is not None:
                parts.append(f"close {close_angle:.1f}")
            if lift_angle is not None:
                parts.append(f"lift {lift_angle:.1f}")
            if insert_angle is not None:
                parts.append(f"insert {insert_angle:.1f}")
            if lift_error is not None:
                parts.append(f"err {lift_error:+.1f}")
            angle_text += " | actual " + " / ".join(parts)
        cards.append(
            f"""
            <section class="card">
              <div class="title"><span>{html.escape(name)}</span><strong class="{'success' if success else 'failure'}">{outcome}</strong></div>
              <div class="label-row">
                <span>{html.escape(angle_text)}</span>
                <span>{html.escape(str(style))} / {html.escape(str(variant))}</span>
              </div>
              <video src="{html.escape(rel_path)}" autoplay muted loop controls playsinline></video>
              <div class="meta">{frame_count} frames · {html.escape('all checks passed' if success else failure_reason)}</div>
            </section>
            """
        )
    body = "\n".join(cards)
    with open(index_path, "w") as f:
        f.write(
            f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threading Demo Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #111;
      color: #eee;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 14px 18px;
      background: rgba(17, 17, 17, 0.92);
      border-bottom: 1px solid #333;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 600;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 14px;
      padding: 14px;
    }}
    .card {{
      border: 1px solid #333;
      background: #181818;
      border-radius: 6px;
      overflow: hidden;
    }}
    .title, .meta, .label-row {{
      padding: 8px 10px;
      font-size: 12px;
      color: #bbb;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .title {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; }}
    .title span {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; }}
    .title strong {{ flex: 0 0 auto; font-size: 10px; }}
    .success {{ color: #54c987; }}
    .failure {{ color: #ef766e; }}
    .label-row {{
      display: flex;
      gap: 12px;
      justify-content: space-between;
      color: #f0f0f0;
      background: #222;
      border-top: 1px solid #333;
      border-bottom: 1px solid #333;
    }}
    .label-row span {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    video {{
      display: block;
      width: 100%;
      background: #000;
    }}
  </style>
</head>
<body>
  <header><h1>Threading Demo Review</h1></header>
  <main>
{body}
  </main>
</body>
</html>
"""
        )
    return index_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=str(REPO_ROOT / "threading_scripted_demos"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--environment", type=str, default="Threading")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--cameras", nargs="+", type=str, default=["agentview", "robot0_eye_in_hand"])
    parser.add_argument(
        "--angles",
        nargs="+",
        type=float,
        default=None,
        help="Render only episodes whose target grasp angle matches one of these values.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--skip-frame", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--oldest-first", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--separate", action="store_true")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Render all selected episodes into one MP4 instead of one MP4 per episode.",
    )
    parser.add_argument("--combined-name", type=str, default="all_trajectories.mp4")
    parser.add_argument("--separator-frames", type=int, default=3)
    parser.add_argument(
        "--no-flip-vertical",
        action="store_true",
        help="Disable the default vertical flip used to match the reference annotation videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    output_dir = args.output_dir
    if output_dir is None:
        run_name = time.strftime("render_%Y%m%d_%H%M%S")
        output_dir = os.path.join(dataset_dir, "videos", run_name)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    eps = episode_dirs(dataset_dir)
    if args.angles is not None:
        selected_angles = {float(angle) for angle in args.angles}
        filtered_eps = []
        for ep_dir in eps:
            target_angle = load_policy_stats(ep_dir).get("target_grasp_angle_deg")
            if target_angle is not None and float(target_angle) in selected_angles:
                filtered_eps.append(ep_dir)
        eps = filtered_eps
    if not args.oldest_first:
        eps = list(reversed(eps))
    if args.limit is not None:
        eps = eps[: args.limit]
    if not eps:
        raise RuntimeError(f"No ep_* directories found in {dataset_dir}")

    controller_config = suite.load_composite_controller_config(robot=args.robots[0])
    env = suite.make(
        args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=args.cameras,
        camera_heights=[args.height] * len(args.cameras),
        camera_widths=[args.width] * len(args.cameras),
    )

    try:
        if args.combined:
            if args.separate:
                raise ValueError("--combined and --separate cannot be used together")
            if args.separator_frames < 0:
                raise ValueError("--separator-frames cannot be negative")
            output_path = os.path.join(output_dir, args.combined_name)
            frames = render_compilation(
                env=env,
                episode_paths=eps,
                output_path=output_path,
                cameras=args.cameras,
                width=args.width,
                height=args.height,
                fps=args.fps,
                skip_frame=args.skip_frame,
                flip_vertical=not args.no_flip_vertical,
                separator_frames=args.separator_frames,
            )
            print(f"rendered {frames} compilation frames -> {output_path}")
            return

        rendered_videos = []
        for ep_dir in eps:
            ep_name = os.path.basename(ep_dir)
            suffix = "_".join(args.cameras) if args.separate else "agentview_wrist"
            output_path = os.path.join(output_dir, f"{ep_name}_{suffix}.mp4")
            frames = render_episode(
                env=env,
                ep_dir=ep_dir,
                output_path=output_path,
                cameras=args.cameras,
                width=args.width,
                height=args.height,
                fps=args.fps,
                skip_frame=args.skip_frame,
                separate=args.separate,
                flip_vertical=not args.no_flip_vertical,
            )
            if not args.separate:
                rendered_videos.append((output_path, frames, load_policy_stats(ep_dir)))
            print(f"rendered {frames} frames -> {output_path}")
        if rendered_videos and not args.no_html:
            index_path = write_html_index(output_dir, rendered_videos)
            print(f"wrote review html -> {index_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
