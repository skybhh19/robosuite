#!/usr/bin/env python3
"""Render one episode from stored simulator states at a native resolution."""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
import tempfile

import h5py
import imageio.v2 as imageio
import numpy as np

import robosuite as suite


CAMERAS = ("agentview", "robot0_eye_in_hand")


def demo_names(data):
    return sorted(data, key=lambda name: int(name.rsplit("_", 1)[1]))


def make_env(data, height, width):
    env_args = json.loads(data.attrs["env_args"])
    kwargs = deepcopy(env_args["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        hard_reset=False,
        horizon=700,
        camera_names=list(CAMERAS),
        camera_heights=height,
        camera_widths=width,
    )
    kwargs.pop("camera_height", None)
    kwargs.pop("camera_width", None)
    kwargs.pop("camera_depths", None)
    kwargs.pop("render_gpu_device_id", None)
    return suite.make(env_args["env_name"], **kwargs)


def reset_model(env, demo):
    ep_meta = demo.attrs.get("ep_meta", "{}")
    if isinstance(ep_meta, bytes):
        ep_meta = ep_meta.decode("utf-8")
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(json.loads(ep_meta))
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(demo.attrs["model_file"]))
    env.sim.reset()


def render(source, output_dir, index, height, width, review):
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as src:
        names = demo_names(src["data"])
        if not 0 <= index < len(names):
            raise IndexError(f"episode index {index} outside [0, {len(names)})")
        name = names[index]
        demo = src["data"][name]
        output = output_dir / f"{name}.hdf5"
        if output.exists():
            print(json.dumps({"status": "exists", "output": str(output)}))
            return
        env = make_env(src["data"], height, width)
        try:
            reset_model(env, demo)
            states = demo["states"]
            images = {
                camera: np.empty((len(states), height, width, 3), dtype=np.uint8)
                for camera in CAMERAS
            }
            for step, state in enumerate(states):
                if hasattr(env, "load_phase2_reference_state"):
                    env.load_phase2_reference_state(state)
                else:
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                for camera in CAMERAS:
                    images[camera][step] = env.sim.render(
                        camera_name=camera, width=width, height=height, depth=False
                    )[::-1]
            fd, temporary_name = tempfile.mkstemp(
                prefix=output.name + ".tmp.", dir=output_dir
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                with h5py.File(temporary, "w") as dst:
                    dst.attrs["source"] = str(source)
                    dst.attrs["demo"] = name
                    dst.attrs["height"] = height
                    dst.attrs["width"] = width
                    for camera, values in images.items():
                        dst.create_dataset(
                            camera + "_image",
                            data=values,
                            compression="lzf",
                            chunks=(1, height, width, 3),
                        )
                os.replace(temporary, output)
            finally:
                if temporary.exists():
                    temporary.unlink()
            if review:
                review.mkdir(parents=True, exist_ok=True)
                picks = sorted(set((0, len(states) // 2, len(states) - 1)))
                strips = []
                for step in picks:
                    native = np.concatenate([images[c][step] for c in CAMERAS], axis=1)
                    strips.append(native)
                imageio.imwrite(review / f"{name}_native{width}.png", np.concatenate(strips, axis=0))
            print(json.dumps({"status": "rendered", "demo": name, "frames": len(states), "output": str(output)}))
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    render(args.source.resolve(), args.output_dir.resolve(), args.index, args.height, args.width, args.review)


if __name__ == "__main__":
    main()
