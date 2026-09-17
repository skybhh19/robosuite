#!/usr/bin/env python3
"""Render one fork episode at native 256px and recover online proprioception."""

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

import robosuite as suite


CAMERAS = ("agentview", "robot0_eye_in_hand")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int); parser.add_argument("--height", type=int, default=256); parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source) as source:
        names = sorted(source["data"], key=lambda x: int(x.rsplit("_", 1)[1])); name = names[args.index]
        demo = source["data"][name]; output = args.output_dir / f"{name}.hdf5"
        if output.exists(): print(json.dumps({"status": "exists", "output": str(output)})); return
        meta = json.loads(source["data"].attrs["env_args"]); kwargs = deepcopy(meta["env_kwargs"])
        kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, use_object_obs=True, hard_reset=False,
                      camera_names=list(CAMERAS), camera_heights=args.height, camera_widths=args.width)
        env = suite.make(meta["env_name"], **kwargs)
        try:
            ep_meta = demo.attrs.get("ep_meta", "{}"); ep_meta = ep_meta.decode() if isinstance(ep_meta, bytes) else ep_meta
            env.set_ep_meta(json.loads(ep_meta)); env.reset(); env.reset_from_xml_string(env.edit_model_xml(demo.attrs["model_file"])); env.sim.reset()
            states = demo["states"]; images = {c: np.empty((len(states), args.height, args.width, 3), np.uint8) for c in CAMERAS}
            joint = np.empty((len(states), 7), np.float64); gripper = np.empty((len(states), 2), np.float64)
            for step, state in enumerate(states):
                env.sim.set_state_from_flattened(state); env.sim.forward(); obs = env._get_observations(force_update=True)
                joint[step] = obs["robot0_joint_pos"]; gripper[step] = obs["robot0_gripper_qpos"]
                for camera in CAMERAS: images[camera][step] = env.sim.render(camera_name=camera, width=args.width, height=args.height, depth=False)[::-1]
            label_error = float(np.max(np.abs(joint - demo["obs/robot0_joint_pos"][:])))
            if label_error > 1e-8: raise RuntimeError(f"joint-state parity failed: {label_error}")
            fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".tmp.", dir=args.output_dir); os.close(fd); temporary = Path(temporary_name)
            try:
                with h5py.File(temporary, "w") as shard:
                    shard.attrs["demo"] = name; shard.attrs["joint_state_max_error"] = label_error
                    for camera in CAMERAS: shard.create_dataset(camera + "_image", data=images[camera], compression="lzf", chunks=(1,args.height,args.width,3))
                    shard.create_dataset("robot0_joint_pos", data=joint); shard.create_dataset("robot0_gripper_qpos", data=gripper)
                os.replace(temporary, output)
            finally:
                if temporary.exists(): temporary.unlink()
            if args.review:
                args.review.mkdir(parents=True, exist_ok=True); picks=sorted(set((0,len(states)//2,len(states)-1)))
                imageio.imwrite(args.review/f"{name}_native{args.width}.png", np.concatenate([np.concatenate([images[c][i] for c in CAMERAS],axis=1) for i in picks],axis=0))
            print(json.dumps({"status":"rendered","demo":name,"frames":len(states),"joint_state_max_error":label_error}))
        finally: env.close()


if __name__ == "__main__": main()
