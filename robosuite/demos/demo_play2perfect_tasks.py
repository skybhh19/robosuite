"""Interactively inspect the two Play2Perfect task ports with a Panda arm."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import robosuite as suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("YCBForkInRack", "MultiPartAssembly"),
        default="YCBForkInRack",
    )
    parser.add_argument("--robot", default="Panda")
    args = parser.parse_args()

    env = suite.make(
        args.task,
        robots=args.robot,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)
    try:
        while True:
            env.step(np.zeros(env.action_dim))
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
