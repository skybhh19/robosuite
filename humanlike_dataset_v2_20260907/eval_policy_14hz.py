#!/usr/bin/env python3
"""Evaluate an image BC policy with 14 Hz target updates in a 20 Hz simulator."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.scripts.run_trained_agent import rollout


class RatePolicy(RolloutPolicy):
    def __init__(self, policy, chunk_size=10, action_dim=8):
        self.policy = policy
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.step = 0
        self.last = None
        self.pending = []

    def start_episode(self):
        self.policy.start_episode()
        self.step = 0
        self.last = None
        self.pending = []

    def __call__(self, ob, goal=None):
        update = self.step == 0 or (self.step * 7) // 10 > ((self.step - 1) * 7) // 10
        if update:
            if not self.pending:
                prediction = np.asarray(self.policy(ob=ob, goal=goal), dtype=np.float32)
                expected = self.chunk_size * self.action_dim
                if prediction.size != expected:
                    raise ValueError(f"policy returned {prediction.size} values; expected {expected}")
                self.pending.extend(prediction.reshape(self.chunk_size, self.action_dim))
            self.last = self.pending.pop(0)
        self.step += 1
        return self.last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2026090800)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=True
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint, render=False, render_offscreen=True, verbose=True
    )
    policy = RatePolicy(policy)
    records = []
    with h5py.File(args.output_dir / "rollouts.hdf5", "w") as output:
        data = output.create_group("data")
        for episode in range(args.n_rollouts):
            stats, trajectory = rollout(
                policy=policy, env=env, horizon=args.horizon, render=False,
                video_writer=None, video_skip=5, return_obs=False,
                camera_names=["agentview", "robot0_eye_in_hand"],
            )
            group = data.create_group(f"demo_{episode}")
            for key in ("actions", "states", "rewards", "dones"):
                group.create_dataset(key, data=trajectory[key], compression="gzip")
            success = bool(stats["Success_Rate"])
            records.append({"episode": episode, "success": success,
                            "return": float(stats["Return"]), "horizon": int(stats["Horizon"])})
            print(json.dumps(records[-1]), flush=True)
    summary = {"agent": args.agent, "seed": args.seed, "control_hz": 14,
               "sim_hz": 20, "n_rollouts": len(records),
               "successes": sum(x["success"] for x in records),
               "success_rate": float(np.mean([x["success"] for x in records])),
               "records": records}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
