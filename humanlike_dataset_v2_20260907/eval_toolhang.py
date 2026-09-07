#!/usr/bin/env python3
"""Render separate diagnostic videos and save trajectories for a trained policy."""

import argparse
import json
import types
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.run_trained_agent import rollout


def install_top_mode_policy(rollout_policy):
    """Replace categorical GMM sampling with the mean of its most likely mode."""

    def get_top_mode_action(algo, obs_dict, goal_dict=None):
        distribution = algo.nets["policy"].forward_train(
            obs_dict, goal_dict=goal_dict
        )
        wrapped = hasattr(distribution, "base_dist")
        base = distribution.base_dist if wrapped else distribution
        logits = base.mixture_distribution.logits
        component = base.component_distribution
        normal = component.base_dist if hasattr(component, "base_dist") else component
        means = normal.loc
        mode = logits.argmax(dim=-1)
        gather_index = mode[..., None, None].expand(*mode.shape, 1, means.shape[-1])
        action = means.gather(dim=-2, index=gather_index).squeeze(-2)
        if wrapped:
            action = torch.tanh(action) * distribution.scale
        return action

    rollout_policy.policy.get_action = types.MethodType(
        get_top_mode_action, rollout_policy.policy
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--video-skip", type=int, default=2)
    parser.add_argument("--top-mode", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=True
    )
    if args.top_mode:
        install_top_mode_policy(policy)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        render=False,
        render_offscreen=True,
        verbose=True,
    )

    records = []
    hdf_path = args.output_dir / "rollouts.hdf5"
    with h5py.File(hdf_path, "w") as output:
        data = output.create_group("data")
        total = 0
        for episode in range(args.n_rollouts):
            video_path = videos_dir / f"episode_{episode:03d}.mp4"
            if args.skip_video:
                stats, traj = rollout(
                    policy=policy,
                    env=env,
                    horizon=args.horizon,
                    render=False,
                    video_writer=None,
                    video_skip=args.video_skip,
                    return_obs=False,
                    camera_names=["agentview", "robot0_eye_in_hand"],
                )
            else:
                with imageio.get_writer(video_path, fps=20 // args.video_skip) as writer:
                    stats, traj = rollout(
                        policy=policy,
                        env=env,
                        horizon=args.horizon,
                        render=False,
                        video_writer=writer,
                        video_skip=args.video_skip,
                        return_obs=False,
                        camera_names=["agentview", "robot0_eye_in_hand"],
                    )

            group = data.create_group(f"demo_{episode}")
            group.create_dataset("actions", data=traj["actions"], compression="gzip")
            group.create_dataset("states", data=traj["states"], compression="gzip")
            group.create_dataset("rewards", data=traj["rewards"], compression="gzip")
            group.create_dataset("dones", data=traj["dones"], compression="gzip")
            group.attrs["num_samples"] = len(traj["actions"])
            total += len(traj["actions"])

            states = traj["states"]
            tool_z = states[:, 23 + 2]
            frame_xyz = states[:, 16:19]
            tool_xyz = states[:, 23:26]
            record = {
                "episode": episode,
                "success": bool(stats["Success_Rate"]),
                "horizon": int(stats["Horizon"]),
                "return": float(stats["Return"]),
                "video": None if args.skip_video else str(video_path.relative_to(args.output_dir)),
                "tool_lift_m": float(tool_z.max() - tool_z[0]),
                "min_tool_frame_body_distance_m": float(
                    np.linalg.norm(tool_xyz - frame_xyz, axis=1).min()
                ),
            }
            records.append(record)
            print(json.dumps(record), flush=True)

        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps(env.serialize(), sort_keys=True)

    summary = {
        "agent": args.agent,
        "seed": args.seed,
        "top_mode": args.top_mode,
        "n_rollouts": args.n_rollouts,
        "successes": sum(item["success"] for item in records),
        "success_rate": sum(item["success"] for item in records) / len(records),
        "records": records,
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
