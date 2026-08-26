#!/usr/bin/env python3
"""Evaluate a ToolHang robomimic policy with canonical single-reset semantics.

Unlike ``robomimic.scripts.run_trained_agent``, this evaluator never calls
``reset_to`` after ``env.reset``. ToolHang OSC demonstrations define frame zero
inside ``ToolHangWrenchOnly.reset`` after controller settling and arm-state
restoration, so a second state reset invalidates the controller state.
"""

import argparse
import json
import types
from copy import deepcopy
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch


def rollout_single_reset(policy, env, horizon, video_writer, video_skip, camera_names):
    """Run one episode after exactly one call to ``env.reset``."""

    policy.start_episode()
    observation = env.reset()
    states, actions, rewards, dones = [], [], [], []
    success = False
    total_reward = 0.0

    for step_index in range(horizon):
        states.append(env.get_state()["states"])
        action = policy(ob=observation)
        next_observation, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        total_reward += reward
        success = success or bool(env.is_success()["task"])

        if video_writer is not None and step_index % video_skip == 0:
            frames = [env.render(mode="rgb_array", height=512, width=512, camera_name=name) for name in camera_names]
            video_writer.append_data(np.concatenate(frames, axis=1))

        if done or success:
            break
        observation = deepcopy(next_observation)

    trajectory = {
        "actions": np.asarray(actions),
        "states": np.asarray(states),
        "rewards": np.asarray(rewards),
        "dones": np.asarray(dones),
    }
    statistics = {
        "return": float(total_reward),
        "horizon": len(actions),
        "success": bool(success),
    }
    return statistics, trajectory


def install_top_mode_policy(rollout_policy):
    """Use the mean of the most likely GMM component instead of sampling."""

    def get_top_mode_action(algo, obs_dict, goal_dict=None):
        distribution = algo.nets["policy"].forward_train(obs_dict, goal_dict=goal_dict)
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

    rollout_policy.policy.get_action = types.MethodType(get_top_mode_action, rollout_policy.policy)


def validate_environment(env, expected_settle_steps):
    """Reject a checkpoint environment with non-canonical ToolHang reset settings."""

    raw_env = getattr(env, "env", None)
    if raw_env is None or raw_env.__class__.__name__ != "ToolHangWrenchOnly":
        raise RuntimeError("Expected a robomimic ToolHangWrenchOnly environment, got " f"{type(raw_env).__name__}")
    actual = int(raw_env.RESET_CONTROLLER_SETTLE_STEPS)
    if actual != expected_settle_steps:
        raise RuntimeError(
            f"ToolHang reset mismatch: expected {expected_settle_steps} settle steps, " f"found {actual}"
        )


def save_trajectory(group, trajectory):
    for key in ("actions", "states", "rewards", "dones"):
        group.create_dataset(key, data=trajectory[key], compression="gzip")
    group.attrs["num_samples"] = len(trajectory["actions"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="robomimic checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--seed", type=int, default=2026082601)
    parser.add_argument("--video-skip", type=int, default=4)
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=["agentview", "robot0_eye_in_hand"],
    )
    parser.add_argument("--top-mode", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--expected-reset-settle-steps", type=int, default=10)
    args = parser.parse_args()

    if args.n_rollouts <= 0 or args.horizon <= 0 or args.video_skip <= 0:
        parser.error("n-rollouts, horizon, and video-skip must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"
    if not args.skip_video:
        videos_dir.mkdir(exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=True)
    if args.top_mode:
        install_top_mode_policy(policy)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        render=False,
        render_offscreen=True,
        verbose=True,
    )
    validate_environment(env, args.expected_reset_settle_steps)

    records = []
    with h5py.File(args.output_dir / "rollouts.hdf5", "w") as output:
        data = output.create_group("data")
        total = 0
        for episode in range(args.n_rollouts):
            video_path = videos_dir / f"episode_{episode:03d}.mp4"
            if args.skip_video:
                statistics, trajectory = rollout_single_reset(
                    policy, env, args.horizon, None, args.video_skip, args.camera_names
                )
            else:
                with imageio.get_writer(video_path, fps=max(1, 20 // args.video_skip)) as writer:
                    statistics, trajectory = rollout_single_reset(
                        policy,
                        env,
                        args.horizon,
                        writer,
                        args.video_skip,
                        args.camera_names,
                    )

            group = data.create_group(f"demo_{episode}")
            save_trajectory(group, trajectory)
            total += statistics["horizon"]
            record = {
                "episode": episode,
                **statistics,
                "video": None if args.skip_video else str(video_path.relative_to(args.output_dir)),
            }
            records.append(record)
            print(json.dumps(record), flush=True)

        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps(env.serialize(), sort_keys=True)

    successes = sum(record["success"] for record in records)
    summary = {
        "agent": args.agent,
        "seed": args.seed,
        "reset_mode": "single_env_reset",
        "reset_controller_settle_steps": args.expected_reset_settle_steps,
        "top_mode": args.top_mode,
        "n_rollouts": args.n_rollouts,
        "successes": successes,
        "success_rate": successes / args.n_rollouts,
        "records": records,
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
