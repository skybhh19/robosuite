"""Episode-level, low-frequency motion profiles calibrated against DROID data."""

from __future__ import annotations

import hashlib

import numpy as np


def profile(seed, stage):
    key = int.from_bytes(
        hashlib.sha256(f"{seed}:{stage}:human-v2".encode()).digest()[:4], "little"
    )
    rng = np.random.RandomState(key)
    first = rng.uniform(-0.008, 0.008, 3)
    second = rng.uniform(-0.006, 0.006, 3) - 0.45 * first
    first[2] = abs(first[2])
    return {
        "duration_scale": float(rng.uniform(0.72, 0.92)),
        "residual_gain": float(rng.uniform(1.35, 1.60)),
        "command_alpha": float(rng.uniform(0.45, 0.65)),
        "warp_a": float(rng.uniform(-0.32, 0.32)),
        "warp_b": float(rng.uniform(-0.12, 0.12)),
        "offset_first": first.tolist(),
        "offset_second": second.tolist(),
    }


def smooth_progress(value, config):
    """Continuous, bounded low-frequency time warp with fixed endpoints."""
    value = float(np.clip(value, 0.0, 1.0))
    warped = (
        value
        + config["warp_a"] * np.sin(2.0 * np.pi * value) / (2.0 * np.pi)
        + config["warp_b"] * np.sin(4.0 * np.pi * value) / (4.0 * np.pi)
    )
    return float(np.clip(warped, 0.0, 1.0))


def path_offset(progress, config, cutoff=1.0):
    """Two bounded corrective lobes with an unchanged contact endpoint."""
    if progress <= 0.0 or progress >= cutoff:
        return np.zeros(3)
    local = progress / cutoff
    first = np.asarray(config["offset_first"], dtype=float)
    second = np.asarray(config["offset_second"], dtype=float)
    envelope = np.sin(np.pi * local) ** 2
    return envelope * (first + np.sin(2.0 * np.pi * local) * second)


def patch_threading():
    from robosuite.scripts.collect_threading_scripted_grasp_angle import (
        ThreadingScriptedPolicy,
    )

    class HumanThreadingPolicy(ThreadingScriptedPolicy):
        human_seed = 0

        def _run_bounded_trajectory_stage(
            self,
            env,
            target_at_progress,
            gripper,
            nominal_steps,
            policy_state,
            stats,
            subgoal,
            completion_fn,
            *args,
            **kwargs,
        ):
            env.stage = subgoal
            target = target_at_progress
            if subgoal in ("aim_continuous", "lift_arc"):
                config = profile(self.human_seed, subgoal)
                stats.setdefault("human_motion_v2", {})[subgoal] = config
                original_steps = int(nominal_steps)
                nominal_steps = int(np.ceil(original_steps * config["duration_scale"]))
                if kwargs.get("max_steps") is not None:
                    nominal_steps = min(
                        nominal_steps,
                        max(original_steps, int(kwargs["max_steps"]) - 4),
                    )
                cutoff = 0.58 if subgoal == "aim_continuous" else 1.0

                def human_target(value):
                    progress = smooth_progress(value, config)
                    position, quaternion = target_at_progress(progress)
                    return (
                        np.asarray(position) + path_offset(progress, config, cutoff),
                        quaternion,
                    )

                target = human_target
            return super()._run_bounded_trajectory_stage(
                env,
                target,
                gripper,
                nominal_steps,
                policy_state,
                stats,
                subgoal,
                completion_fn,
                *args,
                **kwargs,
            )

    return HumanThreadingPolicy
