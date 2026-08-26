# ToolHang Phase-2 Scripted Collection

`ToolHangWrenchOnly` isolates the wrench-hanging phase of ToolHang. On every
reset, the stand and frame are placed in their assembled configuration and
fixed with MuJoCo mocap welds. The wrench remains a normal simulated free body:
the policy never writes its pose or velocity after execution starts.

The environment is registered by importing `robosuite`, so it can be created
with:

```python
import robosuite as suite

env = suite.make(
    "ToolHangWrenchOnly",
    robots=["Panda"],
    initialization_noise=None,
)
```

A normal `env.reset()` samples:

- Panda joint positions around a task-specific home pose with per-joint
  Gaussian noise (sigma 0.02 rad, clipped to +/-0.06 rad);
- a fully open gripper;
- the wrench XY position and yaw within the native ToolHang workspace; and
- a small assembled-fixture translation and yaw.

For OSC demonstrations, reset then executes 10 zero-delta controller-settling
steps, restores the sampled arm joint positions, zeros arm velocity, and
updates and resets the controller. This is the recorded frame-zero semantics;
evaluation must call `env.reset()` exactly once and must not subsequently call
`reset_to(state_dict)`.

## Policy and collection

The geometric policy generates gated geometric stages:
pregrasp, vertical descend, close, lift verification, transfer/rotation,
preinsert, linear insertion, release, and retreat. The production dataset uses
the real 7-D OSC_POSE action sent to the environment: normalized world-frame
delta position, delta rotation, and gripper command.

Collect 220 frozen states, obtain two strict-success candidates per state, and
retain the better candidate for the first 100 eligible full and partial states:

```bash
python robosuite/scripts/collect_tool_hang_two_candidate_pool.py \
  --output-dir output/tool_hang_osc_balanced200 \
  --states 220 \
  --final-states 200 \
  --successes-per-state 2 \
  --max-attempts 20 \
  --max-regime-success-rate-gap 0.05 \
  --seed 20260827 \
  --assignment-seed 20260827 \
  --controller-backend osc_pose \
  --policy-motion-style high_arc
```

All physical reset states are generated before the 50/50 full-visible and
partial-hidden labels are assigned. The two regimes use equally wide,
continuous 10 mm grasp intervals: `[-5, 5]` mm for full and `[35, 45]` mm for
partial. Shared hanging and motion parameters are identical. A failed attempt
is retried without changing its physical reset, and only native successes that
pass every stage and smoothness gate are eligible. The completed collection is
rejected when the per-attempt full/partial success-rate gap exceeds five
percentage points.

The collector writes:

- `raw_demos/ep_*/state_*.npz`, `model.xml`, and `policy_stats.json`;
- `labels.csv`, mapping state IDs to episode directories, grasp coordinates,
  grasp regimes, and motion styles;
- `state_manifest.json`, preserving the sampled reset states; and
- `tool_hang_wrench_joint_summary.json`, containing collection and quality
  statistics.

Each OSC NPZ `action_infos` record stores the action actually executed; actions
are not relabeled from a joint trajectory after collection.

## Canonical evaluation

Evaluate a robomimic checkpoint with the single-reset evaluator:

```bash
python robosuite/scripts/evaluate_tool_hang_policy.py \
  --agent path/to/model.pth \
  --output-dir output/tool_hang_policy_eval \
  --n-rollouts 100 \
  --horizon 700 \
  --seed 2026082601 \
  --skip-video
```

Remove `--skip-video` to render the centered agent view and wrist view side by
side. The evaluator rejects a ToolHang environment whose configured controller
settle count is not 10. It always uses one `env.reset()` and never calls
`reset_to`.

The reset audit for the published OSC dataset measured a maximum frame-zero
state error of `1.18e-7`. After the first recorded action, arm qpos and qvel
errors were `4.2e-17` and `2.2e-15`.

## Published dataset

The audited 200-demo release is available at:

```text
/iliad/u/jasonyan/projects/robosuite/tool_hang_osc_v1_balanced_100_per_regime_20260825
```

It contains `dataset/demo.hdf5`, `dataset/observability_labels.csv`, corrected
low-dimensional and two-camera RGB derivatives, collection metadata, and
SHA-256 checksums.
