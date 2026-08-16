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

## Policy and collection

The geometric policy commands absolute joint positions through gated stages:
pregrasp, vertical descend, close, lift verification, transfer/rotation,
preinsert, linear insertion, release, and retreat. Five related transfer
families provide controlled path diversity while converging to the same
preinsert and insertion geometry.

Collect a balanced set of fixed reset states with one retained strict success
per state:

```bash
python robosuite/scripts/collect_tool_hang_balanced_state_retries.py \
  --states 100 \
  --max-retries 20 \
  --max-replacements-per-slot 20 \
  --seed 10000 \
  --assignment-seed 10001 \
  --output-dir output/tool_hang_success100
```

All physical reset states are generated before the 50/50 full-visible and
partial-hidden grasp labels are assigned. A failed state is retried without
changing its physical reset; only after all retries are exhausted is that slot
replaced by another screened state. Only native successes that pass every
stage and smoothness gate are retained.

The collector writes:

- `raw_demos/ep_*/state_*.npz`, `model.xml`, and `policy_stats.json`;
- `labels.csv`, mapping state IDs to episode directories, grasp coordinates,
  grasp regimes, and motion styles;
- `state_manifest.json`, preserving the sampled reset states; and
- `tool_hang_wrench_joint_summary.json`, containing collection and quality
  statistics.

Each NPZ `action_infos` record includes the absolute joint-position target,
current joint position, raw joint delta, reference-scaled joint delta, and the
gripper action.
