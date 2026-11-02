# Fork V17 environment

`environment.json` is the environment metadata exported with the V17 dataset
of 200 demonstrations (100 Full / 100 Partial, 48,724 native frames).
Load this explicit configuration rather than the defaults of `YCBForkInRack`.

```python
import json
from pathlib import Path
import robosuite

path = Path(robosuite.__file__).parent / "configs/fork_v17/environment.json"
config = json.loads(path.read_text())
env = robosuite.make(config["env_name"], **config["env_kwargs"])
obs = env.reset()
```

## Contract

- Panda, 20 Hz; seven **absolute joint-position targets** and one gripper command.
  Raw robosuite gripper commands are +1 (close) and -1 (open). The training
  dataset maps that last dimension to 1/0; convert it back for environment use.
- Agentview and wrist RGB are 256 x 256. Wrist FOV is 50 degrees.
  Both labels use exactly the same camera configuration and physical parameters.
- Geometry profile: `legacy_precision_guided`; contact profile: `moderate`;
  pad dynamics: `low_slip_stiff`; no extra fork extension or mass redistribution.
- Gripper slew is 12 command units/second and position gain is 1500.
- Full and Partial differ in the collector's grasp point (centers 0 and -65 mm,
  each with a 6 mm-wide band), not in an environment label switch. `env.reset()`
  alone does not choose a grasp point or reproduce the dataset's scene manifest.

For exact evaluation/replay, use the dataset's initial-state manifest and
per-demonstration variation with this configuration. Rebuilding an MJCF string
alone is not an exact native replay contract: serialized MuJoCo floats can be
rounded. This release publishes the environment, not the V17 collection pipeline
or the training data. Default resets are not a replacement for benchmark resets.

The exported VLA images use the evaluation convention (vertical flip of the raw
robosuite camera observations). The environment itself retains robosuite's raw
image convention; avoid applying a second flip in a training loader.

The environment pose-success check uses 1.8 cm / 20 degrees. It is not by itself
the dataset's stricter released-and-stable trajectory acceptance check.

## Smoke test

From a source checkout, install robosuite's dependencies and run:

```bash
python -m robosuite.scripts.check_fork_v17_environment
# Linux headless rendering (requires a working OSMesa installation):
MUJOCO_GL=osmesa python -m robosuite.scripts.check_fork_v17_environment --render
```

The test checks construction, native rate, eight-dimensional actions, camera
extrinsics/FOV, and stepping. `--render` additionally checks both RGB outputs;
it does not demonstrate task completion or trained-policy performance.

Compared with the V16 dataset environment metadata, the changes are 14 to 20 Hz,
wrist FOV 46 to 50 degrees, parent-frame camera X +2 mm, and local camera pitch
+3 degrees. The small Euler pan/roll differences reflect rotation composition.
Panda, fork/holder physics, and external camera settings are unchanged.

Source metadata SHA256:
`f14bac76700067e1ba239053f7fae47cffb9f3f1d0a7cb7c6d25e0a251fbbf5e`.
The environment and its supporting modules were taken from the frozen runtime
used to generate that dataset, not a later experimental branch.
