# YCB Fork balanced-v5 environment

The balanced-v5 Fork dataset and its policies must be run with the opt-in
balanced contact profile:

```python
env = robosuite.make(
    "YCBForkInRack",
    robots="Panda",
    grasp_contact_profile="balanced",
    camera_names=["agentview", "robot0_eye_in_hand"],
)
```

The profile is shared by Full and Partial episodes; the observability label
does not select a different controller, camera, geometry, or contact model.
It applies the physics recorded in the balanced-v5 HDF5 model XML:

- standard Fork and rack geometry (no v6 `thick_wide` geometry);
- Fork handle friction `(4.0, 0.3, 0.02)` and `condim=6`;
- Fork neck friction `(4.0, 0.08, 0.02)` and `condim=6`;
- Panda finger/pad friction `(8.0, 0.08, 0.02)` and `condim=6`;
- gripper actuator force limits `[-100, 100]`;
- wrist-camera field of view `40` degrees.

The default `grasp_contact_profile="legacy"` remains unchanged for older Fork
datasets. Do not evaluate balanced-v5 policies with the later v6 geometry or
adaptive-grasp collection code.
