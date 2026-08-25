# ToolHang OSC paired collection

The phase-2 collector uses real `OSC_POSE` actions at 20 Hz. Full and partial
share frozen physical reset states and all motion parameters; only the
continuous grasp interval changes:

- full: `[-5, 5]` mm from the black-handle center;
- partial: `[35, 45]` mm toward the ring.

Both intervals are 10 mm wide. The fixture is fixed, the external camera is
object-height, and the wrist camera uses the centered Panda pose. Every state
must produce two strict-success candidates before the best replayable candidate
is selected.

```bash
python -m robosuite.scripts.collect_tool_hang_two_candidate_pool \
  --output-dir /path/to/output \
  --states 220 \
  --final-states 200 \
  --successes-per-state 2 \
  --max-attempts 20 \
  --controller-backend osc_pose \
  --high-hole-height-m 0.060 \
  --seat-along-fraction 0.10 \
  --hang-yaw-deg 0 \
  --policy-motion-style high_arc \
  --keep-rejected-candidates
```

The registered environment does not advance OSC during reset. Object sensors
query the current simulator at sample time and cache absolute poses before
relative poses, which keeps state-to-observation conversion correct after a
per-demo XML reload.
