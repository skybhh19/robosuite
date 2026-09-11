# ToolHang VLA observability v2

Goal: produce 300 demonstrations as 150 matched Full/Partial pairs with a
larger hook pose distribution than v1. The release keeps only pairs where both
scripts succeed from the same frozen state and the preinsert wrist-camera check
confirms that the wrench hole is visible for Full and hidden for Partial.

Pilot candidates:

| Variant | Fixture X | Fixture Y | Fixture yaw |
|---|---:|---:|---:|
| `moderate_v2` | +/-5.5 cm | +/-4 cm | -30 to +35 deg |
| `d08_like_v2` | +/-7 cm | +/-5 cm | -15 to +45 deg |

Both use the original uniform grip friction 2.0. The v2 policy allows up to 12
measured insertion corrections, versus 8 in v1, and gives wide-pose waypoint
tracking a little more time. Full and Partial use identical policy parameters;
only the grasp position and resulting wrist-camera occlusion differ.

Production contains 150 pairs / 300 demos. The `all` mask contains all 300,
while `fully_observable` and `partially_observable` each contain 150. Train and
validation masks are split by pair.

## Pilot decision

`moderate_v2` was selected for production. On 40 frozen pairs, physical
success was 29/40 Full and 28/40 Partial; accepted success was 25/40 and 26/40.
Nineteen pairs succeeded in both regimes and 17 also passed the strict camera
label check. The D08-like range was less balanced (30/40 Full versus 25/40
Partial) and produced only 12 strictly eligible pairs.

A linear episode-grouped holdout diagnostic predicting absolute action from
current joint position remained high, as expected for absolute joint control.
For delta action, R2 was 0.110 in `moderate_v2`, compared with 0.125 in v1 and
0.137 in `d08_like_v2`. This diagnostic is supporting evidence only; it does
not predict the final VLA ordering.

The production pool uses 450 frozen pairs and selects the first 150 pairs that
pass physical, quality, paired-state, and camera-label checks. This fixed-order
rule does not rank trajectories by smoothness or model score.

## Production release

The completed release is:

`/iris/u/jasonyan/data/toolhang_vla_upgrade_v2_20260912/production_observability_v2/dataset_image84_14hz.hdf5`

It contains 300 demos (150 Full and 150 Partial), with 240 train and 60 valid
demos split by pair. The 450-state collection produced 227 pairs accepted in
both regimes and 212 pairs that also passed the strict visibility gate; the
first 150 in fixed pair order were released. First-attempt physical success was
352/450 Full and 365/450 Partial. The difference was -2.9 percentage points,
with paired bootstrap 95% interval [-8.2, +2.2] points.

The selected poses cover x [-5.39, +5.39] cm, y [-3.97, +3.95] cm, and yaw
[-29.74, +34.97] degrees. Exact initial simulator states match within every
Full/Partial pair, and train/valid pair isolation passed. Episode-grouped
delta-action R2 is 0.105, down from 0.125 in v1; absolute-action R2 remains high
at 0.993 because the controller target is an absolute joint position.

Final validation passed all image, state, action, action-chunk, mask, and
resampling checks. The 14 Hz file SHA-256 is
`7701f2f432d19900e24415892229fa8e3ff52bb87e18915b57d25a4e460d3d4c`.
Collection evidence supports a stronger observability contrast, but the VLA
Full / All / Partial ordering must still be measured by training and rollout.
