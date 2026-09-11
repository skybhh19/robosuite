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
