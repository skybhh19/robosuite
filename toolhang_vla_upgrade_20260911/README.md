# ToolHang VLA dataset upgrade

This experiment tests two changes independently:

1. Randomize the assembled stand and hook pose for every matched initial state.
2. Increase sliding friction on the wrench's black grip from 2.0 to 4.0.

Full and Partial members of a pair reuse the same robot, wrench, and fixture
pose. The policy receives the same privileged geometry in both regimes. Only
the grasp location, and therefore wrist-camera observability, differs.

The first pilot uses 40 paired states per condition:

| Variant | Fixture X | Fixture Y | Fixture yaw | Grip friction |
|---|---:|---:|---:|---:|
| `fixed_f2` | 0 | 0 | 0 | 2.0 |
| `random_f2` | +/-3.5 cm | +/-2.5 cm | +/-20 deg | 2.0 |
| `random_f4` | +/-3.5 cm | +/-2.5 cm | +/-20 deg | 4.0 |
| `random_center_f4` | +/-3.5 cm | +/-2.5 cm | +/-20 deg | 4.0 in central +/-2 cm; 2.0 elsewhere |

`random_f2` isolates fixture randomization. Comparing `random_f4` against it
isolates grip friction. The random-fixture variants use the same seeds and
therefore the same reset variations.

`random_center_f4` keeps the Partial grasp range at the original friction and
places the Full grasp range entirely inside a higher-friction center segment.
The three grip collision boxes are adjacent and non-overlapping, and their
combined volume and mass equal the original single grip box.

## Pilot result

The selected production condition is `production_random_f2`: randomized
fixture pose with the original grip friction of 2.0. On 40 pairs, first-attempt
physical success was 34/40 Full and 37/40 Partial. With at most two independent
policy attempts it was 40/40 Full and 39/40 Partial. Uniform friction 4.0 and
center-only friction 4.0 were both rejected because neither reliably improved
Full. Exact counts are in `pilot_results.json`.

Run on Sherlock with `pilot_job.sh`. Do not scale to the final dataset until
all three pilots finish, videos show the intended pose coverage, and Full and
Partial completion remain close enough for a balanced paired release.
