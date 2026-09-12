# ToolHang collection failure analysis

The unit is one first-attempt scripted rollout from a frozen candidate state. Final released HDF5 files contain only paired accepted successes, so their internal success rate is 100%.

| Source pool | Regime | Physical success | Accepted after quality checks | Physical failures | Quality-only rejects |
|---|---:|---:|---:|---:|---:|
| 400-demo v1, 320 pairs | Full | 270/320 (84.4%) | 263/320 (82.2%) | 50 | 7 |
| 400-demo v1, 320 pairs | Partial | 278/320 (86.9%) | 267/320 (83.4%) | 42 | 11 |
| 300-demo v2, 450 pairs | Full | 352/450 (78.2%) | 307/450 (68.2%) | 98 | 45 |
| 300-demo v2, 450 pairs | Partial | 365/450 (81.1%) | 330/450 (73.3%) | 85 | 35 |

In v1, both regimes most often failed during insertion (33 each). Full additionally had 13 descend failures; Partial had four preinsert and two release/retreat failures.

In v2, insertion remained the largest category (55 Full, 58 Partial). Full had 36 descend failures. Partial instead had more late approach failures: 16 transfer/rotate and 10 preinsert.

A descend failure means the gripper did not complete the approach and grasp waypoint. Transfer/rotate or preinsert means the wrench was acquired but its hole did not reach a usable pose near the hook. Insert failures reached the hook area but did not seat deeply or accurately enough. Release/retreat failures were close to completion but the wrench did not remain successfully hanging after opening and retreat.

The broader v2 hook range intentionally reduced raw collection success. The released 300 demos are 150 pairs where both regimes succeeded and the camera label was verified. Collection success still favored Partial by 2.9 percentage points, so collection evidence does not support a claim that Full is physically easier.
