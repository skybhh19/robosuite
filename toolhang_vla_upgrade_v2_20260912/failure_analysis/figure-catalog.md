# Figure catalog

## `figures/failure_stages.png`

- Purpose: compare where Full and Partial first-attempt rollouts stopped in each dataset source pool.
- Data: raw rollout JSON files from the v1 320-pair and v2 450-pair production pools.
- Reader should notice: insertion dominates both regimes; v2 Full has a distinct descend-failure group, while v2 Partial has more transfer/preinsert failures.
- Meaning: the Full/Partial difference is not only visual; the different grasp positions create different mechanical failure patterns.
- Caveat: counts describe scripted collection and do not measure trained VLA performance.
