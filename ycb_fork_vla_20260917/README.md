# YCB fork VLA dataset (2026-09-17)

This production run collects 150 `full_visible` and 150 `partial_hidden`
demonstrations with the variation configuration approved after the v2 preview.

The two regimes are deliberately launched as separate jobs with independent
seeds (`202609171` and `202609172`). This keeps their initial states and
continuous path parameters unpaired while preserving the same balanced
discrete distribution in each regime:

- 10 motion styles
- 5 bend variants
- 2 routing sides
- 100 discrete path families per regime

Every family appears exactly once among the first 100 accepted trajectories in
each regime. The final 50 trajectories repeat discrete families with newly
sampled continuous scene and motion parameters.

Accepted demonstrations must satisfy all collector gates: task success,
9-of-9 aperture visibility for Full or 0-of-9 for Partial, final pose error at
most 5 mm and 2 degrees, the Threading smoothness checks, and exactly one grasp
and release.

Cluster output root:

`/iris/u/jasonyan/data/ycb_fork_vla300_variation_v2_20260917`

