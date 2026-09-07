# Human-calibrated scripted datasets (`human_v8` / `human_v4`)

This run produces one balanced 400-demo dataset for `Threading_D06_Hard` and one for
`ToolHangWrenchOnly`. Each contains 200 Full and 200 Partial trajectories selected as
matched pairs from a larger fixed-budget collection pool.

Server root:

```text
/iris/u/jasonyan/data/humanlike_dataset_v2_20260907/
```

## Exact collection code

The released datasets were generated from these frozen server copies:

```text
Threading environment:
/iris/u/jasonyan/repos/robosuite-threading-humanlike-v4-20260907/robosuite/environments/manipulation/threading.py

ToolHang environment:
/iris/u/jasonyan/repos/robosuite-toolhang-humanlike-v4-20260907/robosuite/environments/manipulation/tool_hang_wrench_only.py
```

Both datasets use `collect.py` as the common Full/Partial driver and
`human_motion.py` for the episode-level motion variation. Threading calls
`collect_threading_scripted_grasp_angle.py` through `patch_threading()`.
ToolHang calls `collect_tool_hang_wrench_joint.py`; `patch_toolhang.py` was
applied once to the frozen ToolHang source copy before collection.

The production entry points are:

```bash
# Threading_D06_Hard human_v8
sbatch threading_v7_pipeline_job.sh prepare human_v8 202609081
sbatch --array=0-259 threading_v7_pipeline_job.sh run human_v8 202609081

# ToolHangWrenchOnly human_v4
sbatch production_job.sh toolhang prepare 320
sbatch --array=0-319 production_job.sh toolhang run 320
```

Use the frozen server copies above to reproduce the released files exactly.
The normal repository paths contain the corresponding maintained environments
and base policies, but may continue to receive later development changes.

The collection code does not change either environment, object geometry, camera,
success condition, or observability definition. It changes only free-space expert
motion before the original contact endpoints. Each episode uses bounded intermediate
spatial corrections, smooth low-frequency timing variation, and a lagged target
correction. Full and Partial trials from the same pair start from the same saved state
and share the same sampled motion profile. Failures are retained in the pool statistics;
the released balanced datasets use the first 200 pairs for which both trials pass the
pre-existing acceptance condition.

## Files

The final Threading directory is `production_v8/threading`; the final ToolHang directory
is `production/toolhang`. Each contains:

- `dataset_state.hdf5`: all 400 trajectories at the simulator's 20 Hz control rate.
- `dataset_image84.hdf5`: the same trajectories with 84x84 `agentview_image` and
  `robot0_eye_in_hand_image` observations.
- `dataset_image84_14hz.hdf5`: a deterministic 14 Hz view aligned to the measured DROID
  collection rate. It includes `source_step_index` and a within-episode, terminal-padded
  10-step `action_chunks` target.
- `summary.json`: fixed-budget collection outcomes and the deterministic selection.
- `retry_completion_audit.json`: first-attempt and at-most-two-attempt
  completion rates on the same 260 frozen states. Retry trajectories are audit-only
  and are excluded from the released dataset.
- `temporal_audit.json` and `correlation_audit.json`: 20 Hz motion diagnostics.
- `human_rate_temporal_audit.json` and `human_rate_correlation_audit.json`: comparable
  14 Hz diagnostics against the two real-robot references.
- `final_validation.json`: episode/mask counts, pair-split isolation, action/state/image
  resampling parity, finite-value checks, byte sizes, and SHA-256 hashes.

Expected masks are `all=400`, `fully_observable=200`,
`partially_observable=200`, `train=320`, and `valid=80`. Train/valid splitting is done
at the matched-pair level.

The fixed-budget production pool outcomes are:

| Environment | Pairs attempted | Full physical success | Partial physical success | Matched accepted pairs |
|---|---:|---:|---:|---:|
| Threading D06 (`human_v8`) | 260 | 241/260 (92.69%) | 260/260 (100.00%) | 241 |
| ToolHang | 320 | 291/320 (90.94%) | 296/320 (92.50%) | 259 |

Threading's first-attempt Full-minus-Partial difference is -7.31 percentage points
(paired bootstrap 95% interval [-10.77, -4.23]); it does not support equal
first-attempt difficulty. With the predeclared fixed maximum of two independently
seeded attempts per state and regime, Full completes 254/260 (97.69%) and Partial
260/260 (100.00%). The resulting -2.31-point difference has a paired interval of
[-4.23, -0.77], inside the one-sided 5-point generation-completion margin. This is
reported as retry-bounded collection comparability, not first-attempt equivalence.
The independent 60-pair v8 calibration was 59/60 Full and 60/60 Partial.

ToolHang's paired first-attempt difference is -1.56 points (paired bootstrap 95%
interval [-5.94, 2.81]). Its point estimate is close, but this sample does not establish
a 5-point equivalence bound because the lower confidence limit extends past -5 points.
Within two independently seeded attempts, Full completes 316/320 (98.75%) and Partial
317/320 (99.06%); the -0.31-point difference has a paired interval of
[-1.88, +1.25], fully inside a symmetric +/-5-point bound.

## Calibration and evaluation

Motion calibration uses the 8D absolute-joint-plus-gripper representation shared with
the real DROID wrench-on-hook and tool-holder datasets. The audit reports target
increments, target-state residuals, acceleration, direction reversals, exact holds,
episode lengths, direct correlations, and episode-held-out ridge predictability from
the current and previous robot state. These diagnostics are descriptive; high absolute
target/state R2 is partly inherent to an absolute joint target and is not by itself a
measure of observability.

The final Full-versus-All check trains image-only BC-GMM policies from the two 84x84 RGB
cameras, with no low-dimensional observation. It uses the established 10-step action
chunk setup, three policy seeds per condition, a frozen epoch-300 checkpoint, and 100
matched-seed rollouts per policy. The result is written to `bc/report.json`, including
per-seed differences and a paired hierarchical bootstrap interval. The comparison is
compute-matched: both conditions receive the same number of gradient updates, so each
Full-200 example is sampled more often than each All-400 example.

At 14 Hz, Threading has a median episode-level direction-reversal rate of 2.53%,
between the real tool-holder reference (2.14%) and real wrench reference (3.14%); its
old scripted dataset was 1.51% at 20 Hz. ToolHang is less human-matched on this measure:
the new 14 Hz rate is 5.49%, versus 4.32% for the old 20 Hz data. Sampling rate itself
changes reversal and acceleration metrics, so the complete 20 Hz/14 Hz comparison is
saved in `evidence/old_new_temporal.json`. Real references differ in task and phase
composition; these values establish motion calibration context rather than distributional
equivalence to human demonstrations.
