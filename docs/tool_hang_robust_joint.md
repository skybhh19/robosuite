# ToolHang robust joint script

Select `--policy-preset robust_joint` in either
`collect_tool_hang_wrench_joint.py` or `collect_tool_hang_two_candidate_pool.py`.
This selects joint-position control and the same six improvements for Full and
Partial: transfer retiming, bounded grasp-yaw IK fallback, lift endpoint retiming,
release support pivot, correction target memory, and its unseated-only gate.
The underlying implementations were developed as V12; this preset makes them
available together through the normal collection entry points. Legacy defaults
remain available for reproducing old runs. Explicit CLI parameters override preset
defaults and describe a different configuration from the evaluated preset.

The contact correction retains the unachieved target instead of restarting every
correction at the stalled measured EEF. Each new correction is at most 4 mm, and
the accumulated target lead is capped at 8 mm. The stronger correction is selected
only when the tool is held and measured ring geometry is not already seated.
The original insertion depths, grasp intervals, success checks and quality gates
are unchanged. Tool pose is never teleported during execution.

## Run the script

```bash
python robosuite/scripts/collect_tool_hang_wrench_joint.py \
  --policy-preset robust_joint --variation --grasp-profile full_visible \
  --num-rollouts 100 --require-ph-quality \
  --summary-dir output/tool_hang_robust_full
```

Use `--grasp-profile partial_hidden` and a separate output directory for Partial.
These commands sample independent runs; use the paired evaluator below for a
matched comparison and separate physical and strict-quality denominators.
For dataset collection, add `--policy-preset robust_joint` to the two-candidate
collector. Final retained dataset counts are not per-attempt success rates.

## Fixed-budget paired evaluation

```bash
python robosuite/scripts/evaluate_tool_hang_robust_joint.py prepare \
  --output output/tool_hang_robust_eval --pairs 200 --seed 202609031
python robosuite/scripts/evaluate_tool_hang_robust_joint.py run \
  --output output/tool_hang_robust_eval --shard-count 1 --shard-index 0
python robosuite/scripts/evaluate_tool_hang_robust_joint.py summarize \
  --output output/tool_hang_robust_eval
```

Preparation freezes 200 physical reset states before outcomes are observed.
Both policies and both grasp regimes run once per state (800 rollouts), with
identical reset physics and matched randomization. Four path styles and five
continuous grasp bins are balanced. Full uses [-5, 5] mm; Partial uses [45, 55] mm.
The current pool keeps the assembled fixture fixed and randomizes the robot and
wrench; this does not establish performance under additional hook randomization.

The evaluator saves every result, counts policy exceptions as failures, refuses
implicit retries, checks source hashes and pairing, and refuses incomplete
summaries. `summary.json` reports physical completion, quality acceptance, 95%
Wilson intervals, failure stages, and paired gains/losses separately for each
regime. Physical success requires all stages and native environment success,
including the release persistence check. Strict acceptance additionally requires
the existing trajectory-quality checks; it does not imply image or raw replay
audits have passed. The 90% target is assessed on each regime's physical point
estimate, not asserted as a population lower confidence bound.

This evaluation uses the current source snapshot and records the Python, MuJoCo
and NumPy versions. Historical V12 statistics use a different environment version
and are not substituted for this evaluation.

The completed 2026-09-03 paired evaluation measured physical completion of
188/200 (94.0%) Full and 186/200 (93.0%) Partial; strict-quality acceptance was
175/200 (87.5%) and 179/200 (89.5%). See the
[complete result and scope](../output/tool_hang_robust_joint_eval_20260903/report.md).
