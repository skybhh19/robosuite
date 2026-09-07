"""Populate the reproducible audit notebook from saved JSON/CSV evidence."""
from pathlib import Path

import nbformat as nbf


repo = Path(__file__).resolve().parents[1]
out = repo / "output/jupyter-notebook/humanlike_dataset_observability_audit.ipynb"
nb = nbf.read(out, as_version=4)
cells = []
cells.append(nbf.v4.new_markdown_cell("""# Humanlike Dataset Observability Audit

This notebook rebuilds the compact comparisons used to release the 400-demo
Threading and ToolHang datasets. It separates first-attempt success, completion
within a fixed two-attempt collection budget, action/state predictability, temporal
motion statistics, file integrity, and the image-only Full-versus-All BC experiment.
"""))
cells.append(nbf.v4.new_markdown_cell("""## Scope and interpretation

- Unit of difficulty analysis: a frozen simulator initial state evaluated under both observability regimes.
- Released grain: 200 matched successful pairs per environment, or 400 episodes.
- Retry trajectories measure data-generation completion and are excluded from training.
- Absolute joint targets naturally correlate strongly with current joint state; comparisons across OSC/delta and absolute action spaces are descriptive, not causal.
"""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path.cwd()
if not (REPO / 'humanlike_dataset_v2_20260907').exists():
    REPO = REPO.parents[1]
EVIDENCE = REPO / 'humanlike_dataset_v2_20260907/evidence'
BASELINE = REPO / 'analysis/observability_dataset_audit_20260907/summary.csv'

def load_json(path):
    with open(path) as f:
        return json.load(f)
"""))
cells.append(nbf.v4.new_markdown_cell("""## Retry-bounded generation is comparable; Threading first-attempt success is not

The grouped bars show the point estimates. The exact paired bootstrap intervals are
printed below because equivalence depends on the interval, not only the bar height.
"""))
cells.append(nbf.v4.new_code_cell("""rows = []
for env in ['threading', 'toolhang']:
    audit = load_json(EVIDENCE / env / 'retry_completion_audit.json')
    for protocol in ['first_attempt', 'within_two_attempts']:
        for regime in ['full', 'partial']:
            rows.append({
                'environment': env,
                'protocol': protocol,
                'regime': regime,
                'success_rate': audit[protocol][f'{regime}_rate'],
                'successes': audit[protocol][regime],
                'pairs': audit[protocol]['n_pairs'],
            })
success = pd.DataFrame(rows)
display(success)
for env in ['threading', 'toolhang']:
    audit = load_json(EVIDENCE / env / 'retry_completion_audit.json')
    print(env, {p: {
        'full_minus_partial': audit[p]['full_minus_partial'],
        'paired_bootstrap95': audit[p]['paired_bootstrap95']
    } for p in ['first_attempt', 'within_two_attempts']})

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
colors = {'full': '#2563eb', 'partial': '#d97706'}
for ax, (env, part) in zip(axes, success.groupby('environment', sort=False)):
    protocols = ['first_attempt', 'within_two_attempts']
    x = range(len(protocols))
    for offset, regime in [(-0.18, 'full'), (0.18, 'partial')]:
        vals = [part[(part.protocol == p) & (part.regime == regime)].success_rate.iloc[0] for p in protocols]
        ax.bar([i + offset for i in x], vals, width=.34, color=colors[regime], label=regime.title())
        for i, v in enumerate(vals): ax.text(i + offset, v + .004, f'{v:.1%}', ha='center', fontsize=9)
    ax.set_title(env.title())
    ax.set_xticks(list(x), ['First attempt', 'Within 2 attempts'])
    ax.set_ylim(.85, 1.015)
    ax.grid(axis='y', color='#dddddd', linewidth=.7)
axes[0].set_ylabel('Physical success rate')
axes[1].legend(frameon=False)
fig.suptitle('Scripted collection success on paired frozen states')
fig.tight_layout()
"""))
cells.append(nbf.v4.new_markdown_cell("""## Threading gains human-scale corrections; ToolHang remains more variable

The rows below compare episode medians at each dataset's stated rate. Rate changes affect
finite differences, so the 20 Hz old/new rows are the direct collector comparison and the
14 Hz rows provide context against the approximately 14 Hz real-robot references.
"""))
cells.append(nbf.v4.new_code_cell("""temporal = load_json(EVIDENCE / 'old_new_temporal.json')
motion_rows = []
for name, payload in temporal.items():
    metric = payload['metrics']
    motion_rows.append({
        'dataset': name,
        'median_steps': metric['steps']['median'],
        'median_target_delta': metric['target_delta_median']['median'],
        'median_target_residual': metric['target_residual_median']['median'],
        'median_p90_acceleration': metric['target_accel_p90']['median'],
        'median_direction_reverse': metric['direction_reverse']['median'],
        'median_strong_reverse': metric['strong_direction_reverse']['median'],
    })
motion = pd.DataFrame(motion_rows)
display(motion)
"""))
cells.append(nbf.v4.new_markdown_cell("""## Absolute action/state correlation is high in both script and real data

This table uses episode-held-out ridge R² from the original audit. Action representations
differ across rows, so PH/MH values cannot be ranked directly against absolute-joint data.
The residual column is the more informative same-representation comparison for the scripted
and real absolute-joint datasets.
"""))
cells.append(nbf.v4.new_code_cell("""corr = pd.read_csv(BASELINE)
display(corr)

new_rows = []
for env in ['threading', 'toolhang']:
    p = EVIDENCE / env / 'human_rate_correlation_audit.json'
    if not p.exists(): p = EVIDENCE / env / 'correlation_audit.json'
    x = load_json(p)
    new_rows.append({
        'dataset': f'{env} humanlike',
        'episodes': x['episodes'],
        'steps': x['steps'],
        'q_to_a_r2': x['a_heldout_ridge_r2']['q']['mean'],
        'qprev_to_a_r2': x['a_heldout_ridge_r2']['q_prev']['mean'],
        'aprev_to_a_r2': x['a_heldout_ridge_r2']['a_prev']['mean'],
        'q_to_residual_r2': x['residual_heldout_ridge_r2']['q']['mean'],
    })
display(pd.DataFrame(new_rows))
"""))
cells.append(nbf.v4.new_markdown_cell("""## Final dataset integrity and BC observability check

Validation must show exactly 400 episodes, 200 per observability regime, pair-isolated
train/validation masks, aligned action/state/image lengths, and matching hashes. The BC
experiment then compares image-only Full-200 against image-only All-400 with identical
architecture, optimization, seeds, checkpoints, and rollout states.
"""))
cells.append(nbf.v4.new_code_cell("""validations = []
for env in ['threading', 'toolhang']:
    p = EVIDENCE / env / 'final_validation.json'
    if p.exists(): validations.append({'environment': env, **load_json(p)})
display(pd.json_normalize(validations) if validations else 'Final validation pending')

bc_path = EVIDENCE / 'bc_report.json'
if bc_path.exists():
    bc = load_json(bc_path)
    display(pd.json_normalize(bc))
else:
    print('BC report pending')
"""))
cells.append(nbf.v4.new_markdown_cell("""## Limits and next checks

1. Retry-bounded completion is a collection-pipeline result; it does not erase the lower Threading Full first-attempt rate.
2. Human likeness is supported by time-normalized motion summaries, but the real references are different tasks and do not provide a causal ground truth.
3. Full > All in image-only BC is the direct acceptance test for useful observability separation. If it fails consistently across seeds, the dataset should not be described as meeting the observability goal even when all file checks pass.
"""))

nb.cells = cells
nb.metadata.setdefault('kernelspec', {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'})
nbf.write(nb, out)
print(out)
