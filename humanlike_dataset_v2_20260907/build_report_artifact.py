"""Build the canonical Data Analytics report artifact from saved evidence."""
from datetime import datetime, timezone
import csv
import json
import sqlite3
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVIDENCE = HERE / "evidence"
OUT = HERE / "report"
OUT.mkdir(exist_ok=True)


def load(path):
    with open(path) as stream:
        return json.load(stream)


def materialize(rows, table):
    """Run the final report projection in SQLite and preserve that exact SQL as provenance."""
    columns = list(rows[0])
    kinds = {
        key: ("REAL" if any(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
                            for row in rows) else "TEXT")
        for key in columns
    }
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        f"CREATE TABLE {table} (" + ", ".join(f'\"{key}\" {kinds[key]}' for key in columns) + ")"
    )
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO {table} VALUES ({placeholders})",
        [[row.get(key) for key in columns] for row in rows],
    )
    query = f"SELECT {', '.join(columns)} FROM {table}"
    result = [dict(row) for row in connection.execute(query)]
    connection.close()
    return result, query


retry = {env: load(EVIDENCE / env / "retry_completion_audit.json") for env in ("threading", "toolhang")}
validation = {env: load(EVIDENCE / env / "final_validation.json") for env in ("threading", "toolhang")}
bc = load(EVIDENCE / "bc_report.json")
temporal = load(EVIDENCE / "old_new_temporal.json")
generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

success_delta = []
for env in ("threading", "toolhang"):
    for protocol, label in (("first_attempt", "First attempt"), ("within_two_attempts", "Within two attempts")):
        value = retry[env][protocol]
        success_delta.append({
            "comparison": f"{env.title()} · {label}",
            "environment": env.title(),
            "protocol": label,
            "difference": value["full_minus_partial"],
            "ci_low": value["paired_bootstrap95"][0],
            "ci_high": value["paired_bootstrap95"][1],
            "pairs": value["n_pairs"],
        })

bc_rows = []
for env, value in bc["groups"].items():
    for condition in ("all", "full"):
        bc_rows.append({
            "environment": env.title(),
            "condition": condition.title(),
            "success_rate": value[condition]["mean"],
            "seed_rates": ", ".join(f"{x:.1%}" for x in value[condition]["values"]),
            "full_minus_all": value["full_minus_all"],
            "ci_low": value["paired_hierarchical_bootstrap95"][0],
            "ci_high": value["paired_hierarchical_bootstrap95"][1],
            "passed": "Yes" if value["passed"] else "No",
        })

validation_rows = []
for env, value in validation.items():
    validation_rows.append({
        "environment": env.title(),
        "passed": "Yes" if value["passed"] else "No",
        "episodes": value["episodes"],
        "full": value["mask_counts"]["fully_observable"],
        "partial": value["mask_counts"]["partially_observable"],
        "train": value["mask_counts"]["train"],
        "valid": value["mask_counts"]["valid"],
        "steps_14hz": value["steps_14hz"],
        "sha256_14hz": value["files"]["dataset_image84_14hz.hdf5"]["sha256"],
    })

motion_rows = []
for name, label in (
    ("old_thread", "Threading old · 20 Hz"),
    ("new_thread_20hz", "Threading new · 20 Hz"),
    ("new_thread_14hz", "Threading new · 14 Hz"),
    ("old_tool", "ToolHang old · 20 Hz"),
    ("new_tool_20hz", "ToolHang new · 20 Hz"),
    ("new_tool_14hz", "ToolHang new · 14 Hz"),
    ("real_wrench", "Real wrench · ≈14 Hz"),
    ("real_holder", "Real holder · ≈14 Hz"),
):
    metrics = temporal[name]["metrics"]
    motion_rows.append({
        "dataset": label,
        "median_steps": metrics["steps"]["median"],
        "target_delta": metrics["target_delta_median"]["median"],
        "target_residual": metrics["target_residual_median"]["median"],
        "acceleration_p90": metrics["target_accel_p90"]["median"],
        "direction_reverse": metrics["direction_reverse"]["median"],
        "strong_reverse": metrics["strong_direction_reverse"]["median"],
    })

correlation_rows = []
with open(REPO / "analysis/observability_dataset_audit_20260907/summary.csv") as stream:
    for row in csv.DictReader(stream):
        correlation_rows.append({
            "dataset": row.get("dataset") or row.get("name"),
            "action": row.get("action") or row.get("representation"),
            "q_to_a": float(row["q"]),
            "qprev_to_a": float(row["q_previous"]),
            "aprev_to_a": float(row["action_previous"]),
            "q_to_residual": (float(row["residual_q"]) if row.get("residual_q") else None),
        })
for env in ("threading", "toolhang"):
    x = load(EVIDENCE / env / "human_rate_correlation_audit.json")
    correlation_rows.append({
        "dataset": f"{env.title()} humanlike, 14 Hz",
        "action": "absolute joint",
        "q_to_a": x["a_heldout_ridge_r2"]["q"]["mean"],
        "qprev_to_a": x["a_heldout_ridge_r2"]["q_prev"]["mean"],
        "aprev_to_a": x["a_heldout_ridge_r2"]["a_prev"]["mean"],
        "q_to_residual": x["residual_heldout_ridge_r2"]["q"]["mean"],
    })

success_delta, success_sql = materialize(success_delta, "success_delta")
motion_rows, motion_sql = materialize(motion_rows, "motion_diagnostics")
bc_rows, bc_sql = materialize(bc_rows, "bc_rollouts")
validation_rows, validation_sql = materialize(validation_rows, "dataset_validation")
correlation_rows, correlation_sql = materialize(correlation_rows, "correlation_audit")

source_notebook = {
    "id": "audit_notebook",
    "label": "Executed observability audit notebook",
    "path": "output/jupyter-notebook/humanlike_dataset_observability_audit.executed.ipynb",
    "query": {"engine": "SQLite", "sql": correlation_sql, "tables_used": ["correlation_audit"], "description": "Final projection of reviewed episode-held-out correlation metrics."},
}
source_bc = {"id": "bc_report", "label": "Image-only BC Full-versus-All report", "path": "humanlike_dataset_v2_20260907/evidence/bc_report.json", "query": {"engine": "SQLite", "sql": bc_sql, "tables_used": ["bc_rollouts"], "description": "Final projection of matched-rollout BC success metrics."}}
source_thread = {"id": "threading_final", "label": "Threading strict final validation", "path": "humanlike_dataset_v2_20260907/evidence/threading/final_validation.json", "query": {"engine": "SQLite", "sql": validation_sql, "tables_used": ["dataset_validation"], "description": "Final projection of strict release-validation results."}}
source_tool = {"id": "toolhang_final", "label": "ToolHang strict final validation", "path": "humanlike_dataset_v2_20260907/evidence/toolhang/final_validation.json", "query": {"engine": "SQLite", "sql": validation_sql, "tables_used": ["dataset_validation"], "description": "Final projection of strict release-validation results."}}
source_motion = {"id": "motion_audit", "label": "Old, new, and real-robot temporal audit", "path": "humanlike_dataset_v2_20260907/evidence/old_new_temporal.json", "query": {"engine": "SQLite", "sql": motion_sql, "tables_used": ["motion_diagnostics"], "description": "Final projection of episode-level temporal diagnostics."}}
source_success = {"id": "success_audit", "label": "Paired Full-versus-Partial completion audits", "path": "humanlike_dataset_v2_20260907/evidence", "query": {"engine": "SQLite", "sql": success_sql, "tables_used": ["success_delta"], "description": "Final projection of paired success-rate differences and bootstrap intervals."}}
source_validation = {"id": "validation_audit", "label": "Combined strict release validation", "path": "humanlike_dataset_v2_20260907/evidence", "query": {"engine": "SQLite", "sql": validation_sql, "tables_used": ["dataset_validation"], "description": "Final projection of both strict release-validation records."}}
sources = [source_notebook, source_bc, source_thread, source_tool, source_motion, source_success, source_validation]

t = retry["threading"]
h = retry["toolhang"]
tg = bc["groups"]["threading"]
hg = bc["groups"]["toolhang"]
summary = (
    "## Technical summary\n\n"
    f"- Both released datasets contain **400 validated episodes**: 200 matched Full and 200 matched Partial trajectories.\n"
    f"- With at most two independently seeded collection attempts, Threading Full-minus-Partial is **{t['within_two_attempts']['full_minus_partial']:.2%}** "
    f"(paired 95% interval {t['within_two_attempts']['paired_bootstrap95'][0]:.2%} to {t['within_two_attempts']['paired_bootstrap95'][1]:.2%}); "
    f"ToolHang is **{h['within_two_attempts']['full_minus_partial']:.2%}** "
    f"({h['within_two_attempts']['paired_bootstrap95'][0]:.2%} to {h['within_two_attempts']['paired_bootstrap95'][1]:.2%}).\n"
    f"- Image-only BC Full-minus-All is **{tg['full_minus_all']:.2%}** for Threading and **{hg['full_minus_all']:.2%}** for ToolHang. "
    f"The frozen criterion {'passes' if bc['passed'] else 'fails'} across both environments.\n"
    "- Absolute action/state correlation remains high, including in real robot absolute-joint data. The BC experiment is the direct test of whether images retain useful observability information."
)

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "Humanlike Scripted Dataset Audit",
        "description": "Difficulty, motion, integrity, and image-only BC evidence for two 400-demo datasets.",
        "generatedAt": generated,
        "charts": [
            {
                "id": "success_delta_chart", "title": "Full-minus-Partial collection success",
                "subtitle": "Paired frozen states; point difference by first-attempt and two-attempt protocol",
                "type": "bar", "dataset": "success_delta", "sourceId": "success_audit",
                "valueFormat": "percent", "layout": "full",
                "encodings": {
                    "x": {"field": "comparison", "type": "nominal", "label": "Environment and protocol"},
                    "y": {"field": "difference", "type": "quantitative", "label": "Full minus Partial", "format": "percent"},
                    "tooltip": [
                        {"field": "ci_low", "type": "quantitative", "label": "95% low", "format": "percent"},
                        {"field": "ci_high", "type": "quantitative", "label": "95% high", "format": "percent"},
                        {"field": "pairs", "type": "quantitative", "label": "Paired states"},
                    ],
                },
            },
            {
                "id": "bc_chart", "title": "Image-only BC rollout success",
                "subtitle": "Mean across three policy seeds; 100 matched-seed rollouts per policy",
                "type": "bar", "dataset": "bc", "sourceId": "bc_report", "valueFormat": "percent", "layout": "full",
                "encodings": {
                    "x": {"field": "environment", "type": "nominal", "label": "Environment"},
                    "y": {"field": "success_rate", "type": "quantitative", "label": "Success rate", "format": "percent"},
                    "color": {"field": "condition", "type": "nominal", "label": "Training data"},
                    "tooltip": [
                        {"field": "seed_rates", "type": "nominal", "label": "Per-seed rates"},
                        {"field": "full_minus_all", "type": "quantitative", "label": "Full minus All", "format": "percent"},
                    ],
                },
            },
        ],
        "tables": [
            {
                "id": "motion_table", "title": "Motion diagnostics across collection sources",
                "subtitle": "Episode medians; rates differ, so compare like-for-like rows before using human references as context",
                "dataset": "motion", "sourceId": "motion_audit", "defaultSort": {"field": "dataset", "direction": "asc"},
                "columns": [
                    {"field": "dataset", "label": "Dataset", "type": "text"},
                    {"field": "median_steps", "label": "Steps", "format": "number"},
                    {"field": "target_delta", "label": "Median |Δ target|", "format": "number"},
                    {"field": "target_residual", "label": "Median |target − q|", "format": "number"},
                    {"field": "acceleration_p90", "label": "P90 acceleration", "format": "number"},
                    {"field": "direction_reverse", "label": "Direction reversal", "format": "percent"},
                    {"field": "strong_reverse", "label": "Strong reversal", "format": "percent"},
                ],
            },
            {
                "id": "correlation_table", "title": "Episode-held-out action predictability",
                "subtitle": "Ridge R²; action representations differ across dataset families",
                "dataset": "correlation", "sourceId": "audit_notebook", "defaultSort": {"field": "q_to_a", "direction": "desc"},
                "columns": [
                    {"field": "dataset", "label": "Dataset", "type": "text"},
                    {"field": "action", "label": "Action", "type": "text"},
                    {"field": "q_to_a", "label": "q(t) → a(t)", "format": "number"},
                    {"field": "qprev_to_a", "label": "q(t-1) → a(t)", "format": "number"},
                    {"field": "aprev_to_a", "label": "a(t-1) → a(t)", "format": "number"},
                    {"field": "q_to_residual", "label": "q(t) → a(t)-q(t)", "format": "number"},
                ],
            },
            {
                "id": "validation_table", "title": "Released dataset integrity",
                "subtitle": "Strict counts and 14 Hz file identity",
                "dataset": "validation", "sourceId": "validation_audit", "defaultSort": {"field": "environment", "direction": "asc"},
                "columns": [
                    {"field": "environment", "label": "Environment", "type": "text"},
                    {"field": "passed", "label": "Passed", "type": "text"},
                    {"field": "episodes", "label": "Episodes", "format": "number"},
                    {"field": "full", "label": "Full", "format": "number"},
                    {"field": "partial", "label": "Partial", "format": "number"},
                    {"field": "train", "label": "Train", "format": "number"},
                    {"field": "valid", "label": "Valid", "format": "number"},
                    {"field": "steps_14hz", "label": "14 Hz steps", "format": "number"},
                    {"field": "sha256_14hz", "label": "14 Hz SHA-256", "type": "text"},
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# Humanlike Scripted Dataset Audit"},
            {"id": "summary", "type": "markdown", "body": summary},
            {"id": "difficulty", "type": "markdown", "body": "## Retry-bounded generation is comparable; Threading first-attempt success is lower\n\nThe two-attempt protocol is the closest scripted analogue to a human recollecting a failed demonstration. Retry trajectories are excluded from the released data, so they improve the completion-rate audit without changing training distributions. Threading's first-attempt gap remains a visible limitation."},
            {"id": "success_chart_block", "type": "chart", "chartId": "success_delta_chart", "layout": "full"},
            {"id": "motion", "type": "markdown", "sourceId": "motion_audit", "body": "## Threading moved toward human-like correction frequency; ToolHang remains a caveat\n\nAt 14 Hz, Threading's median direction-reversal rate is 2.53%, between the real holder reference at 2.14% and real wrench at 3.14%. ToolHang reaches 5.49%, above its old 20 Hz value of 4.32% and both real references. This supports the intended reduction in uniformly smooth Threading motion, while ToolHang should be treated as more variable than the available human references."},
            {"id": "motion_table_block", "type": "table", "tableId": "motion_table", "layout": "full"},
            {"id": "correlation", "type": "markdown", "body": "## State/action correlation alone does not diagnose observability\n\nAbsolute joint commands remain highly predictable from joint state in scripted and real robot data. Cross-representation PH/MH comparisons are useful context, but cannot establish that lower R² is more human. Residual predictability and the image-only policy experiment provide stronger checks."},
            {"id": "correlation_table_block", "type": "table", "tableId": "correlation_table", "layout": "full"},
            {"id": "bc", "type": "markdown", "sourceId": "bc_report", "body": "## Image-only Full-versus-All BC is the release decision\n\nBoth conditions use the same cameras, architecture, optimization budget, action chunks, policy seeds, and matched rollout seeds. The only training change is the All-400 versus Full-200 mask. The chart reports point means; the source report also preserves per-seed results and paired hierarchical bootstrap intervals."},
            {"id": "bc_chart_block", "type": "chart", "chartId": "bc_chart", "layout": "full"},
            {"id": "scope", "type": "markdown", "body": "## Scope and metric definitions\n\nA pair is one frozen simulator state evaluated once under Full and Partial. Physical success uses each environment's existing success condition. The released dataset selects the first 200 pairs where both first attempts pass the existing quality gate. Two-attempt completion counts a pair-regime successful if either of at most two independently seeded motion realizations succeeds."},
            {"id": "validation_table_block", "type": "table", "tableId": "validation_table", "layout": "full"},
            {"id": "method", "type": "markdown", "body": "## Model and validation design\n\nThe collector preserves the environments, cameras, object geometry, success checks, and joint-position action semantics. It varies bounded free-space waypoints, timing, residual gain, and low-frequency command lag. Final data are resampled deterministically from 20 Hz to 14 Hz with within-episode terminal-padded 10-step action chunks. BC-GMM trains for 300 epochs from two RGB cameras and no low-dimensional robot state."},
            {"id": "limits", "type": "markdown", "body": "## Limitations and robustness\n\nThreading does not have equal first-attempt difficulty; its two-attempt interval is inside the one-sided 5-point completion margin. The real robot references differ in task and stage composition, so temporal proximity is descriptive. PH/MH use OSC or delta actions and cannot be compared numerically as if action semantics were identical. BC is compute-matched, so both conditions receive the same number of gradient updates while each Full-200 example is sampled more often than each All-400 example. A positive mean BC difference establishes the requested empirical ordering on this protocol, not a universal observability theorem."},
            {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n- Use the validated 14 Hz files and masks recorded here.\n- Preserve the frozen BC rollout states for future collector comparisons.\n- Add phase-aligned real robot labels before making stronger claims about human motion similarity.\n- Keep first-attempt and retry-bounded success as separate release metrics."},
            {"id": "questions", "type": "markdown", "body": "## Further questions\n\nThe next useful test is whether the density score's Full-versus-Partial separation agrees with image-only BC on held-out paired states. A disagreement would localize the remaining issue to the density estimator or score definition rather than the dataset alone."},
        ],
    },
    "snapshot": {
        "version": 1, "generatedAt": generated, "status": "ready",
        "datasets": {"success_delta": success_delta, "motion": motion_rows, "bc": bc_rows, "validation": validation_rows, "correlation": correlation_rows},
    },
    "sources": sources,
}
(OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
print(OUT / "artifact.json")
