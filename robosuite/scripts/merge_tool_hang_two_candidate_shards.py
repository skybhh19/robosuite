"""Merge and audit exactly 100 full + 100 partial ToolHang winners."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--extra", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load(source, root):
    rows = []
    for progress in sorted(root.glob("shard_*/collection_progress.json")):
        shard = progress.parent
        data = json.loads(progress.read_text())
        for record in data["state_records"].values():
            if record.get("status") != "eligible":
                continue
            winner = record["winner"]
            rows.append(
                {
                    "source": source,
                    "source_root": root,
                    "shard": shard.name,
                    "state_id": int(record["state_id"]),
                    "regime": record["assigned_regime"],
                    "winner": winner,
                }
            )
    return sorted(rows, key=lambda x: (x["state_id"], x["shard"]))


def video_frames(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(result.stdout.strip())


def main():
    a = args()
    output = a.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output is not empty: {output}")
    raw_out = output / "raw_demos"
    video_out = output / "videos"
    raw_out.mkdir(parents=True, exist_ok=True)
    video_out.mkdir(parents=True, exist_ok=True)

    base = load("base", a.base.resolve())
    extra = load("extra", a.extra.resolve())
    full = [x for x in base + extra if x["regime"] == "full_visible"][:100]
    partial = [x for x in base if x["regime"] == "partial_hidden"][:100]
    if len(full) != 100 or len(partial) != 100:
        raise RuntimeError(f"Insufficient winners: full={len(full)} partial={len(partial)}")
    selected = full + partial
    audits = []
    labels = []

    for index, row in enumerate(selected, 1):
        winner = row["winner"]
        source_shard = row["source_root"] / row["shard"]
        source_episode = source_shard / "raw_demos" / winner["episode_dir"]
        source_video = source_shard / "videos" / winner["video"]
        demo_id = f"{row['source']}_{row['state_id']:03d}"
        target_episode = raw_out / f"demo_{demo_id}"
        target_video = video_out / f"demo_{demo_id}.mp4"
        shutil.copytree(source_episode, target_episode)
        shutil.copy2(source_video, target_video)

        stats_path = target_episode / "policy_stats.json"
        stats = json.loads(stats_path.read_text())
        stats["dataset_demo_id"] = demo_id
        stats["source_pool"] = row["source"]
        stats["source_shard"] = row["shard"]
        stats["candidate_count"] = int(winner.get("candidate_count", 0))
        stats["selected_candidate_retry"] = int(winner["selected_candidate_retry"])
        stats["rejected_candidate_retries"] = winner["rejected_candidate_retries"]
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        insert = next(x for x in stats["stage_checks"] if x["name"] == "insert")
        debug = insert["tool_debug"]
        integrity = stats.get("recording_integrity", {})
        npz_files = sorted(target_episode.glob("state_*.npz"))
        if len(npz_files) != 1:
            raise RuntimeError(f"{demo_id}: expected one NPZ, got {len(npz_files)}")
        data = np.load(npz_files[0], allow_pickle=True)
        states = len(data["states"])
        actions = len(data["action_infos"])
        training_steps = int(stats["training_recorded_steps"])
        frames = video_frames(target_video)
        checks = {
            "two_candidates": int(stats.get("candidate_count", 0)) >= 2,
            "accepted": bool(stats.get("accepted")),
            "native_success": bool(stats.get("native_success")),
            "tool_on_frame": bool(stats.get("tool_on_frame")),
            "zero_pose_assist": int(stats.get("wrench_pose_assist_count", -1)) == 0,
            "persistent_release": next(
                x for x in stats["stage_checks"] if x["name"] == "release_retreat"
            ).get("persistent_success_run", 0) >= 20,
            "pre_release_contact": bool(debug.get("hole_frame_contact")),
            "pre_release_straddle": bool(debug.get("hole_straddles_hook")),
            "pre_release_line": float(debug.get("line_distance_m", 1.0)) <= 0.005,
            "pre_release_hold": int(
                insert.get(
                    "pre_release_seated_count",
                    insert.get("pre_release_seated_run", 0),
                )
            )
            >= max(10, int(insert.get("pre_release_seated_required", 10))),
            "recording_integrity": bool(integrity) and all(integrity.values()),
            "npz_states_actions": states == actions + 1,
            "npz_actions_policy": actions == training_steps,
            "video_frame_parity": frames == training_steps + 1,
        }
        if not all(checks.values()):
            raise RuntimeError(f"{demo_id}: failed audit {checks}")
        audits.append({"demo_id": demo_id, "checks": checks, "frames": frames})
        labels.append(
            {
                "demo_id": demo_id,
                "regime": row["regime"],
                "source_pool": row["source"],
                "source_state_id": row["state_id"],
                "motion_style": stats["variation"]["motion_style"],
                "grasp_offset_local_x_m": stats["variation"]["grasp_offset_local_x_m"],
                "selected_candidate_retry": stats["selected_candidate_retry"],
                "rejected_candidate_retries": json.dumps(stats["rejected_candidate_retries"]),
                "episode_dir": target_episode.name,
                "video": target_video.name,
            }
        )

    with (output / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    (output / "audit.json").write_text(json.dumps({"count": 200, "audits": audits}, indent=2) + "\n")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "count": 200,
                "full_visible": 100,
                "partial_hidden": 100,
                "selection": "within-state best of two; deterministic first eligible states",
                "all_audits_passed": True,
            },
            indent=2,
        ) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
