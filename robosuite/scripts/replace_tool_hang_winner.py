#!/usr/bin/env python3
"""Replace one within-state winner with a replay-verified sibling candidate."""

import argparse
import json
from pathlib import Path
import shutil


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--state-id", type=int, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    args = parser.parse_args()

    replay = json.loads((args.replacement_root / "open_loop_replay.json").read_text())
    if replay.get("episodes") != 1 or not replay.get("all_passed"):
        raise RuntimeError("replacement candidate has not passed open-loop replay")
    replacement_episode = next((args.replacement_root / "raw_demos").glob("ep_*"))
    replacement_stats = json.loads((replacement_episode / "policy_stats.json").read_text())
    if int(replacement_stats.get("state_id", -1)) != args.state_id:
        raise RuntimeError("replacement state id differs")

    regime_root = args.pilot_root / args.backend / args.regime
    matches = []
    for progress_path in regime_root.glob("shard_*/collection_progress.json"):
        progress = json.loads(progress_path.read_text())
        record = progress.get("state_records", {}).get(str(args.state_id))
        if record is not None:
            matches.append((progress_path, progress, record))
    if len(matches) != 1:
        raise RuntimeError(f"expected one original state record, found {len(matches)}")
    progress_path, progress, record = matches[0]
    shard_root = progress_path.parent
    old_winner = record["winner"]
    old_episode = shard_root / "raw_demos" / old_winner["episode_dir"]
    old_video = shard_root / "videos" / old_winner["video"]
    new_video_source = args.replacement_root / "videos" / f"rollout_{args.state_id:03d}.mp4"
    if not old_episode.is_dir() or not old_video.is_file() or not new_video_source.is_file():
        raise RuntimeError("winner episode or video is missing")

    backup = args.pilot_root / "replaced_winner_backup" / args.backend / args.regime
    backup.mkdir(parents=True, exist_ok=True)
    backup_episode = backup / old_episode.name
    backup_video = backup / old_video.name
    if backup_episode.exists() or backup_video.exists():
        raise RuntimeError("backup target already exists")
    shutil.move(str(old_episode), str(backup_episode))
    shutil.copy2(old_video, backup_video)
    old_video.unlink()

    destination_episode = shard_root / "raw_demos" / replacement_episode.name
    shutil.copytree(replacement_episode, destination_episode)
    shutil.copy2(new_video_source, old_video)
    replacement_stats.update(
        {
            "candidate_count": int(old_winner.get("candidate_count", 2)),
            "selected_candidate_retry": int(replacement_stats["retry"]),
            "rejected_candidate_retries": [int(old_winner["retry"])],
            "selection_rule": "replay_parity_then_lexicographic_within_frozen_state",
            "episode_dir": destination_episode.name,
            "video": old_video.name,
        }
    )
    write_json(destination_episode / "policy_stats.json", replacement_stats)
    record["winner"] = replacement_stats
    record["replay_rejected_winner"] = {
        "episode_dir": old_winner["episode_dir"],
        "retry": int(old_winner["retry"]),
        "backup_dir": str(backup_episode),
    }
    progress["state_records"][str(args.state_id)] = record
    write_json(progress_path, progress)

    summary_path = shard_root / "tool_hang_wrench_joint_summary.json"
    summary = json.loads(summary_path.read_text())
    indexes = [
        index
        for index, row in enumerate(summary.get("rollouts", []))
        if int(row.get("state_id", -1)) == args.state_id
    ]
    if len(indexes) != 1:
        raise RuntimeError("state is missing or duplicated in shard summary")
    summary["rollouts"][indexes[0]] = replacement_stats
    write_json(summary_path, summary)
    print(destination_episode)


if __name__ == "__main__":
    main()
