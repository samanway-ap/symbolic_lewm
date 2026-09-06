"""One-off extraction: pull per-episode (task, length, action-spread,
involves_gripper) records from lerobot/droid_100 and dump them as JSON for
scope/task_families.py to consume. Run with a Python that has `lerobot`
installed (the le-wm venv, which already has it):

    <le-wm venv python> _extract_droid_episode_records.py <out.json>

No scene/building id exists as a column in droid_100 -- see repo_audit.json
and the note this prints. scene_key is left "" for every episode.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main(out_path: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot/droid_100")
    eps = ds.meta.episodes
    tasks_col = eps["tasks"]
    lengths_col = eps["length"]

    # action array, keyed by episode_index, straight from the parquet column
    # (cheap: 32k rows x 7 floats, no video decode needed)
    ep_idx_all = np.asarray(ds.hf_dataset["episode_index"])
    action_all = np.asarray(ds.hf_dataset["action"])

    records = []
    for ep_id in range(len(tasks_col)):
        task_list = tasks_col[ep_id]
        task = task_list[0] if task_list else ""
        length = int(lengths_col[ep_id])

        mask = ep_idx_all == ep_id
        actions = action_all[mask]  # (T, 7): 6 arm dims + 1 gripper dim (DROID convention)
        action_spread = float(np.nanstd(actions, axis=0).mean()) if len(actions) else 0.0

        gripper_channel = actions[:, -1] if actions.shape[-1] >= 1 and len(actions) else np.array([])
        involves_gripper = bool(
            len(gripper_channel) and (gripper_channel.max() - gripper_channel.min()) > 0.5
        )

        records.append({
            "episode_id": ep_id,
            "task": task,
            "scene_key": "",  # not available in droid_100 -- see repo_audit.json
            "length": length,
            "action_spread": action_spread,
            "involves_gripper": involves_gripper,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {len(records)} episode records -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "droid100_episode_records.json")
