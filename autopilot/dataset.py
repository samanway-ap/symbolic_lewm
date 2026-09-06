"""Builds a plain (pixels, raw_action) sample list for a set of episode ids
-- no FSM, no state_id/predicates (v6 has none of that). Compatible with
`training/finetune.py`'s `_collate`. The `continue` arm draws directly from
`replay_train` (the original training-split episodes) instead of calling
retrieval at all.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM

N_POSITIONS = 20


@dataclass
class ArmDataset:
    samples: list[dict]
    pipeline: "ActionPipeline"
    episode_ids: list[int]

    def __len__(self):
        return len(self.samples)


def build_dataset_from_episodes(episode_ids: list[int], pipeline: "ActionPipeline",
                                    n_positions: int = N_POSITIONS, max_windows_per_episode: int = 8,
                                    seed: int = 3072) -> ArmDataset:
    if not episode_ids:
        return ArmDataset(samples=[], pipeline=pipeline, episode_ids=[])
    n_raw = n_positions * FRAMESKIP
    actions = load_actions_for_episodes(episode_ids, max_frames=n_raw)
    windows = stream_many_windows(episode_ids, num_frames=n_positions, frameskip=FRAMESKIP,
                                     max_workers=16, per_episode_timeout_s=60.0)
    ok = sorted(set(episode_ids) & set(windows.keys()) & set(actions.keys()))
    ok = [e for e in ok if windows[e].shape[0] == n_positions and actions[e].shape[0] >= n_raw]

    rng = np.random.default_rng(seed)
    samples = []
    for eid in ok:
        pix = windows[eid]
        raw_full = actions[eid][:n_raw].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)
        max_t = n_positions - HISTORY_SIZE - 1
        if max_t <= 0:
            continue
        ts = rng.choice(max_t, size=min(max_windows_per_episode, max_t), replace=False)
        for t in ts:
            t = int(t)
            samples.append({
                "pixels": pix[t:t + HISTORY_SIZE + 1].copy(),
                "raw_action": raw_full[t:t + HISTORY_SIZE + 1].copy(),
                "episode": int(eid), "pos": t,
            })
    return ArmDataset(samples=samples, pipeline=pipeline, episode_ids=ok)


def build_replay_dataset(replay_train_ids: list[int], target_n_samples: int, pipeline: "ActionPipeline",
                             seed: int = 3072) -> ArmDataset:
    """`continue` arm: sample enough replay_train episodes to reach roughly
    the same number of training samples the other arms' K-trajectory
    curricula produce -- equal-compute in sample count, not just in step
    count (which is separately held constant by the training loop itself)."""
    rng = np.random.default_rng(seed)
    per_episode = 8
    n_needed_episodes = max(1, -(-target_n_samples // per_episode))
    chosen = sorted(rng.choice(replay_train_ids, size=min(n_needed_episodes, len(replay_train_ids)),
                                  replace=False).tolist())
    return build_dataset_from_episodes(chosen, pipeline, max_windows_per_episode=per_episode, seed=seed)
