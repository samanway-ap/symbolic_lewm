"""Shared data prep for Phase C's three predicate families: collect
TRAIN-split-only latent trajectories (real ViT encodings, not model
rollouts) + their aligned action embeddings, respecting episode boundaries
(no cross-episode "next" pairs).

Everything here uses TRAIN-split episodes ONLY (episode_split.json), per
instruction: Phase C's pool construction must not touch the test-20% split
that Phase F's acceptance gate and Phase I's evaluation are reserved for.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_episode_frames
from oracle.lewm_g import (
    ActionPipeline, FRAMESKIP, RAW_ACTION_DIM,
    encode_action_embeddings_batch, encode_pixel_windows_batch,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass
class TrajectoryPool:
    episode_ids: list[int]
    latents: dict[int, np.ndarray]      # eid -> (T,D) real latent trace
    act_embeddings: dict[int, np.ndarray]  # eid -> (T-1,D) action_encoder outputs, act_emb[t] drives latents[t]->latents[t+1]
    raw_actions: dict[int, np.ndarray]  # eid -> (T-1,FRAMESKIP,7) raw actions in droid_100 convention, same alignment
    pipeline: ActionPipeline


def train_episode_pool() -> list[int]:
    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    return split["train_episode_ids"]


def collect_trajectories(
    episode_ids: list[int],
    n_positions: int = 40,
    max_workers: int = 16,
    seed: int = 3072,
) -> TrajectoryPool:
    """Streams+encodes n_positions subsampled frames (FRAMESKIP=2) per
    episode from real video, plus the raw actions driving each transition."""
    n_raw_needed = n_positions * FRAMESKIP

    print(f"loading actions for {len(episode_ids)} candidate episodes...", flush=True)
    actions_all = load_actions_for_episodes(episode_ids, max_frames=n_raw_needed)
    usable_ids = [e for e in episode_ids if e in actions_all and actions_all[e].shape[0] >= n_raw_needed]
    print(f"usable (enough raw actions): {len(usable_ids)}", flush=True)

    print("streaming real frames...", flush=True)
    windows: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(stream_episode_frames, eid, "observation.images.exterior_1_left", 0, n_positions, FRAMESKIP): eid
            for eid in usable_ids
        }
        for fut in as_completed(futs):
            eid = futs[fut]
            try:
                windows[eid] = fut.result()
            except Exception as e:
                print(f"  episode {eid} stream failed: {e!r}", file=sys.stderr)

    final_ids = sorted(windows.keys())
    print(f"final usable episodes: {len(final_ids)}", flush=True)

    pixel_stack = np.stack([windows[e] for e in final_ids])  # (N,n_positions,h,w,3)
    all_raw = np.concatenate([actions_all[e][:n_raw_needed] for e in final_ids], axis=0)
    pipeline = ActionPipeline.fit(all_raw)

    print("encoding latent traces...", flush=True)
    emb_stack = encode_pixel_windows_batch(pixel_stack).numpy()  # (N,n_positions,D)

    n_transitions = n_positions - 1
    action_windows = np.stack([
        actions_all[e][:n_transitions * FRAMESKIP].reshape(n_transitions, FRAMESKIP, RAW_ACTION_DIM)
        for e in final_ids
    ])
    print("encoding action embeddings...", flush=True)
    act_emb_stack = encode_action_embeddings_batch(action_windows, pipeline).numpy()  # (N,n_transitions,D)

    latents = {e: emb_stack[i] for i, e in enumerate(final_ids)}
    act_embeddings = {e: act_emb_stack[i] for i, e in enumerate(final_ids)}
    raw_actions = {
        e: pipeline.to_convention(action_windows[i].reshape(-1, RAW_ACTION_DIM).astype(np.float64)).reshape(n_transitions, FRAMESKIP, RAW_ACTION_DIM)
        for i, e in enumerate(final_ids)
    }
    return TrajectoryPool(episode_ids=final_ids, latents=latents, act_embeddings=act_embeddings,
                           raw_actions=raw_actions, pipeline=pipeline)


def flatten_transitions(pool: TrajectoryPool):
    """Returns (Z_t, Z_next, A_emb_t, ep_of_row): flattened across episodes,
    respecting episode boundaries (Z_t[i]->Z_next[i] is always a real
    single-model-step transition within one episode)."""
    zt, znext, aemb, ep_ids = [], [], [], []
    for e in pool.episode_ids:
        lat = pool.latents[e]
        act = pool.act_embeddings[e]
        zt.append(lat[:-1])
        znext.append(lat[1:])
        aemb.append(act)
        ep_ids.extend([e] * (len(lat) - 1))
    return np.concatenate(zt), np.concatenate(znext), np.concatenate(aemb), np.array(ep_ids)
