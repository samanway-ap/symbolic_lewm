"""Phase I.2 -- PRIMARY metric: offline plan ranking (plan §12.2). Reframed
from "did the plan succeed" to "can the model's objective identify the
correct action sequence among distractors" -- see §12.1 for why the naive
self-scored CEM-latent-distance proxy is rejected (never used here). Every
score is against a REAL future frame (`z_goal`), never a model rollout
target.

`pred_scale` exists ONLY for §16's metric non-gameability check
(eval/nongameability.py) -- it artificially scales the model's OWN
predicted/candidate embeddings before scoring (z_goal stays at real scale).
Default 1.0 = the actual metric used everywhere else in Phase I.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import (
    ActionPipeline, DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM,
    encode_action_embeddings_batch, encode_pixel_windows_batch, get_model, rollout_batch,
)

K_NEGATIVES = 63
N_EVAL_SEGMENTS = 500
HORIZON_LETTERS = 10  # H=10 only, per this rerun's budget (plan sweeps {5,10,20})


# Phase I evaluates 6 arms x 3 seeds on plan-ranking AND reachability, plus the
# non-gameability check -- ~48 calls to this function, every one of them with the
# SAME (episode_ids, horizon, seed). The streaming/action-loading half is a pure
# function of exactly those three, and streams all 637 held-out episodes, so
# without a cache Phase I would re-decode the entire test split 48 times. Cached
# in-process AND on disk; the disk cache additionally guarantees every arm is
# scored on bit-identical frames rather than on 48 independent HTTP-range
# decodes that could differ if a fetch transiently failed.
_SEG_CACHE: dict = {}
_SEG_CACHE_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _load_or_stream(episode_ids: list[int], horizon: int, n_positions: int, n_raw_needed: int):
    key = (horizon, len(episode_ids), int(sum(episode_ids)))
    if key in _SEG_CACHE:
        return _SEG_CACHE[key]

    disk = _SEG_CACHE_DIR / f"eval_stream_cache_h{horizon}_n{len(episode_ids)}.npz"
    if disk.exists():
        blob = np.load(disk, allow_pickle=False)
        usable = blob["usable"].tolist()
        # Read each member ONCE. NpzFile.__getitem__ re-reads and re-decompresses
        # the whole member array on every access, and every returned view keeps its
        # own fresh parent alive -- so indexing inside the comprehension would
        # allocate len(usable) x 1.43 GiB instead of 1.43 GiB total.
        windows_all = blob["windows"]
        actions_all = blob["actions"]
        windows = {int(e): windows_all[k] for k, e in enumerate(usable)}
        actions = {int(e): actions_all[k] for k, e in enumerate(usable)}
        print(f"  eval segments: loaded {len(usable)} episodes from cache {disk.name}", flush=True)
    else:
        actions_all = load_actions_for_episodes(episode_ids, max_frames=n_raw_needed)
        usable = [e for e in episode_ids if e in actions_all and actions_all[e].shape[0] >= n_raw_needed]
        windows_all = stream_many_windows(usable, num_frames=n_positions, frameskip=FRAMESKIP, max_workers=16)
        usable = sorted(set(usable) & set(windows_all.keys()))
        windows = {int(e): windows_all[e] for e in usable}
        actions = {int(e): actions_all[e][:n_raw_needed] for e in usable}
        np.savez(disk, usable=np.asarray(usable, dtype=np.int64),
                 windows=np.stack([windows[e] for e in usable]),
                 actions=np.stack([actions[e] for e in usable]))
        print(f"  eval segments: streamed {len(usable)} episodes -> cache {disk.name}", flush=True)

    _SEG_CACHE[key] = (usable, windows, actions)
    return usable, windows, actions


def _build_segments(episode_ids: list[int], n_segments: int, horizon: int, seed: int,
                      restrict_episodes: set | None = None):
    """`restrict_episodes` narrows the SCORED episodes without changing the
    cache key: Phase K's LOCAL metric evaluates cell-local subsets, and
    re-streaming per cell (instead of filtering the already-cached full
    held-out set) would cost hours and, worse, score each cell on an
    independently-decoded copy of the same frames."""
    rng = np.random.default_rng(seed + 401)
    n_positions = HISTORY_SIZE + horizon + 1
    n_raw_needed = n_positions * FRAMESKIP

    usable, windows, actions = _load_or_stream(episode_ids, horizon, n_positions, n_raw_needed)
    if restrict_episodes is not None:
        usable = [e for e in usable if int(e) in restrict_episodes]
    n_take = min(n_segments, len(usable))
    if n_take == 0:
        return [], usable, windows, actions, n_positions, n_raw_needed
    chosen = sorted(rng.choice(usable, size=n_take, replace=False).tolist())
    return chosen, usable, windows, actions, n_positions, n_raw_needed


def evaluate_plan_ranking(
    model_state: dict, episode_ids: list[int],
    n_segments: int = N_EVAL_SEGMENTS, horizon: int = HORIZON_LETTERS,
    k_negatives: int = K_NEGATIVES, seed: int = 3072, pred_scale: float = 1.0,
    restrict_episodes: set | None = None,
) -> dict:
    model = get_model()
    model.load_state_dict(model_state)
    model.eval()

    chosen, usable, windows, actions, n_positions, n_raw_needed = _build_segments(
        episode_ids, n_segments, horizon, seed, restrict_episodes=restrict_episodes)
    if not chosen:
        return {"mrr": 0.0, "top1_acc": 0.0, "top5_acc": 0.0, "n_segments": 0, "horizon": horizon}

    all_raw = np.concatenate([actions[e][:n_raw_needed] for e in usable], axis=0)
    pipeline = ActionPipeline.fit(all_raw)

    pix_ctx = np.stack([windows[e][:HISTORY_SIZE] for e in chosen])
    raw_ctx = np.stack([
        actions[e][:n_raw_needed].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)[:HISTORY_SIZE]
        for e in chosen
    ])
    with torch.no_grad():
        init_emb_all = encode_pixel_windows_batch(pix_ctx).to(DEVICE)
        init_act_emb_all = encode_action_embeddings_batch(raw_ctx, pipeline).to(DEVICE)
        pix_goal = np.stack([windows[e][HISTORY_SIZE + horizon - 1] for e in chosen])[:, None]
        z_goal_all = encode_pixel_windows_batch(pix_goal).to(DEVICE)[:, 0]

    rng = np.random.default_rng(seed + 402)
    ranks, top1, top5 = [], [], []
    for idx, eid in enumerate(chosen):
        raw_full = actions[eid][:n_raw_needed].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)
        true_seq = raw_full[HISTORY_SIZE:HISTORY_SIZE + horizon]

        candidates = [true_seq]
        other_eids = [e for e in chosen if e != eid]
        n_other = min(max(1, k_negatives // 3), len(other_eids))
        for oe in rng.choice(other_eids, size=n_other, replace=False):
            raw_o = actions[oe][:n_raw_needed].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)
            candidates.append(raw_o[HISTORY_SIZE:HISTORY_SIZE + horizon])

        max_shift = max(1, n_positions - HISTORY_SIZE - horizon)
        n_shift = min(max(1, k_negatives // 3), 2 * max_shift)
        for _ in range(n_shift):
            shift = int(rng.integers(-max_shift, max_shift + 1))
            start = HISTORY_SIZE + shift
            if start < 0 or start + horizon > n_positions or shift == 0:
                continue
            candidates.append(raw_full[start:start + horizon])

        n_perturb = max(0, (k_negatives + 1) - len(candidates))
        scale = 0.15 * (np.abs(true_seq).mean() + 1e-3)
        for _ in range(n_perturb):
            candidates.append(true_seq + rng.normal(scale=scale, size=true_seq.shape))
        candidates = candidates[:k_negatives + 1]

        cand_arr = np.stack(candidates)  # (n_cand, horizon, FRAMESKIP, 7)
        n_cand = cand_arr.shape[0]
        init_emb_b = init_emb_all[idx:idx + 1].expand(n_cand, -1, -1).contiguous()
        init_act_emb_b = init_act_emb_all[idx:idx + 1].expand(n_cand, -1, -1).contiguous()
        with torch.no_grad():
            trace = rollout_batch(init_emb_b, init_act_emb_b, cand_arr, pipeline)
        final = trace[:, -1].to(DEVICE) * pred_scale
        dist = torch.norm(final - z_goal_all[idx], dim=1).cpu().numpy()
        order = np.argsort(dist)
        rank = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)
        top1.append(int(rank == 1))
        top5.append(int(rank <= 5))

    ranks = np.array(ranks)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "top1_acc": float(np.mean(top1)),
        "top5_acc": float(np.mean(top5)),
        "n_segments": len(ranks), "horizon": horizon, "k_negatives": k_negatives,
        "pred_scale": pred_scale,
    }
