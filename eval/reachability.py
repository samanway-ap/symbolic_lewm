"""Phase I.3 -- secondary metric (reported, not optimised): reachability
AUC. Given (zeta_t, z_g), discriminate reachable-within-H (real positives:
the model's own predicted rollout under the TRUE action sequence, scored
against the REAL frame it actually reached) from not (the same rollout
scored against a real frame from a DIFFERENT episode -- not reachable from
zeta_t by construction). Tests the implicit reachability structure a
planner depends on, per plan §12.3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.plan_ranking import _build_segments
from oracle.lewm_g import (
    ActionPipeline, DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM,
    encode_action_embeddings_batch, encode_pixel_windows_batch, get_model, rollout_batch,
)


def evaluate_reachability(model_state: dict, episode_ids: list[int], n_segments: int = 200,
                            horizon: int = 10, seed: int = 3072) -> dict:
    from sklearn.metrics import roc_auc_score

    model = get_model()
    model.load_state_dict(model_state)
    model.eval()

    chosen, usable, windows, actions, n_positions, n_raw_needed = _build_segments(
        episode_ids, n_segments, horizon, seed)
    if len(chosen) < 2:
        return {"auc": 0.5, "n_segments": len(chosen)}

    all_raw = np.concatenate([actions[e][:n_raw_needed] for e in usable], axis=0)
    pipeline = ActionPipeline.fit(all_raw)

    pix_ctx = np.stack([windows[e][:HISTORY_SIZE] for e in chosen])
    raw_ctx = np.stack([
        actions[e][:n_raw_needed].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)[:HISTORY_SIZE]
        for e in chosen
    ])
    raw_true = np.stack([
        actions[e][:n_raw_needed].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)[HISTORY_SIZE:HISTORY_SIZE + horizon]
        for e in chosen
    ])
    with torch.no_grad():
        init_emb = encode_pixel_windows_batch(pix_ctx).to(DEVICE)
        init_act_emb = encode_action_embeddings_batch(raw_ctx, pipeline).to(DEVICE)
        pix_goal = np.stack([windows[e][HISTORY_SIZE + horizon - 1] for e in chosen])[:, None]
        z_goal = encode_pixel_windows_batch(pix_goal).to(DEVICE)[:, 0]  # (N,D), real frames

        trace = rollout_batch(init_emb, init_act_emb, raw_true, pipeline)
        # rollout_batch returns a CPU tensor; z_goal lives on DEVICE. plan_ranking.py
        # already does this .to(DEVICE) -- this file was missing it.
        pred_final = trace[:, -1].to(DEVICE)  # (N,D), model's own predicted rollout under TRUE actions

    n = len(chosen)
    rng = np.random.default_rng(seed + 501)
    shift = rng.permutation(n)
    shift = np.array([s if s != i else (s + 1) % n for i, s in enumerate(shift)])  # ensure no self-pairing

    pos_dist = torch.norm(pred_final - z_goal, dim=1).cpu().numpy()
    neg_dist = torch.norm(pred_final - z_goal[shift], dim=1).cpu().numpy()

    scores = np.concatenate([-pos_dist, -neg_dist])  # higher score = "more reachable"
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    auc = float(roc_auc_score(labels, scores))
    return {"auc": auc, "n_segments": n, "horizon": horizon}
