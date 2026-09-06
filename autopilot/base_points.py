"""Base-point selection for a node's geometry probe. Policies named in v6
S3: `high_W_ratio`, `high_residual`, `diverse_scene`. All draw candidates
from `route_val` only (never retrieval_pool or confirm_*), matching v6 S4.1
("Select base points using only training/retrieval data.").

Operationalisation (not given verbatim by v6, chosen and documented here):
  high_W_ratio  -- cheap dim(W)/dim(T) computed for a larger candidate pool,
                   keep the top N.
  high_residual -- largest true-start one-letter prediction error
                   ||ghat(z,a)-z'||, a model-residual proxy standing in for
                   the FSM-based "abstraction-failure rate" v5/A0 used
                   (there is no FSM in v6).
  diverse_scene -- one candidate per distinct episode, maximising the
                   number of distinct episodes represented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a0_label_repair.horizon_decomposition import real_window_state
from autopilot.geometry import compute_T_V_W
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import (
    DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, encode_pixel_windows_batch,
)

N_CANDIDATE_SCAN = 300
N_BASE_POINTS = 20
N_POSITIONS = 16   # per-candidate-episode context length streamed


def _stream_candidates(episode_ids: list[int], actions: dict, n_positions: int = N_POSITIONS):
    windows = stream_many_windows(episode_ids, num_frames=n_positions, frameskip=FRAMESKIP,
                                     max_workers=16, per_episode_timeout_s=60.0)
    ok = sorted(set(episode_ids) & set(windows.keys()) & set(actions.keys()))
    ok = [e for e in ok if windows[e].shape[0] == n_positions
           and actions[e].shape[0] >= n_positions * FRAMESKIP]
    return ok, windows


def _real_word_converted(pipeline, raw_full: np.ndarray, t: int, length: int = 1) -> list[np.ndarray]:
    """raw_full: (N_POSITIONS, FRAMESKIP, RAW_ACTION_DIM) -- already reshaped
    by position, NOT a flat (n_raw, RAW_ACTION_DIM) array, so this indexes
    by position directly (no second multiplication by FRAMESKIP)."""
    segs = []
    for k in range(length):
        raw_seg = raw_full[t + k]                     # (FRAMESKIP, RAW_ACTION_DIM)
        segs.append(pipeline.to_convention(raw_seg.astype(np.float64)))
    return segs


def select_base_points(policy: str, route_val_ids: list[int], pipeline, model, U_m: np.ndarray,
                          real_latents_pool: np.ndarray, seed: int, n_base_points: int | None = None,
                          suffix_len: int = 1) -> list[dict]:
    n_base_points = n_base_points if n_base_points is not None else N_BASE_POINTS  # module global at
                                                                                      # call time, see retrieval.scan_pool
    rng = np.random.default_rng(seed)
    scan_ids = sorted(rng.choice(route_val_ids, size=min(N_CANDIDATE_SCAN, len(route_val_ids)),
                                    replace=False).tolist())
    n_raw = N_POSITIONS * FRAMESKIP
    actions = load_actions_for_episodes(scan_ids, max_frames=n_raw)
    ok, windows = _stream_candidates(scan_ids, actions)
    if not ok:
        return []

    pixel_stack = np.stack([windows[e] for e in ok])
    emb_stack = encode_pixel_windows_batch(pixel_stack).numpy()   # (N, N_POSITIONS, D)

    t = HISTORY_SIZE - 1   # position with a full HISTORY_SIZE context and a next real letter
    candidates = []
    for i, eid in enumerate(ok):
        raw_full = actions[eid][:n_raw].reshape(N_POSITIONS, FRAMESKIP, RAW_ACTION_DIM)
        act_norm = pipeline(raw_full[:HISTORY_SIZE].reshape(-1, RAW_ACTION_DIM))
        act_flat = act_norm.reshape(1, HISTORY_SIZE, FRAMESKIP * RAW_ACTION_DIM)
        a = torch.from_numpy(act_flat).float().to(DEVICE)
        with torch.no_grad():
            act_emb = model.action_encoder(a)[0].cpu()
        from oracle.lewm_g import LeWMWindowState
        emb = torch.from_numpy(emb_stack[i, :HISTORY_SIZE]).float()
        state0 = LeWMWindowState(emb=emb, act_emb=act_emb)
        z0 = emb_stack[i, HISTORY_SIZE - 1]
        z_next_real = emb_stack[i, HISTORY_SIZE]

        with torch.no_grad():
            pred = model.predict(state0.emb.unsqueeze(0).to(DEVICE), state0.act_emb.unsqueeze(0).to(DEVICE))
            z_pred = pred[0, -1].cpu().numpy()
        residual = float(np.linalg.norm(z_pred - z_next_real))

        candidates.append({"eid": eid, "state0": state0, "z0": z0, "residual": residual, "raw_full": raw_full})

    if policy == "diverse_scene":
        chosen = candidates[:n_base_points]   # already one-per-episode by construction (ok is a set of distinct eids)
    elif policy == "high_residual":
        chosen = sorted(candidates, key=lambda c: -c["residual"])[:n_base_points]
    elif policy == "high_W_ratio":
        scored = []
        for c in candidates:
            word = [_real_word_converted(pipeline, c["raw_full"], t, suffix_len)]
            g = compute_T_V_W(model, c["state0"], c["z0"], word, pipeline, U_m, real_latents_pool,
                                 k_local=128)   # cheap screen: smaller k for speed
            scored.append((g["W_ratio"], c))
        scored.sort(key=lambda x: -x[0])
        chosen = [c for _, c in scored[:n_base_points]]
    else:
        raise ValueError(f"unknown base-point policy {policy!r}")

    return chosen


def attach_full_geometry(base_points: list[dict], model, pipeline, U_m: np.ndarray,
                            real_latents_pool: np.ndarray, k_local: int = 256, suffix_len: int = 1) -> list[dict]:
    """Full-fidelity (T,V,W) per selected base point (k_local=256, per v6
    SS4), attached as bp['geometry'] -- separate from the cheap k=128 screen
    `high_W_ratio` uses internally to rank candidates. suffix_len=2 is the
    "two-letter proxy" branch (v6 SS3, tried only after the one-letter
    family fails at every m -- which is what actually happened, see
    preregistration/autopilot_final_report.json's TREE_EXHAUSTED result)."""
    out = []
    for bp in base_points:
        word = [_real_word_converted(pipeline, bp["raw_full"], HISTORY_SIZE - 1, suffix_len)]
        g = compute_T_V_W(model, bp["state0"], bp["z0"], word, pipeline, U_m, real_latents_pool, k_local=k_local)
        out.append({**bp, "geometry": g})
    return out
