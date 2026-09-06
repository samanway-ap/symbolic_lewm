"""v6 SS6: evaluation. Primary metric E_W(theta) -- normalised projected
one-letter error in the node's FROZEN target subspace, from TRUE cached
starts only (never a reset substitute). Diagnostics (bit fidelity, switch
recall, false-switch rate, balanced accuracy, persistence margin, joint
accuracy) are computed too but are `diagnostic_only: true` -- v5 SS4/SS6:
"No diagnostic-only field may call an A0 repair or cancel a valid training
result."
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a0_label_repair.horizon_decomposition import real_window_state
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM
from training.finetune import _collate

EPS = 1e-8
N_EVAL_POSITIONS = 12


def pool_W_basis(base_points: list[dict], energy: float = 0.95) -> np.ndarray:
    rows = [bp["geometry"]["W_basis"] for bp in base_points if bp["geometry"]["W_basis"].shape[0] > 0]
    if not rows:
        return np.zeros((0, base_points[0]["geometry"]["T_basis"].shape[1])) if base_points else np.zeros((0, 1))
    G = np.concatenate(rows, axis=0)
    _, s, Vt = np.linalg.svd(G, full_matrices=False)
    cum = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
    r = int(np.searchsorted(cum, energy) + 1)
    return Vt[:max(1, r)]


def build_eval_slice(episode_ids: list[int], n_per_episode: int, seed: int,
                        n_positions: int = N_EVAL_POSITIONS) -> dict:
    """Frozen once per node: real (episode, t) transitions -- pixel context
    + real action + real next latent -- built directly, no FSM/reset."""
    rng = np.random.default_rng(seed)
    n_raw = n_positions * FRAMESKIP
    actions = load_actions_for_episodes(episode_ids, max_frames=n_raw)
    windows = stream_many_windows(episode_ids, num_frames=n_positions, frameskip=FRAMESKIP,
                                     max_workers=16, per_episode_timeout_s=60.0)
    ok = sorted(set(episode_ids) & set(windows.keys()) & set(actions.keys()))
    ok = [e for e in ok if windows[e].shape[0] == n_positions and actions[e].shape[0] >= n_raw]

    samples = []
    for eid in ok:
        pix = windows[eid]
        raw_full = actions[eid][:n_raw].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)
        max_t = n_positions - HISTORY_SIZE - 1
        if max_t <= 0:
            continue
        ts = rng.choice(max_t, size=min(n_per_episode, max_t), replace=False)
        for t in ts:
            t = int(t)
            samples.append({"eid": eid, "t": t,
                              "pixels": pix[t:t + HISTORY_SIZE + 1].copy(),
                              "raw_action": raw_full[t:t + HISTORY_SIZE + 1].copy()})
    return {"samples": samples, "episode_ids": ok}


def evaluate_E_W(model, eval_slice: dict, pipeline, W_basis: np.ndarray, U8: np.ndarray, B8: np.ndarray) -> dict:
    """Returns per-episode aggregates for E_W, E_all, and diagnostics, so
    CIs can be recomputed later without re-running the model."""
    model.eval()
    W_t = torch.from_numpy(W_basis).float().to(DEVICE) if W_basis.shape[0] > 0 else None

    # global_guard-style full-latent variance estimate for tr Cov(P_W z_{t+1}),
    # from the eval slice's own real targets (fixed once, not re-estimated per arm)
    by_eid = {}
    with torch.no_grad():
        for s in eval_slice["samples"]:
            batch = _collate([s], pipeline, DEVICE)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            info = model.encode(batch)
            emb, act_emb = info["emb"], info["act_emb"]
            ctx_emb, ctx_act = emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE]
            tgt_emb = emb[:, 1:]
            pred_emb = model.predict(ctx_emb, ctx_act)

            err = (pred_emb - tgt_emb)[0, -1]        # (D,)
            real = tgt_emb[0, -1]
            pred = pred_emb[0, -1]

            e_all = float(err.pow(2).mean().item())
            if W_t is not None and W_t.shape[0] > 0:
                err_w = W_t @ err
                e_w_num = float(err_w.pow(2).sum().item())
            else:
                e_w_num = float(err.pow(2).sum().item())

            z_real_np = real.cpu().numpy()
            z_pred_np = pred.cpu().numpy()
            bits_real = (z_real_np @ U8.T) > B8
            bits_pred = (z_pred_np @ U8.T) > B8

            by_eid.setdefault(s["eid"], []).append({
                "e_all": e_all, "e_w_num": e_w_num,
                "bits_match": (bits_real == bits_pred).tolist(), "joint_match": bool((bits_real == bits_pred).all()),
                "z_real_w": (W_t.cpu().numpy() @ z_real_np) if W_t is not None and W_t.shape[0] > 0 else z_real_np,
            })

    all_zw = np.concatenate([np.stack([r["z_real_w"] for r in v]) for v in by_eid.values()]) if by_eid else np.zeros((0, 1))
    tr_cov_w = float(np.trace(np.cov(all_zw.T))) if all_zw.shape[0] > 1 else 1.0
    denom = tr_cov_w + EPS

    per_episode = {}
    for eid, recs in by_eid.items():
        e_w_vals = [r["e_w_num"] / denom for r in recs]
        e_all_vals = [r["e_all"] for r in recs]
        per_episode[eid] = {
            "E_W": float(np.mean(e_w_vals)), "E_all": float(np.mean(e_all_vals)),
            "joint_accuracy": float(np.mean([r["joint_match"] for r in recs])),
            "bit_fidelity": np.mean([r["bits_match"] for r in recs], axis=0).tolist(),
            "n_samples": len(recs),
        }
    return {"per_episode": per_episode, "tr_cov_w": tr_cov_w}


def assert_slice_nonempty(eval_slice: dict, name: str, min_episodes: int = 1) -> None:
    """Bug fix, code audit 2026-09-06 (P1-8): an empty or degenerate eval/
    guard slice used to fail SILENTLY -- `evaluate_E_W` returns
    `per_episode={}`, and every downstream mean-over-empty-dict collapses to
    a default (0.0 or a `np.zeros(1)` placeholder) that then flows straight
    into gates and gains as if it were a real, informative measurement.
    Call this right after building a slice so a starved slice aborts the run
    instead of silently producing a meaningless pass/fail."""
    n = len(eval_slice.get("episode_ids", []))
    if n < min_episodes:
        raise RuntimeError(
            f"IMPLEMENTATION_FAILURE: {name} has only {n} episode(s) (< min_episodes={min_episodes}) -- "
            f"failing closed rather than silently evaluating on a degenerate slice")


def paired_bootstrap_ci(vals_a: np.ndarray, vals_b: np.ndarray, seed: int, n_boot: int = 1000):
    rng = np.random.default_rng(seed)
    n = len(vals_a)
    diffs = vals_a - vals_b
    if n == 0:
        return 0.0, 0.0, 0.0
    boots = np.array([diffs[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return float(diffs.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
