"""v7 SS6: E_N evaluation. The metric itself is IDENTICAL in form to the
completed v6 tree's E_W (normalised projected one-step error) --
autopilot/evaluate.py's `evaluate_E_W` is already fully generic over the
target basis, so it is reused UNCHANGED, called with B_N (or W_all, or the
node's shared cell basis -- evaluation always uses the SAME frozen basis
across all arms; only training curricula differ) instead of W_proxy. What is
new here is building the eval slice ITSELF: it must be restricted to real
route_val transitions that actually match the node's (region, action_letter)
cell, which the completed v6 tree never needed (it evaluated on an
unconditional route_val sample). Output format matches
autopilot/evaluate.py's `build_eval_slice` exactly (`{"samples": [...],
"episode_ids": [...]}`, each sample `{"eid","t","pixels","raw_action"}` with
RAW (unconverted) actions -- conversion happens downstream in `_collate`),
so `evaluate_E_W` needs no changes at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.need_common import AlphabetLookup  # noqa: E402
from autopilot.need_geometry import N_POSITIONS, assign_region  # noqa: E402
from oracle.droid_actions import load_actions_for_episodes  # noqa: E402
from oracle.droid_streaming import stream_many_windows  # noqa: E402
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, encode_pixel_windows_batch  # noqa: E402

N_CELL_EVAL_SCAN = 400


def build_cell_eval_slice(episode_ids: list[int], region_id: int, action_letter: str, anchors: np.ndarray,
                             lever0_basis: dict, pipeline: ActionPipeline, alphabet: AlphabetLookup, seed: int,
                             n_per_episode: int = 4, n_scan: int = N_CELL_EVAL_SCAN,
                             n_positions: int = N_POSITIONS, model=None) -> dict:
    """Frozen once per node (reused for pre/post-training eval within every
    stage, exactly like the completed v6 tree's `eval_slice`).

    `model` defaults to None (encode_pixel_windows_batch's own epoch-20
    fallback). Bug fix, code audit 2026-09-06 (TWO_ARM_EXPERIMENT_REQUIRED_
    CHANGES.md Change A): callers evaluating an epoch-7-trained model MUST
    pass it explicitly, so region assignment (which anchors/lever0_basis
    were themselves built in epoch-7 coordinates for) and the eval targets
    this function encodes are computed in the SAME coordinate system."""
    rng = np.random.default_rng(seed)
    scan_ids = sorted(rng.choice(episode_ids, size=min(n_scan, len(episode_ids)), replace=False).tolist())
    n_raw = n_positions * FRAMESKIP
    actions = load_actions_for_episodes(scan_ids, max_frames=n_raw)
    windows = stream_many_windows(scan_ids, num_frames=n_positions, frameskip=FRAMESKIP,
                                     max_workers=16, per_episode_timeout_s=60.0)
    ok = sorted(set(scan_ids) & set(windows.keys()) & set(actions.keys()))
    ok = [e for e in ok if windows[e].shape[0] == n_positions and actions[e].shape[0] >= n_raw]
    if not ok:
        return {"samples": [], "episode_ids": []}
    pixel_stack = np.stack([windows[e] for e in ok])
    emb_stack = encode_pixel_windows_batch(pixel_stack, model=model).numpy()

    samples = []
    used_eids = []
    for i, eid in enumerate(ok):
        pix = windows[eid]
        raw_full = actions[eid][:n_raw].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)
        cand_ts = []
        for t in range(HISTORY_SIZE - 1, n_positions - 1):
            z0 = emb_stack[i, t][None, :]
            region = int(assign_region(z0, anchors, lever0_basis)[0])
            if region != region_id:
                continue
            raw_seg_converted = pipeline.to_convention(raw_full[t].astype(np.float64))
            if alphabet.letter(raw_seg_converted) != action_letter:
                continue
            cand_ts.append(t)
        if not cand_ts:
            continue
        chosen = rng.choice(cand_ts, size=min(n_per_episode, len(cand_ts)), replace=False)
        for t in chosen:
            t = int(t)
            t_start = t - HISTORY_SIZE + 1
            samples.append({"eid": int(eid), "t": t_start,
                              "pixels": pix[t_start:t_start + HISTORY_SIZE + 1].copy(),
                              "raw_action": raw_full[t_start:t_start + HISTORY_SIZE + 1].copy()})
        used_eids.append(int(eid))
    return {"samples": samples, "episode_ids": used_eids}


def full_latent_predictions(model, slice_: dict, pipeline: ActionPipeline) -> np.ndarray:
    """Stacked one-step predicted embeddings on a frozen slice -- the raw
    material for the effective-rank forgetting check (SS6: "lowers effective
    rank by more than 10%"), which `evaluate.evaluate_E_W` doesn't expose
    (it only returns scalar per-episode aggregates)."""
    import torch
    from oracle.lewm_g import DEVICE
    from training.finetune import _collate
    model.eval()
    preds = []
    with torch.no_grad():
        for s in slice_["samples"]:
            batch = _collate([s], pipeline, DEVICE)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            info = model.encode(batch)
            emb, act_emb = info["emb"], info["act_emb"]
            pred_emb = model.predict(emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE])
            preds.append(pred_emb[0, -1].cpu().numpy())
    return np.stack(preds) if preds else np.zeros((0, 1))


def effective_rank(Z: np.ndarray) -> float:
    """Participation ratio (sum(eig)^2 / sum(eig^2)) of cov(Z) --
    operationalisation of "effective rank," not given verbatim by v7."""
    if Z.shape[0] < 2:
        return 0.0
    eigvals = np.linalg.eigvalsh(np.cov(Z.T))
    eigvals = np.clip(eigvals, 0, None)
    s = eigvals.sum()
    if s < 1e-12:
        return 0.0
    return float(s ** 2 / (eigvals ** 2).sum())
