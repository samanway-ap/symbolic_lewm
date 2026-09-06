"""v7 SS3: regions, action-conditioned support, and the cheap N(r,a,1)
construction. Builds directly on autopilot/geometry.py's score_grad_bank /
local_tangent (reused unchanged) and base_points.py's streaming pattern, but
keys everything to a discrete (region, action_letter) cell instead of a
single base point + fixed U_m word.

Operationalisations chosen here (not given verbatim by v7, documented per
its own convention -- see base_points.py's docstring for precedent):
  - "near a region anchor" for SUPPORT/gradient-start purposes reuses plain
    Euclidean local_tangent kNN (as the completed v6 tree did); only region
    ASSIGNMENT (which anchor a transition belongs to) uses the Lever-0
    residualized metric, per SS3.1's explicit instruction.
  - "normalized full-latent residual energy" for the pre-gradient cell
    ranking (SS3.3) is normalized by ONE global trace(Cov(z_next)) computed
    over the whole scanned replay_train pool, so cells are comparable on a
    common scale without a second full-pool pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.common import stable_seed  # noqa: E402
from autopilot.geometry import intersect_tangent_with_Vperp, local_tangent, score_grad_bank  # noqa: E402
from autopilot.need_common import AlphabetLookup, project_residualized  # noqa: E402
from oracle.droid_actions import load_actions_for_episodes  # noqa: E402
from oracle.droid_streaming import stream_many_windows  # noqa: E402
from retrieval.geometric import action_magnitude, path_length  # noqa: E402
from oracle.lewm_g import (  # noqa: E402
    DEVICE, FRAMESKIP, HISTORY_SIZE, LeWMWindowState, RAW_ACTION_DIM, encode_pixel_windows_batch,
)

N_POSITIONS = 20
N_SUPPORT_SCAN_REPLAY = 600
N_SUPPORT_SCAN_ROUTE = 400
MIN_FIT_TRANSITIONS = 64
MIN_FIT_EPISODES = 8
MIN_ROUTE_TRANSITIONS = 32
MIN_ROUTE_EPISODES = 4
MIN_REGIONS = 6
MAX_REGIONS = 8
GRADIENT_STARTS_MAX = 64
GRADIENT_STARTS_MIN = 32
PREGRADIENT_CELL_CAP = 12
PREGRADIENT_PER_REGION_CAP = 3
TANGENT_ENERGY = 0.95
GRADIENT_ENERGY = 0.95
NEED_ENERGY = 0.80
NEED_RANK_MIN, NEED_RANK_MAX = 1, 4
CANDIDATE_CELLS = 3
M_ACTION_DIRS = 4   # fixed m=4, per v7's own "SCRAPPED: search over m" ruling


def farthest_point_anchors(pool_z: np.ndarray, lever0_basis: dict, n_anchors: int, seed: int) -> np.ndarray:
    """Deterministic farthest-point sampling on `pool_z` (real replay_train
    latents) in the Lever-0 residualized metric. Licensed fallback per v7
    SS3.1 / loop v6 SS3.1: no persisted v6 base-point catalogue survives
    (none was ever written to disk by the completed C00-C06 tree), so all
    `n_anchors` anchors come from this mechanical supplement rather than 0
    "surviving" ones -- logged as a scientifically-neutral adjustment."""
    proj = project_residualized(lever0_basis, pool_z)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(proj)))
    chosen = [first]
    d2 = ((proj - proj[first]) ** 2).sum(axis=1)
    for _ in range(n_anchors - 1):
        nxt = int(np.argmax(d2))
        chosen.append(nxt)
        d2 = np.minimum(d2, ((proj - proj[nxt]) ** 2).sum(axis=1))
    return pool_z[chosen]


def assign_region(z: np.ndarray, anchors: np.ndarray, lever0_basis: dict) -> np.ndarray:
    """z: (N, D) -> (N,) nearest-anchor index, distance in the residualized metric."""
    zp = project_residualized(lever0_basis, z)
    ap = project_residualized(lever0_basis, anchors)
    d2 = ((zp[:, None, :] - ap[None, :, :]) ** 2).sum(axis=-1)
    return d2.argmin(axis=1)


def scan_transitions(episode_ids: list[int], seed: int, n_scan: int, pipeline, model, alphabet: AlphabetLookup,
                        n_positions: int = N_POSITIONS) -> list[dict]:
    """Streams `n_scan` episodes and returns EVERY valid (t -> t+1) transition
    from each (not just one per episode) -- reuses the same per-episode
    streaming cost the old pipeline paid for a single transition, per
    common.py's efficiency precedent (build_node_datasets)."""
    rng = np.random.default_rng(seed)
    scan_ids = sorted(rng.choice(episode_ids, size=min(n_scan, len(episode_ids)), replace=False).tolist())
    n_raw = n_positions * FRAMESKIP
    actions = load_actions_for_episodes(scan_ids, max_frames=n_raw)
    windows = stream_many_windows(scan_ids, num_frames=n_positions, frameskip=FRAMESKIP,
                                     max_workers=16, per_episode_timeout_s=60.0)
    ok = sorted(set(scan_ids) & set(windows.keys()) & set(actions.keys()))
    ok = [e for e in ok if windows[e].shape[0] == n_positions and actions[e].shape[0] >= n_raw]
    if not ok:
        return []
    pixel_stack = np.stack([windows[e] for e in ok])
    emb_stack = encode_pixel_windows_batch(pixel_stack).numpy()   # (N, n_positions, D)

    records = []
    for i, eid in enumerate(ok):
        raw_full = actions[eid][:n_raw].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)
        # per-EPISODE (not per-transition) non-degeneracy signal: the fixed
        # PATH_FLOOR/ACTION_FLOOR in need_retrieval.py were carried over from
        # the completed v6 tree, which measured them over a full ~20-position
        # window -- a single (t -> t+1) transition's path length would almost
        # always read as near-zero against that same floor, so both floors
        # are checked against the WHOLE streamed window, once per episode.
        ep_path_length = path_length(emb_stack[i])
        ep_action_magnitude = action_magnitude(raw_full)
        for t in range(HISTORY_SIZE - 1, n_positions - 1):
            ctx_raw = raw_full[t - HISTORY_SIZE + 1:t + 1]                      # (HISTORY_SIZE, FRAMESKIP, RAW_ACTION_DIM)
            act_norm = pipeline(ctx_raw.reshape(-1, RAW_ACTION_DIM))
            act_flat = act_norm.reshape(1, HISTORY_SIZE, FRAMESKIP * RAW_ACTION_DIM)
            a = torch.from_numpy(act_flat).float().to(DEVICE)
            with torch.no_grad():
                act_emb = model.action_encoder(a)[0].cpu()
            emb = torch.from_numpy(emb_stack[i, t - HISTORY_SIZE + 1:t + 1]).float()
            state0 = LeWMWindowState(emb=emb, act_emb=act_emb)
            z0 = emb_stack[i, t]
            z_next = emb_stack[i, t + 1]
            with torch.no_grad():
                pred = model.predict(state0.emb.unsqueeze(0).to(DEVICE), state0.act_emb.unsqueeze(0).to(DEVICE))
                z_pred = pred[0, -1].cpu().numpy()
            raw_seg_converted = pipeline.to_convention(raw_full[t].astype(np.float64))
            records.append({
                "eid": int(eid), "t": int(t), "z0": z0, "z_next": z_next, "state0": state0,
                "raw_seg_converted": raw_seg_converted, "letter": alphabet.letter(raw_seg_converted),
                "residual": z_pred - z_next,
                "episode_path_length": ep_path_length, "episode_action_magnitude": ep_action_magnitude,
            })
    return records


def assign_and_count(records: list[dict], anchors: np.ndarray, lever0_basis: dict) -> None:
    """Attaches `region` to every record in-place."""
    if not records:
        return
    z0 = np.stack([r["z0"] for r in records])
    regions = assign_region(z0, anchors, lever0_basis)
    for r, reg in zip(records, regions):
        r["region"] = int(reg)


def support_table(records: list[dict]) -> dict:
    """(region, letter) -> {transitions, episodes}."""
    table: dict[tuple, dict] = {}
    for r in records:
        key = (r["region"], r["letter"])
        e = table.setdefault(key, {"transitions": 0, "episodes": set()})
        e["transitions"] += 1
        e["episodes"].add(r["eid"])
    return {k: {"transitions": v["transitions"], "n_episodes": len(v["episodes"])} for k, v in table.items()}


def build_cell_gradient_basis(records_for_cell: list[dict], model, pipeline, U_m: np.ndarray, seed: int) -> dict:
    """SS3.2: V(r,a,1) from real-word score gradients at up to
    GRADIENT_STARTS_MAX true starts, episode-balanced. Reduces toward
    GRADIENT_STARTS_MIN only under the E0 timing rule (loop v6 SS7.1) --
    handled by the caller passing a smaller n_starts."""
    rng = np.random.default_rng(seed)
    by_eid: dict[int, list[dict]] = {}
    for r in records_for_cell:
        by_eid.setdefault(r["eid"], []).append(r)
    eids = sorted(by_eid.keys())
    rng.shuffle(eids)

    chosen = []
    i = 0
    while len(chosen) < GRADIENT_STARTS_MAX and eids:
        eid = eids[i % len(eids)]
        pool = by_eid[eid]
        chosen.append(pool[int(rng.integers(0, len(pool)))])
        i += 1
        if i >= len(eids) * 4:   # exhausted reasonable balanced draws
            break

    grad_rows = []
    for rec in chosen:
        word = [rec["raw_seg_converted"]]
        grad_rows.extend(score_grad_bank(model, rec["state0"], word, pipeline, U_m))

    d = U_m.shape[1]
    if not grad_rows:
        return {"V_basis": np.zeros((0, d)), "n_starts": len(chosen), "n_episodes": len(set(r["eid"] for r in chosen))}
    G = np.stack(grad_rows)
    _, s, Vt = np.linalg.svd(G, full_matrices=False)
    cum = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
    r_ = int(np.searchsorted(cum, GRADIENT_ENERGY) + 1)
    return {"V_basis": Vt[:max(1, r_)], "n_starts": len(chosen), "n_episodes": len(set(r["eid"] for r in chosen))}


def project_T_into_Vperp(T_basis: np.ndarray, V_basis: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Name kept for call-site compatibility; computes the actual
    intersection T \\cap V^perp now (see `intersect_tangent_with_Vperp`'s
    docstring for the bug this replaced -- the old body here projected T's
    rows into V^perp and orthonormalized the projected IMAGE, which is a
    different, generally larger subspace than the true intersection)."""
    return intersect_tangent_with_Vperp(T_basis, V_basis, rtol=tol)


def build_need_basis(records_for_cell: list[dict], W_basis: np.ndarray) -> dict:
    """SS3.3: error-weighted N(r,a,1) via thin SVD of B_W^T e_j -- no d x d
    covariance ever formed.

    Uses the UNCENTERED second moment E[ee^T] of the projected residuals,
    not the centered covariance (bug fix, code audit 2026-09-06): if the
    frozen model consistently under/over-predicts along some direction, that
    MEAN residual is exactly the systematic, learnable error the need space
    is supposed to find -- the previous `X - X.mean(...)` centering step
    deleted it before the SVD ever saw it, keeping only directions of
    residual VARIANCE around that (removed) mean. `mean_bias_energy` is
    reported separately as a diagnostic: the fraction of total residual
    energy that was concentrated in the mean (large values mean the fix
    changes B_N a lot relative to the old centered version)."""
    d = W_basis.shape[1] if W_basis.ndim == 2 else 0
    if W_basis.shape[0] == 0 or not records_for_cell:
        return {"B_N": np.zeros((0, d)), "q": 0, "eta_c": 0.0, "energy_zero": True, "mean_bias_energy_frac": 0.0}

    E = np.stack([r["residual"] for r in records_for_cell])           # (n, D)
    Znext = np.stack([r["z_next"] for r in records_for_cell])          # (n, D)
    X = E @ W_basis.T                                                    # (n, dim_W)
    if X.shape[0] < 2 or not np.any(np.abs(X) > 1e-12):
        return {"B_N": np.zeros((0, d)), "q": 0, "eta_c": 0.0, "energy_zero": True, "mean_bias_energy_frac": 0.0}

    mean_bias_energy_frac = float((X.mean(axis=0) ** 2).sum() / max((X ** 2).sum() / X.shape[0], 1e-12))

    _, s, Rt = np.linalg.svd(X, full_matrices=False)
    cum = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
    q = int(np.searchsorted(cum, NEED_ENERGY) + 1)
    q = int(np.clip(q, NEED_RANK_MIN, min(NEED_RANK_MAX, W_basis.shape[0])))
    B_N = Rt[:q] @ W_basis                                               # (q, D)

    zw = Znext @ W_basis.T
    tr_cov_w = float(np.trace(np.cov(zw.T))) if zw.shape[0] > 1 else 1.0
    eta_c = float((X ** 2).sum() / X.shape[0] / max(tr_cov_w, 1e-12))
    return {"B_N": B_N, "q": q, "eta_c": eta_c, "energy_zero": False,
              "mean_bias_energy_frac": mean_bias_energy_frac}


def random_subspace_inside(container_basis: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """`random_dir` arm per v7 SS5: a seeded, dimension-`dim` (== q, the
    conditional cell's own need-space rank) random subspace INSIDE
    `container_basis` (== W(r,a,1), not T -- v7's redefinition of the
    completed v6 tree's `random_dir`, which lived inside T instead)."""
    rng = np.random.default_rng(seed)
    dim_c = container_basis.shape[0]
    d = container_basis.shape[1] if container_basis.ndim == 2 else 0
    if dim <= 0 or dim_c == 0:
        return np.zeros((0, d))
    dim = min(dim, dim_c)
    coeffs = rng.normal(size=(dim_c, dim))
    raw = container_basis.T @ coeffs
    Q, _ = np.linalg.qr(raw)
    return Q.T[:dim]


def build_cell_geometry(records_for_cell: list[dict], anchor_z: np.ndarray,
                           real_latents_pool: np.ndarray, model, pipeline, U_m: np.ndarray, seed: int,
                           k_local: int = 256) -> dict | None:
    """Full T/V/W/N for one (region, action) cell."""
    T_basis, neigh, neigh_idx = local_tangent(anchor_z, real_latents_pool, k_local, TANGENT_ENERGY)
    grad = build_cell_gradient_basis(records_for_cell, model, pipeline, U_m, seed)
    W_basis = project_T_into_Vperp(T_basis, grad["V_basis"])
    if W_basis.shape[0] == 0:
        return None
    need = build_need_basis(records_for_cell, W_basis)
    if need["energy_zero"] or need["B_N"].shape[0] == 0:
        return None
    return {"T_basis": T_basis, "V_basis": grad["V_basis"], "W_basis": W_basis, "B_N": need["B_N"],
            "dim_T": int(T_basis.shape[0]), "dim_V": int(grad["V_basis"].shape[0]),
            "dim_W": int(W_basis.shape[0]), "dim_N": int(need["B_N"].shape[0]),
            "q": need["q"], "eta_c": need["eta_c"],
            "gradient_starts": grad["n_starts"], "gradient_episodes": grad["n_episodes"]}


def build_global_W_control(region_id: int, sibling_action_records: dict[str, list[dict]], anchor_z: np.ndarray,
                              real_latents_pool: np.ndarray, model, pipeline, U_m: np.ndarray, seed: int,
                              q_cap: int, k_local: int = 256) -> dict:
    """SS3.3 tail: global_W = T_r \\cap (span_a' V(r,a',1))^perp, action-pooled
    across every supported action in the region (not just the candidate's
    own action), same projected-residual SVD/rank rule, capped by q_cap
    (the conditional cell's own q, per v7's "capped by available dimension")."""
    T_basis, _, _ = local_tangent(anchor_z, real_latents_pool, k_local, TANGENT_ENERGY)
    V_rows = []
    for a_letter, recs in sibling_action_records.items():
        grad = build_cell_gradient_basis(recs, model, pipeline, U_m, stable_seed(seed, a_letter))
        if grad["V_basis"].shape[0] > 0:
            V_rows.append(grad["V_basis"])

    if not V_rows:
        return {"W_all_basis": np.zeros((0, T_basis.shape[1])), "empty": True}

    V_all = np.concatenate(V_rows, axis=0)
    _, s, Vt = np.linalg.svd(V_all, full_matrices=False)
    cum = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
    r_ = int(np.searchsorted(cum, GRADIENT_ENERGY) + 1)
    V_all_basis = Vt[:max(1, r_)]
    W_all = project_T_into_Vperp(T_basis, V_all_basis)
    if W_all.shape[0] == 0:
        return {"W_all_basis": np.zeros((0, T_basis.shape[1])), "empty": True}

    all_recs = [r for recs in sibling_action_records.values() for r in recs]
    need = build_need_basis(all_recs, W_all)
    if need["energy_zero"] or need["B_N"].shape[0] == 0:
        return {"W_all_basis": W_all, "empty": True}
    q = min(need["q"], q_cap, W_all.shape[0])
    B_N_all = need["B_N"][:q]
    return {"W_all_basis": W_all, "B_N_all": B_N_all, "q": q, "empty": False}
