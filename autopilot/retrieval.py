"""v6 SS5: retrieval + arms. Continuous analogue of retrieval/geometric.py's
four filters (that module keys "near" to exact tessellation-cell membership;
there is no tessellation in v6, so "near a base point" is operationalised
as latent distance below a locality radius derived from the SAME 256-NN
neighbourhood already computed for T -- documented here, not given verbatim
by the v6 text). Reuses `greedy_dispersion` unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.geometry import random_subspace_like
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import FRAMESKIP, RAW_ACTION_DIM, encode_pixel_windows_batch
from retrieval.geometric import action_magnitude, greedy_dispersion, longest_run, path_length

N_POOL_SCAN = 500     # smoke-test empirical qual rate ~72% at N=2000 for m=2 (W_ratio~1.0,
                       # thin V_proxy at small m); 500 leaves ample headroom over K=200 while
                       # cutting scan wall-time roughly 4x
N_POSITIONS = 20
ALIGNMENT_MIN = 0.5
DWELL_MIN = 12
CONTAINMENT_MIN = 0.80
EPS = 1e-8


def locality_radius(base_points: list[dict]) -> float:
    """Median distance from each base point to its own 256-NN neighbours --
    the natural local scale "near this base point" should mean, reused
    directly from the geometry already computed for T."""
    dists = []
    for bp in base_points:
        g = bp["geometry"]
        dists.extend(np.linalg.norm(g["neigh_z"] - g["base_z"][None, :], axis=1).tolist())
    return float(np.median(dists)) if dists else 1.0


def scan_pool(pool_episode_ids: list[int], seed: int, n_scan: int | None = None,
                n_positions: int = N_POSITIONS, exclude_used: set[int] | None = None) -> dict:
    n_scan = n_scan if n_scan is not None else N_POOL_SCAN   # module global read at CALL time,
                                                                # not bound as a default at import time,
                                                                # so tests can monkeypatch N_POOL_SCAN
    rng = np.random.default_rng(seed)
    exclude_used = exclude_used or set()
    eligible = [e for e in pool_episode_ids if e not in exclude_used]
    scan_ids = sorted(rng.choice(eligible, size=min(n_scan, len(eligible)), replace=False).tolist())
    n_raw = n_positions * FRAMESKIP
    actions = load_actions_for_episodes(scan_ids, max_frames=n_raw)
    windows = stream_many_windows(scan_ids, num_frames=n_positions, frameskip=FRAMESKIP,
                                     max_workers=16, per_episode_timeout_s=60.0)
    ok = sorted(set(scan_ids) & set(windows.keys()) & set(actions.keys()))
    ok = [e for e in ok if windows[e].shape[0] == n_positions and actions[e].shape[0] >= n_raw]
    print(f"    scan_pool: {len(scan_ids)} sampled, {len(ok)} usable (streamed + have actions)", flush=True)
    if not ok:
        return {"episode_ids": [], "Z": np.zeros((0, n_positions, 1)), "raw": np.zeros((0, n_positions, FRAMESKIP, RAW_ACTION_DIM))}
    pixel_stack = np.stack([windows[e] for e in ok])
    Z = encode_pixel_windows_batch(pixel_stack).numpy()
    raw = np.stack([actions[e][:n_raw].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM) for e in ok])
    return {"episode_ids": ok, "Z": Z, "raw": raw}


def qualify_and_score(scanned: dict, base_points: list[dict], W_or_random_basis_per_bp,
                         radius: float, path_floor: float, action_floor: float) -> dict:
    """W_or_random_basis_per_bp: list of (dim_w, D) bases, one per base
    point, aligned by index with `base_points`. Returns per-episode
    attrition + qualifying episodes with their P_W-projected mean
    displacement (for dispersion selection)."""
    eids, Z, raw = scanned["episode_ids"], scanned["Z"], scanned["raw"]
    base_zs = np.stack([bp["geometry"]["base_z"] for bp in base_points]) if base_points else np.zeros((0, Z.shape[-1] if Z.size else 1))

    attrition = {"considered": len(eids), "locality": 0, "alignment": 0, "non_degenerate": 0, "novelty": len(eids)}
    qualifiers = []
    for i, eid in enumerate(eids):
        z_traj = Z[i]                                  # (n_positions, D)
        if base_zs.shape[0] == 0:
            continue
        d_to_bp = np.linalg.norm(z_traj[:, None, :] - base_zs[None, :, :], axis=-1)   # (n_pos, n_bp)
        nearest_bp = d_to_bp.argmin(axis=1)
        inside = d_to_bp.min(axis=1) <= radius
        containment = float(inside.mean())
        dwell = longest_run(inside)
        if not (containment >= CONTAINMENT_MIN and dwell >= DWELL_MIN):
            continue
        attrition["locality"] += 1

        plen = path_length(z_traj)
        amag = action_magnitude(raw[i])
        if not (plen >= path_floor and amag >= action_floor):
            continue
        attrition["non_degenerate"] += 1

        deltas = np.diff(z_traj, axis=0)                # (n_pos-1, D)
        mean_delta = deltas.mean(axis=0)
        norm_delta = np.linalg.norm(mean_delta)
        # score against the nearest base point's own basis
        bp_idx = int(np.bincount(nearest_bp[inside]).argmax()) if inside.any() else int(nearest_bp[0])
        basis = W_or_random_basis_per_bp[bp_idx]
        if basis.shape[0] == 0 or norm_delta < EPS:
            rho = 0.0
        else:
            proj = basis.T @ (basis @ mean_delta)
            rho = float(np.linalg.norm(proj) / (norm_delta + EPS))
        if rho < ALIGNMENT_MIN:
            continue
        attrition["alignment"] += 1

        qualifiers.append({"eid": eid, "rho": rho, "mean_delta": mean_delta, "proj_delta": basis.T @ (basis @ mean_delta) if basis.shape[0] else mean_delta * 0})

    attrition["selected_pool_before_dispersion"] = len(qualifiers)
    return {"attrition": attrition, "qualifiers": qualifiers}


def select_K(qualifiers: list[dict], K: int, seed: int) -> list[int]:
    if not qualifiers:
        return []
    reps = np.stack([q["proj_delta"] for q in qualifiers])
    idx = greedy_dispersion(reps, K, seed=seed)
    return [qualifiers[i]["eid"] for i in idx]


def build_curriculum(arm: str, base_points: list[dict], pool_episode_ids: list[int], K: int, seed: int,
                        path_floor: float, action_floor: float, used_episodes: set[int]) -> dict:
    """Returns {episode_ids, attrition, angular_coverage}. `arm` in
    {orthogonal, random_dir, random_traj}. `continue` is handled entirely in
    dataset.py (it reads replay_train directly, no retrieval)."""
    radius = locality_radius(base_points)
    scanned = scan_pool(pool_episode_ids, seed=seed, exclude_used=used_episodes)
    if not scanned["episode_ids"]:
        return {"episode_ids": [], "attrition": {"considered": 0}, "angular_coverage": None}

    if arm == "orthogonal":
        bases = [bp["geometry"]["W_basis"] for bp in base_points]
    elif arm == "random_dir":
        bases = [random_subspace_like(bp["geometry"]["W_basis"], bp["geometry"]["T_basis"], seed=seed + i)
                  for i, bp in enumerate(base_points)]
    elif arm == "random_traj":
        rng = np.random.default_rng(seed)
        eligible = []
        for i, eid in enumerate(scanned["episode_ids"]):
            z_traj = scanned["Z"][i]
            plen = path_length(z_traj)
            amag = action_magnitude(scanned["raw"][i])
            if plen >= path_floor and amag >= action_floor:
                eligible.append(eid)
        chosen = sorted(rng.choice(eligible, size=min(K, len(eligible)), replace=False).tolist()) if eligible else []
        return {"episode_ids": chosen, "attrition": {"considered": len(scanned["episode_ids"]),
                                                          "non_degenerate": len(eligible), "selected": len(chosen)},
                  "angular_coverage": None}
    else:
        raise ValueError(f"build_curriculum: unknown arm {arm!r}")

    result = qualify_and_score(scanned, base_points, bases, radius, path_floor, action_floor)
    chosen_eids = select_K(result["qualifiers"], K, seed)
    result["attrition"]["selected"] = len(chosen_eids)

    coverage = None
    if len(chosen_eids) >= 2:
        chosen_dirs = np.stack([q["proj_delta"] for q in result["qualifiers"] if q["eid"] in chosen_eids])
        norms = np.linalg.norm(chosen_dirs, axis=1, keepdims=True)
        norms[norms < EPS] = EPS
        unit = chosen_dirs / norms
        cos_sim = unit @ unit.T
        coverage = float(1 - np.mean(cos_sim[np.triu_indices(len(unit), k=1)])) if len(unit) > 1 else None

    return {"episode_ids": chosen_eids, "attrition": result["attrition"], "angular_coverage": coverage}
