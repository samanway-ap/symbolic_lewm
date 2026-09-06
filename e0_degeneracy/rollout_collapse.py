"""E0.1 (rollout collapse) + E0.2 (reachable cell count), from ONE shared
batch of random-word rollouts from a fixed base point z0.

Design choice (pre-registered): 500 words of length 24, sampled ONCE, rolled
out autoregressively for the full 24 steps in one batched call; every prefix
length ell in 1..24 is read off the SAME trace by slicing at position
ell-1, rather than redrawing 200 independent words per length. This tracks
how the SAME trajectories evolve as they lengthen -- which is what "decays
toward zero with word length" means -- and costs one rollout, not 24.

Word symbols are drawn from the k*=2 action alphabet's medoid segments,
which are stored in droid_100 CONVENTION already -- rolled out via
`oracle.lewm_g.rollout_batch_from_convention` (normalizer-only), never the
full `rollout_batch` (which would double-convert).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import ActionPipeline, DEVICE, get_model, rollout_batch_from_convention
from oracle.production_oracle import build_reset_windows, load_alphabet_symbols

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
N_WORDS = 500
MAX_LEN = 24
SEED = 3072
BASE_RESET = "r0"


def _fit_pipeline(seed: int = SEED) -> ActionPipeline:
    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    train_pool = sorted(set(split["train_episode_ids"]) - reset_ids)
    rng = np.random.default_rng(seed)
    fit_ids = sorted(rng.choice(train_pool, size=min(200, len(train_pool)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    all_raw = np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0)
    return ActionPipeline.fit(all_raw)


def sample_random_words(symbols_dict: dict[str, np.ndarray], n_words: int, max_len: int, seed: int):
    names = list(symbols_dict.keys())
    segs = np.stack([symbols_dict[n] for n in names])   # (n_symbols, FRAMESKIP, 7)
    rng = np.random.default_rng(seed + 909)
    idx = rng.integers(0, len(names), size=(n_words, max_len))
    raw = segs[idx]   # (n_words, max_len, FRAMESKIP, 7)
    return raw, idx, names


def run(model=None, base_reset: str = BASE_RESET, n_words: int = N_WORDS,
         max_len: int = MAX_LEN, seed: int = SEED) -> dict:
    model = model or get_model()
    pipeline = _fit_pipeline(seed)
    reset_windows = build_reset_windows(pipeline)
    if base_reset not in reset_windows:
        base_reset = sorted(reset_windows.keys())[0]
    z0 = reset_windows[base_reset]

    symbols_dict = load_alphabet_symbols(ALPHABET_PATH)
    raw, sym_idx, sym_names = sample_random_words(symbols_dict, n_words, max_len, seed)

    init_emb = z0.emb.unsqueeze(0).expand(n_words, -1, -1).clone()
    init_act = z0.act_emb.unsqueeze(0).expand(n_words, -1, -1).clone()
    trace = rollout_batch_from_convention(init_emb, init_act, raw, pipeline, model=model).numpy()
    # trace: (n_words, max_len, D) -- trace[:, ell-1, :] is the state after `ell` symbols

    return {"trace": trace, "sym_idx": sym_idx, "sym_names": sym_names,
             "z0": z0.emb[-1].numpy(), "base_reset": base_reset, "pipeline": pipeline}


def collapse_metrics(trace: np.ndarray, global_mean: np.ndarray) -> dict:
    """trace: (n_words, max_len, D). Returns per-ell total variance (trace of
    the cross-word covariance) and mean distance from the global latent mean,
    for ell = 1..max_len."""
    n_words, max_len, D = trace.shape
    total_var, mean_dist, mean_pairwise = [], [], []
    for ell in range(1, max_len + 1):
        pts = trace[:, ell - 1, :]                       # (n_words, D)
        total_var.append(float(pts.var(axis=0).sum()))
        mean_dist.append(float(np.linalg.norm(pts - global_mean[None, :], axis=1).mean()))
        # mean pairwise distance on a subsample -- O(n^2), keep n small
        sub = pts[:100]
        d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        mean_pairwise.append(float(d[np.triu_indices(len(sub), k=1)].mean()))
    return {"ell": list(range(1, max_len + 1)), "total_variance": total_var,
             "mean_dist_from_global_mean": mean_dist, "mean_pairwise_distance": mean_pairwise}


def reachable_cells(trace: np.ndarray, z0: np.ndarray, tess_params_path: Path, m_values=(2, 4, 6, 8)) -> dict:
    """Union of tessellation cell ids visited across the whole (word, step)
    grid, at each swept m -- compared against occupied-cell counts already
    measured on real data (tessellation.json)."""
    from tessellation.hyperplanes import Tessellation
    tp = np.load(tess_params_path)
    flat = np.concatenate([z0[None, :], trace.reshape(-1, trace.shape[-1])], axis=0)
    out = {}
    for m in m_values:
        t = Tessellation(U=tp[f"U_m{m}"], B=tp[f"B_m{m}"], m=m, occupied={}, seed=SEED)
        ids = t.cell_ids(flat)
        out[str(m)] = {"n_reachable": int(len(np.unique(ids))), "n_nominal": 2 ** m}
    return out


def reachable_machine_states(sym_idx: np.ndarray, sym_names: list[str], machine, base_reset: str) -> dict:
    """How many of the discovered machine's states are reached from z0 under
    this same word set (the actual SYMBOLS used, not the resulting labels --
    the machine transitions on symbols) -- the Moore-counting-bound sanity
    check applied to the CURRENT (already known degenerate) machine,
    essentially free since the word set already exists."""
    n_words, max_len = sym_idx.shape
    visited = set()
    for i in range(n_words):
        machine.reset_to_initial()
        machine.step(base_reset)
        visited.add(machine.current_state.state_id)
        for t in range(max_len):
            machine.step(sym_names[sym_idx[i, t]])
            visited.add(machine.current_state.state_id)
    return {"n_visited": len(visited), "n_total_states": machine.size, "visited": sorted(visited)}
