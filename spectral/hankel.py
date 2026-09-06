"""Phase S -- spectral pre-test. Cheap Hankel-rank probe that should predict
automaton size BEFORE paying for an extraction.

Construction (per plan-of-record, and per preregistration.md's fixed gate):
  rows = action-symbol prefixes of length 1..3
  cols = action-symbol suffixes of length 1..3
  entry(p,q) = mean +/-1-encoded label observed at word p+q
  ONE MATRIX PER PREDICATE/BIT -- never the joint tuple, since a joint tuple
  conflates |S| independent binary functions into one categorical one and its
  rank stops meaning anything.

Short words are deliberate: with a 12-symbol alphabet, only short prefixes
repeat often enough in real data to estimate a cell at all. Cells with < 5
observations are dropped and reported as fill fraction; all-empty rows/cols
are removed; remaining holes are filled with 0, the neutral value under
+/-1 encoding (a hole is "no evidence", not "label -1").

Three matrices per predicate:
  DATA   -- from real held-out trajectories (the ground truth)
  NULL   -- labels shuffled WITHIN EACH COLUMN, 20x (the falsification test:
            if DATA's rank isn't clearly below NULL's, there is no low-rank
            structure, only a low-rank-looking artifact)
  MODEL  -- same words, labels from stepping the frozen theta_0 oracle. The
            DATA-vs-MODEL rank GAP is diagnostic: a much lower MODEL rank
            means g invented structure the data does not support.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_OBS_PER_CELL = 5
MAX_WORD_LEN = 3
N_NULL_SHUFFLES = 20


def enumerate_words(alphabet: list[str], max_len: int = MAX_WORD_LEN) -> list[tuple]:
    out = []
    for L in range(1, max_len + 1):
        out.extend(product(alphabet, repeat=L))
    return out


def collect_data_observations(corpus, predicate_bit_fn, max_len: int = MAX_WORD_LEN):
    """Walk every real held-out trajectory and record, for every split of every
    observed window into (prefix, suffix), the +/-1 label at the END of the
    window. Returns {(prefix,suffix): [labels]}.

    Alignment: symbols[t] drives z_t -> z_{t+1}, so a word symbols[t:t+L]
    starting at position t ends at latent z_{t+L}; the label is phi(z_{t+L}).
    """
    obs = defaultdict(list)
    n_ep, n_pos = corpus.latents.shape[0], corpus.latents.shape[1]
    n_sym = corpus.symbols.shape[1]
    for i in range(n_ep):
        names = corpus.symbol_seq(i)
        labels = predicate_bit_fn(corpus.latents[i])       # (n_pos,) in {0,1}
        pm1 = np.where(labels > 0, 1.0, -1.0)
        for t in range(n_sym):
            max_total = min(2 * max_len, n_sym - t)
            for total in range(2, max_total + 1):
                end = t + total
                if end >= n_pos:
                    break
                word = tuple(names[t:end])
                lab = pm1[end]
                for cut in range(1, total):
                    if cut > max_len or (total - cut) > max_len:
                        continue
                    obs[(word[:cut], word[cut:])].append(lab)
    return obs


def collect_model_observations(cells: list[tuple], oracle, reset_names: list[str], predicate_index: int):
    """Same (prefix,suffix) cells, labels from the frozen theta_0 oracle
    instead of real data: evaluate (reset,)+prefix+suffix and average the
    +/-1 label over the 8 canonical reset windows. The oracle's own prefix
    cache makes this far cheaper than the word count suggests."""
    out = {}
    for (p, q) in cells:
        vals = []
        for r in reset_names:
            lab = oracle.evaluate((r,) + p + q)
            if isinstance(lab, tuple) and len(lab) > predicate_index:
                vals.append(1.0 if lab[predicate_index] else -1.0)
        if vals:
            out[(p, q)] = float(np.mean(vals))
    return out


def build_matrix(obs: dict, min_obs: int = MIN_OBS_PER_CELL):
    """obs: {(prefix,suffix): [labels]} -> (matrix, row_keys, col_keys, stats)."""
    kept = {k: float(np.mean(v)) for k, v in obs.items() if len(v) >= min_obs}
    if not kept:
        return None, [], [], {"n_cells_observed": len(obs), "n_cells_kept": 0, "fill_fraction": 0.0}
    rows = sorted({p for (p, _) in kept}, key=lambda w: (len(w), w))
    cols = sorted({q for (_, q) in kept}, key=lambda w: (len(w), w))
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: j for j, c in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)), dtype=np.float64)
    mask = np.zeros_like(M, dtype=bool)
    for (p, q), v in kept.items():
        M[ri[p], ci[q]] = v
        mask[ri[p], ci[q]] = True
    stats = {
        "n_cells_observed": len(obs), "n_cells_kept": len(kept),
        "shape": [len(rows), len(cols)],
        "fill_fraction": float(mask.sum() / mask.size),
        "median_obs_per_kept_cell": float(np.median([len(v) for v in obs.values() if len(v) >= min_obs])),
    }
    return M, rows, cols, stats, mask


def effective_rank(M: np.ndarray, energies=(0.90, 0.95, 0.99)) -> dict:
    if M is None or M.size == 0 or not np.any(M):
        return {f"rank_{e}": 0 for e in energies} | {"singular_values": []}
    s = np.linalg.svd(M, compute_uv=False)
    total = (s ** 2).sum()
    if total <= 0:
        return {f"rank_{e}": 0 for e in energies} | {"singular_values": []}
    cum = np.cumsum(s ** 2) / total
    out = {f"rank_{e}": int(np.searchsorted(cum, e) + 1) for e in energies}
    out["singular_values"] = s[:25].tolist()
    out["n_singular_values"] = int(len(s))
    return out


def null_ranks(M: np.ndarray, mask: np.ndarray, n_shuffles: int = N_NULL_SHUFFLES,
                 seed: int = 3072, energies=(0.90, 0.95, 0.99)) -> dict:
    """Shuffle observed labels WITHIN EACH COLUMN (preserving which cells are
    observed and each column's label multiset), then recompute rank."""
    rng = np.random.default_rng(seed)
    acc = {e: [] for e in energies}
    for _ in range(n_shuffles):
        S = M.copy()
        for j in range(M.shape[1]):
            idx = np.flatnonzero(mask[:, j])
            if len(idx) > 1:
                S[idx, j] = M[rng.permutation(idx), j]
        r = effective_rank(S, energies)
        for e in energies:
            acc[e].append(r[f"rank_{e}"])
    out = {}
    for e in energies:
        a = np.array(acc[e], dtype=float)
        out[f"null_rank_{e}"] = {
            "mean": float(a.mean()), "sd": float(a.std()),
            "p05": float(np.percentile(a, 5)), "min": float(a.min()), "max": float(a.max()),
        }
    return out


def gate_verdict(data_rank_95: float, null_stats_95: dict) -> dict:
    """The criterion FIXED IN preregistration.md BEFORE this ran: the gate
    passes iff data rank at 0.95 is (i) below the 5th percentile of the
    shuffled-null rank distribution AND (ii) <= 0.8 * the null mean. Both a
    distributional and an effect-size test, so that a trivially rank-1 matrix
    whose null is ALSO rank-1 correctly fails."""
    p05 = null_stats_95["p05"]
    mean = null_stats_95["mean"]
    cond_dist = data_rank_95 < p05
    cond_effect = data_rank_95 <= 0.8 * mean
    return {
        "data_rank_0.95": data_rank_95,
        "null_p05": p05, "null_mean": mean,
        "below_null_p05": bool(cond_dist),
        "at_most_80pct_of_null_mean": bool(cond_effect),
        "passes": bool(cond_dist and cond_effect),
    }
