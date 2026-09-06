"""STEP 2 of the exploratory Phase G/H/I run: the `time_binned_sample_prio`
control arm's signal (pre-registered in preregistration.md before any
training run).

Why this arm exists
-------------------
The automaton state after t letters is a DETERMINISTIC FUNCTION of the
action history. On stereotyped demonstration data the action history is
tightly coupled to episode PHASE (approach -> grasp -> pull -> release), so
`state_id` may be little more than a proxy for "how far into the episode we
are". If that is all it is, `sample_prio` beating `control` would show only
that non-uniform, temporally-coherent weighting helps -- nothing at all
about symbolic content.

The existing shuffled twin CANNOT separate those two explanations: a random
permutation destroys temporal coherence and symbolic content at the same
time. This arm destroys only the symbolic content, keeping the temporal
coherence, by conditioning the identical weights on binned normalized
timestep instead of on automaton state.

What "statistically matched" means here, exactly
------------------------------------------------
The rarity weights this module returns are a RANK-REMAPPING of the real
arm's own rarity values onto time-binned scores. The returned array is
therefore a PERMUTATION of `real_rarity` -- bit-identical multiset, hence
identical mean, variance, min, max and sparsity -- differing only in WHICH
sample receives which weight. The flag surrogate raises exactly the same
NUMBER of samples as the real abstraction-failure flag, so `(1 + kappa *
flag)` has the same count of raised samples at every annealing step, and
the shared cosine schedule is applied to it verbatim.

So across the three sample-prio arms the weight multiset is identical and
only the conditioning differs:
    sample_prio              -> conditioned on automaton state
    shuffled_sample_prio     -> conditioned on nothing (random permutation)
    time_binned_sample_prio  -> conditioned on episode phase (6 time bins)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signals.edge_stats import compute_edge_rarity

N_TIME_BINS = 6  # matches the accepted machine's state count


def time_bin_of(sample: dict, n_bins: int = N_TIME_BINS) -> int:
    """Normalized timestep within the episode, binned. `n_pos_in_episode` is
    that episode's own usable length, so this is a phase, not an absolute
    step index."""
    n = max(1, int(sample.get("n_pos_in_episode", 1)))
    return int(np.clip((int(sample["pos"]) * n_bins) // n, 0, n_bins - 1))


def _rank_remap(raw_scores: np.ndarray, target_values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign `target_values`'s sorted multiset to samples in ascending order
    of `raw_scores`. Ties in raw_scores (very common -- many samples share a
    (bin,symbol) key) are broken by a fixed seeded permutation so no
    systematic index/episode ordering leaks into the assignment.

    Returns an array that is exactly a permutation of `target_values`."""
    tiebreak = rng.random(len(raw_scores))
    order = np.lexsort((tiebreak, raw_scores))  # ascending, ties randomised
    out = np.empty_like(target_values)
    out[order] = np.sort(target_values)
    return out


def build_time_binned_signals(samples: list[dict], real_rarity: np.ndarray,
                              real_flag: np.ndarray, seed: int = 3072,
                              n_bins: int = N_TIME_BINS) -> dict:
    rng = np.random.default_rng(seed + 303)
    bins = np.array([time_bin_of(s, n_bins) for s in samples], dtype=np.int64)

    # (1) rarity: same visitation->rarity machinery as the real arm, keyed on
    #     time bin instead of automaton state.
    visitation: dict[tuple, int] = {}
    for b, s in zip(bins, samples):
        key = (int(b), s["symbol"])
        visitation[key] = visitation.get(key, 0) + 1
    rarity_table = compute_edge_rarity(visitation)
    raw_rarity = np.array([rarity_table[(int(b), s["symbol"])] for b, s in zip(bins, samples)],
                          dtype=np.float64)
    tb_rarity = _rank_remap(raw_rarity, real_rarity.astype(np.float64), rng).astype(np.float32)

    # (2) flag surrogate: raise the same NUMBER of samples, chosen by per-bin
    #     empirical failure propensity (ties inside a bin broken by the same
    #     seeded scheme). Deterministic function of the time bin, up to ties.
    n_flags = int(real_flag.sum())
    bin_rate = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        m = bins == b
        bin_rate[b] = float(real_flag[m].mean()) if m.any() else 0.0
    prop = bin_rate[bins]
    tiebreak = rng.random(len(samples))
    order = np.lexsort((tiebreak, -prop))  # descending propensity
    tb_flag = np.zeros(len(samples), dtype=np.float32)
    tb_flag[order[:n_flags]] = 1.0

    diag = {
        "n_bins": n_bins,
        "bin_counts": np.bincount(bins, minlength=n_bins).tolist(),
        "per_bin_real_failure_rate": bin_rate.round(4).tolist(),
        "n_flags_real": n_flags,
        "n_flags_time_binned": int(tb_flag.sum()),
        "rarity_multiset_identical_to_real": bool(
            np.allclose(np.sort(tb_rarity), np.sort(real_rarity.astype(np.float32)))),
        "rarity_assignment_differs_from_real": bool(not np.allclose(tb_rarity, real_rarity)),
        "n_distinct_time_binned_keys": len(rarity_table),
        # How much automaton state and episode phase actually overlap -- the
        # empirical version of this arm's whole rationale. Reported either way.
        "normalized_mutual_information_state_vs_timebin": _nmi(
            np.array([str(s["state_id"]) for s in samples]), bins),
    }
    return {"rarity": tb_rarity, "flag": tb_flag, "bins": bins, "diagnostics": diag}


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information between the automaton-state labelling and
    the time-bin labelling of the same samples. ~1.0 means the automaton state
    IS episode phase (this arm is then the decisive control); ~0.0 means they
    are unrelated (the concern motivating this arm does not apply)."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(a, b))
