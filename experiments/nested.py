"""Nested tessellation — the capability ladder's foundation.

The Phase-T tessellation was NOT nested: `fit_tessellation` drew a fresh RNG
stream per m (`seed + 7919*m`), so the m=4 hyperplanes had nothing to do with
the m=2 ones and level k+1 did not refine level k. The capability ladder's
monotonicity (S_1 >= S_2 >= ... ) is a THEOREM only under nesting, so it has
to be rebuilt properly here.

Draw m_max directions ONCE, in fixed order, each with a median offset fitted
on the train split. Level k uses the prefix u_1..u_k. Nesting is then
automatic: appending a hyperplane can only split existing cells, never move a
point across an existing boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

M_MAX = 8
MIN_TRAIN_PER_CELL = 20


def fit_nested(train_Z: np.ndarray, m_max: int = M_MAX, seed: int = 3072):
    """Returns (U, B): U (m_max,d) unit directions, B (m_max,) median offsets."""
    rng = np.random.default_rng(seed + 20260824)
    d = train_Z.shape[1]
    U = rng.standard_normal((m_max, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)      # uniform on the unit sphere
    B = np.median(train_Z @ U.T, axis=0)                # equal-mass split per hyperplane
    return U, B


def bits_at(Z: np.ndarray, U: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    return (Z @ U[:k].T) > B[:k][None, :]


def cell_ids_at(Z: np.ndarray, U: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    b = bits_at(Z, U, B, k)
    return (b * (1 << np.arange(k))[None, :]).sum(axis=1).astype(np.int64)


def occupancy_at(Z: np.ndarray, U: np.ndarray, B: np.ndarray, k: int, min_count: int = MIN_TRAIN_PER_CELL):
    ids = cell_ids_at(Z, U, B, k)
    uniq, counts = np.unique(ids, return_counts=True)
    occupied = {int(c): int(n) for c, n in zip(uniq, counts) if n >= min_count}
    return occupied, ids


def verify_nesting(Z: np.ndarray, U: np.ndarray, B: np.ndarray, m_max: int = M_MAX) -> bool:
    """Every level-(k+1) cell must sit inside exactly one level-k cell. This is
    guaranteed by construction; asserted anyway because the previous
    tessellation silently violated it."""
    for k in range(1, m_max):
        lo = cell_ids_at(Z, U, B, k)
        hi = cell_ids_at(Z, U, B, k + 1)
        for c in np.unique(hi):
            parents = np.unique(lo[hi == c])
            if len(parents) != 1:
                return False
    return True
