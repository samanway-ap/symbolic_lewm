"""E1.2 -- local effective rank at sampled base points.

At each base point, take the N nearest REAL latents (from the held-out
corpus), local-PCA them, report the number of components needed for 95%
energy. Reported as a DISTRIBUTION over base points (mean/median/min/max/all
values), not just the median -- per EXPERIMENTS v2's explicit instruction.

Assumption A5 (the load-bearing one): local effective rank << ambient dim d.
If it is ~= d everywhere, there is no local low-rank structure to speak of
and orthogonal expansion has nothing to act on.
"""
from __future__ import annotations

import numpy as np


def local_effective_rank(base_point: np.ndarray, real_latents: np.ndarray, k: int = 256,
                            energy: float = 0.95) -> dict:
    d2 = ((real_latents - base_point[None, :]) ** 2).sum(axis=1)
    idx = np.argpartition(d2, min(k, len(d2) - 1))[:k]
    neigh = real_latents[idx]
    mean = neigh.mean(axis=0)
    centered = neigh - mean
    # SVD of the centered neighbourhood -- cheaper and more stable than
    # forming the DxD covariance matrix for D=192, k=256
    s = np.linalg.svd(centered, compute_uv=False)
    total = (s ** 2).sum()
    if total <= 0:
        return {"effective_rank": 0, "n_neighbors": int(len(idx))}
    cum = np.cumsum(s ** 2) / total
    rank = int(np.searchsorted(cum, energy) + 1)
    return {"effective_rank": rank, "n_neighbors": int(len(idx)), "ambient_dim": int(real_latents.shape[1])}


def local_rank_distribution(base_points: np.ndarray, real_latents: np.ndarray, k: int = 256,
                               energy: float = 0.95) -> dict:
    ranks = [local_effective_rank(bp, real_latents, k=k, energy=energy)["effective_rank"] for bp in base_points]
    ranks = np.array(ranks)
    return {
        "ranks": ranks.tolist(), "n_base_points": int(len(ranks)),
        "mean": float(ranks.mean()), "median": float(np.median(ranks)),
        "min": int(ranks.min()), "max": int(ranks.max()), "std": float(ranks.std()),
        "ambient_dim": int(real_latents.shape[1]),
        "mean_ratio_to_ambient": float(ranks.mean() / real_latents.shape[1]),
    }
