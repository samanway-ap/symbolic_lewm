"""E1.1 -- global intrinsic dimension of the reachable latent set.

Two independent estimators, both reported (per the plan's "estimate via
TwoNN... or Levina-Bickel MLE" -- doing both is stronger than picking one):

TwoNN (Facco, Rodriguez, Laio 2017): for each point, mu_i = r2/r1 (ratio of
distances to the 2nd and 1st nearest neighbors, EXCLUDING the point itself).
Under the local-uniform-density assumption, mu follows a Pareto distribution
with shape = intrinsic dimension d; the closed-form MLE is
d_hat = N / sum(log(mu_i)). The paper recommends discarding the largest ~10%
of mu_i (points where the local-uniformity assumption is most strained) for
robustness -- done here, controllable via `discard_frac`.

Levina-Bickel MLE (2005): at a point x_i with k nearest-neighbor distances
r_1 <= ... <= r_k, m_hat_i(k) = [ (1/(k-1)) sum_{j=1}^{k-1} log(r_k/r_j) ]^-1.
Averaged over points and over a range of k (the paper's own recommendation,
since single-k estimates are noisy) for stability.
"""
from __future__ import annotations

import numpy as np


def _pairwise_knn_distances(X: np.ndarray, k: int, chunk: int = 500) -> np.ndarray:
    """X: (N,D). Returns (N,k) sorted distances to the k NEAREST OTHER points
    (self excluded), via chunked brute-force -- exact, and fast enough at
    N ~ a few thousand, D ~ 192 (the regime E1 actually runs in)."""
    N = X.shape[0]
    out = np.empty((N, k), dtype=np.float64)
    X64 = X.astype(np.float64)
    sq = (X64 ** 2).sum(axis=1)
    for i in range(0, N, chunk):
        block = X64[i:i + chunk]
        d2 = sq[i:i + chunk, None] + sq[None, :] - 2 * block @ X64.T
        np.clip(d2, 0, None, out=d2)
        idx = np.arange(i, min(i + chunk, N))
        d2[np.arange(len(idx)), idx] = np.inf   # exclude self
        part = np.partition(d2, k, axis=1)[:, :k]
        part.sort(axis=1)
        out[i:i + chunk] = np.sqrt(part)
    return out


def twonn(X: np.ndarray, discard_frac: float = 0.1) -> dict:
    knn = _pairwise_knn_distances(X, k=2)
    r1, r2 = knn[:, 0], knn[:, 1]
    valid = r1 > 1e-12
    mu = r2[valid] / r1[valid]
    log_mu = np.log(mu)
    order = np.argsort(log_mu)
    n_keep = int(len(log_mu) * (1 - discard_frac))
    kept = log_mu[order[:n_keep]]
    d_hat = len(kept) / kept.sum() if kept.sum() > 0 else float("nan")
    return {"estimator": "TwoNN", "d_hat": float(d_hat), "n_points": int(X.shape[0]),
             "n_kept_after_discard": int(n_keep), "discard_frac": discard_frac}


def levina_bickel(X: np.ndarray, k_range: tuple[int, int] = (10, 20)) -> dict:
    k_max = k_range[1]
    knn = _pairwise_knn_distances(X, k=k_max)
    estimates_by_k = []
    for k in range(k_range[0], k_range[1] + 1):
        rk = knn[:, k - 1]
        rj = knn[:, :k - 1]
        valid = rk > 1e-12
        log_ratios = np.log(rk[valid, None] / np.clip(rj[valid], 1e-12, None))
        m_i = 1.0 / log_ratios.mean(axis=1)
        estimates_by_k.append(float(np.mean(m_i)))
    d_hat = float(np.mean(estimates_by_k))
    return {"estimator": "Levina-Bickel MLE", "d_hat": d_hat, "n_points": int(X.shape[0]),
             "k_range": list(k_range), "estimates_by_k": estimates_by_k}


def estimate_intrinsic_dimension(X: np.ndarray, seed: int = 3072, max_points: int = 4000) -> dict:
    """Subsamples to `max_points` for tractability (pairwise distances are
    O(N^2)); both estimators computed on the SAME subsample for comparability."""
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
    return {"ambient_dim": int(X.shape[1]), "n_points_used": int(X.shape[0]),
             "twonn": twonn(X), "levina_bickel": levina_bickel(X)}
