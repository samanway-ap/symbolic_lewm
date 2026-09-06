"""Phase T -- uniform tessellation of the latent space.

A literal grid is impossible (2^192 cells at one bit per axis), so "uniform"
is operationalised in the two senses that actually matter here:

  ROTATION-INVARIANT: directions u_i drawn uniform on the unit sphere, NOT
    from PCA. SIGReg deliberately flattens the eigenspectrum, so principal
    directions carry no signal in this latent space -- PCA gridding would be
    slicing along axes that mean nothing.
  EQUAL-MASS: offset b_i = MEDIAN of u_i . z over TRAIN latents, so every
    single hyperplane splits the data 50/50 by construction. Hyperplanes
    through the centroid without offsets give wildly unequal cell masses and
    wreck every downstream statistic that assumes comparable cell counts.

cell(z) = m-bit sign vector [ 1(u_i . z > b_i) ]_i. With m bits there are
2^m nominal cells but the data lies on a low-dimensional manifold, so most
are empty; only OCCUPIED cells (>= 20 train latents) are real labels.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_TRAIN_PER_CELL = 20


@dataclass
class Tessellation:
    U: np.ndarray        # (m, d) unit directions
    B: np.ndarray        # (m,) median offsets
    m: int
    occupied: dict       # {cell_id: train_count} for occupied cells only
    seed: int

    def bits(self, Z: np.ndarray) -> np.ndarray:
        """Z: (N,d) -> (N,m) bool"""
        return (Z @ self.U.T) > self.B[None, :]

    def cell_ids(self, Z: np.ndarray) -> np.ndarray:
        b = self.bits(Z)
        powers = (1 << np.arange(self.m))[None, :]
        return (b * powers).sum(axis=1).astype(np.int64)

    @staticmethod
    def id_to_bits(cell_id: int, m: int) -> str:
        return "".join(str((cell_id >> i) & 1) for i in range(m))


def fit_tessellation(train_Z: np.ndarray, m: int, seed: int = 3072) -> Tessellation:
    rng = np.random.default_rng(seed + 7919 * m)
    d = train_Z.shape[1]
    U = rng.standard_normal((m, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)   # uniform on the unit sphere
    proj = train_Z @ U.T                             # (N,m)
    B = np.median(proj, axis=0)                      # equal-mass split per hyperplane
    t = Tessellation(U=U, B=B, m=m, occupied={}, seed=seed)
    ids = t.cell_ids(train_Z)
    uniq, counts = np.unique(ids, return_counts=True)
    t.occupied = {int(c): int(n) for c, n in zip(uniq, counts) if n >= MIN_TRAIN_PER_CELL}
    return t


def sweep_report(train_Z: np.ndarray, m_values=(2, 4, 6, 8), seed: int = 3072):
    """Returns (results, tessellations). `best_m` selection is the rule fixed
    in preregistration.md BEFORE this was run: largest m with >=4 occupied
    cells AND occupied cells holding >=90% of train latents; fall back to
    largest m with >=4 occupied cells."""
    results, tess = [], {}
    N = train_Z.shape[0]
    for m in m_values:
        t = fit_tessellation(train_Z, m, seed=seed)
        tess[m] = t
        occ_mass = sum(t.occupied.values())
        results.append({
            "m": m, "n_cells_nominal": 2 ** m, "n_cells_occupied": len(t.occupied),
            "occupied_mass": occ_mass, "occupied_mass_frac": occ_mass / N,
            "min_occupancy": min(t.occupied.values()) if t.occupied else 0,
            "median_occupancy": int(np.median(list(t.occupied.values()))) if t.occupied else 0,
            "max_occupancy": max(t.occupied.values()) if t.occupied else 0,
        })

    qualifying = [r for r in results if r["n_cells_occupied"] >= 4 and r["occupied_mass_frac"] >= 0.90]
    if qualifying:
        best_m = max(r["m"] for r in qualifying)
        rule = "largest m with >=4 occupied cells and >=90% occupied mass"
    else:
        fallback = [r for r in results if r["n_cells_occupied"] >= 4]
        best_m = max(r["m"] for r in fallback) if fallback else results[0]["m"]
        rule = "fallback: largest m with >=4 occupied cells (90% mass rule unmet by any m)"
    return results, tess, best_m, rule


# ---------- secondary comparison: equal-mass Voronoi ----------
@dataclass
class VoronoiPartition:
    centroids: np.ndarray   # (N_c, d), sampled TRAIN latents -> approx equal mass by construction
    occupied: dict
    seed: int

    def cell_ids(self, Z: np.ndarray, batch: int = 4096) -> np.ndarray:
        out = np.empty(Z.shape[0], dtype=np.int64)
        for i in range(0, Z.shape[0], batch):
            chunk = Z[i:i + batch]
            d2 = ((chunk[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
            out[i:i + batch] = d2.argmin(1)
        return out


def fit_voronoi(train_Z: np.ndarray, n_centroids: int, seed: int = 3072) -> VoronoiPartition:
    """Centroids are SAMPLED train latents, so cell mass is approximately equal
    by construction (sampling from the data distribution itself) -- the
    matched-comparison analogue of the median-offset trick above."""
    rng = np.random.default_rng(seed + 104729)
    idx = rng.choice(train_Z.shape[0], size=min(n_centroids, train_Z.shape[0]), replace=False)
    v = VoronoiPartition(centroids=train_Z[idx].copy(), occupied={}, seed=seed)
    ids = v.cell_ids(train_Z)
    uniq, counts = np.unique(ids, return_counts=True)
    v.occupied = {int(c): int(n) for c, n in zip(uniq, counts) if n >= MIN_TRAIN_PER_CELL}
    return v


def make_cell_label_fn(t: Tessellation):
    """phi(z) = cell bits, as a tuple -- the Phase T2 label function, drop-in
    compatible with the existing predicate label_fn contract (window -> tuple)."""
    def label_fn(window):
        z = window.emb[-1].numpy()[None, :]
        return tuple(int(x) for x in t.bits(z)[0])
    return label_fn
