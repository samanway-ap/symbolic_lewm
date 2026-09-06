"""Minimal, generic action-segment codebook (SYMBOLIC_ABSTRACTION_PLAN.md,
Phase B.3): fit(...) clusters segment vectors and stores, per symbol, the
MEDOID (the actual nearest real segment, never the k-means centroid, which
may be dynamically unrealisable), assign(...) does nearest-medoid decoding.

This module is intentionally generic over the segment representation -- it
does not do the real Phase B.1/B.2 work (temporal-window slicing, gripper
special-casing on real Franka action data). That requires the real dataset
and is out of scope for the pre-experiment tests, which only need to check
that the *codebook mechanism itself* is deterministic: given a fixed set of
segment vectors and a fixed seed, the resulting medoids, symbol ids, and
content hash are identical across repeated fits -- including across a fresh
Python process, which is the only way to rule out determinism that
accidentally depends on process-local state (hash randomisation, BLAS
threading nondeterminism, etc.) rather than the algorithm itself.
"""
from __future__ import annotations

import hashlib

import numpy as np


class Codebook:
    def __init__(self, medoids: np.ndarray) -> None:
        # Canonical ordering: sort rows lexicographically so that the same
        # SET of medoids always yields the same symbol ids, regardless of
        # the (algorithm-internal, not guaranteed stable) label order the
        # clustering happened to emit them in.
        order = np.lexsort(medoids.T[::-1])
        self.medoids = np.ascontiguousarray(medoids[order])

    @classmethod
    def fit(cls, segments: np.ndarray, n_codes: int, seed: int) -> "Codebook":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_codes, random_state=seed, n_init=10).fit(segments)
        medoids = []
        for c in range(n_codes):
            members = segments[km.labels_ == c]
            if len(members) == 0:
                continue
            centroid = km.cluster_centers_[c]
            d = np.linalg.norm(members - centroid, axis=1)
            medoids.append(members[int(np.argmin(d))])
        return cls(np.stack(medoids))

    def assign(self, segment: np.ndarray) -> int:
        d = np.linalg.norm(self.medoids - segment, axis=1)
        return int(np.argmin(d))

    def assign_batch(self, segments: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(segments[:, None, :] - self.medoids[None, :, :], axis=2)
        return np.argmin(d, axis=1)

    def content_hash(self) -> str:
        # round to kill float noise across BLAS backends / platforms before
        # hashing, so the hash reflects the codebook's *content*, not its
        # floating-point provenance.
        rounded = np.round(self.medoids, 6)
        return hashlib.sha256(rounded.tobytes()).hexdigest()

    @property
    def n_codes(self) -> int:
        return self.medoids.shape[0]
