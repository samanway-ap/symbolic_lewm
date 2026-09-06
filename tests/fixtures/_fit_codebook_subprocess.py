"""Self-contained script: generate a fixed synthetic segment pool, fit a
Codebook, print its content hash and a fixed query set's assignments as
JSON. Invoked as a fresh subprocess by test_alphabet_determinism.py so that
determinism is checked across process boundaries, not just in-process
(ruling out hash randomisation / module-level caching artifacts).

Usage: python _fit_codebook_subprocess.py <seed>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from alphabet.codebook import Codebook

N_SEGMENTS = 300
DIM = 12
N_CODES = 6
DATA_SEED = 424242  # the segment POOL is fixed regardless of fit seed


def make_segments(seed: int = DATA_SEED, n: int = N_SEGMENTS, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(N_CODES, dim))
    labels = rng.integers(0, N_CODES, size=n)
    return centers[labels] + rng.normal(0, 0.3, size=(n, dim))


def main() -> None:
    fit_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    segments = make_segments()
    cb = Codebook.fit(segments, n_codes=N_CODES, seed=fit_seed)

    query = make_segments(seed=DATA_SEED + 1, n=20)
    out = {
        "content_hash": cb.content_hash(),
        "n_codes": cb.n_codes,
        "assignments": cb.assign_batch(query).tolist(),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
