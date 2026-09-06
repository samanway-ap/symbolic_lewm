"""Non-triviality index components: suffix depth and memory multiplier.

Suffix depth reuses signals/relevance_subspace.py's `compute_distinguishing_
suffixes` (AALpy's own validated BFS, `find_distinguishing_seq`, per pair)
unchanged -- it already returns the actual witness words, so depth is just
`len(seq)` per pair. Reported as a full distribution (median, max), not a
single number, since the median IS the non-triviality threshold's input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signals.relevance_subspace import compute_distinguishing_suffixes


def suffix_depth_distribution(machine, alphabet: list[str]) -> dict:
    pairs = compute_distinguishing_suffixes(machine, alphabet)
    depths = [len(seq) for seq in pairs.values()]
    if not depths:
        return {"n_pairs": 0, "median": 0.0, "max": 0, "mean": 0.0, "depths": []}
    return {"n_pairs": len(depths), "median": float(np.median(depths)), "max": int(max(depths)),
             "mean": float(np.mean(depths)), "depths": depths}


def memory_multiplier(n_states: int, n_reached_cells: int) -> float:
    return n_states / max(1, n_reached_cells)
