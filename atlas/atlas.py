"""Phase R -- the atlas, built on tessellation cells.

Regions ARE occupied cells at best_m: no radius parameter, no overlap,
exhaustive coverage of the occupied latent space, and containment is EXACT
(a bit-vector match) rather than an approximate distance test.

Three things this file is careful about, each of which was a specific
instruction:

1. ELIGIBILITY vs HOMOGENEITY. A cell with too few held-out evaluation
   segments is UNDERPOWERED, which is a completely different claim from
   HOMOGENEOUS. Cells below the >=50-segment floor merge with their nearest
   neighbour (flip the least-informative bit) until they clear it, and every
   merge is recorded in the atlas so a merged region is never mistaken for a
   primitive one.

2. BEHAVIOURAL COHERENCE. A cell is only meaningful if its points behave
   alike under the frozen DISCOVERED-predicate machine. Coherence = the
   fraction of the cell's points sharing the modal automaton state. Near
   chance -> BLOCKED_INCOHERENT, skipped, because a geometrically compact
   region that mixes behaviours is not a region worth targeting.

3. SEED WEIGHTING IS EXTERNALLY GROUNDED. Cells are sampled by their
   ABSTRACTION-FAILURE RATE on real held-out data -- timesteps where the
   frozen machine's predicted next label disagrees with phi of the real next
   latent. Deliberately NOT weighted by the machine's own lack of resolution:
   "the model is unsure here" is unfalsifiable from inside the model, while
   "the model's symbolic prediction is contradicted by real data" is not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
# RGK_SUFFIX keeps a supplementary (coarser-m) run's atlas in its own file so
# it can never overwrite or silently merge into the pre-registered primary.
ATLAS_PATH = OUT_DIR / f"atlas{os.environ.get('RGK_SUFFIX', '')}.json"

MIN_HELDOUT_SEGMENTS = 50
MIN_CANDIDATE_TRAJECTORIES = 200
COHERENCE_CHANCE_MARGIN = 0.10   # must beat 1/n_states by this much to count as coherent

STATUSES = ("UNEXPLORED", "ACTIVE", "IMPROVED", "BLOCKED_HOMOGENEOUS",
            "BLOCKED_UNDERPOWERED", "BLOCKED_INCOHERENT")


def load_atlas() -> dict:
    if ATLAS_PATH.exists():
        return json.loads(ATLAS_PATH.read_text())
    return {"cells": {}, "rounds_log": []}


def save_atlas(atlas: dict) -> None:
    ATLAS_PATH.write_text(json.dumps(atlas, indent=2, default=float))


def new_record(cell_id: int, bits: str, occupancy: int) -> dict:
    return {
        "cell_bits": bits, "cell_id": int(cell_id), "occupancy": int(occupancy),
        "status": "UNEXPLORED", "rounds_attempted": 0, "trajectory_ids_used": [],
        "metric_history": [], "block_reason": None,
        "eligibility_counts": {}, "coherence_fraction": None, "merge_history": [],
    }


def hamming_neighbours(cell_id: int, m: int) -> list[int]:
    return [cell_id ^ (1 << i) for i in range(m)]


def merge_underpowered(
    cell_counts: dict[int, int], m: int, min_segments: int = MIN_HELDOUT_SEGMENTS,
    bit_informativeness: np.ndarray | None = None,
):
    """Merge cells below the held-out segment floor into their nearest
    neighbour, flipping the LEAST-INFORMATIVE bit first (informativeness =
    how strongly that bit splits held-out mass; a bit that barely separates
    anything is the safest one to erase). Returns (groups, merge_log) where
    groups maps representative_cell_id -> [member cell ids]."""
    groups = {c: [c] for c in cell_counts}
    counts = dict(cell_counts)
    merge_log = []
    order = np.argsort(bit_informativeness) if bit_informativeness is not None else np.arange(m)

    changed = True
    while changed:
        changed = False
        small = sorted([c for c, n in counts.items() if n < min_segments], key=lambda c: counts[c])
        for c in small:
            if c not in counts:
                continue
            target = None
            for bit in order:                      # least-informative bit first
                cand = c ^ (1 << int(bit))
                if cand in counts and cand != c:
                    target = cand
                    break
            if target is None:
                continue
            merge_log.append({"merged": format(c, f"0{m}b"), "into": format(target, f"0{m}b"),
                                "merged_count": counts[c], "target_count_before": counts[target],
                                "reason": f"below {min_segments} held-out segments"})
            counts[target] += counts[c]
            groups[target] = groups[target] + groups[c]
            del counts[c], groups[c]
            changed = True
    return groups, counts, merge_log


def bit_informativeness(bits: np.ndarray) -> np.ndarray:
    """bits: (N,m) bool. Informativeness of bit i = how balanced its split is
    (2*|p-0.5| inverted), so a bit that is nearly constant on this data scores
    near 0 and is the first candidate to be erased when merging."""
    p = bits.mean(axis=0)
    return 1.0 - 2.0 * np.abs(p - 0.5)


def coherence_fraction(states_in_cell: np.ndarray, n_machine_states: int) -> tuple[float, bool]:
    """Fraction of the cell's points sharing the MODAL automaton state, and
    whether that clears chance (1/n_states) by the required margin."""
    if states_in_cell.size == 0:
        return 0.0, False
    _, counts = np.unique(states_in_cell, return_counts=True)
    frac = float(counts.max() / states_in_cell.size)
    chance = 1.0 / max(1, n_machine_states)
    return frac, bool(frac >= chance + COHERENCE_CHANCE_MARGIN)


def seed_weights(failure_rate_by_cell: dict[int, float], unexplored: list[int]) -> np.ndarray:
    """Sampling weights over UNEXPLORED cells, proportional to externally
    grounded abstraction-failure rate. Uniform fallback if every rate is 0."""
    w = np.array([max(0.0, failure_rate_by_cell.get(c, 0.0)) for c in unexplored], dtype=float)
    if w.sum() <= 0:
        return np.full(len(unexplored), 1.0 / max(1, len(unexplored)))
    return w / w.sum()
