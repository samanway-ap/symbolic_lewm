"""Phase G.3 -- edge rarity. Pure helper: turns raw (state,symbol)
visitation counts (gathered once, alongside G.1's state labeling and G.4's
abstraction-failure flag, by training/data.py's single real pass over
held-out TRAIN episodes against the frozen accepted automaton) into
inverse-frequency weights, clipped to [0.2, 5.0] per plan §10 G.3.

G.1 (automaton state q_t) and G.4 (abstraction-failure flag) are not
separate modules -- they fall out directly of running the machine along
the observed abstract-action sequence, which training/data.py already does
once to build the fine-tune sample table itself (state_id and
abstraction_failure_flag are just fields on each sample). Keeping them
inline there (rather than a second redundant pass here) avoids re-running
real pixel encoding twice for the same episodes.
"""
from __future__ import annotations

import numpy as np

RARITY_CLIP = (0.2, 5.0)


def compute_edge_rarity(visitation: dict[tuple, int]) -> dict[tuple, float]:
    """visitation: {(state_id, symbol): count} over a held-out sample.
    Rarity is inverse frequency relative to the MEAN count across observed
    (state,symbol) keys (so a key visited at exactly the average rate gets
    weight ~1.0), clipped to [0.2, 5.0]."""
    if not visitation:
        return {}
    total = sum(visitation.values())
    mean_freq = total / len(visitation)
    return {
        key: float(np.clip(mean_freq / max(count, 1), *RARITY_CLIP))
        for key, count in visitation.items()
    }
