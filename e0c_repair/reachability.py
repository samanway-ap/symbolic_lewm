"""Reached-cell count from the REAL oracle, not from the extracted machine's
own transition graph.

Checking a machine's reachable-output-label count against ITS OWN state
graph is tautological (a state's output is a function of the state, so the
count of distinct outputs among reachable states is trivially <= |Q| by
construction -- it can never expose under-resolution). The Moore bound this
project cares about is only meaningful measured against what the REAL
model+label-function actually produces, independent of what the extracted
machine claims. So: roll out random words from ALL 8 canonical reset
symbols (enlarging the reachable set past a single base point, per
SELF_IMPROVEMENT_LOOP.md v2 SS2's explicit instruction) through the real
frozen model, and report the union of REAL tessellation-cell labels
observed -- this is what `|Q_k}| >=` is actually checked against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.lewm_g import rollout_batch_from_convention


def reached_cells_from_real_oracle(
    reset_windows: dict, symbols_dict: dict, tess, pipeline,
    n_words_per_reset: int = 100, word_len: int = 16, seed: int = 3072,
) -> dict:
    reset_names = sorted(reset_windows.keys())
    names = list(symbols_dict.keys())
    segs = np.stack([symbols_dict[n] for n in names])
    rng = np.random.default_rng(seed + 4242)

    all_init_emb, all_init_act, all_raw = [], [], []
    for r in reset_names:
        w = reset_windows[r]
        idx = rng.integers(0, len(names), size=(n_words_per_reset, word_len))
        raw = segs[idx]
        all_raw.append(raw)
        all_init_emb.append(w.emb.unsqueeze(0).expand(n_words_per_reset, -1, -1))
        all_init_act.append(w.act_emb.unsqueeze(0).expand(n_words_per_reset, -1, -1))

    import torch
    init_emb = torch.cat(all_init_emb, dim=0)
    init_act = torch.cat(all_init_act, dim=0)
    raw = np.concatenate(all_raw, axis=0)

    trace = rollout_batch_from_convention(init_emb.clone(), init_act.clone(), raw, pipeline).numpy()
    flat = trace.reshape(-1, trace.shape[-1])
    bits = (flat @ tess.U.T) > tess.B[None, :]
    powers = (1 << np.arange(tess.m))[None, :]
    cell_ids = (bits * powers).sum(axis=1)
    reached = set(np.unique(cell_ids).tolist())
    return {"n_reached": len(reached), "reached_cell_ids": sorted(reached),
             "n_resets_used": len(reset_names), "n_words_per_reset": n_words_per_reset, "word_len": word_len}
