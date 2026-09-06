"""E0.3 -- retroactive baselines for every predicate/tessellation bit.

For a binary label sequence with switch rate s = P(label[t+1] != label[t]):
  persistence baseline  = 1 - s          (predict "same as last step")
  majority baseline     = max(p, 1-p)    (predict the globally more common class)

Computed on the SAME cached held-out corpus (`corpus_heldout.npz`, 355
episodes x 80 positions, canonical test-20% split) the tessellation phase
already built and cached -- an apples-to-apples resolution match with how
the official 0.916 trace_fidelity was measured, not a different-resolution
number reused from Phase C's discovery-time switch_rate.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import Corpus, build_corpus, test_ids
from predicates.functions import build_predicate_fns
from tessellation.hyperplanes import Tessellation

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def label_sequences(corpus: Corpus, bit_fn) -> np.ndarray:
    """bit_fn(z) -> 0/1 for a single latent. Returns (n_ep, n_pos) int array."""
    n_ep, n_pos, _ = corpus.latents.shape
    out = np.zeros((n_ep, n_pos), dtype=np.int8)
    for i in range(n_ep):
        out[i] = [bit_fn(corpus.latents[i, t]) for t in range(n_pos)]
    return out


def switch_persistence_majority(labels: np.ndarray) -> dict:
    """labels: (n_ep, n_pos) binary."""
    switches = (labels[:, 1:] != labels[:, :-1])
    s = float(switches.mean())
    p1 = float(labels.mean())
    return {
        "switch_rate": s,
        "persistence_baseline": 1.0 - s,
        "majority_baseline": max(p1, 1.0 - p1),
        "p_label_1": p1,
        "n_observations": int(labels.size),
    }


def discovered_predicate_baselines(corpus: Corpus, subset: list[str]) -> dict:
    all_fns = build_predicate_fns()
    out = {}
    for name in subset:
        fn = all_fns[name]
        labels = label_sequences(corpus, fn)
        out[name] = switch_persistence_majority(labels)
    return out


def tessellation_baselines(corpus: Corpus, tess_params_path: Path, m_values=(2, 4, 6, 8)) -> dict:
    tp = np.load(tess_params_path)
    out = {}
    for m in m_values:
        t = Tessellation(U=tp[f"U_m{m}"], B=tp[f"B_m{m}"], m=m, occupied={}, seed=3072)
        out[str(m)] = {}
        for i in range(m):
            def bit_fn(z, i=i, t=t):
                return int((float(z @ t.U[i]) > float(t.B[i])))
            labels = label_sequences(corpus, bit_fn)
            # MUST match tessellation/guard.py's CELL_PREDICATE_PREFIX ("cell_h"),
            # since this is joined against tessellation_fsm.json's per_predicate_fidelity
            # keys elsewhere -- a prior mismatch here ("h{i}" vs "cell_h{i}") made
            # that join silently match nothing, and all() over the resulting empty
            # comparison set vacuously returned True. Confirmed and corrected
            # 2026-09-02; see preregistration.md.
            out[str(m)][f"cell_h{i}"] = switch_persistence_majority(labels)
    return out


def compare_to_measured_fidelity(baselines: dict, measured: dict) -> dict:
    """measured: {predicate_name: trace_fidelity} (per-predicate, official).
    Returns, per predicate, whether the measured fidelity CLEARS both
    baselines -- the GATE E0 condition."""
    out = {}
    for name, base in baselines.items():
        m = measured.get(name)
        if m is None:
            continue
        best_baseline = max(base["persistence_baseline"], base["majority_baseline"])
        out[name] = {
            "measured_fidelity": m, "persistence_baseline": base["persistence_baseline"],
            "majority_baseline": base["majority_baseline"],
            "beats_persistence": m > base["persistence_baseline"],
            "beats_majority": m > base["majority_baseline"],
            "beats_best_baseline": m > best_baseline,
            "margin_over_best_baseline": m - best_baseline,
        }
    return out
