"""Ladder A0 Step 1 (EXPERIMENTS_orthogonal_expansion.md v4 SS4, "diagnose").
Per bit at every m: switch rate, block determinism (+ entropy), absolute
fidelity/persistence/majority, occupancy. All on cached latents+symbols --
no model calls, matching the doc's "~15 min, all seconds" framing.

Switch rate and occupancy are NOT new measurements -- they already exist in
artifacts/e0b_report.json and artifacts/tessellation.json respectively. This
recomputes switch rate independently (cheap, pooled across train+heldout, a
straightforward cross-check) and adds the one genuinely missing measurement:
BLOCK DETERMINISM, "specified in the original plan, never computed for the
tessellation" per the v4 doc.

Block determinism definition (this project's operationalisation, not given
verbatim by the doc): for every (cell, action-symbol) pair type observed at
least MIN_SUPPORT times on real transitions, is its successor cell always
the same? Reports three numbers, since "fraction of pair TYPES" and
"fraction of individual TRANSITIONS" answer different questions and can
diverge a lot when mass is concentrated in a few pair types:
  - type_determinism: fraction of (cell,action) TYPES with a unique successor
  - transition_weighted_determinism: fraction of individual TRANSITIONS whose
    outcome matches their (cell,action) type's majority successor
  - mean_successor_entropy: mean Shannon entropy of the successor
    distribution per (cell,action) type, weighted by type frequency
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import Corpus, build_corpus, test_ids, train_ids
from e0_degeneracy.baselines import switch_persistence_majority
from tessellation.hyperplanes import Tessellation

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SEED = 3072
MIN_SUPPORT = 3          # a (cell,action) pair seen only once/twice can't fail determinism meaningfully
SWITCH_RATE_BAND = (0.02, 0.25)
MIN_BLOCK_DETERMINISM = 0.70


def cell_ids(latents_flat_by_ep: np.ndarray, tess: Tessellation) -> np.ndarray:
    """latents_flat_by_ep: (n_ep, n_pos, D). Returns (n_ep, n_pos) int cell ids
    (0..2^m - 1) packing all m bits, matching e0c_repair/reachability.py's
    convention (bit i contributes 2**i)."""
    bits = (latents_flat_by_ep @ tess.U.T) > tess.B[None, None, :]
    powers = (1 << np.arange(tess.m))[None, None, :]
    return (bits * powers).sum(axis=-1)


def block_determinism(corpus: Corpus, cells: np.ndarray, min_support: int = MIN_SUPPORT) -> dict:
    """cells: (n_ep, n_pos) int. corpus.symbols: (n_ep, n_pos-1) int8 symbol
    index. Pools (cell_t, action_t) -> [cell_{t+1}, ...] across ALL episodes."""
    pair_successors: dict[tuple, list[int]] = defaultdict(list)
    n_ep, n_pos = cells.shape
    for i in range(n_ep):
        n_sym = min(corpus.symbols.shape[1], n_pos - 1)
        for t in range(n_sym):
            key = (int(cells[i, t]), int(corpus.symbols[i, t]))
            pair_successors[key].append(int(cells[i, t + 1]))

    n_types_total = len(pair_successors)
    n_types_supported = 0
    n_types_unique = 0
    n_trans_supported = 0
    n_trans_matching_majority = 0
    entropy_weighted_sum = 0.0
    for key, succs in pair_successors.items():
        n = len(succs)
        if n < min_support:
            continue
        n_types_supported += 1
        n_trans_supported += n
        counts = np.array(list(np.unique(succs, return_counts=True)[1]), dtype=np.float64)
        if counts.size == 1:
            n_types_unique += 1
        p = counts / counts.sum()
        ent = float(-(p * np.log2(p)).sum())
        entropy_weighted_sum += ent * n
        n_trans_matching_majority += int(counts.max())

    return {
        "n_pair_types_total": n_types_total,
        "n_pair_types_supported": n_types_supported,
        "min_support": min_support,
        "n_transitions_supported": n_trans_supported,
        "type_determinism": n_types_unique / n_types_supported if n_types_supported else None,
        "transition_weighted_determinism": (n_trans_matching_majority / n_trans_supported
                                              if n_trans_supported else None),
        "mean_successor_entropy": (entropy_weighted_sum / n_trans_supported
                                      if n_trans_supported else None),
    }


def main():
    train = build_corpus(train_ids(), 60, "train")
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=False)
    tp = np.load(OUT_DIR / "tessellation_params.npz")

    e0b_path = OUT_DIR / "e0b_report.json"
    e0b = json.loads(e0b_path.read_text()) if e0b_path.exists() else None

    report = {}
    for m in (2, 4, 6, 8):
        tess = Tessellation(U=tp[f"U_m{m}"], B=tp[f"B_m{m}"], m=m, occupied={}, seed=SEED)
        bits_out = {}

        # --- switch rate / persistence / majority, pooled train+heldout, fresh cross-check ---
        for i in range(m):
            def bit_fn(z, i=i, t=tess):
                return int((float(z @ t.U[i]) > float(t.B[i])))
            labels_train = np.stack([[bit_fn(train.latents[e, p]) for p in range(train.n_positions)]
                                        for e in range(train.n_episodes)])
            labels_held = np.stack([[bit_fn(heldout.latents[e, p]) for p in range(heldout.n_positions)]
                                       for e in range(heldout.n_episodes)])
            spm_train = switch_persistence_majority(labels_train)
            spm_held = switch_persistence_majority(labels_held)
            bits_out[f"cell_h{i}"] = {"switch_rate_train": spm_train["switch_rate"],
                                         "switch_rate_heldout": spm_held["switch_rate"],
                                         "in_band_train": SWITCH_RATE_BAND[0] <= spm_train["switch_rate"] <= SWITCH_RATE_BAND[1],
                                         "in_band_heldout": SWITCH_RATE_BAND[0] <= spm_held["switch_rate"] <= SWITCH_RATE_BAND[1]}
            if e0b and str(m) in e0b.get("e0b2", {}):
                prior = e0b["e0b2"][str(m)]["bits"].get(f"h{i}")
                if prior:
                    bits_out[f"cell_h{i}"]["mean_fidelity_e0b2"] = prior["mean_fidelity"]
                    bits_out[f"cell_h{i}"]["mean_persistence_e0b2"] = prior["mean_persistence"]
                    bits_out[f"cell_h{i}"]["margin_e0b2"] = prior["margin_mean_diff"]

        # --- block determinism, on the FULL cell id (all m bits jointly), pooled train+heldout ---
        cells_train = cell_ids(train.latents, tess)
        cells_held = cell_ids(heldout.latents, tess)

        det_train = block_determinism(train, cells_train)
        det_held = block_determinism(heldout, cells_held)

        # --- occupancy: already computed in tessellation.json; just surface it here ---
        tess_meta = json.loads((OUT_DIR / "tessellation.json").read_text())
        occ_row = [r for r in tess_meta["sweep"] if r["m"] == m][0]

        report[str(m)] = {
            "bits": bits_out,
            "block_determinism_train": det_train,
            "block_determinism_heldout": det_held,
            "occupancy": {"n_cells_occupied": occ_row["n_cells_occupied"],
                            "n_cells_nominal": occ_row["n_cells_nominal"],
                            "min_occupancy": occ_row["min_occupancy"],
                            "meets_min_20": occ_row["min_occupancy"] >= 20},
        }

        print(f"=== m={m} ===", flush=True)
        for bit, v in bits_out.items():
            print(f"  {bit}: switch_rate train={v['switch_rate_train']:.4f} "
                  f"heldout={v['switch_rate_heldout']:.4f} "
                  f"in_band(train)={v['in_band_train']} in_band(heldout)={v['in_band_heldout']}", flush=True)
        print(f"  block determinism (train):  type={det_train['type_determinism']:.4f} "
              f"trans_weighted={det_train['transition_weighted_determinism']:.4f} "
              f"mean_entropy={det_train['mean_successor_entropy']:.4f} "
              f"(n_types_supported={det_train['n_pair_types_supported']}/{det_train['n_pair_types_total']})",
              flush=True)
        print(f"  block determinism (heldout): type={det_held['type_determinism']:.4f} "
              f"trans_weighted={det_held['transition_weighted_determinism']:.4f} "
              f"mean_entropy={det_held['mean_successor_entropy']:.4f} "
              f"(n_types_supported={det_held['n_pair_types_supported']}/{det_held['n_pair_types_total']})",
              flush=True)
        print(f"  occupancy: min={occ_row['min_occupancy']} occupied={occ_row['n_cells_occupied']}"
              f"/{occ_row['n_cells_nominal']}", flush=True)

    (OUT_DIR / "a0_diagnose_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote artifacts/a0_diagnose_report.json", flush=True)


if __name__ == "__main__":
    main()
