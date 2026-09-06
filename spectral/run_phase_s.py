"""Phase S driver -- runs the spectral pre-test for BOTH label functions
(discovered 3-predicate set, and the tessellation at every swept m), builds
the shuffled null and the model-oracle matrix, and applies the GATE.

The gate depends ONLY on the discovered-predicate data-Hankel rank vs its
own shuffled null (preregistration.md). Tessellation spectra are computed
for comparison and are never allowed to influence the gate.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import build_corpus, test_ids
from oracle.droid_actions import load_actions_for_episodes
from oracle.latent_oracle import LatentOracle
from oracle.lewm_g import ActionPipeline
from oracle.production_oracle import (
    _step_fn_from_convention, alphabet_symbol_names, build_reset_windows,
    load_alphabet_symbols, reset_symbol_names,
)
from predicates.functions import build_predicate_fns
from spectral.hankel import (
    build_matrix, collect_data_observations, collect_model_observations,
    effective_rank, gate_verdict, null_ranks,
)
from tessellation.hyperplanes import Tessellation

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
MODEL_MATRIX_MAX_CELLS = 4000   # oracle evals are real model steps; cap for cost


def _discovered_bit_fns(subset):
    """Returns [f(latents (T,D)) -> (T,) 0/1] for each discovered predicate,
    vectorised over a trajectory's positions."""
    all_fns = build_predicate_fns()
    out = []
    for name in subset:
        fn = all_fns[name]
        out.append(lambda Z, fn=fn: np.array([fn(Z[t]) for t in range(Z.shape[0])], dtype=np.int8))
    return out


def _tessellation_bit_fns(t: Tessellation):
    out = []
    for i in range(t.m):
        out.append(lambda Z, i=i: (t.bits(Z)[:, i]).astype(np.int8))
    return out


def analyse(label_set_name: str, bit_fns, corpus, bit_names, want_model=False,
              oracle=None, reset_names=None):
    per_bit = []
    for bi, fn in enumerate(bit_fns):
        obs = collect_data_observations(corpus, fn)
        built = build_matrix(obs)
        if built[0] is None:
            per_bit.append({"bit": bit_names[bi], "stats": built[3], "note": "no cell reached min observations"})
            continue
        M, rows, cols, stats, mask = built
        dr = effective_rank(M)
        nr = null_ranks(M, mask, seed=SEED + bi)
        entry = {"bit": bit_names[bi], "stats": stats, "data": dr, "null": nr}

        if want_model and oracle is not None:
            # index by position, never list.index() -- rows/cols run to ~1884
            # each, so .index() inside the loop would be O(n^3)
            ri, cj = np.nonzero(mask)
            cells_idx = list(zip(ri.tolist(), cj.tolist()))
            if len(cells_idx) > MODEL_MATRIX_MAX_CELLS:
                rng = np.random.default_rng(SEED)
                sel = rng.choice(len(cells_idx), size=MODEL_MATRIX_MAX_CELLS, replace=False)
                cells_idx = [cells_idx[k] for k in sel]
            cells = [(rows[i], cols[j]) for i, j in cells_idx]
            mobs = collect_model_observations(cells, oracle, reset_names, bi)
            Mm = np.zeros_like(M)
            pos = {(rows[i], cols[j]): (i, j) for i, j in cells_idx}
            for key, v in mobs.items():
                i, j = pos[key]
                Mm[i, j] = v
            entry["model"] = effective_rank(Mm) | {"n_cells_evaluated": len(mobs)}
        per_bit.append(entry)
        print(f"  [{label_set_name}/{bit_names[bi]}] shape={stats['shape']} fill={stats['fill_fraction']:.4f} "
              f"data_rank95={dr['rank_0.95']} null_rank95={nr['null_rank_0.95']['mean']:.1f}"
              f"{' model_rank95=' + str(entry['model']['rank_0.95']) if 'model' in entry else ''}", flush=True)
    return per_bit


def main():
    print("=== Phase S: spectral pre-test ===", flush=True)
    corpus = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)

    with open(OUT_DIR / "best_machine.pkl", "rb") as f:
        best = pickle.load(f)
    subset = best["subset"]
    print(f"discovered predicate set: {subset}", flush=True)

    # frozen theta_0 oracle for the MODEL matrix
    rng = np.random.default_rng(SEED)
    fit_ids = sorted(rng.choice(corpus.episode_ids, size=min(200, len(corpus.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    all_raw = np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0)
    pipeline = ActionPipeline.fit(all_raw)
    reset_windows = build_reset_windows(pipeline)
    symbols = load_alphabet_symbols(ALPHABET_PATH)
    step_raw = _step_fn_from_convention(pipeline)
    all_fns = build_predicate_fns()
    fns = [all_fns[n] for n in subset]

    def step_fn(w, s):
        return step_raw(w, symbols[s])

    def label_fn(w):
        z = w.emb[-1].numpy()
        return tuple(f(z) for f in fns)

    oracle = LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=reset_windows)
    reset_names = reset_symbol_names()

    print("\n--- discovered 3-predicate label set (THE GATE) ---", flush=True)
    disc = analyse("discovered", _discovered_bit_fns(subset), corpus, subset,
                     want_model=True, oracle=oracle, reset_names=reset_names)

    valid = [b for b in disc if "data" in b]
    if not valid:
        verdict = {"passes": False, "reason": "no discovered-predicate Hankel cell reached min observations"}
        mean_data_95 = None
    else:
        mean_data_95 = float(np.mean([b["data"]["rank_0.95"] for b in valid]))
        null_agg = {
            "mean": float(np.mean([b["null"]["null_rank_0.95"]["mean"] for b in valid])),
            "p05": float(np.mean([b["null"]["null_rank_0.95"]["p05"] for b in valid])),
        }
        verdict = gate_verdict(mean_data_95, null_agg)
    print(f"\nGATE: {json.dumps(verdict, indent=2)}", flush=True)

    print("\n--- tessellation label sets (comparison only; cannot affect the gate) ---", flush=True)
    tess_params = np.load(OUT_DIR / "tessellation_params.npz")
    tess_meta = json.loads((OUT_DIR / "tessellation.json").read_text())
    tess_report = {}
    for m in (2, 4, 6, 8):
        t = Tessellation(U=tess_params[f"U_m{m}"], B=tess_params[f"B_m{m}"], m=m, occupied={}, seed=SEED)
        names = [f"h{i}" for i in range(m)]
        tess_report[str(m)] = analyse(f"tess_m{m}", _tessellation_bit_fns(t), corpus, names)

    out = {
        "config": {"min_obs_per_cell": 5, "max_word_len": 3, "n_null_shuffles": 20,
                     "n_heldout_episodes": int(corpus.n_episodes), "n_positions": int(corpus.n_positions),
                     "n_letters": int(corpus.symbols.size)},
        "discovered": {"subset": subset, "per_bit": disc, "mean_data_rank_0.95": mean_data_95},
        "gate": verdict,
        "tessellation": tess_report,
        "best_m": tess_meta["best_m"],
    }
    (OUT_DIR / "spectral_report.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote spectral_report.json", flush=True)
    print(f"\n=== PHASE S GATE: {'PASS -- proceeding to T2' if verdict.get('passes') else 'FAIL -- STOP'} ===", flush=True)


if __name__ == "__main__":
    main()
    import sys as _s, os as _o
    _s.stdout.flush()
    _o._exit(0)
