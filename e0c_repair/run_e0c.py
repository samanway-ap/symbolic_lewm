"""E0c driver: repair the extractor, sweep m in {2,4,6,8} x 3 seeds with the
repaired oracle, assert monotonicity (primary check), compute the
non-triviality index (memory multiplier, suffix depth) against REAL
oracle-reached cells (not occupied), check seed-isomorphism via AALpy's
`bisimilar`, and select k0 by the largest-m-that-qualifies rule -- all per
SELF_IMPROVEMENT_LOOP.md v2 SS2 and EXPERIMENTS_orthogonal_expansion.md v2
SS3 E0c, both pre-registered before this ran.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aalpy.utils import bisimilar
from corpus.latent_corpus import build_corpus, test_ids
from e0c_repair.reachability import reached_cells_from_real_oracle
from e0c_repair.scaled_pac_oracle import ScaledPacOracle
from e0c_repair.suffix_depth import memory_multiplier, suffix_depth_distribution
from learning.extract import extract
from oracle.droid_actions import load_actions_for_episodes
from oracle.latent_oracle import LatentOracle
from oracle.lewm_g import ActionPipeline
from oracle.production_oracle import (
    _step_fn_from_convention, alphabet_symbol_names, build_reset_windows,
    load_alphabet_symbols, reset_symbol_names,
)
from tessellation.hyperplanes import Tessellation

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MACHINES_DIR = OUT_DIR / "e0c_machines"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
SEEDS_PER_LEVEL = [0, 1, 2]
M_VALUES = (2, 4, 6, 8)

# Repaired oracle config (preregistration.md's E0c entry)
PAC_EPSILON = 0.02
PAC_DELTA = 0.05
MAX_LEARNING_ROUNDS = 40
MIN_WORDS_COEF = 4.0
MIN_MEMORY_MULTIPLIER = 1.3
MIN_SUFFIX_DEPTH = 1


def eq_oracle_factory(alphabet, sul, epsilon, delta):
    return ScaledPacOracle(alphabet, sul, epsilon=epsilon, delta=delta,
                              min_walk_len=4, max_walk_len=20, min_words_per_round_coef=MIN_WORDS_COEF)


def main():
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    tp = np.load(OUT_DIR / "tessellation_params.npz")

    rng = np.random.default_rng(SEED)
    fit_ids = sorted(rng.choice(heldout.episode_ids, size=min(200, len(heldout.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    pipeline = ActionPipeline.fit(np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0))
    reset_windows = build_reset_windows(pipeline)
    symbols_dict = load_alphabet_symbols(ALPHABET_PATH)
    full_alphabet = reset_symbol_names() + alphabet_symbol_names(ALPHABET_PATH)
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(w, s):
        return step_fn_raw(w, symbols_dict[s])

    per_level = {}
    for m in M_VALUES:
        print(f"\n{'=' * 60}\n=== m={m} ===\n{'=' * 60}", flush=True)
        tess = Tessellation(U=tp[f"U_m{m}"], B=tp[f"B_m{m}"], m=m, occupied={}, seed=SEED)
        subset_names = [f"cell_h{i}" for i in range(m)]

        def label_fn(w, tess=tess):
            z = w.emb[-1].numpy()
            return tuple(int((float(z @ tess.U[i]) > float(tess.B[i]))) for i in range(tess.m))

        machines, n_states_list, wall_times = [], [], []
        for seed_idx in SEEDS_PER_LEVEL:
            random.seed(SEED + 1000 * m + seed_idx)   # reproducible-per-seed, distinct across seeds
            oracle = LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=reset_windows)
            t0 = time.time()
            res = extract(subset_names, oracle, full_alphabet, max_learning_rounds=MAX_LEARNING_ROUNDS,
                            pac_epsilon=PAC_EPSILON, pac_delta=PAC_DELTA, max_states_hard=2000,
                            eq_oracle_factory=eq_oracle_factory)
            dt = time.time() - t0
            print(f"  seed={seed_idx}: status={res.status} n_states={res.n_states} "
                  f"n_queries={res.n_queries} time={dt:.1f}s", flush=True)
            machines.append(res.machine if res.status == "converged" else None)
            n_states_list.append(res.n_states if res.status == "converged" else None)
            wall_times.append(dt)
            if res.status == "converged" and res.machine is not None:
                # Persisted so E1 (and anything later) can load the machine k0
                # selection settles on WITHOUT re-running this expensive
                # extraction a second time.
                import pickle
                MACHINES_DIR.mkdir(parents=True, exist_ok=True)
                with open(MACHINES_DIR / f"m{m}_seed{seed_idx}.pkl", "wb") as f:
                    pickle.dump({"machine": res.machine, "m": m, "seed_idx": seed_idx,
                                  "subset": subset_names}, f)

        valid_machines = [mm for mm in machines if mm is not None]
        seeds_isomorphic = None
        if len(valid_machines) >= 2:
            seeds_isomorphic = all(bool(bisimilar(valid_machines[0], mm)) for mm in valid_machines[1:])
            print(f"  seed isomorphism (all vs seed 0): {seeds_isomorphic}", flush=True)
        else:
            print("  fewer than 2 converged machines -- cannot check seed isomorphism", flush=True)

        print("  computing reached cells from the REAL oracle (8 resets)...", flush=True)
        reach = reached_cells_from_real_oracle(reset_windows, symbols_dict, tess, pipeline, seed=SEED)
        print(f"  reached_cells={reach['n_reached']} (from {reach['n_resets_used']} resets x "
              f"{reach['n_words_per_reset']} words x {reach['word_len']} letters)", flush=True)

        rep_machine = valid_machines[0] if valid_machines else None
        suffix_dist = suffix_depth_distribution(rep_machine, full_alphabet) if rep_machine else None
        mu = memory_multiplier(n_states_list[0], reach["n_reached"]) if n_states_list[0] is not None else None
        if suffix_dist:
            print(f"  suffix depth: median={suffix_dist['median']:.1f} max={suffix_dist['max']} "
                  f"(n_pairs={suffix_dist['n_pairs']})", flush=True)
        if mu is not None:
            print(f"  memory multiplier mu = |Q|/reached = {n_states_list[0]}/{reach['n_reached']} = {mu:.3f}",
                  flush=True)

        per_level[m] = {
            "n_states_per_seed": n_states_list, "wall_times_s": wall_times,
            "seeds_isomorphic": seeds_isomorphic, "reached_cells": reach["n_reached"],
            "reached_cell_ids": reach["reached_cell_ids"],
            "suffix_depth_median": suffix_dist["median"] if suffix_dist else None,
            "suffix_depth_max": suffix_dist["max"] if suffix_dist else None,
            "memory_multiplier": mu,
            "moore_bound_vs_reached_ok": (n_states_list[0] >= reach["n_reached"]) if n_states_list[0] is not None else None,
        }
        (OUT_DIR / "e0c_report.json").write_text(json.dumps(per_level, indent=2, default=float))

    # ---------------- monotonicity assertion (primary check) ----------------
    print(f"\n{'=' * 60}\n=== monotonicity + non-triviality + k0 selection ===\n{'=' * 60}", flush=True)
    medians = {}
    for m in M_VALUES:
        vals = [v for v in per_level[m]["n_states_per_seed"] if v is not None]
        medians[m] = float(np.median(vals)) if vals else None
    print(f"  median |Q| per level: {medians}", flush=True)
    ordered = [medians[m] for m in M_VALUES if medians[m] is not None]
    monotone_ok = all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))
    print(f"  MONOTONICITY (|Q_2|<=|Q_4|<=|Q_6|<=|Q_8|, median over seeds): "
          f"{'HOLDS' if monotone_ok else 'VIOLATED'}", flush=True)

    qualifying = []
    for m in M_VALUES:
        pl = per_level[m]
        mu, sd, iso = pl["memory_multiplier"], pl["suffix_depth_median"], pl["seeds_isomorphic"]
        ok = (mu is not None and mu >= MIN_MEMORY_MULTIPLIER and
               sd is not None and sd > MIN_SUFFIX_DEPTH and iso is True)
        pl["qualifies_non_triviality"] = ok
        print(f"  m={m}: mu={mu}, suffix_depth_median={sd}, seeds_isomorphic={iso} -> qualifies={ok}", flush=True)
        if ok:
            qualifying.append(m)

    if monotone_ok and qualifying:
        k0 = max(qualifying)
        route = f"E1 at k0={k0}"
    elif monotone_ok and not qualifying:
        all_mu_near_1 = all((per_level[m]["memory_multiplier"] or 0) < 1.3 for m in M_VALUES)
        if all_mu_near_1:
            route = ("monotonicity holds but mu~=1.0 at every level -- route to lengthening the horizon "
                       "before concluding no memory exists (may be a horizon artefact, not a property)")
        else:
            route = "monotonicity holds, no level clears both non-triviality thresholds -- see per-level detail"
        k0 = None
    else:
        route = ("monotonicity STILL VIOLATED after repair -- failure is in the LEARNER, not sampling; "
                   "swap TTT's equivalence oracle for the abstraction-refinement variant (original plan "
                   "SS7.2(b), never implemented) and re-test")
        k0 = None

    if k0 is None and len([m for m in M_VALUES if per_level[m]["seeds_isomorphic"]]) == 1:
        only_m = [m for m in M_VALUES if per_level[m]["seeds_isomorphic"]][0]
        if only_m == 2:
            route += ". Only m=2 reproducible: E1 may run there, but its numbers are a LOWER BOUND, not an estimate."
            k0 = 2

    print(f"\n  k0 = {k0}", flush=True)
    print(f"  ROUTE: {route}", flush=True)

    out = {
        "config": {"pac_epsilon": PAC_EPSILON, "pac_delta": PAC_DELTA,
                     "max_learning_rounds": MAX_LEARNING_ROUNDS, "min_words_per_round_coef": MIN_WORDS_COEF,
                     "min_memory_multiplier": MIN_MEMORY_MULTIPLIER, "min_suffix_depth": MIN_SUFFIX_DEPTH},
        "per_level": per_level, "median_n_states": medians, "monotone_ok": monotone_ok,
        "qualifying_levels": qualifying, "k0": k0, "route": route,
    }
    (OUT_DIR / "e0c_report.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote e0c_report.json", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
