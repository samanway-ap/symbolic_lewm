"""Phase F gate #5: 3-seed reproducibility. "3 extractions, different
query seeds, agree up to isomorphism" (plan §9). Never implemented in the
original run -- moot there since trace_fidelity/coverage already failed.
Implemented now since this rerun's fixes may actually clear those gates.

Re-extracts the OFFICIALLY ACCEPTED predicate subset 2 more times (query
randomness differs run-to-run because AALpy's PacOracle draws from
Python's global `random` state, which has already advanced by the time
this script's own extraction calls run -- no explicit seed threading
needed) and checks each result is bisimilar to the accepted machine via
AALpy's own `aalpy.utils.bisimilar` (bisimilarity implies isomorphism of
the reachable/minimal part for deterministic Moore machines -- the
practically meaningful notion here, not a weaker stand-in for it).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aalpy.utils import bisimilar
from learning.extract import extract
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import ActionPipeline
from oracle.latent_oracle import LatentOracle
from oracle.production_oracle import (
    _step_fn_from_convention, alphabet_symbol_names, build_reset_windows,
    load_alphabet_symbols, reset_symbol_names,
)
from predicates.functions import build_predicate_fns, make_label_fn

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
N_EXTRA_EXTRACTIONS = 2


def main():
    with open(OUT_DIR / "best_machine.pkl", "rb") as f:
        best = pickle.load(f)
    accepted_machine, subset = best["machine"], best["subset"]
    print(f"reproducibility check: re-extracting subset={subset} {N_EXTRA_EXTRACTIONS} more times "
          f"(accepted machine: {accepted_machine.size} states)", flush=True)

    all_fns = build_predicate_fns()
    label_fn = make_label_fn(subset, all_fns)

    alphabet_names = alphabet_symbol_names(ALPHABET_PATH)
    reset_names = reset_symbol_names()
    full_alphabet = reset_names + alphabet_names

    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    train_pool = sorted(set(split["train_episode_ids"]) - reset_ids)

    rng = np.random.default_rng(SEED)
    fit_ids = sorted(rng.choice(train_pool, size=300, replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    all_raw = np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0)
    pipeline = ActionPipeline.fit(all_raw)

    reset_windows = build_reset_windows(pipeline)
    symbols = load_alphabet_symbols(ALPHABET_PATH)
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(window, symbol):
        return step_fn_raw(window, symbols[symbol])

    results = []
    all_bisimilar = True
    for i in range(N_EXTRA_EXTRACTIONS):
        oracle = LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=reset_windows)
        result = extract(subset, oracle, full_alphabet, max_learning_rounds=10)
        ok = False
        if result.status == "converged" and result.machine is not None:
            ok = bool(bisimilar(accepted_machine, result.machine))
        all_bisimilar = all_bisimilar and ok
        print(f"  re-extraction {i + 1}/{N_EXTRA_EXTRACTIONS}: status={result.status} "
              f"n_states={result.n_states} bisimilar_to_accepted={ok}", flush=True)
        results.append({"status": result.status, "n_states": result.n_states, "bisimilar_to_accepted": ok})

    out = {"subset": subset, "accepted_n_states": accepted_machine.size,
           "re_extractions": results, "reproducible": all_bisimilar}
    (OUT_DIR / "reproducibility_check.json").write_text(json.dumps(out, indent=2))
    print(f"\nreproducibility gate: {'PASS' if all_bisimilar else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    import os
    os._exit(0)
