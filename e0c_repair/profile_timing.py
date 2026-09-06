"""Timing probe for the repaired oracle, per HANDOFF_ORTHOGONAL_EXPANSION.md
SS4 option 1: "Instrument ScaledPacOracle.find_cex to print num_test_cases
per round and time per round; confirm whether cost is dominated by the
round-scaling term, the size floor, or word length."

Runs ONLY m=2, seed=0, capped at a handful of rounds (not the full 40) --
the point is per-round wall-clock and num_test_cases, not a converged
machine. Prints one line per equivalence round so a resuming session can
decide options 2/3 from this repo's own real numbers instead of the
back-of-envelope in the handoff.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aalpy.learning_algs import run_Lsharp
from corpus.latent_corpus import build_corpus, test_ids
from e0c_repair.scaled_pac_oracle import ScaledPacOracle
from learning.sul import LeWMSUL
from oracle.droid_actions import load_actions_for_episodes
from oracle.latent_oracle import LatentOracle
from oracle.lewm_g import ActionPipeline
from oracle.production_oracle import (
    _step_fn_from_convention, alphabet_symbol_names, build_reset_windows,
    load_alphabet_symbols, reset_symbol_names,
)
from tessellation.hyperplanes import Tessellation

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
M = 2
PROBE_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6

PAC_EPSILON = 0.02
PAC_DELTA = 0.05
MIN_WORDS_COEF = 4.0


class LoggingScaledPacOracle(ScaledPacOracle):
    def find_cex(self, hypothesis):
        t0 = time.time()
        n_states_before = len(hypothesis.states)
        cex = super().find_cex(hypothesis)
        dt = time.time() - t0
        print(f"    round={self.round:3d}  n_states_hyp={n_states_before:4d}  "
              f"num_test_cases={self.last_num_test_cases:8.1f}  "
              f"cex_found={'yes' if cex else 'no '}  wall_time={dt:7.2f}s  "
              f"per_word={dt / max(self.last_num_test_cases, 1) * 1000:6.2f}ms", flush=True)
        return cex


def main():
    print(f"probe: m={M}, epsilon={PAC_EPSILON}, delta={PAC_DELTA}, "
          f"min_words_coef={MIN_WORDS_COEF}, capped at {PROBE_ROUNDS} rounds", flush=True)

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

    tess = Tessellation(U=tp[f"U_m{M}"], B=tp[f"B_m{M}"], m=M, occupied={}, seed=SEED)

    def label_fn(w, tess=tess):
        z = w.emb[-1].numpy()
        return tuple(int((float(z @ tess.U[i]) > float(tess.B[i]))) for i in range(tess.m))

    import random
    random.seed(SEED)
    oracle = LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=reset_windows)
    sul = LeWMSUL(oracle)
    eq_oracle = LoggingScaledPacOracle(full_alphabet, sul, epsilon=PAC_EPSILON, delta=PAC_DELTA,
                                          min_walk_len=4, max_walk_len=20,
                                          min_words_per_round_coef=MIN_WORDS_COEF)

    t_start = time.time()
    try:
        result = run_Lsharp(full_alphabet, sul, eq_oracle, automaton_type="moore",
                              max_learning_rounds=PROBE_ROUNDS, print_level=0, return_data=True)
        machine, info = result if isinstance(result, tuple) else (result, {})
        print(f"\nconverged early at n_states={machine.size} "
              f"total_time={time.time() - t_start:.1f}s n_queries={sul.num_queries}", flush=True)
    except Exception as e:  # noqa: BLE001 -- probe hitting the round cap raises inside AALpy; that's fine, we already logged
        print(f"\nprobe stopped ({type(e).__name__}: {e}) after "
              f"total_time={time.time() - t_start:.1f}s n_queries={sul.num_queries}", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
