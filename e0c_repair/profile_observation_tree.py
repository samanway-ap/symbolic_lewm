"""cProfile pass over the SAME m=2/seed=0 extraction the timing probe ran,
per preregistration.md's "E0c trajectory-stability check" entry: the
explosion (11->74 states, round 4->5) is now believed to be REAL structure,
not noise, so the remaining blocker is purely AALpy's own `ObservationTree`
bookkeeping (`build_hypothesis`/`process_counter_example` in
`aalpy/learning_algs/deterministic/LSharp.py`) not scaling gracefully with
hypothesis size. This finds the SPECIFIC superlinear operation rather than
just confirming bookkeeping-not-oracle-sampling is the cost (already known).

Capped at the same round count as `profile_timing.py` (default 7) so this
profiles exactly the segment that stalled >9 minutes previously, not a fresh
unbounded run.
"""
from __future__ import annotations

import cProfile
import io
import pstats
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
PROBE_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
SEPARATION_RULE = sys.argv[2] if len(sys.argv) > 2 else "ADS"

PAC_EPSILON = 0.02
PAC_DELTA = 0.05
MIN_WORDS_COEF = 4.0


def main():
    print(f"cProfile probe: m={M}, capped at {PROBE_ROUNDS} rounds, separation_rule={SEPARATION_RULE}", flush=True)

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
    eq_oracle = ScaledPacOracle(full_alphabet, sul, epsilon=PAC_EPSILON, delta=PAC_DELTA,
                                  min_walk_len=4, max_walk_len=20, min_words_per_round_coef=MIN_WORDS_COEF)

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    try:
        run_Lsharp(full_alphabet, sul, eq_oracle, automaton_type="moore", separation_rule=SEPARATION_RULE,
                    max_learning_rounds=PROBE_ROUNDS, print_level=2, return_data=True)
    finally:
        profiler.disable()
    dt = time.time() - t0
    print(f"\ntotal wall time: {dt:.1f}s, n_queries={sul.num_queries}", flush=True)

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("cumulative")
    stats.print_stats(35)
    print("\n=== top 35 by CUMULATIVE time ===", flush=True)
    print(buf.getvalue(), flush=True)

    buf2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=buf2)
    stats2.sort_stats("tottime")
    stats2.print_stats(25)
    print("\n=== top 25 by SELF (tottime) ===", flush=True)
    print(buf2.getvalue(), flush=True)

    out_name = f"e0c_profile_{SEPARATION_RULE.lower()}.pstats"
    profiler.dump_stats(str(OUT_DIR / out_name))
    print(f"wrote artifacts/{out_name}", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
