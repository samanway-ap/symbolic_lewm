"""Phase E: greedy ADD-ONLY growth (sizes 3..8), still time-constrained
(drop/CEGAR, plan §8.3, remain out of scope for this rerun -- see
preregistration.md's FIX 1 entry). Each add-step now does what plan §8.2
point 2 actually specifies: rank the remaining pool by a cheap,
extraction-free proxy (search/greedy.py; FIX 1, 2026-08-21 -- trace-
fidelity of a majority-vote table, not block-nondeterminism reduction),
run real extractions on only the top-3 ranked candidates, and commit
whichever of those 3 real extractions has the best J(S). Budget ~20
extractions total. Each extraction is a real Angluin run against the real
oracle -- the first (seed) one IS Phase D's M4 milestone.

J(S) = n_states + 1000*max(0, 0.85 - trace_fidelity) + 0.1*|S| ; never
trade fidelity for compactness (the 1000x penalty dominates whenever
fidelity is below threshold). `trace_fidelity` here is the FIX 3b
per-predicate metric from gate/acceptance.py, measured at the FIX 3a
distinguishing-suffix horizon -- not the original run's joint/L_max
metric.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate.acceptance import evaluate, load_alphabet_feats
from learning.extract import extract
from oracle.lewm_g import ActionPipeline
from oracle.production_oracle import (
    alphabet_symbol_names, build_reset_windows, load_alphabet_symbols,
    reset_symbol_names, _step_fn_from_convention,
)
from oracle.droid_actions import load_actions_for_episodes
from oracle.latent_oracle import LatentOracle
from predicates.functions import all_predicate_names, build_predicate_fns, make_label_fn
from search.greedy import PROXY_TOPK, block_nondeterminism_proxy, build_proxy_transitions, rank_by_proxy

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
MAX_ACTIVE = 8
FIDELITY_THRESH = 0.85
SEED = 3072
BUDGET_EXTRACTIONS = 20


def ranked_candidates() -> list[str]:
    summary = json.loads((OUT_DIR / "predicate_pool_summary.json").read_text())
    cands = summary["candidates"]
    # slow_feature has a "score" concept baked into hygiene ordering already
    # (discover.py appends in score order within family); VQ/bisim don't
    # carry an explicit score, so just keep discovery order.
    names = [f"{c['family']}__{c['name']}" for c in cands]
    return names


def main():
    print("loading predicate functions + alphabet...", flush=True)
    all_fns = build_predicate_fns()
    candidates = [n for n in ranked_candidates() if n in all_fns]
    print(f"{len(candidates)} candidates available for search", flush=True)

    alphabet_names = alphabet_symbol_names(ALPHABET_PATH)
    reset_names = reset_symbol_names()
    full_alphabet = reset_names + alphabet_names

    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    scope_ids = json.loads((OUT_DIR / "scope_episode_ids.json").read_text())
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    train_pool = sorted(set(split["train_episode_ids"]) - reset_ids)

    # fixed pipeline for the whole search (fit once on a broad train sample)
    rng = np.random.default_rng(SEED)
    fit_ids = sorted(rng.choice(train_pool, size=300, replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    all_raw = np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0)
    pipeline = ActionPipeline.fit(all_raw)
    print("fitted global ActionPipeline for search", flush=True)

    reset_windows = build_reset_windows(pipeline)
    symbols = load_alphabet_symbols(ALPHABET_PATH)
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(window, symbol):
        return step_fn_raw(window, symbols[symbol])

    # fidelity-scoring episode sample (train, disjoint-ish from discovery's
    # own 177; not the canonical test-20% -- that's reserved for Phase F)
    fidelity_pool = sorted(set(train_pool) - set(fit_ids))
    fidelity_sample = sorted(rng.choice(fidelity_pool, size=80, replace=False).tolist())

    # FIX 1: cheap extraction-free proxy-transition dataset for add-step
    # ranking, held out from the fidelity-scoring sample above too
    print("\nbuilding proxy-transition dataset (add-step ranking, FIX 1)...", flush=True)
    alphabet_feats, alphabet_medoid_names = load_alphabet_feats(ALPHABET_PATH)
    proxy_pool_ids = sorted(set(fidelity_pool) - set(fidelity_sample))
    proxy_data = build_proxy_transitions(proxy_pool_ids, alphabet_feats, alphabet_medoid_names, seed=SEED)
    print(f"  proxy dataset: {proxy_data['n_episodes']} episodes, {proxy_data['n_transitions']} transitions "
          f"({len(proxy_data['fit_idx'])} fit / {len(proxy_data['ho_idx'])} held-out)", flush=True)

    search_log = []
    extractions_used = 0
    best = None

    def run_extraction(subset: list[str]):
        nonlocal extractions_used
        label_fn = make_label_fn(subset, all_fns)
        oracle = LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=reset_windows)
        result = extract(subset, oracle, full_alphabet, max_learning_rounds=10)
        extractions_used += 1
        print(f"  [extraction {extractions_used}/{BUDGET_EXTRACTIONS}] |S|={len(subset)} subset={subset} "
              f"status={result.status} n_states={result.n_states} n_queries={result.n_queries} "
              f"time={result.wall_time_s:.1f}s", flush=True)
        entry = {"size": len(subset), "subset": list(subset), "status": result.status,
                  "n_states": result.n_states, "n_queries": result.n_queries,
                  "wall_time_s": result.wall_time_s}
        machine = None
        J = float("inf")
        if result.status == "converged" and result.machine is not None:
            fid = evaluate(result.machine, label_fn, all_fns, subset, fidelity_sample, reset_windows, ALPHABET_PATH)
            entry.update(fid)
            entry["block_nondeterminism_proxy_diagnostic"] = block_nondeterminism_proxy(subset, all_fns, proxy_data)
            J = result.n_states + 1000 * max(0, FIDELITY_THRESH - fid["trace_fidelity"]) + 0.1 * len(subset)
            entry["J"] = J
            print(f"    trace_fidelity(per-pred,H={fid['fidelity_horizon']})={fid['trace_fidelity']:.3f} "
                  f"joint_trace_fidelity={fid['joint_trace_fidelity']:.3f} "
                  f"coverage={fid['reachable_state_coverage']:.3f} "
                  f"block_nondeterminism={fid['block_nondeterminism']:.3f} J={J:.2f}", flush=True)
            machine = result.machine
        else:
            entry["J"] = float("inf")
        search_log.append({k: v for k, v in entry.items() if k != "machine"})
        (OUT_DIR / "search_log.jsonl").write_text("\n".join(json.dumps(e) for e in search_log))
        return entry, machine, J

    # seed: 3 highest-scoring family-(ii)-first candidates (plan §8.2 point 1)
    S = candidates[:3]
    remaining_pool = [c for c in candidates if c not in S]
    print(f"\n=== seed extraction: |S|={len(S)} {S} ===", flush=True)
    entry, machine, J = run_extraction(S)
    if machine is not None:
        best = {**entry, "machine": machine}

    while len(S) < MAX_ACTIVE and extractions_used < BUDGET_EXTRACTIONS and remaining_pool:
        print(f"\n=== add-step: |S|={len(S)} -> {len(S) + 1}; ranking {len(remaining_pool)} remaining "
              f"by FIX-1 proxy fidelity ===", flush=True)
        ranked = rank_by_proxy(remaining_pool, S, all_fns, proxy_data, topk=PROXY_TOPK)
        print("  top proxy candidates: " + ", ".join(f"{p}(proxy={score:.3f})" for p, score in ranked), flush=True)

        step_attempts = []
        for p, proxy_score in ranked:
            if extractions_used >= BUDGET_EXTRACTIONS:
                print("  extraction budget exhausted mid-step", flush=True)
                break
            entry, machine, J = run_extraction(S + [p])
            entry["proxy_fidelity_score"] = proxy_score
            step_attempts.append((p, entry, machine, J))
            if best is None or J < best["J"]:
                best = {**entry, "machine": machine} if machine is not None else best

        if not step_attempts:
            break
        p_star, entry_star, machine_star, J_star = min(step_attempts, key=lambda x: x[3])
        S = S + [p_star]
        remaining_pool = [c for c in remaining_pool if c != p_star]
        print(f"  committed: added '{p_star}' (J={J_star:.2f}) -> |S|={len(S)}", flush=True)

    print("\n=== search complete ===", flush=True)
    print(f"extractions used: {extractions_used}/{BUDGET_EXTRACTIONS}", flush=True)
    if best is None:
        print("NO CONVERGED EXTRACTION -- search failed to produce any candidate machine", flush=True)
        (OUT_DIR / "search_result.json").write_text(json.dumps({"status": "no_candidate"}, indent=2))
        return

    print(f"best: |S|={best['size']} n_states={best['n_states']} "
          f"trace_fidelity={best['trace_fidelity']:.3f} J={best['J']:.2f}", flush=True)
    (OUT_DIR / "search_result.json").write_text(json.dumps(
        {k: v for k, v in best.items() if k != "machine"}, indent=2))
    import pickle
    with open(OUT_DIR / "best_machine.pkl", "wb") as f:
        pickle.dump({"machine": best["machine"], "subset": best["subset"]}, f)
    print("wrote search_result.json + best_machine.pkl")


if __name__ == "__main__":
    main()
    # Defensive: a lingering non-daemon thread (observed twice this session,
    # source not isolated) can keep the interpreter alive well after all
    # real work is flushed to disk. Everything main() produces is already
    # written -- force immediate exit rather than risk another silent hang.
    import sys
    sys.stdout.flush()
    import os
    os._exit(0)
