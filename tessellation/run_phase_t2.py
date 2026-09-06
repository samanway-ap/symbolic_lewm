"""Phase T2 -- DIAGNOSTIC FSM extraction from the tessellation.

No new machinery: phi(z) = cell(z), reusing the existing SUL / L# / PAC
oracle / fidelity / diagnostics unchanged. Two changes for tessellation
extractions ONLY:
  - max_states = 2000 (we are MEASURING blowup, not passing a gate)
  - the resulting machines are diagnostic-only, enforced by
    tessellation/guard.py, not by convention

Scientific payload: put these on the SAME (labels, states, fidelity) axes as
the Phase E predicate-discovery Pareto front. A geometrically neutral
partition is not a Markov partition, so it SHOULD blow up relative to the
6-state predicate machine. If it instead produces a comparably small
machine, that is flagged prominently -- it would mean predicate discovery
was inheriting coarse geometry rather than finding behavioural structure,
which calls Phase C into question.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import build_corpus, test_ids
from gate.acceptance import evaluate
from learning.extract import extract
from oracle.droid_actions import load_actions_for_episodes
from oracle.latent_oracle import LatentOracle
from oracle.lewm_g import ActionPipeline
from oracle.production_oracle import (
    _step_fn_from_convention, alphabet_symbol_names, build_reset_windows,
    load_alphabet_symbols, reset_symbol_names,
)
from tessellation.guard import CELL_PREDICATE_PREFIX, mark_diagnostic
from tessellation.hyperplanes import Tessellation

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
TESSELLATION_MAX_STATES = 2000


def cell_fn_dict(t: Tessellation):
    """{synthetic_predicate_name: phi_i(z)->0/1}, shaped exactly like
    predicates/functions.py's contract so gate/acceptance.py's `evaluate`
    works on tessellation labels with zero modification."""
    fns = {}
    for i in range(t.m):
        name = f"{CELL_PREDICATE_PREFIX}{i}"
        fns[name] = (lambda z, i=i, t=t: int((float(z @ t.U[i]) > float(t.B[i]))))
    return fns


def main():
    tess_params = np.load(OUT_DIR / "tessellation_params.npz")
    tess_meta = json.loads((OUT_DIR / "tessellation.json").read_text())
    best_m = int(tess_meta["best_m"])

    corpus = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    rng = np.random.default_rng(SEED)
    fit_ids = sorted(rng.choice(corpus.episode_ids, size=min(200, len(corpus.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    all_raw = np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0)
    pipeline = ActionPipeline.fit(all_raw)
    reset_windows = build_reset_windows(pipeline)
    symbols = load_alphabet_symbols(ALPHABET_PATH)
    step_raw = _step_fn_from_convention(pipeline)
    full_alphabet = reset_symbol_names() + alphabet_symbol_names(ALPHABET_PATH)

    def step_fn(w, s):
        return step_raw(w, symbols[s])

    fidelity_sample = sorted(rng.choice(corpus.episode_ids, size=min(150, len(corpus.episode_ids)),
                                          replace=False).tolist())

    rows = []
    for m in (2, 4, 6, 8):
        t = Tessellation(U=tess_params[f"U_m{m}"], B=tess_params[f"B_m{m}"], m=m, occupied={}, seed=SEED)
        occ = tess_meta["sweep"][[r["m"] for r in tess_meta["sweep"]].index(m)]["n_cells_occupied"]
        fns = cell_fn_dict(t)
        subset = list(fns.keys())

        def label_fn(w, fns=fns, subset=subset):
            z = w.emb[-1].numpy()
            return tuple(fns[n](z) for n in subset)

        oracle = LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=reset_windows)
        print(f"\n=== T2 extraction: tessellation m={m} ({occ} occupied cells) ===", flush=True)
        res = extract(subset, oracle, full_alphabet, max_learning_rounds=10,
                        max_states_hard=TESSELLATION_MAX_STATES)
        print(f"  status={res.status} n_states={res.n_states} n_queries={res.n_queries} "
              f"time={res.wall_time_s:.1f}s", flush=True)

        row = {"m": m, "n_labels_bits": m, "n_occupied_cells": occ, "status": res.status,
               "n_states": res.n_states, "n_queries": res.n_queries, "wall_time_s": res.wall_time_s}
        if res.status == "converged" and res.machine is not None:
            fid = evaluate(res.machine, label_fn, fns, subset, fidelity_sample, reset_windows, ALPHABET_PATH)
            row.update({k: v for k, v in fid.items() if k != "per_predicate_fidelity"})
            row["per_predicate_fidelity"] = fid.get("per_predicate_fidelity", {})
            row["states_per_occupied_cell"] = res.n_states / max(1, occ)
            print(f"  trace_fidelity(per-bit)={fid['trace_fidelity']:.3f} "
                  f"joint={fid['joint_trace_fidelity']:.3f} coverage={fid['reachable_state_coverage']:.3f} "
                  f"states/occupied_cell={row['states_per_occupied_cell']:.2f}", flush=True)
        rows.append(mark_diagnostic(row))

    # ---- comparison against the Phase E predicate-discovery front ----
    pred_front = [json.loads(l) for l in (OUT_DIR / "search_log.jsonl").read_text().splitlines() if l.strip()]
    pred_rows = [{"n_labels_bits": e["size"], "n_states": e["n_states"],
                    "trace_fidelity": e.get("trace_fidelity"),
                    "joint_trace_fidelity": e.get("joint_trace_fidelity"),
                    "states_per_label": e["n_states"] / max(1, e["size"])} for e in pred_front]

    matched = []
    for r in rows:
        same = [p for p in pred_rows if p["n_labels_bits"] == r["n_labels_bits"]]
        if same and r.get("trace_fidelity") is not None:
            best_pred = min(same, key=lambda p: p["n_states"])
            matched.append({
                "n_labels_bits": r["n_labels_bits"],
                "tessellation_states": r["n_states"], "predicate_states": best_pred["n_states"],
                "tessellation_fidelity": r["trace_fidelity"], "predicate_fidelity": best_pred["trace_fidelity"],
                "state_ratio_tess_over_pred": r["n_states"] / max(1, best_pred["n_states"]),
            })

    out = {
        "max_states_used": TESSELLATION_MAX_STATES,
        "best_m": best_m,
        "tessellation_fsms": rows,
        "predicate_front": pred_rows,
        "matched_label_count_comparison": matched,
        "guard": "diagnostic-only; tessellation/guard.py refuses these in Phase F and Phase K",
    }
    (OUT_DIR / "tessellation_fsm.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote tessellation_fsm.json", flush=True)


if __name__ == "__main__":
    main()
    import sys as _s, os as _o
    _s.stdout.flush()
    _o._exit(0)
