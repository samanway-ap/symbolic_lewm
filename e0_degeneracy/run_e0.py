"""E0 driver: rollout collapse (E0.1), reachable cells (E0.2), retroactive
baselines (E0.3), and GATE E0. Everything downstream of this project is
uninterpretable until this passes -- per EXPERIMENTS_orthogonal_expansion.md.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import build_corpus, test_ids
from e0_degeneracy.baselines import (
    compare_to_measured_fidelity, discovered_predicate_baselines, tessellation_baselines,
)
from e0_degeneracy.rollout_collapse import collapse_metrics, reachable_cells, reachable_machine_states, run

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    print("=== E0.1/E0.2: rollout-collapse probe (500 words x 24 steps from z0=r0) ===", flush=True)
    result = run()
    trace, sym_idx, sym_names = result["trace"], result["sym_idx"], result["sym_names"]

    train_blob = np.load(OUT_DIR / "corpus_train.npz")
    global_mean = train_blob["latents"].reshape(-1, train_blob["latents"].shape[-1]).mean(axis=0)

    metrics = collapse_metrics(trace, global_mean)
    print("  ell : total_var : mean_dist_from_global_mean : mean_pairwise_dist", flush=True)
    for i in [0, 3, 7, 11, 15, 19, 23]:
        print(f"    {metrics['ell'][i]:3d} : {metrics['total_variance'][i]:8.4f} : "
              f"{metrics['mean_dist_from_global_mean'][i]:8.4f} : {metrics['mean_pairwise_distance'][i]:8.4f}",
              flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(metrics["ell"], metrics["total_variance"], "o-")
    axes[0].set_xlabel("word length (letters)"); axes[0].set_ylabel("total variance across 500 words")
    axes[0].set_title("E0.1(a): cross-word variance vs length"); axes[0].grid(alpha=0.3)
    axes[1].plot(metrics["ell"], metrics["mean_dist_from_global_mean"], "o-", color="tab:orange")
    axes[1].set_xlabel("word length (letters)"); axes[1].set_ylabel("mean dist from global latent mean")
    axes[1].set_title("E0.1(b): distance from global mean vs length"); axes[1].grid(alpha=0.3)
    axes[2].plot(metrics["ell"], metrics["mean_pairwise_distance"], "o-", color="tab:green")
    axes[2].set_xlabel("word length (letters)"); axes[2].set_ylabel("mean pairwise distance (100-word subsample)")
    axes[2].set_title("E0.1(c): pairwise spread vs length"); axes[2].grid(alpha=0.3)
    plt.suptitle("E0.1 rollout collapse probe (z0=r0, 500 random words, nested prefixes)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "e0_rollout_collapse.png", dpi=140)
    print("  saved e0_rollout_collapse.png", flush=True)

    print("\n=== E0.2: reachable cells vs occupied cells ===", flush=True)
    reach = reachable_cells(trace, result["z0"], OUT_DIR / "tessellation_params.npz")
    tess_meta = json.loads((OUT_DIR / "tessellation.json").read_text())
    occ_by_m = {str(r["m"]): r["n_cells_occupied"] for r in tess_meta["sweep"]}
    for m, r in reach.items():
        print(f"  m={m}: reachable from z0={r['n_reachable']}/{r['n_nominal']} nominal, "
              f"vs occupied-in-real-data={occ_by_m.get(m)}", flush=True)

    with open(OUT_DIR / "best_machine.pkl", "rb") as f:
        best = pickle.load(f)
    machine, subset = best["machine"], best["subset"]
    mstates = reachable_machine_states(sym_idx, sym_names, machine, result["base_reset"])
    print(f"  discovered machine ({machine.size} states): reached {mstates['n_visited']} "
          f"states from z0 under the same 500 words: {mstates['visited']}", flush=True)

    print("\n=== E0.3: retroactive baselines ===", flush=True)
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    disc_baselines = discovered_predicate_baselines(heldout, subset)
    for name, b in disc_baselines.items():
        print(f"  {name}: switch_rate={b['switch_rate']:.4f} persistence={b['persistence_baseline']:.4f} "
              f"majority={b['majority_baseline']:.4f}", flush=True)

    official = json.loads((OUT_DIR / "acceptance_check_official.json").read_text())
    measured = official.get("per_predicate_fidelity", {})
    comparison = compare_to_measured_fidelity(disc_baselines, measured)
    print("\n  measured vs baseline, per discovered predicate:", flush=True)
    for name, c in comparison.items():
        print(f"    {name}: measured={c['measured_fidelity']:.4f} best_baseline="
              f"{max(c['persistence_baseline'], c['majority_baseline']):.4f} "
              f"beats_best_baseline={c['beats_best_baseline']} margin={c['margin_over_best_baseline']:+.4f}",
              flush=True)
    joint_beats_best = official.get("trace_fidelity", 0.0)
    overall_best_baseline = max(
        max(c["persistence_baseline"], c["majority_baseline"]) for c in disc_baselines.values())
    aggregate_beats = joint_beats_best > overall_best_baseline
    print(f"\n  AGGREGATE: official trace_fidelity={joint_beats_best:.4f} vs worst-case single-predicate "
          f"best-baseline={overall_best_baseline:.4f} -> beats={aggregate_beats}", flush=True)

    print("\n  tessellation bits (all swept m):", flush=True)
    tess_base = tessellation_baselines(heldout, OUT_DIR / "tessellation_params.npz")
    t2 = json.loads((OUT_DIR / "tessellation_fsm.json").read_text())
    t2_fidelity_by_m = {str(r["m"]): r.get("per_predicate_fidelity", {}) for r in t2.get("tessellation_fsms", [])}
    tess_comparison = {}
    for m, bits in tess_base.items():
        measured_m = t2_fidelity_by_m.get(m, {})
        tess_comparison[m] = compare_to_measured_fidelity(bits, measured_m)
        for name, c in tess_comparison[m].items():
            print(f"    m={m} {name}: measured={c['measured_fidelity']:.4f} best_baseline="
                  f"{max(c['persistence_baseline'], c['majority_baseline']):.4f} "
                  f"beats_best_baseline={c['beats_best_baseline']}", flush=True)

    # ---------------- GATE E0 ----------------
    var_decays = metrics["total_variance"][-1] < 0.05 * max(metrics["total_variance"][0], 1e-9)
    disc_all_beat = all(c["beats_best_baseline"] for c in comparison.values()) and aggregate_beats
    tess_all_bits = [c for m in tess_comparison.values() for c in m.values()]
    if not tess_all_bits:
        # An empty comparison set must NEVER read as a pass -- all() over zero
        # items is vacuously True, which is exactly how a name mismatch between
        # this file's bit keys and tessellation_fsm.json's ("h{i}" vs the
        # correct "cell_h{i}") silently reported PASS with zero comparisons
        # actually made. Confirmed and fixed 2026-09-02; see preregistration.md.
        raise RuntimeError("tessellation baseline comparison matched ZERO bits -- "
                            "this must not silently pass. Check key naming against "
                            "tessellation_fsm.json's per_predicate_fidelity.")
    tess_all_beat = all(c["beats_best_baseline"] for c in tess_all_bits)
    gate_pass = (not var_decays) and disc_all_beat and tess_all_beat

    verdict = {
        "variance_decays_to_near_zero": bool(var_decays),
        "variance_first": metrics["total_variance"][0], "variance_last": metrics["total_variance"][-1],
        "discovered_predicates_all_beat_baseline": bool(disc_all_beat),
        "discovered_aggregate_beats_baseline": bool(aggregate_beats),
        "tessellation_bits_all_beat_baseline": bool(tess_all_beat),
        "GATE_E0_PASSES": bool(gate_pass),
    }
    print(f"\n=== GATE E0: {'PASS' if gate_pass else 'FAIL -- STOP, per EXPERIMENTS doc'} ===", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)

    out = {
        "config": {"n_words": len(sym_idx), "max_len": trace.shape[1], "base_reset": result["base_reset"]},
        "collapse_metrics": metrics,
        "reachable_cells": reach, "occupied_cells_real_data": occ_by_m,
        "reachable_machine_states": mstates,
        "discovered_predicate_baselines": disc_baselines,
        "discovered_fidelity_vs_baseline": comparison,
        "official_joint_trace_fidelity": joint_beats_best,
        "official_joint_vs_worst_baseline": {"beats": aggregate_beats,
                                                "worst_case_baseline": overall_best_baseline},
        "tessellation_baselines": tess_base,
        "tessellation_fidelity_vs_baseline": tess_comparison,
        "gate": verdict,
    }
    (OUT_DIR / "e0_report.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote e0_report.json + e0_rollout_collapse.png", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
