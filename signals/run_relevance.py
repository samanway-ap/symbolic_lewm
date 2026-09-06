"""STEP 1 of the exploratory Phase G/H/I run: compute the relevance
subspace and apply the PRE-REGISTERED r/d decision rule BEFORE any arm is
trained (preregistration.md, "Phase G/H/I EXPLORATORY RUN" -> STEP 1).

Rule, fixed before measurement:
  r/d >  0.8  -> latent_subspace is vacuous; DROP it and its shuffled twin.
  r/d <= 0.8  -> keep both.

Writes artifacts/arm_decision.json, which training/finetune.py reads to
build its arm list -- the decision is a recorded artifact, not a judgement
call re-made at training time.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signals.relevance_subspace import compute_relevance_subspace
from training.data import build_finetune_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CACHE = OUT_DIR / "finetune_stream_cache.npz"
R_OVER_D_THRESHOLD = 0.8


def load_dataset():
    with open(OUT_DIR / "best_machine.pkl", "rb") as f:
        best = pickle.load(f)
    machine, subset = best["machine"], best["subset"]
    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    train_pool = sorted(set(split["train_episode_ids"]) - reset_ids)
    alphabet_path = OUT_DIR / "alphabet_trainfit.json"
    dataset = build_finetune_dataset(machine, subset, train_pool, alphabet_path, cache_path=CACHE)
    return machine, subset, dataset, alphabet_path


def main():
    machine, subset, dataset, alphabet_path = load_dataset()
    print(f"machine: |S|={len(subset)} n_states={machine.size} (Phase-F-REJECTED, exploratory run)", flush=True)

    relevance = compute_relevance_subspace(machine, subset, dataset, alphabet_path)
    np.savez(OUT_DIR / "relevance_subspace.npz", P_V=relevance["P_V"],
             explained_variance_curve=np.array(relevance["explained_variance_curve"]))
    summary = {k: v for k, v in relevance.items() if k != "P_V"}
    (OUT_DIR / "relevance_subspace_summary.json").write_text(json.dumps(summary, indent=2))

    r, d, rd = relevance["r"], relevance["d"], relevance["r_over_d"]
    vacuous = rd > R_OVER_D_THRESHOLD
    decision = {
        "r": r, "d": d, "r_over_d": rd,
        "threshold": R_OVER_D_THRESHOLD,
        "latent_subspace_vacuous": bool(vacuous),
        "rule": "pre-registered before measurement: r/d > 0.8 -> drop latent_subspace + shuffled twin",
        "arms": (["control", "sample_prio", "shuffled_sample_prio", "time_binned_sample_prio"]
                 if vacuous else
                 ["control", "sample_prio", "shuffled_sample_prio", "time_binned_sample_prio",
                  "latent_subspace", "shuffled_latent_subspace"]),
        "n_gradient_rows": relevance["n_gradient_rows"],
        "n_pairs_probed": len(relevance["per_pair"]),
        "n_samples": len(dataset.samples),
    }
    (OUT_DIR / "arm_decision.json").write_text(json.dumps(decision, indent=2))

    print("\n=== STEP 1 DECISION ===", flush=True)
    print(f"r={r}  d={d}  r/d={rd:.4f}  (threshold {R_OVER_D_THRESHOLD})", flush=True)
    print(f"latent_subspace vacuous: {vacuous}", flush=True)
    print(f"ARMS -> {decision['arms']}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
