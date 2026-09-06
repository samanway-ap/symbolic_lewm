"""Phase F: official acceptance check, on the canonical TEST-20% split
(not the train sample search/run_search.py used for its own scoring)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate.acceptance import evaluate
from oracle.lewm_g import ActionPipeline
from oracle.production_oracle import build_reset_windows
from oracle.droid_actions import load_actions_for_episodes
from predicates.functions import build_predicate_fns, make_label_fn

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
GATES = {"n_states_max": 50, "trace_fidelity_min": 0.85, "block_nondeterminism_max": 0.10, "coverage_min": 0.90}


def main():
    with open(OUT_DIR / "best_machine.pkl", "rb") as f:
        best = pickle.load(f)
    machine, subset = best["machine"], best["subset"]
    print(f"evaluating best candidate: |S|={len(subset)} n_states={machine.size}", flush=True)

    all_fns = build_predicate_fns()
    label_fn = make_label_fn(subset, all_fns)

    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    test_ids = split["test_episode_ids"]
    rng = np.random.default_rng(SEED + 999)
    test_sample = sorted(rng.choice(test_ids, size=min(150, len(test_ids)), replace=False).tolist())

    fit_actions = load_actions_for_episodes(test_sample[:100], max_frames=40)
    all_raw = np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0)
    pipeline = ActionPipeline.fit(all_raw)
    reset_windows = build_reset_windows(pipeline)

    result = evaluate(machine, label_fn, all_fns, subset, test_sample, reset_windows, ALPHABET_PATH)
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n[FIX 3a/3b] gated trace_fidelity (per-predicate, horizon={result['fidelity_horizon']})="
          f"{result['trace_fidelity']:.3f} vs. joint_trace_fidelity (diagnostic only, full horizon)="
          f"{result['joint_trace_fidelity']:.3f}", flush=True)

    gates_passed = {
        "n_states": machine.size <= GATES["n_states_max"],
        "trace_fidelity": result["trace_fidelity"] >= GATES["trace_fidelity_min"],
        "block_nondeterminism": result["block_nondeterminism"] <= GATES["block_nondeterminism_max"],
        "coverage": result["reachable_state_coverage"] >= GATES["coverage_min"],
    }
    all_pass = all(gates_passed.values())
    print(f"\ngates: {gates_passed}", flush=True)
    print(f"ALL GATES PASS (excl. reproducibility): {all_pass}", flush=True)

    out = {"subset": subset, "n_states": machine.size, **result, "gates_passed": gates_passed,
           "kill_criterion_F_triggered": not all_pass}
    (OUT_DIR / "acceptance_check_official.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    import os
    os._exit(0)
