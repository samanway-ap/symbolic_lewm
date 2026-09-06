"""Fast, tiny-scale run-through of the full pipeline (E0 -> probe -> Stage 1
only, one candidate) to catch import/shape/interface bugs BEFORE committing
real GPU-hours to the unattended run. Not a scientific result -- sizes are
cut far below anything statistically meaningful.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import autopilot.base_points as bp_mod
import autopilot.retrieval as retr_mod
import autopilot.evaluate as ev_mod
import autopilot.controller as ctrl_mod

# shrink every scan/pool size for a fast smoke pass
bp_mod.N_CANDIDATE_SCAN = 40
bp_mod.N_BASE_POINTS = 3
retr_mod.N_POOL_SCAN = 60
ev_mod.N_EVAL_POSITIONS = 12
ctrl_mod.CANDIDATE_QUEUE = [{"id": "SMOKE", "m": 2, "suffix": 1, "base_policy": "high_W_ratio", "K": 5}]
ctrl_mod.MAX_GPU_HOURS = 0.15   # ~9 minutes, enough for one tiny stage
ctrl_mod.MAX_TRAINED_NODES = 1


def main():
    from autopilot.common import build_splits, assert_disjoint_splits, load_frozen_directions, load_partial_model
    import copy

    print("=== SMOKE: E0 ===", flush=True)
    manifest = build_splits()
    assert_disjoint_splits(manifest)
    U8, B8 = load_frozen_directions()
    model = load_partial_model()
    theta0_state = copy.deepcopy(model.state_dict())
    pipeline = ctrl_mod.fit_pipeline(manifest["episode_ids"]["route_val"][:50] + manifest["episode_ids"]["replay_train"][:50])

    print("=== SMOKE: eval slices (tiny) ===", flush=True)
    eval_slice = ev_mod.build_eval_slice(manifest["episode_ids"]["route_val"][:15], n_per_episode=2, seed=3072)
    guard_slice = ev_mod.build_eval_slice(manifest["episode_ids"]["global_guard"][:8], n_per_episode=2, seed=3073)
    print(f"  eval_slice samples={len(eval_slice['samples'])}  guard samples={len(guard_slice['samples'])}", flush=True)
    assert len(eval_slice["samples"]) > 0, "SMOKE FAIL: empty eval slice"

    print("=== SMOKE: probe_node ===", flush=True)
    real_latents_pool = ctrl_mod.build_real_latents_pool(manifest["episode_ids"]["route_val"][:40], pipeline, n_episodes=40)
    print(f"  real_latents_pool shape={real_latents_pool.shape}", flush=True)
    cand = ctrl_mod.CANDIDATE_QUEUE[0]
    probe = ctrl_mod.probe_node(cand, manifest, model, pipeline, U8, real_latents_pool)
    assert probe is not None, "SMOKE FAIL: probe_node returned None"
    print(f"  probe OK: {len(probe['base_points'])} base points", flush=True)

    print("=== SMOKE: retrieval (tiny K) + node datasets (built ONCE) ===", flush=True)
    used = set()
    arm_datasets = ctrl_mod.build_node_datasets(cand, probe, manifest, cand["K"], pipeline, used)
    assert arm_datasets is not None, "SMOKE FAIL: build_node_datasets returned None (starved)"
    for arm, ds in arm_datasets.items():
        print(f"  {arm}: {len(ds.samples)} samples", flush=True)

    print("=== SMOKE: run_stage (tiny: 1 seed, 5 updates), reusing the SAME arm_datasets twice ===", flush=True)
    budget = ctrl_mod.GPUBudget(1.0)
    stage = ctrl_mod.run_stage(cand, probe, arm_datasets, model, pipeline, theta0_state, U8, B8,
                                  n_updates=5, seeds=[0], eval_slice=eval_slice,
                                  global_guard_slice=guard_slice, budget=budget, stage_name="smoke")
    print(f"  stage(1st call) status={stage.get('status')} gains={stage.get('gains')}", flush=True)
    stage2 = ctrl_mod.run_stage(cand, probe, arm_datasets, model, pipeline, theta0_state, U8, B8,
                                   n_updates=5, seeds=[0], eval_slice=eval_slice,
                                   global_guard_slice=guard_slice, budget=budget, stage_name="smoke2")
    print(f"  stage(2nd call, reused datasets) status={stage2.get('status')} gains={stage2.get('gains')}", flush=True)
    assert stage["status"] in ("EVALUATED", "STARVED"), f"SMOKE FAIL: unexpected status {stage}"
    assert stage2["status"] in ("EVALUATED", "STARVED"), f"SMOKE FAIL: unexpected status {stage2}"

    print("\n=== SMOKE TEST PASSED ===", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
