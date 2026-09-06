"""Targeted smoke test for the NEW suffix=2 (two-letter proxy) code path
only -- the rest of the pipeline is unchanged from smoke_test.py, which
already passed. Tiny scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import autopilot.base_points as bp_mod
import autopilot.retrieval as retr_mod
import autopilot.controller as ctrl_mod

bp_mod.N_CANDIDATE_SCAN = 40
bp_mod.N_BASE_POINTS = 3
retr_mod.N_POOL_SCAN = 60
cand = {"id": "SMOKE05", "m": 4, "suffix": 2, "base_policy": "high_W_ratio", "K": 5}


def main():
    from autopilot.common import build_splits, load_frozen_directions, load_partial_model

    print("=== SMOKE-SUFFIX2: setup ===", flush=True)
    manifest = build_splits()
    U8, B8 = load_frozen_directions()
    model = load_partial_model()
    pipeline = ctrl_mod.fit_pipeline(manifest["episode_ids"]["route_val"][:50] + manifest["episode_ids"]["replay_train"][:50])

    print("=== SMOKE-SUFFIX2: probe_node with suffix=2 ===", flush=True)
    real_latents_pool = ctrl_mod.build_real_latents_pool(manifest["episode_ids"]["route_val"][:40], pipeline, n_episodes=40)
    probe = ctrl_mod.probe_node(cand, manifest, model, pipeline, U8, real_latents_pool)
    assert probe is not None, "SMOKE FAIL: probe_node returned None for suffix=2"
    print(f"  probe OK: {len(probe['base_points'])} base points", flush=True)
    for bp in probe["base_points"][:3]:
        g = bp["geometry"]
        print(f"    dim_T={g['dim_T']} dim_V={g['dim_V']} dim_W={g['dim_W']} W_ratio={g['W_ratio']:.3f}", flush=True)

    print("\n=== SMOKE-SUFFIX2 PASSED ===", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
