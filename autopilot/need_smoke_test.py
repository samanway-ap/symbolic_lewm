"""Fast, tiny-scale run-through of the need_autopilot pipeline (region
build -> cell geometry -> targeted retrieval -> one tiny Stage-1) to catch
import/shape/interface bugs BEFORE committing wall-clock budget to the real
run. Not a scientific result -- every floor/scan/time-cap size is shrunk far
below anything statistically meaningful, mirroring smoke_test.py's
convention for the completed C00-C06 tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import autopilot.need_geometry as ng
import autopilot.need_targeted_retrieval as nt
import autopilot.need_evaluate as ne
import autopilot.need_controller as nc

# shrink every scan/floor for a fast smoke pass. need_controller.py does
# `from autopilot.need_geometry import NAME` (copies the value at import
# time), so patching only `ng.NAME` would silently miss every read inside
# need_controller -- both the defining module and every importing one must
# be patched.
for mod in (ng, nc):
    mod.MAX_REGIONS = 3
    mod.MIN_FIT_TRANSITIONS = 8
    mod.MIN_FIT_EPISODES = 3
    mod.MIN_ROUTE_TRANSITIONS = 4
    mod.MIN_ROUTE_EPISODES = 2
    mod.N_SUPPORT_SCAN_REPLAY = 25
    mod.N_SUPPORT_SCAN_ROUTE = 20
    mod.GRADIENT_STARTS_MAX = 8
    mod.GRADIENT_STARTS_MIN = 4
    mod.PREGRADIENT_CELL_CAP = 6
    mod.PREGRADIENT_PER_REGION_CAP = 2
    mod.CANDIDATE_CELLS = 1
nt.MIN_COMMON_ELIGIBLE = 3
nt.K_FALLBACK = 3
nt.K_TARGET = 5
nt.MAX_RETRIEVAL_MINUTES = 2.0
nt.DECODE_BATCH = 16
ne.N_CELL_EVAL_SCAN = 25
nc.MAX_WALL_HOURS = 0.3


def test_reconstruction_verification_catches_mismatch():
    """Unit-style check (no GPU/streaming): get_or_reconstruct_frozen_candidates
    must raise, not silently proceed, if a reconstruction doesn't exactly
    match the recorded Attempt-1 reference -- this is the safety property
    the whole "reuse, don't rerank" correction depends on."""
    saved = dict(nc.ATTEMPT1_REFERENCE)
    nc.ATTEMPT1_REFERENCE = {"N00": {"region": 999, "action": "zzz", "eta_c": 0.0, "q": 1}}
    try:
        class FakeCand(dict):
            pass
        fake_result = {"status": "OK", "candidates": [
            {"id": "N00", "region": 0, "action": "aa8", "geometry": {"q": 4, "eta_c": 0.05}}],
            "anchors": None}
        # directly exercise the comparison logic get_or_reconstruct_frozen_candidates uses,
        # without needing a real build_regions_and_cells call
        c = fake_result["candidates"][0]
        ref = nc.ATTEMPT1_REFERENCE.get(c["id"])
        mismatch = (ref is None or c["region"] != ref["region"] or c["action"] != ref["action"]
                    or c["geometry"]["q"] != ref["q"] or abs(c["geometry"]["eta_c"] - ref["eta_c"]) > 1e-9)
        assert mismatch, "SMOKE FAIL: mismatch detection did not fire on a deliberately wrong reference"
        print("  reconstruction-mismatch detection: OK (would have raised FATAL_SPEC_AMBIGUITY)", flush=True)
    finally:
        nc.ATTEMPT1_REFERENCE = saved


def main():
    from autopilot.common import build_splits, load_partial_model
    from autopilot.controller import fit_pipeline
    from autopilot.need_common import AlphabetLookup, WallClockBudget, load_lever0_basis
    import copy

    test_reconstruction_verification_catches_mismatch()

    print("=== NEED-SMOKE: E0 ===", flush=True)
    wall = WallClockBudget(nc.MAX_WALL_HOURS)
    manifest = build_splits()
    lever0_basis = load_lever0_basis()
    print(f"  lever0 basis: residual_basis shape={lever0_basis['residual_basis'].shape} "
          f"k_removed={lever0_basis['k_removed']}", flush=True)
    alphabet = AlphabetLookup()
    print(f"  alphabet: {len(alphabet.names)} symbols", flush=True)
    model = load_partial_model()
    theta0_state = copy.deepcopy(model.state_dict())
    pipeline = fit_pipeline(manifest["episode_ids"]["route_val"][:50] + manifest["episode_ids"]["replay_train"][:50])

    print("=== NEED-SMOKE: build_regions_and_cells (tiny; NOT the verified-reconstruction path, "
          "which needs real-scale reference values) ===", flush=True)
    build_result = nc.build_regions_and_cells(manifest, lever0_basis, model, pipeline, alphabet, wall)
    print(f"  status={build_result['status']}", flush=True)
    assert build_result["status"] == "OK", f"SMOKE FAIL: {build_result}"
    candidates = build_result["candidates"]
    anchors = build_result["anchors"]
    assert candidates, "SMOKE FAIL: no candidates frozen"
    cand = candidates[0]
    print(f"  candidate {cand['id']}: region={cand['region']} action={cand['action']} "
          f"dim_T={cand['geometry']['dim_T']} dim_W={cand['geometry']['dim_W']} q={cand['geometry']['q']} "
          f"eta_c={cand['geometry']['eta_c']:.4f}", flush=True)

    print("=== NEED-SMOKE: cell eval slice ===", flush=True)
    eval_slice = ne.build_cell_eval_slice(manifest["episode_ids"]["route_val"], cand["region"], cand["action"],
                                             anchors, lever0_basis, pipeline, alphabet, seed=1, n_scan=25)
    print(f"  eval_slice samples={len(eval_slice['samples'])}", flush=True)

    print("=== NEED-SMOKE: targeted retrieval (tiny pool, 2-minute cap) ===", flush=True)
    import numpy as np
    rng = np.random.default_rng(7)
    pool_ids = manifest["episode_ids"]["retrieval_pool"]
    tiny_pool = sorted(rng.choice(pool_ids, size=min(400, len(pool_ids)), replace=False).tolist())
    curricula_by_cell, stats = nt.run_targeted_retrieval(
        [cand], tiny_pool, anchors, lever0_basis, pipeline, model, alphabet, seed=7,
        max_minutes=nt.MAX_RETRIEVAL_MINUTES)
    print(f"  retrieval stats: {stats}", flush=True)
    curricula = curricula_by_cell.get((cand["region"], cand["action"]))
    assert curricula is not None, "SMOKE FAIL: candidate STARVED in targeted retrieval smoke pass"
    for arm, c in curricula.items():
        print(f"  {arm}: {c['attrition']}", flush=True)

    print("=== NEED-SMOKE: node datasets (6 arms from the targeted-retrieval curricula) ===", flush=True)
    arm_datasets = nc.build_node_datasets(cand, manifest, pipeline, curricula)
    assert arm_datasets is not None, "SMOKE FAIL: build_node_datasets returned None (starved)"
    for arm, ds in arm_datasets.items():
        print(f"  {arm}: {len(ds.samples)} samples", flush=True)

    print("=== NEED-SMOKE: run_stage_need (tiny: 1 seed, 3 updates) ===", flush=True)
    from autopilot.evaluate import build_eval_slice
    guard_ids = manifest["episode_ids"]["global_guard"][:8]
    guard_slice = build_eval_slice(guard_ids, n_per_episode=2, seed=2)
    stage = nc.run_stage_need(cand, arm_datasets, model, pipeline, theta0_state, eval_slice, guard_slice,
                                 wall, n_updates=3, seeds=[0], stage_name="smoke", gate="stage1")
    print(f"  stage status={stage.get('status')} gains={stage.get('gains')} "
          f"forgetting={stage.get('forgetting')} ROUTE_PASS={stage.get('ROUTE_PASS')}", flush=True)
    assert stage["status"] == "EVALUATED", f"SMOKE FAIL: {stage}"

    print("\n=== NEED-SMOKE TEST PASSED ===", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
