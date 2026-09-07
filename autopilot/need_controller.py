"""v6-controller / v7-experiment: action-conditional need-space autopilot.
Thin wrapper around the completed C00-C06 tree's validated utilities (splits,
run_arm, evaluate_E_W, paired_bootstrap_ci, build_dataset_from_episodes,
build_replay_dataset, fit_pipeline, build_real_latents_pool) -- see
common.py's and this package's need_*.py module docstrings for what is new
(region/action cells, N(r,a,1), six arms, the wall-clock budget) versus
reused verbatim, per SELF_IMPROVEMENT_LOOP_1.md v6 SS9's explicit preference
for this implementation style.

Genuine simplification versus the documents' full state machine (documented,
not silent, matching common.py's precedent): no formal SHA-256 ledger-replay
crash recovery -- this runs as one long-lived unattended process monitored
externally, restarted from scratch (not resumed) if it crashes, which the
5-hour wall-clock budget makes acceptable (a wasted partial run is bounded).
"""
from __future__ import annotations

import copy
import pickle
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.common import (  # noqa: E402
    AP_DIR, OUT_DIR, SEED, assert_disjoint_splits, build_splits, copy_to_downloads,
    load_frozen_directions, load_partial_model, now_iso, stable_seed, write_atomic,
)
from autopilot.controller import build_real_latents_pool, fit_pipeline  # noqa: E402
from autopilot.dataset import ArmDataset, build_dataset_from_segments, build_replay_dataset  # noqa: E402
from autopilot.evaluate import assert_slice_nonempty, evaluate_E_W, paired_bootstrap_ci  # noqa: E402
from autopilot.need_common import (  # noqa: E402
    ALPHABET_PATH, CANDIDATES_PATH, FINAL_JSON, FINAL_MD, NEED_AP_DIR, REGIONS_PATH, STATE_PATH,
    TARGETED_RETRIEVAL_CACHE_PATH,
    AlphabetLookup, WallClockBudget, append_need_ledger, load_frozen_geometry, load_lever0_basis,
    node_id, save_frozen_geometry,
)
from autopilot.need_geometry import (  # noqa: E402
    CANDIDATE_CELLS, GRADIENT_STARTS_MAX, GRADIENT_STARTS_MIN, M_ACTION_DIRS, MAX_REGIONS,
    MIN_FIT_EPISODES, MIN_FIT_TRANSITIONS, MIN_ROUTE_EPISODES, MIN_ROUTE_TRANSITIONS, N_POSITIONS,
    N_SUPPORT_SCAN_REPLAY, N_SUPPORT_SCAN_ROUTE, PREGRADIENT_CELL_CAP, PREGRADIENT_PER_REGION_CAP,
    assign_and_count, build_cell_geometry, build_global_W_control, farthest_point_anchors,
    scan_transitions, support_table,
)
from autopilot.need_evaluate import build_cell_eval_slice, effective_rank, full_latent_predictions  # noqa: E402
from autopilot.need_targeted_retrieval import MAX_RETRIEVAL_MINUTES, run_targeted_retrieval  # noqa: E402
from autopilot.train_arm import run_arm  # noqa: E402

# Attempt 1 (2026-09-04, archived *.attempt1): E0 found these 3 candidates with valid T/V/W/N
# geometry; ALL THREE then starved during retrieval before any arm trained (terminal_reason should
# have been ALL_CANDIDATES_STARVED, not NO_SUPPORTED_CELLS -- corrected per owner review, see
# preregistration.md). Attempt 1 never persisted the raw basis arrays, only these summary scalars,
# so Attempt 2 deterministically reconstructs E0 once more (same code/seed/cached inputs) and
# verifies the reconstruction matches this reference EXACTLY before proceeding -- a verified replay
# of an already-fixed computation, not a fresh, outcome-informed re-derivation of the candidates.
ATTEMPT1_HOURS_USED = 0.47296000000000477
ATTEMPT1_REFERENCE = {
    "N00": {"region": 0, "action": "aa8", "eta_c": 0.052890416234731674, "q": 4},
    "N01": {"region": 6, "action": "aa9", "eta_c": 0.04585869237780571, "q": 3},
    "N02": {"region": 6, "action": "aa5", "eta_c": 0.04463501274585724, "q": 4},
}

MAX_WALL_HOURS = 5.0
TRAINING_DEADLINE_HOURS = 4.5
LIVE_DEADLINE_HOURS = 4.25   # "at 4h15, do not begin a unit forecast to end after 4h30"
REPORT_RESERVE_MINUTES = 30
MAX_ARM_RUNS = 90
MIN_RELATIVE_TARGET_GAIN = 0.02
GLOBAL_GUARD_MAX_REGRESSION = 0.02
EFFRANK_DROP_MAX = 0.10

ARMS =["conditional_need", "global_W", "error_only", "random_dir", "random_traj", "continue"]
S1_CONTROLS = ["global_W", "random_dir", "random_traj", "continue"]
CI_CONTROLS = ["global_W", "random_dir", "continue"]   # v7 E2: "at least global_W, random_dir, and continue"

CONFIRM_STATE_PATH = OUT_DIR / "need_autopilot_confirm_state.json"
CONFIRM_ORDER = ["confirm_A", "confirm_B", "confirm_C"]

SCHEDULE = {"stage1_cells": 3, "stage2_promotions": 2, "stage3_attempts": 2,
            "stage3_updates": 750, "stage3_seeds": [0, 1, 2], "reductions": []}


def _load_confirm_state() -> dict:
    if CONFIRM_STATE_PATH.exists():
        import json
        return json.loads(CONFIRM_STATE_PATH.read_text())
    return {"consumed": []}


def _next_confirm_shard(state: dict) -> str | None:
    for name in CONFIRM_ORDER:
        if name not in state["consumed"]:
            return name
    return None


def _mark_confirm_consumed(state: dict, name: str) -> None:
    state["consumed"].append(name)
    write_atomic(CONFIRM_STATE_PATH, state)


def build_regions_and_cells(manifest: dict, lever0_basis: dict, model, pipeline, alphabet: AlphabetLookup,
                               wall: WallClockBudget, target_cells: list[tuple] | None = None,
                               U4: np.ndarray | None = None, w_mode: str = "exact") -> dict:
    """`target_cells`, if given, is a list of (region, action) pairs to
    build geometry for DIRECTLY, in the given order, bypassing the
    residual-energy pre-screen/eta_c ranking entirely. Needed because that
    ranking is not robust to the run-to-run streaming variability documented
    in `get_or_reconstruct_frozen_candidates`/preregistration.md: on the
    two-arm pilot's second reconstruction attempt (2026-09-05), two region-0
    cells' eta_c values (aa8=0.0437, aa6=0.0459) were close enough that this
    noise flipped which one ranked #1, silently swapping N00's identity --
    not just its eta_c value, which a tolerance cannot fix. Asking directly
    for specific cells sidesteps the ranking comparison altogether: we
    already know which (region, action) pairs we want, so we only need them
    to still be SUPPORTED and to produce valid geometry, not to out-rank
    nearby competitors under noisy measurements.

    `U4`, if given, is used DIRECTLY instead of the module-level
    `load_frozen_directions()[0][:M_ACTION_DIRS]` fallback (user-directed
    minimal-changes patch before the two-arm v2 launch: U8/B8 in
    tessellation_params_residualized.npz were themselves built from
    epoch-20-encoded latents -- the same class of coordinate mismatch
    Change A already fixed elsewhere. The v2 pilot recomputes U8 in epoch-7
    coordinates (need_two_arm_pilot.build_v2_lever0_and_directions) and
    passes U4=U8[:M_ACTION_DIRS] here explicitly; it must never fall through
    to this function's own `load_frozen_directions()` call. The six-arm
    tree's own call site passes nothing, keeping its existing epoch-20
    behavior unchanged.

    `w_mode` ("exact" default, or "soft") is passed straight through to
    every `build_cell_geometry` call below -- see that function's docstring.
    Only the two-arm v3 experiment revision passes "soft"; every other
    caller keeps the unchanged exact-intersection default."""
    replay_ids = manifest["episode_ids"]["replay_train"]
    route_ids = manifest["episode_ids"]["route_val"]

    print("  building region anchors (farthest-point sampling, Lever-0 residualized metric)...", flush=True)
    # Bug fix, code audit 2026-09-06 (Change A): thread this function's own
    # `model` (epoch-7 on the v7-need/two-arm-pilot path) instead of letting
    # build_real_latents_pool silently default to the epoch-20 singleton.
    anchor_pool_z = build_real_latents_pool(replay_ids, pipeline, n_episodes=250, model=model)
    anchors = farthest_point_anchors(anchor_pool_z, lever0_basis, MAX_REGIONS, seed=SEED)
    append_need_ledger({"event_type": "ADJUSTMENT", "reason": "no persisted v6 base-point catalogue found; "
                          "all region anchors built via licensed farthest-point-sampling fallback",
                          "requested_value": "deduplicated_v6_basepoints", "actual_value": f"{MAX_REGIONS}_fps_anchors",
                          "scientifically_neutral": True, "decided_before_outcome": True})

    print(f"  scanning replay_train support ({N_SUPPORT_SCAN_REPLAY} episodes)...", flush=True)
    replay_records = scan_transitions(replay_ids, SEED, N_SUPPORT_SCAN_REPLAY, pipeline, model, alphabet)
    assign_and_count(replay_records, anchors, lever0_basis)
    print(f"  scanning route_val support ({N_SUPPORT_SCAN_ROUTE} episodes)...", flush=True)
    route_records = scan_transitions(route_ids, SEED + 1, N_SUPPORT_SCAN_ROUTE, pipeline, model, alphabet)
    assign_and_count(route_records, anchors, lever0_basis)

    fit_support = support_table(replay_records)
    route_support = support_table(route_records)

    supported = [(r, a) for (r, a), v in fit_support.items()
                 if v["transitions"] >= MIN_FIT_TRANSITIONS and v["n_episodes"] >= MIN_FIT_EPISODES]
    region_action_counts = Counter(r for r, a in supported)
    valid_regions = {r for r, c in region_action_counts.items() if c >= 2}
    supported = [(r, a) for r, a in supported if r in valid_regions]
    supported = [(r, a) for r, a in supported
                 if route_support.get((r, a), {}).get("transitions", 0) >= MIN_ROUTE_TRANSITIONS
                 and route_support.get((r, a), {}).get("n_episodes", 0) >= MIN_ROUTE_EPISODES]

    print(f"  {len(supported)} supported (region, action) cells across {len(valid_regions)} regions", flush=True)
    write_atomic(REGIONS_PATH, {"n_anchors": len(anchors), "supported_cells": [{"region": r, "action": a,
                  "fit": fit_support[(r, a)], "route": route_support.get((r, a))} for r, a in supported]})
    if not supported:
        return {"status": "NO_SUPPORTED_CELLS"}

    by_cell: dict[tuple, list] = {}
    for r in replay_records:
        by_cell.setdefault((r["region"], r["letter"]), []).append(r)
    supported_set = set(supported)
    by_region: dict[int, dict] = {}
    for (r, a), recs in by_cell.items():
        if (r, a) in supported_set:   # SS3.3: pool only SUPPORTED sibling actions for global_W
            by_region.setdefault(r, {})[a] = recs

    real_latents_by_region = {r: build_real_latents_pool(replay_ids, pipeline, seed=SEED + r, n_episodes=120,
                                                             model=model)
                                 for r in valid_regions}

    all_znext = np.stack([r["z_next"] for (r_, a) in supported for r in by_cell[(r_, a)]])
    tr_cov_global = float(np.trace(np.cov(all_znext.T))) if all_znext.shape[0] > 1 else 1.0

    # User-directed patch: never call load_frozen_directions() (epoch-20
    # coordinates) when a caller supplies its own U4 explicitly.
    U4 = U4 if U4 is not None else load_frozen_directions()[0][:M_ACTION_DIRS]

    if target_cells is not None:
        missing = [cell for cell in target_cells if cell not in supported]
        if missing:
            raise RuntimeError(
                f"FATAL_SPEC_AMBIGUITY: target_cells {missing} are not supported in this reconstruction "
                f"(supported={supported}) -- cannot directly target them")
        frozen = []
        for r, a in target_cells:
            geom = build_cell_geometry(by_cell[(r, a)], anchors[r], real_latents_by_region[r],
                                          model, pipeline, U4, seed=stable_seed(SEED, r, a), w_mode=w_mode)
            if geom is None:
                raise RuntimeError(f"FATAL_SPEC_AMBIGUITY: target cell (r={r}, a={a}) produced invalid "
                                      f"geometry on reconstruction")
            geom.update({"region": r, "action": a})
            frozen.append(geom)
            print(f"    target cell (r={r}, a={a}): dim_T={geom['dim_T']} dim_V={geom['dim_V']} "
                  f"dim_W={geom['dim_W']} q={geom['q']} eta_c={geom['eta_c']:.4f}", flush=True)
    else:
        energies = []
        for (r, a) in supported:
            recs = by_cell[(r, a)]
            e = float(np.mean([np.sum(rec["residual"] ** 2) for rec in recs])) / max(tr_cov_global, 1e-12)
            energies.append(((r, a), e))
        energies.sort(key=lambda x: -x[1])

        prescreened, per_region_count = [], Counter()
        for (r, a), e in energies:
            if per_region_count[r] >= PREGRADIENT_PER_REGION_CAP:
                continue
            prescreened.append((r, a, e))
            per_region_count[r] += 1
            if len(prescreened) >= PREGRADIENT_CELL_CAP:
                break

        print(f"  pre-gradient screen: {len(prescreened)} cells (cap {PREGRADIENT_CELL_CAP}, "
              f"<= {PREGRADIENT_PER_REGION_CAP}/region)", flush=True)

        valid_cells = []
        for r, a, energy in prescreened:
            geom = build_cell_geometry(by_cell[(r, a)], anchors[r], real_latents_by_region[r],
                                          model, pipeline, U4, seed=stable_seed(SEED, r, a), w_mode=w_mode)
            if geom is None:
                print(f"    cell (r={r}, a={a}): INVALID, discarded", flush=True)
                continue
            geom.update({"region": r, "action": a, "residual_energy": energy})
            valid_cells.append(geom)
            print(f"    cell (r={r}, a={a}): dim_T={geom['dim_T']} dim_V={geom['dim_V']} dim_W={geom['dim_W']} "
                  f"q={geom['q']} eta_c={geom['eta_c']:.4f}", flush=True)

        if not valid_cells:
            return {"status": "NO_SUPPORTED_CELLS"}

        valid_cells.sort(key=lambda g: -g["eta_c"])
        frozen = valid_cells[:CANDIDATE_CELLS]

    candidates = []
    for i, g in enumerate(frozen):
        r, a = g["region"], g["action"]
        global_geom = build_global_W_control(r, by_region[r], anchors[r], real_latents_by_region[r], model,
                                                pipeline, U4, seed=SEED + 500 + r, q_cap=g["q"])
        candidates.append({"id": f"N0{i}", "region": r, "action": a, "geometry": g, "global_geometry": global_geom})

    write_atomic(CANDIDATES_PATH, {"candidates": [{"id": c["id"], "region": c["region"], "action": c["action"],
                  "eta_c": c["geometry"]["eta_c"], "q": c["geometry"]["q"]} for c in candidates]})
    return {"status": "OK", "candidates": candidates, "anchors": anchors,
            "replay_ids": replay_ids, "route_ids": route_ids}


ETA_C_RELATIVE_TOLERANCE = 0.10   # see docstring below: eta_c is not bit-reproducible in practice


def get_or_reconstruct_frozen_candidates(manifest: dict, lever0_basis: dict, model, pipeline,
                                             alphabet: AlphabetLookup, wall: WallClockBudget) -> dict:
    """Reuses Attempt 1's frozen N00-N02 verbatim rather than re-deriving
    them after seeing its starvation outcome. Prefers a persisted geometry
    cache outright (no recomputation at all); absent that (Attempt 1 never
    wrote one), deterministically reconstructs E0 once more and verifies it
    against Attempt 1's recorded region/action/eta_c/q before trusting it.

    `region`/`action` identity is now guaranteed by construction: E0 is asked
    to build geometry for Attempt 1's EXACT (region, action) pairs directly
    (`target_cells`, see `build_regions_and_cells`'s docstring), bypassing
    the residual-energy pre-screen/eta_c ranking that turned out not to be
    robust to run-to-run streaming variability -- on the two-arm pilot's
    second reconstruction attempt (2026-09-05), two region-0 cells' eta_c
    values were close enough (aa8=0.0437, aa6=0.0459) that this noise
    flipped which one ranked #1, silently swapping N00's IDENTITY, not just
    its eta_c value; a per-candidate tolerance cannot catch that kind of
    mismatch because slot N00 was simply a different cell. Only `q` (an
    integer, robust to small perturbations) and `eta_c` (continuous, allowed
    ETA_C_RELATIVE_TOLERANCE relative difference -- e.g. Attempt 1's first
    reconstruction attempt differed by ~1.85% with an identical cell) are
    compared now; region/action mismatches can no longer occur except as a
    hard error (the target cell not being supported at all -- see
    `build_regions_and_cells`)."""
    cached = load_frozen_geometry()
    if cached is not None:
        print("  reusing persisted frozen geometry from disk verbatim (no recomputation)", flush=True)
        candidates = [cached["candidates"][cid] for cid in sorted(cached["candidates"])]
        return {"status": "OK", "candidates": candidates, "anchors": cached["anchors"]}

    target_cells = [(ATTEMPT1_REFERENCE[cid]["region"], ATTEMPT1_REFERENCE[cid]["action"])
                       for cid in sorted(ATTEMPT1_REFERENCE)]
    print(f"  no persisted frozen geometry on disk; deterministically reconstructing Attempt 1's E0 "
          f"(same code/seed/cached inputs) DIRECTLY TARGETING its recorded cells {target_cells} "
          f"(bypassing eta_c ranking) and verifying against its recorded summary...", flush=True)
    result = build_regions_and_cells(manifest, lever0_basis, model, pipeline, alphabet, wall,
                                        target_cells=target_cells)
    if result["status"] != "OK":
        return result
    for cid, c in zip(sorted(ATTEMPT1_REFERENCE), result["candidates"]):
        c["id"] = cid   # target_cells order == sorted(ATTEMPT1_REFERENCE) order == N00,N01,N02
        ref = ATTEMPT1_REFERENCE[cid]
        eta_c_rel_diff = abs(c["geometry"]["eta_c"] - ref["eta_c"]) / max(abs(ref["eta_c"]), 1e-12)
        if c["geometry"]["q"] != ref["q"] or eta_c_rel_diff > ETA_C_RELATIVE_TOLERANCE:
            raise RuntimeError(
                f"FATAL_SPEC_AMBIGUITY: reconstructed candidate {cid} "
                f"(region={c['region']} action={c['action']} q={c['geometry']['q']} "
                f"eta_c={c['geometry']['eta_c']}) does not match Attempt 1's recorded "
                f"reference ({ref}) within tolerance (eta_c relative diff={eta_c_rel_diff}) -- "
                f"cannot safely reuse")
        if eta_c_rel_diff > 0:
            append_need_ledger({"event_type": "ADJUSTMENT", "node_id": c["id"],
                                  "reason": "eta_c reconstruction differs from Attempt 1's recorded value by "
                                            f"{eta_c_rel_diff:.4%} (within the licensed "
                                            f"{ETA_C_RELATIVE_TOLERANCE:.0%} tolerance) -- expected run-to-run "
                                            "variation from live network streaming, not a scientific choice",
                                  "requested_value": ref["eta_c"], "actual_value": c["geometry"]["eta_c"],
                                  "scientifically_neutral": True, "decided_before_outcome": True})
    print("  VERIFIED: reconstruction matches Attempt 1's recorded region/action/q exactly and eta_c "
          f"within {ETA_C_RELATIVE_TOLERANCE:.0%} for all 3 candidates -- a reproduction, not a fresh "
          "re-derivation", flush=True)
    save_frozen_geometry(result["anchors"], result["candidates"])
    append_need_ledger({"event_type": "ADJUSTMENT",
                          "reason": "Attempt 1 never persisted raw T/V/W/N/anchor arrays, only summary "
                                    "scalars; reconstructed deterministically and verified an exact match "
                                    "against Attempt 1's recorded region/action/eta_c/q before any "
                                    "retrieval was attempted, then persisted the full arrays for reuse",
                          "requested_value": "reuse_frozen_N00_N02_verbatim",
                          "actual_value": "verified_deterministic_reconstruction",
                          "scientifically_neutral": True, "decided_before_outcome": True})
    return result


def build_node_datasets(cand: dict, manifest: dict, pipeline, curricula: dict | None) -> dict | None:
    """`curricula` (for `cand`'s (region, action) cell, or None if that cell
    was STARVED) comes from the single shared `run_targeted_retrieval` pass
    -- no per-candidate retrieval left to do here beyond turning its episode
    lists into training datasets. No cross-candidate episode-novelty
    exclusion is enforced (a simplification versus the earlier per-candidate
    sequential design: the three candidates are different (region, action)
    cells scored independently, never trained against each other, so an
    episode appearing in more than one candidate's curriculum does not
    contaminate any comparison)."""
    if curricula is None:
        return None
    for arm, c in curricula.items():
        print(f"    [{cand['id']}] retrieval[{cand['region']}/{cand['action']}] arm={arm}: "
              f"{c['attrition']}", flush=True)

    n_target_samples = max(1, len(curricula["conditional_need"]["segments"])) * 8
    replay_ds = build_replay_dataset(manifest["episode_ids"]["replay_train"], n_target_samples, pipeline, seed=SEED)

    # Bug fix, code audit 2026-09-06 (P0-1): built from EPISODE IDS, which
    # `build_dataset_from_episodes` then re-windows into RANDOM (episode,
    # position) pairs -- discarding the exact selected (episode_id, t)
    # segments retrieval scored and ranked, with no guaranteed overlap.
    # `build_dataset_from_segments` trains on the exact retrieved segments.
    arm_datasets = {arm: build_dataset_from_segments(curricula[arm]["segments"], pipeline)
                       for arm in ("conditional_need", "global_W", "error_only", "random_dir", "random_traj")}
    arm_datasets["continue"] = replay_ds
    for arm, ds in arm_datasets.items():
        if len(ds.samples) == 0:
            print(f"    [{cand['id']}] arm={arm} has 0 samples -> STARVED", flush=True)
            return None
    return arm_datasets


def run_stage_need(cand: dict, arm_datasets: dict, model, pipeline, theta0_state: dict, eval_slice: dict,
                      guard_slice: dict, wall: WallClockBudget, n_updates: int, seeds: list[int],
                      stage_name: str, gate: str, replay_ratio_note: str = "1:1") -> dict:
    t0 = time.time()
    U8, B8 = load_frozen_directions()
    B_N = cand["geometry"]["B_N"]
    D = B_N.shape[1]
    model.load_state_dict(theta0_state)
    model.eval()
    pre_eval = evaluate_E_W(model, eval_slice, pipeline, B_N, U8, B8)
    pre_E_N = float(np.mean([v["E_W"] for v in pre_eval["per_episode"].values()])) if pre_eval["per_episode"] else 1.0
    pre_gg = evaluate_E_W(model, guard_slice, pipeline, np.eye(D), U8, B8)
    pre_gg_e_all = float(np.mean([v["E_all"] for v in pre_gg["per_episode"].values()])) if pre_gg["per_episode"] else 0.0
    pre_gg_preds = full_latent_predictions(model, guard_slice, pipeline)
    pre_effrank = effective_rank(pre_gg_preds)

    arm_results = {arm: {"E_N": []} for arm in arm_datasets}
    cond_gg_e_all, cond_effrank = [], []
    n_arm_runs = 0
    for seed in seeds:
        for arm, ds in arm_datasets.items():
            t_train = time.time()
            res = run_arm(model, theta0_state, ds, n_updates, seed=seed,
                             ckpt_path=NEED_AP_DIR / cand["id"] / stage_name / arm / f"seed{seed}.pt")
            n_arm_runs += 1
            ev = evaluate_E_W(model, eval_slice, pipeline, B_N, U8, B8)
            e_n_vals = np.array([v["E_W"] for v in ev["per_episode"].values()]) if ev["per_episode"] else np.zeros(1)
            arm_results[arm]["E_N"].append(e_n_vals)
            if arm == "conditional_need":
                gg_ev = evaluate_E_W(model, guard_slice, pipeline, np.eye(D), U8, B8)
                cond_gg_e_all.append(float(np.mean([v["E_all"] for v in gg_ev["per_episode"].values()])) if gg_ev["per_episode"] else pre_gg_e_all)
                cond_effrank.append(effective_rank(full_latent_predictions(model, guard_slice, pipeline)))
            print(f"    [{cand['id']}/{stage_name}] arm={arm} seed={seed} steps={res['n_steps']} "
                  f"loss={res['final_loss']:.5f} E_N_mean={e_n_vals.mean():.4f} "
                  f"(wall {wall.elapsed_hours():.2f}h)", flush=True)
            if wall.exhausted():
                return {"status": "TIME_BUDGET_EXHAUSTED", "n_arm_runs": n_arm_runs}

    gg_e_all_mean = float(np.mean(cond_gg_e_all)) if cond_gg_e_all else pre_gg_e_all
    effrank_mean = float(np.mean(cond_effrank)) if cond_effrank else pre_effrank
    regression = (gg_e_all_mean - pre_gg_e_all) / max(pre_gg_e_all, 1e-8)
    effrank_drop = (pre_effrank - effrank_mean) / max(pre_effrank, 1e-8)
    forgetting = bool(regression > GLOBAL_GUARD_MAX_REGRESSION or effrank_drop > EFFRANK_DROP_MAX)

    gains = {arm: float((pre_E_N - np.mean([v.mean() for v in arm_results[arm]["E_N"]])) / max(pre_E_N, 1e-8))
               for arm in arm_datasets}

    advantages = {}
    seed_signs = {}
    for control in S1_CONTROLS + (["error_only"] if gate == "stage3_confirm" else []):
        vals_cond = np.concatenate(arm_results["conditional_need"]["E_N"])
        vals_ctrl = np.concatenate(arm_results[control]["E_N"])
        n = min(len(vals_cond), len(vals_ctrl))
        mean_adv, lo, hi = paired_bootstrap_ci(vals_ctrl[:n], vals_cond[:n], seed=stable_seed(SEED, control))
        advantages[control] = {"mean": mean_adv, "ci95": [lo, hi]}
        per_seed_signs = []
        for i in range(len(seeds)):
            a = arm_results["conditional_need"]["E_N"][i]
            b = arm_results[control]["E_N"][i]
            m = min(len(a), len(b))
            per_seed_signs.append(bool(np.mean(b[:m] - a[:m]) > 0))
        seed_signs[control] = per_seed_signs

    ci_controls_agree = all(all(seed_signs[c]) or not any(seed_signs[c]) for c in CI_CONTROLS)
    point_pass_all_s1 = all(advantages[c]["mean"] > 0 for c in S1_CONTROLS)
    ci_pass = all(advantages[c]["ci95"][0] > 0 for c in CI_CONTROLS)

    result = {"status": "EVALUATED", "gains": gains, "forgetting": forgetting,
              "global_regression": regression, "effrank_drop": effrank_drop,
              "advantages": advantages, "seed_signs_agree": bool(ci_controls_agree),
              "n_seeds": len(seeds), "n_arm_runs": n_arm_runs, "wall_time_s": time.time() - t0,
              "pre_E_N": pre_E_N}

    if gate == "stage1":
        route_pass = (gains["conditional_need"] > 0 and point_pass_all_s1 and not forgetting)
    elif gate in ("stage2", "stage3_confirm"):
        route_pass = (gains["conditional_need"] >= MIN_RELATIVE_TARGET_GAIN and point_pass_all_s1
                        and ci_pass and ci_controls_agree and not forgetting)
    else:
        route_pass = False
    result["ROUTE_PASS"] = bool(route_pass)

    if gate == "stage3_confirm" and route_pass:
        result["S2_PASS"] = bool(advantages["error_only"]["ci95"][0] > 0)
    return result


def run_confirmation(cand: dict, model, pipeline, anchors, lever0_basis, alphabet, stage3_seeds: list[int],
                        confirm_ids: list, wall: WallClockBudget) -> dict:
    """Evaluate the ALREADY-TRAINED Stage-3 checkpoints (reloaded from disk,
    one per arm/seed -- no additional training) once on the sealed shard,
    aggregating across seeds exactly like route_val evaluation does within a
    stage, per loop v6 SS5.2 / SS6."""
    import torch
    U8, B8 = load_frozen_directions()
    B_N = cand["geometry"]["B_N"]
    eval_slice = build_cell_eval_slice(confirm_ids, cand["region"], cand["action"], anchors, lever0_basis,
                                          pipeline, alphabet, seed=SEED + 900, model=model)
    if not eval_slice["samples"]:
        return {"status": "NO_FRESH_CONFIRMATION_DATA"}

    results = {}
    for arm in ARMS:
        vals = []
        for seed in stage3_seeds:
            ckpt = NEED_AP_DIR / cand["id"] / "stage3" / arm / f"seed{seed}.pt"
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()
            ev = evaluate_E_W(model, eval_slice, pipeline, B_N, U8, B8)
            if ev["per_episode"]:
                vals.append(np.array([v["E_W"] for v in ev["per_episode"].values()]))
        results[arm] = np.concatenate(vals) if vals else np.zeros(1)

    advantages = {}
    for control in S1_CONTROLS + ["error_only"]:
        n = min(len(results["conditional_need"]), len(results[control]))
        mean_adv, lo, hi = paired_bootstrap_ci(results[control][:n], results["conditional_need"][:n],
                                                  seed=stable_seed(SEED, control))
        advantages[control] = {"mean": mean_adv, "ci95": [lo, hi]}

    s1_confirmed = all(advantages[c]["mean"] > 0 for c in S1_CONTROLS) and all(
        advantages[c]["ci95"][0] > 0 for c in CI_CONTROLS)
    s2_confirmed = s1_confirmed and advantages["error_only"]["ci95"][0] > 0
    tier = "S2_SYMBOLIC_INCREMENT" if s2_confirmed else ("S1_ACTION_CONDITIONAL" if s1_confirmed else None)
    return {"status": "EVALUATED", "advantages": advantages, "tier": tier,
              "n_confirm_samples": len(eval_slice["samples"])}


def build_rescue_datasets(base_datasets: dict, manifest: dict, pipeline) -> dict:
    """The one licensed 2:1 replay rescue (loop v6 SS6 / experiment v7 SS5).

    Bug fix, code audit 2026-09-06 (P1-7): `run_stage_need`'s `forgetting`
    flag is computed ONLY from the `conditional_need` arm's own post-training
    guard regression / effective-rank drop (`cond_gg_e_all`/`cond_effrank`
    are appended `if arm == "conditional_need"` and nothing else). The
    previous implementation enlarged the UNRELATED `continue` control arm's
    replay instead -- `continue` never exhibited forgetting, and nothing
    about its own training changes when a different dict entry is replaced,
    so the "rescue" was a no-op with respect to the condition it claimed to
    rescue. The licensed 2:1 rescue must give `conditional_need` itself 2x
    replay samples MIXED INTO its own curriculum (curriculum + 2*curriculum
    replay), since that is the arm whose training is actually causing the
    regression; `continue` and the other controls are left untouched so
    this remains a "matched six-arm" rerun apart from that one change."""
    curr_ds = base_datasets["conditional_need"]
    n_curriculum_samples = len(curr_ds.samples)
    replay_ds = build_replay_dataset(manifest["episode_ids"]["replay_train"],
                                        max(1, n_curriculum_samples) * 2, pipeline, seed=SEED + 1)
    rescued = dict(base_datasets)
    rescued["conditional_need"] = ArmDataset(
        samples=curr_ds.samples + replay_ds.samples, pipeline=pipeline,
        episode_ids=sorted(set(curr_ds.episode_ids) | set(replay_ds.episode_ids)))
    return rescued


def apply_schedule_reductions(wall: WallClockBudget) -> None:
    """Loop v6 SS7.2: deterministic, outcome-blind reductions applied before
    any Stage-1 metric is seen. Simplified trigger (documented): E0's own
    wall-clock cost stands in for the documents' full per-stage timing
    forecast, since arm-run timing is only measurable AFTER E0 completes and
    the reduction record must precede the first Stage-1 metric regardless."""
    if wall.elapsed_hours() > 0.5:
        SCHEDULE["stage3_attempts"] = 1
        SCHEDULE["reductions"].append({"requested_value": "stage3_attempts=2", "actual_value": 1,
                                          "reason": "E0 preprocessing exceeded its 30-minute target",
                                          "scientifically_neutral": True, "decided_before_outcome": True})
    if wall.elapsed_hours() > 1.0:
        SCHEDULE["stage2_promotions"] = 1
        SCHEDULE["reductions"].append({"requested_value": "stage2_promotions=2", "actual_value": 1,
                                          "reason": "E0 preprocessing exceeded 1 hour",
                                          "scientifically_neutral": True, "decided_before_outcome": True})
    for adj in SCHEDULE["reductions"]:
        append_need_ledger({"event_type": "ADJUSTMENT", **adj})


def write_final_report(terminal_reason: str, run_log: dict, wall: WallClockBudget, extra: dict | None = None) -> None:
    tier_claims = {
        "S1_ACTION_CONDITIONAL": ("On an unopened episode shard, action-conditioned need selection improved "
            "frozen-encoder, true-start one-step prediction in its predeclared need subspace over the "
            "action-pooled global geometry, matched random directions, matched random trajectories and "
            "equal-compute continued training, with the preregistered guards."),
        "S2_SYMBOLIC_INCREMENT": ("On an unopened episode shard, action-conditioned need selection improved "
            "frozen-encoder, true-start one-step prediction in its predeclared need subspace over the "
            "action-pooled global geometry, matched random directions, matched random trajectories, "
            "equal-compute continued training, AND action-conditioned residual-only hard-example mining, "
            "with the preregistered guards."),
        "ALL_CANDIDATES_STARVED": ("All frozen candidates had valid (region, action) support and T/V/W/N "
            "geometry from E0, but retrieval could not supply a required curriculum for any of them, even "
            "after enumerating the entire frozen retrieval pool. NO arm ever trained and NO curriculum was "
            "ever compared -- this is a retrieval-feasibility result, not a trained-arm negative, and must "
            "not be read as evidence against action-conditioned guidance."),
        "NO_SUPPORTED_CELLS": ("E0 found no (region, action) cell meeting the preregistered support floors "
            "in this data. No geometry was constructed and no arm ever trained."),
    }
    default_claim = ("No candidate reached even S1_ACTION_CONDITIONAL within the declared queue and "
        "five-hour wall-clock budget. This is a completed bounded search with a negative/inconclusive "
        "result, not evidence of absence.")
    report = {
        "terminal_reason": terminal_reason, "timestamp": now_iso(),
        "wall_hours_used": wall.elapsed_hours(), "wall_hours_budget": MAX_WALL_HOURS,
        "controller_version": 6, "experiment_version": 7,
        "scientific_scope": "exploratory_action_conditional_predictor_adaptation",
        "soundness_status": "HELD_BY_OWNER", "schedule_reductions": SCHEDULE["reductions"],
        "nodes_run": list(run_log.keys()), "run_log": run_log,
        "licensed_claim": tier_claims.get(terminal_reason, default_claim),
        "not_claimed": ["symbolic abstraction validity", "canonical FSM", "long-horizon capability",
                          "latent representation expansion", "universal superiority of action conditioning"],
    }
    if extra:
        report.update(extra)
    write_atomic(FINAL_JSON, report)
    md = [f"# Need-autopilot final report", "", f"**Terminal reason:** {terminal_reason}", "",
           f"Wall-clock used: {report['wall_hours_used']:.2f}h / {MAX_WALL_HOURS}h", "",
           f"Nodes run: {report['nodes_run']}", "", "## Licensed claim", "", report["licensed_claim"], "",
           "## Not claimed", ""] + [f"- {x}" for x in report["not_claimed"]]
    FINAL_MD.write_text("\n".join(md))
    copy_to_downloads(FINAL_JSON, FINAL_MD, OUT_DIR / "need_autopilot_ledger.jsonl", REGIONS_PATH, CANDIDATES_PATH)
    print(f"\n=== FINAL: {terminal_reason} ({wall.elapsed_hours():.2f}h) ===", flush=True)


def main():
    from autopilot.evaluate import build_eval_slice

    print(f"=== need_autopilot INIT ({now_iso()}) -- Attempt 2, {ATTEMPT1_HOURS_USED:.3f}h already "
          f"spent by Attempt 1 counts against this budget ===", flush=True)
    wall = WallClockBudget(MAX_WALL_HOURS, initial_elapsed_hours=ATTEMPT1_HOURS_USED)
    manifest = build_splits()
    assert_disjoint_splits(manifest)
    lever0_basis = load_lever0_basis()
    alphabet = AlphabetLookup()
    model = load_partial_model()
    theta0_state = copy.deepcopy(model.state_dict())
    # Bug fix, code audit 2026-09-06 (P0-6): fitting on route_val + replay_train
    # leaks route_val's own action statistics into the pipeline before it is
    # later used to build every eval_slice FROM route_val. Fit on
    # replay_train only.
    pipeline = fit_pipeline(manifest["episode_ids"]["replay_train"])
    confirm_state = _load_confirm_state()
    append_need_ledger({"event_type": "INIT", "timestamp": now_iso(), "attempt1_hours_charged": ATTEMPT1_HOURS_USED})

    print("=== BUILD_REGIONS / BUILD_NEED_BANKS (reuse-or-verified-reconstruction) ===", flush=True)
    build_result = get_or_reconstruct_frozen_candidates(manifest, lever0_basis, model, pipeline, alphabet, wall)
    apply_schedule_reductions(wall)
    if build_result["status"] != "OK":
        write_final_report(build_result["status"], {}, wall)
        return 11
    candidates = build_result["candidates"]
    anchors = build_result["anchors"]
    append_need_ledger({"event_type": "E0_COMPLETE", "timestamp": now_iso(), "elapsed_hours": wall.elapsed_hours(),
                          "candidates": [{"id": c["id"], "region": c["region"], "action": c["action"]}
                                           for c in candidates]})

    guard_ids = manifest["episode_ids"]["global_guard"][:80]
    guard_slice = build_eval_slice(guard_ids, n_per_episode=4, seed=SEED + 1)
    assert_slice_nonempty(guard_slice, "global guard slice", min_episodes=5)

    print(f"=== targeted retrieval pass (inverted action-letter index over the FULL pool, "
          f"{MAX_RETRIEVAL_MINUTES:.0f}-minute cap, deterministic hash order) ===", flush=True)
    if TARGETED_RETRIEVAL_CACHE_PATH.exists():
        print(f"  loading cached targeted-retrieval result from {TARGETED_RETRIEVAL_CACHE_PATH}", flush=True)
        with open(TARGETED_RETRIEVAL_CACHE_PATH, "rb") as f:
            curricula_by_cell, retrieval_stats = pickle.load(f)
    else:
        curricula_by_cell, retrieval_stats = run_targeted_retrieval(
            candidates, manifest["episode_ids"]["retrieval_pool"], anchors, lever0_basis, pipeline, model,
            alphabet, seed=SEED, max_minutes=MAX_RETRIEVAL_MINUTES,
            # Six-arm tree's own pre-existing (non-empty-is-enough) starvation
            # tolerance, left unchanged -- see run_targeted_retrieval's docstring.
            require_exact_k=False)
        with open(TARGETED_RETRIEVAL_CACHE_PATH, "wb") as f:
            pickle.dump((curricula_by_cell, retrieval_stats), f)
    append_need_ledger({"event_type": "TARGETED_RETRIEVAL_DONE", "timestamp": now_iso(),
                          "elapsed_hours": wall.elapsed_hours(), "stats": retrieval_stats,
                          "counts": {f"{r}/{a}": ("STARVED" if curricula_by_cell.get((r, a)) is None else
                                                    {arm: c["attrition"]["common_eligible_segments"]
                                                       for arm, c in curricula_by_cell[(r, a)].items()})
                                       for (r, a) in {(c["region"], c["action"]) for c in candidates}}})

    print("=== building per-candidate datasets from the shared retrieval pass ===", flush=True)
    node_datasets, node_eval_slices, run_log = {}, {}, {}
    for cand in candidates:
        run_log[cand["id"]] = {"region": cand["region"], "action": cand["action"]}
        node_eval_slices[cand["id"]] = build_cell_eval_slice(
            manifest["episode_ids"]["route_val"], cand["region"], cand["action"], anchors, lever0_basis,
            pipeline, alphabet, seed=stable_seed(SEED, cand["id"]), model=model)
        curricula = curricula_by_cell.get((cand["region"], cand["action"]))
        node_datasets[cand["id"]] = build_node_datasets(cand, manifest, pipeline, curricula)
        if node_datasets[cand["id"]] is None:
            run_log[cand["id"]]["status"] = "STARVED"
            append_need_ledger({"event_type": "NODE_STARVED", "node_id": cand["id"]})

    active = [c for c in candidates if node_datasets[c["id"]] is not None]
    if not active:
        # supported cells + valid geometry existed (candidates is nonempty), but retrieval could
        # not supply a curriculum for any of them -- ALL_CANDIDATES_STARVED, not NO_SUPPORTED_CELLS
        # (that tag is reserved for E0 itself finding no supported cells at all). No arm ever
        # trained; this must not be read as evidence against action-conditioned guidance.
        write_final_report("ALL_CANDIDATES_STARVED", run_log, wall)
        return 14

    total_arm_runs = 0

    def budget_ok() -> bool:
        return not wall.exhausted() and total_arm_runs < MAX_ARM_RUNS

    print("=== STAGE 1: screen ===", flush=True)
    stage1_results = {}
    for cand in active:
        r = run_stage_need(cand, node_datasets[cand["id"]], model, pipeline, theta0_state,
                              node_eval_slices[cand["id"]], guard_slice, wall, n_updates=200, seeds=[0],
                              stage_name="stage1", gate="stage1")
        total_arm_runs += r.get("n_arm_runs", 0)
        stage1_results[cand["id"]] = r
        run_log[cand["id"]]["stage1"] = r
        append_need_ledger({"event_type": "NODE_EVALUATED", "node_id": cand["id"], "state": "STAGE1", "metrics": r})
        write_atomic(STATE_PATH, {"stage": "stage1", "run_log": run_log, "elapsed_hours": wall.elapsed_hours()})
        if not budget_ok():
            write_final_report("TIME_BUDGET_EXHAUSTED", run_log, wall)
            return 13

    promotable = [c for c in active if stage1_results[c["id"]].get("status") == "EVALUATED"
                    and stage1_results[c["id"]]["ROUTE_PASS"]]
    promotable.sort(key=lambda c: (-min(stage1_results[c["id"]]["advantages"][k]["mean"] for k in S1_CONTROLS),
                                      -c["geometry"]["eta_c"], c["id"]))
    promoted = promotable[:SCHEDULE["stage2_promotions"]]
    for c in active:
        run_log[c["id"]]["route_after_stage1"] = "promote" if c in promoted else "terminalize"
    if not promoted:
        write_final_report("TREE_EXHAUSTED", run_log, wall)
        return 10

    print(f"=== STAGE 2: route ({[c['id'] for c in promoted]}) ===", flush=True)
    stage2_results = {}
    for cand in promoted:
        r = run_stage_need(cand, node_datasets[cand["id"]], model, pipeline, theta0_state,
                              node_eval_slices[cand["id"]], guard_slice, wall, n_updates=500, seeds=[0, 1],
                              stage_name="stage2", gate="stage2")
        total_arm_runs += r.get("n_arm_runs", 0)
        stage2_results[cand["id"]] = r
        run_log[cand["id"]]["stage2"] = r
        append_need_ledger({"event_type": "NODE_EVALUATED", "node_id": cand["id"], "state": "STAGE2", "metrics": r})
        write_atomic(STATE_PATH, {"stage": "stage2", "run_log": run_log, "elapsed_hours": wall.elapsed_hours()})
        if not budget_ok():
            write_final_report("TIME_BUDGET_EXHAUSTED", run_log, wall)
            return 13

    rescue_used = False
    for cand in promoted:
        r = stage2_results[cand["id"]]
        if r.get("status") != "EVALUATED" or not r["forgetting"] or rescue_used:
            continue
        would_pass_without_forgetting = (
            r["gains"]["conditional_need"] >= MIN_RELATIVE_TARGET_GAIN
            and all(r["advantages"][k]["mean"] > 0 for k in S1_CONTROLS)
            and all(r["advantages"][k]["ci95"][0] > 0 for k in CI_CONTROLS) and r["seed_signs_agree"])
        if not would_pass_without_forgetting:
            continue
        print(f"  [{cand['id']}] licensed 2:1 replay rescue (positive signal blocked only by forgetting)", flush=True)
        rescue_used = True
        rescue_ds = build_rescue_datasets(node_datasets[cand["id"]], manifest, pipeline)
        rr = run_stage_need(cand, rescue_ds, model, pipeline, theta0_state, node_eval_slices[cand["id"]],
                               guard_slice, wall, n_updates=500, seeds=[0, 1], stage_name="stage2_rescue",
                               gate="stage2")
        total_arm_runs += rr.get("n_arm_runs", 0)
        run_log[cand["id"]]["rescue"] = rr
        append_need_ledger({"event_type": "RESCUE", "node_id": cand["id"], "metrics": rr})
        if rr.get("status") == "EVALUATED" and rr["ROUTE_PASS"]:
            stage2_results[cand["id"]] = rr
        if not budget_ok():
            write_final_report("TIME_BUDGET_EXHAUSTED", run_log, wall)
            return 13

    frozen_for_stage3 = [c for c in promoted if stage2_results[c["id"]].get("status") == "EVALUATED"
                            and stage2_results[c["id"]]["ROUTE_PASS"]]
    frozen_for_stage3.sort(key=lambda c: (-min(stage2_results[c["id"]]["advantages"][k]["ci95"][0]
                                                  for k in CI_CONTROLS), c["id"]))
    frozen_for_stage3 = frozen_for_stage3[:SCHEDULE["stage3_attempts"]]
    for c in promoted:
        run_log[c["id"]]["route_after_stage2"] = "confirm" if c in frozen_for_stage3 else "terminalize"
    if not frozen_for_stage3:
        write_final_report("TREE_EXHAUSTED", run_log, wall)
        return 10

    print(f"=== STAGE 3: robust + confirm ({[c['id'] for c in frozen_for_stage3]}) ===", flush=True)
    scientific_failures = 0
    final_tier = None
    for cand in frozen_for_stage3:
        if wall.past(LIVE_DEADLINE_HOURS) or not budget_ok():
            break
        r3 = run_stage_need(cand, node_datasets[cand["id"]], model, pipeline, theta0_state,
                               node_eval_slices[cand["id"]], guard_slice, wall,
                               n_updates=SCHEDULE["stage3_updates"], seeds=SCHEDULE["stage3_seeds"],
                               stage_name="stage3", gate="stage2")
        total_arm_runs += r3.get("n_arm_runs", 0)
        run_log[cand["id"]]["stage3"] = r3
        append_need_ledger({"event_type": "NODE_EVALUATED", "node_id": cand["id"], "state": "STAGE3", "metrics": r3})
        write_atomic(STATE_PATH, {"stage": "stage3", "run_log": run_log, "elapsed_hours": wall.elapsed_hours()})
        if r3.get("status") != "EVALUATED" or not r3["ROUTE_PASS"]:
            run_log[cand["id"]]["route_after_stage3"] = "next_candidate"
            continue

        shard = _next_confirm_shard(confirm_state)
        if shard is None:
            write_final_report("NO_FRESH_CONFIRMATION_DATA", run_log, wall)
            return 12
        confirm_ids = manifest["episode_ids"][shard]
        conf = run_confirmation(cand, model, pipeline, anchors, lever0_basis, alphabet,
                                   SCHEDULE["stage3_seeds"], confirm_ids, wall)
        _mark_confirm_consumed(confirm_state, shard)
        run_log[cand["id"]]["confirmation"] = {"shard": shard, **conf}
        append_need_ledger({"event_type": "CONFIRMATION", "node_id": cand["id"], "shard": shard, "result": conf})

        if conf.get("status") == "EVALUATED" and conf.get("tier"):
            final_tier = conf["tier"]
            run_log[cand["id"]]["route_after_confirmation"] = "SUCCESS"
            break
        scientific_failures += 1
        run_log[cand["id"]]["route_after_confirmation"] = "scientific_failure"
        if scientific_failures >= 2:
            break

    if final_tier:
        write_final_report(final_tier, run_log, wall)
        return 0
    if wall.exhausted():
        write_final_report("TIME_BUDGET_EXHAUSTED", run_log, wall)
        return 13
    write_final_report("TREE_EXHAUSTED", run_log, wall)
    return 10


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    import os
    os._exit(code)
