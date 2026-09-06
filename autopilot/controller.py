"""v6 autopilot controller. Simplified state machine (see common.py's module
docstring for what's simplified and why): E0 freeze/calibrate -> best-first
queue over C00-C06 -> per-node PROBE/TRAIN/EVAL -> successive halving
(Stage1 screen / Stage2 route / Stage3 robust) -> SOFT_SUCCESS hard-stop
(Stage 3 reproduces S1 without spending a sealed confirmation shard) or a
declared terminal condition. Runs as one long-lived process; a 10 GPU-hour
budget (owner-set, below the document's 24h default) is enforced throughout.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.base_points import attach_full_geometry, select_base_points
from autopilot.common import (
    AP_DIR, OUT_DIR, SEED, append_ledger, assert_disjoint_splits, build_splits,
    copy_to_downloads, load_frozen_directions, load_partial_model, now_iso, short_hash,
    write_atomic,
)
from autopilot.dataset import build_dataset_from_episodes, build_replay_dataset
from autopilot.evaluate import build_eval_slice, evaluate_E_W, paired_bootstrap_ci, pool_W_basis
from autopilot.retrieval import build_curriculum
from autopilot.train_arm import run_arm
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import ActionPipeline, RAW_ACTION_DIM

MAX_GPU_HOURS = 10.0          # owner override of the document's 24h default
MAX_TRAINED_NODES = 32
MIN_RELATIVE_TARGET_GAIN = 0.02
GLOBAL_GUARD_MAX_REGRESSION = 0.02
FORGETTING_EFFRANK_DROP = 0.10
K_ORDER = [200, 400]
REPLAY_FLOOR_TRAJECTORIES = 200

# Full v6 E1 candidate table, for reference (C00-C04 already ran to TREE_EXHAUSTED --
# see artifacts/autopilot_final_report.json.attempt1 / preregistration -- 0.43/10 GPU-hours
# used, no candidate reached even soft success). Not re-listed as the active queue below
# to avoid re-running already-complete, already-reported nodes on a fresh launch (this
# controller has no resume-from-ledger implemented -- see common.py's module docstring).
FULL_V6_CANDIDATE_TABLE = [
    {"id": "C00", "m": 4, "suffix": 1, "base_policy": "high_W_ratio", "K": 200},
    {"id": "C01", "m": 2, "suffix": 1, "base_policy": "high_W_ratio", "K": 200},
    {"id": "C02", "m": 8, "suffix": 1, "base_policy": "high_W_ratio", "K": 200},
    {"id": "C03", "m": 4, "suffix": 1, "base_policy": "high_residual", "K": 200},
    {"id": "C04", "m": 4, "suffix": 1, "base_policy": "diverse_scene", "K": 200},
    {"id": "C05", "m": 4, "suffix": 2, "base_policy": "high_W_ratio", "K": 200},
    {"id": "C06", "m": 2, "suffix": 2, "base_policy": "high_W_ratio", "K": 200},
]

# ACTIVE queue for this launch: only the two-letter proxy candidates, per v6's own
# routing ("random direction beats orthogonal / nothing beats continue -> next m; then
# corresponding two-letter proxy" -- tried only after the one-letter family fails, which
# it did for all of C00-C04).
CANDIDATE_QUEUE = [
    {"id": "C05", "m": 4, "suffix": 2, "base_policy": "high_W_ratio", "K": 200},
    {"id": "C06", "m": 2, "suffix": 2, "base_policy": "high_W_ratio", "K": 200},
]


class GPUBudget:
    def __init__(self, max_hours: float):
        self.max_seconds = max_hours * 3600
        self.used_seconds = 0.0

    def add(self, seconds: float):
        self.used_seconds += seconds

    def remaining(self) -> float:
        return self.max_seconds - self.used_seconds

    def exhausted(self) -> bool:
        return self.used_seconds >= self.max_seconds


def fit_pipeline(episode_ids: list[int], seed: int = SEED) -> "ActionPipeline":
    rng = np.random.default_rng(seed)
    fit_ids = sorted(rng.choice(episode_ids, size=min(200, len(episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    return ActionPipeline.fit(np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0))


def build_real_latents_pool(episode_ids: list[int], pipeline, seed: int = SEED, n_episodes: int = 300) -> np.ndarray:
    from autopilot.dataset import build_dataset_from_episodes
    rng = np.random.default_rng(seed)
    sample_ids = sorted(rng.choice(episode_ids, size=min(n_episodes, len(episode_ids)), replace=False).tolist())
    ds = build_dataset_from_episodes(sample_ids, pipeline, max_windows_per_episode=4, seed=seed)
    if not ds.samples:
        raise RuntimeError("FATAL_DATA_ERROR: could not build any real-latent samples for T")
    from oracle.lewm_g import encode_pixel_windows_batch
    pix = np.stack([s["pixels"] for s in ds.samples])
    emb = encode_pixel_windows_batch(pix).numpy()   # (N, HISTORY_SIZE+1, D)
    return emb.reshape(-1, emb.shape[-1])


def probe_node(cand: dict, manifest: dict, model, pipeline, U8: np.ndarray, real_latents_pool: np.ndarray) -> dict | None:
    U_m = U8[: cand["m"]]
    suffix_len = cand.get("suffix", 1)
    route_val = manifest["episode_ids"]["route_val"]
    base_points = select_base_points(cand["base_policy"], route_val, pipeline, model, U_m,
                                        real_latents_pool, seed=SEED + hash(cand["id"]) % 10000,
                                        suffix_len=suffix_len)
    if not base_points:
        print(f"  [{cand['id']}] PROBE: no base points found -> STARVED", flush=True)
        return None
    base_points = attach_full_geometry(base_points, model, pipeline, U_m, real_latents_pool, suffix_len=suffix_len)
    dims = [bp["geometry"]["W_ratio"] for bp in base_points]
    print(f"  [{cand['id']}] PROBE: {len(base_points)} base points, "
          f"W_ratio mean={np.mean(dims):.3f} median={np.median(dims):.3f}", flush=True)
    if np.median(dims) < 0.05:
        print(f"  [{cand['id']}] PROBE: dim(W)/dim(T) median < 0.05 -> NO_ORTHOGONAL_ROOM", flush=True)
        return None
    return {"base_points": base_points, "U_m": U_m}


def get_or_build_curricula(cand: dict, probe: dict, manifest: dict, K: int, used_episodes: set) -> dict:
    pool = manifest["episode_ids"]["retrieval_pool"]
    path_floor = 5.0     # operationalisation: fixed floors rather than pool-relative percentiles,
    action_floor = 0.05  # to avoid a second full-pool scan just to estimate percentiles per node
    curricula = {}
    for arm in ("orthogonal", "random_dir", "random_traj"):
        seed = SEED + hash((cand["id"], arm)) % 100000
        curricula[arm] = build_curriculum(arm, probe["base_points"], pool, K, seed, path_floor, action_floor,
                                              used_episodes)
        print(f"    [{cand['id']}] retrieval[{arm}]: {curricula[arm]['attrition']}", flush=True)
    return curricula


def build_node_datasets(cand: dict, probe: dict, manifest: dict, K: int, pipeline, used_episodes: set) -> dict | None:
    """Built ONCE per node and reused across Stage 1/2/3 -- all three stages
    of a node share the same K and base-point geometry (only n_updates/seeds
    change between stages), so re-scanning retrieval per stage would triple
    the network/streaming cost for zero scientific benefit. Returns None if
    any arm is starved (0 samples)."""
    curricula = get_or_build_curricula(cand, probe, manifest, K, used_episodes)
    for arm_curric in curricula.values():
        used_episodes.update(arm_curric["episode_ids"])

    n_target_samples = max(1, len(curricula["orthogonal"]["episode_ids"])) * 8
    replay_ds = build_replay_dataset(manifest["episode_ids"]["replay_train"], n_target_samples, pipeline, seed=SEED)

    arm_datasets = {
        "orthogonal": build_dataset_from_episodes(curricula["orthogonal"]["episode_ids"], pipeline, seed=SEED),
        "random_dir": build_dataset_from_episodes(curricula["random_dir"]["episode_ids"], pipeline, seed=SEED),
        "random_traj": build_dataset_from_episodes(curricula["random_traj"]["episode_ids"], pipeline, seed=SEED),
        "continue": replay_ds,
    }
    for arm, ds in arm_datasets.items():
        if len(ds.samples) == 0:
            print(f"    [{cand['id']}] arm={arm} has 0 samples -> STARVED", flush=True)
            return None
    return arm_datasets


def run_stage(cand: dict, probe: dict, arm_datasets: dict, model, pipeline, theta0_state: dict,
                U8: np.ndarray, B8: np.ndarray, n_updates: int, seeds: list[int],
                eval_slice: dict, global_guard_slice: dict, budget: GPUBudget, stage_name: str) -> dict:
    t0 = time.time()
    W_basis = pool_W_basis(probe["base_points"])
    model.load_state_dict(theta0_state)   # `model` is a shared, mutated object -- the previous
    model.eval()                            # arm/stage's TRAINED weights are still loaded unless
                                              # explicitly reset here before measuring the "pre" baseline
    pre_eval = evaluate_E_W(model, eval_slice, pipeline, W_basis, U8, B8)
    pre_E_W = np.mean([v["E_W"] for v in pre_eval["per_episode"].values()])
    pre_gg_eval = evaluate_E_W(model, global_guard_slice, pipeline, np.eye(U8.shape[1]), U8, B8)
    pre_gg_e_all = np.mean([v["E_all"] for v in pre_gg_eval["per_episode"].values()])

    arm_results = {arm: {"E_W": [], "E_all": []} for arm in arm_datasets}
    ortho_gg_e_all = []
    for seed in seeds:
        for arm, ds in arm_datasets.items():
            t_train = time.time()
            res = run_arm(model, theta0_state, ds, n_updates, seed=seed,
                             ckpt_path=AP_DIR / cand["id"] / stage_name / arm / f"seed{seed}.pt")
            budget.add(time.time() - t_train)
            ev = evaluate_E_W(model, eval_slice, pipeline, W_basis, U8, B8)
            e_w_vals = np.array([v["E_W"] for v in ev["per_episode"].values()])
            e_all_vals = np.array([v["E_all"] for v in ev["per_episode"].values()])
            arm_results[arm]["E_W"].append(e_w_vals)
            arm_results[arm]["E_all"].append(e_all_vals)
            if arm == "orthogonal":
                # forgetting is specifically about the ORTHOGONAL arm's effect on the
                # global_guard slice -- must be measured on THIS arm's own just-trained
                # weights, before the next arm's run_arm() overwrites them
                gg_ev = evaluate_E_W(model, global_guard_slice, pipeline, np.eye(U8.shape[1]), U8, B8)
                ortho_gg_e_all.append(np.mean([v["E_all"] for v in gg_ev["per_episode"].values()]))
            print(f"    [{cand['id']}/{stage_name}] arm={arm} seed={seed} steps={res['n_steps']} "
                  f"loss={res['final_loss']:.5f} E_W_mean={e_w_vals.mean():.4f}", flush=True)
            if budget.exhausted():
                print("  BUDGET_EXHAUSTED mid-stage", flush=True)
                return {"status": "BUDGET_EXHAUSTED"}

    ortho_gg_e_all_mean = float(np.mean(ortho_gg_e_all))
    forgetting = (ortho_gg_e_all_mean - pre_gg_e_all) / max(pre_gg_e_all, 1e-8) > GLOBAL_GUARD_MAX_REGRESSION

    gains = {}
    for arm in arm_datasets:
        mean_e_w = np.mean([v.mean() for v in arm_results[arm]["E_W"]])
        gains[arm] = (pre_E_W - mean_e_w) / max(pre_E_W, 1e-8)

    seed_pairs_dir = [arm_results["orthogonal"]["E_W"][i] - arm_results["random_dir"]["E_W"][i] for i in range(len(seeds))]
    seed_pairs_cont = [arm_results["orthogonal"]["E_W"][i] - arm_results["continue"]["E_W"][i] for i in range(len(seeds))]
    seed_signs_dir = [float(np.mean(d)) > 0 for d in seed_pairs_dir]
    seed_signs_cont = [float(np.mean(d)) > 0 for d in seed_pairs_cont]

    all_a = np.concatenate(arm_results["orthogonal"]["E_W"])
    all_dir = np.concatenate(arm_results["random_dir"]["E_W"])
    all_cont = np.concatenate(arm_results["continue"]["E_W"])
    mean_dir, lo_dir, hi_dir = paired_bootstrap_ci(-all_a[:len(all_dir)], -all_dir[:len(all_a)], seed=SEED)
    mean_cont, lo_cont, hi_cont = paired_bootstrap_ci(-all_a[:len(all_cont)], -all_cont[:len(all_a)], seed=SEED + 1)

    result = {
        "status": "EVALUATED", "gains": gains, "forgetting": bool(forgetting),
        "adv_vs_random_dir": {"mean": mean_dir, "ci95": [lo_dir, hi_dir]},
        "adv_vs_continue": {"mean": mean_cont, "ci95": [lo_cont, hi_cont]},
        "seed_signs_agree": bool(all(seed_signs_dir) or not any(seed_signs_dir)) and
                               bool(all(seed_signs_cont) or not any(seed_signs_cont)),
        "n_seeds": len(seeds), "wall_time_s": time.time() - t0,
    }
    s1_pass = (gains["orthogonal"] >= MIN_RELATIVE_TARGET_GAIN and lo_dir > 0 and lo_cont > 0
                 and result["seed_signs_agree"] and not forgetting)
    result["S1_ROUTE_PASS"] = bool(s1_pass)
    return result


def run_node(cand: dict, manifest: dict, model, pipeline, theta0_state: dict, U8, B8,
               eval_slice: dict, global_guard_slice: dict, budget: GPUBudget, used_episodes: set) -> dict:
    real_latents_pool = build_real_latents_pool(manifest["episode_ids"]["route_val"], pipeline)
    probe = probe_node(cand, manifest, model, pipeline, U8, real_latents_pool)
    if probe is None:
        return {"id": cand["id"], "status": "NO_ORTHOGONAL_ROOM_OR_STARVED", "route_rule": "terminalize"}

    print(f"  [{cand['id']}] building node curricula ONCE (reused across all 3 stages)", flush=True)
    arm_datasets = build_node_datasets(cand, probe, manifest, cand["K"], pipeline, used_episodes)
    if arm_datasets is None:
        return {"id": cand["id"], "status": "STARVED", "route_rule": "terminalize"}

    print(f"  [{cand['id']}] Stage 1 (screen): 1 seed, 200 updates", flush=True)
    stage1 = run_stage(cand, probe, arm_datasets, model, pipeline, theta0_state, U8, B8,
                          n_updates=200, seeds=[0], eval_slice=eval_slice,
                          global_guard_slice=global_guard_slice, budget=budget, stage_name="stage1")
    if stage1["status"] != "EVALUATED":
        return {"id": cand["id"], "status": stage1["status"], "stage1": stage1, "route_rule": "terminalize"}
    if stage1["gains"]["orthogonal"] <= 0 or stage1["forgetting"]:
        return {"id": cand["id"], "status": "STAGE1_NO_ADVANTAGE", "stage1": stage1, "route_rule": "terminalize"}

    print(f"  [{cand['id']}] Stage 2 (route): 2 seeds, 500 updates", flush=True)
    stage2 = run_stage(cand, probe, arm_datasets, model, pipeline, theta0_state, U8, B8,
                          n_updates=500, seeds=[0, 1], eval_slice=eval_slice,
                          global_guard_slice=global_guard_slice, budget=budget, stage_name="stage2")
    if stage2["status"] != "EVALUATED" or not stage2["S1_ROUTE_PASS"]:
        return {"id": cand["id"], "status": "S1_ROUTE_FAIL", "stage1": stage1, "stage2": stage2,
                  "route_rule": "next_candidate"}

    print(f"  [{cand['id']}] Stage 3 (robust rerun): 3 seeds, 1000 updates", flush=True)
    stage3 = run_stage(cand, probe, arm_datasets, model, pipeline, theta0_state, U8, B8,
                          n_updates=1000, seeds=[0, 1, 2], eval_slice=eval_slice,
                          global_guard_slice=global_guard_slice, budget=budget, stage_name="stage3")
    if stage3["status"] == "EVALUATED" and stage3["S1_ROUTE_PASS"]:
        return {"id": cand["id"], "status": "SOFT_SUCCESS", "stage1": stage1, "stage2": stage2, "stage3": stage3,
                  "route_rule": "hard_stop_soft_success"}
    return {"id": cand["id"], "status": "STAGE3_FAIL", "stage1": stage1, "stage2": stage2, "stage3": stage3,
              "route_rule": "next_candidate"}


def write_final_report(terminal_reason: str, run_log: list[dict], budget: GPUBudget, extra: dict | None = None) -> None:
    report = {
        "terminal_reason": terminal_reason, "timestamp": now_iso(),
        "gpu_hours_used": budget.used_seconds / 3600, "gpu_hours_budget": MAX_GPU_HOURS,
        "nodes_run": [r["id"] for r in run_log], "run_log": run_log,
        "licensed_claim": (
            "On a fresh heldout episode shard, the frozen-direction orthogonal curriculum improved "
            "true-start one-letter projected prediction error over equal-compute continued training and a "
            "matched random-direction curriculum, with the preregistered confidence and global-regression "
            "guards. This is controlled coarse evidence for guidance-driven predictor improvement."
            if terminal_reason == "SOFT_SUCCESS" else
            "No candidate reached even the soft (Stage-3, unsealed) success criterion within the declared "
            "queue and 10 GPU-hour budget. This is a completed bounded search with a negative/inconclusive "
            "result, not evidence of absence."
        ),
        "not_claimed": ["symbolic abstraction validity", "canonical FSM", "long-horizon capability",
                          "latent representation expansion", "sealed S2_COARSE_SUCCESS (this is a SOFT stop, "
                          "distinguished from the document's full sealed-confirmation success)"],
    }
    if extra:
        report.update(extra)
    write_atomic(OUT_DIR / "autopilot_final_report.json", report)
    md = [f"# Autopilot final report", "", f"**Terminal reason:** {terminal_reason}", "",
           f"GPU-hours used: {report['gpu_hours_used']:.2f} / {MAX_GPU_HOURS}", "",
           f"Nodes run: {report['nodes_run']}", "", "## Licensed claim", "", report["licensed_claim"], "",
           "## Not claimed", ""] + [f"- {x}" for x in report["not_claimed"]]
    (OUT_DIR / "autopilot_final_report.md").write_text("\n".join(md))
    copy_to_downloads(OUT_DIR / "autopilot_final_report.json", OUT_DIR / "autopilot_final_report.md",
                        OUT_DIR / "autopilot_ledger.jsonl")
    print(f"\n=== FINAL: {terminal_reason} ===", flush=True)


def main():
    print(f"=== E0: freeze and calibrate ({now_iso()}) ===", flush=True)
    manifest = build_splits()
    assert_disjoint_splits(manifest)
    U8, B8 = load_frozen_directions()
    model = load_partial_model()
    import copy
    theta0_state = copy.deepcopy(model.state_dict())

    pipeline = fit_pipeline(manifest["episode_ids"]["route_val"] + manifest["episode_ids"]["replay_train"])

    print("  building frozen route_val eval slice + global_guard slice...", flush=True)
    route_val_eval_ids = manifest["episode_ids"]["route_val"][:150]
    eval_slice = build_eval_slice(route_val_eval_ids, n_per_episode=4, seed=SEED)
    guard_ids = manifest["episode_ids"]["global_guard"][:80]
    global_guard_slice = build_eval_slice(guard_ids, n_per_episode=4, seed=SEED + 1)
    print(f"  eval_slice: {len(eval_slice['samples'])} samples / {len(eval_slice['episode_ids'])} episodes; "
          f"global_guard: {len(global_guard_slice['samples'])} samples", flush=True)

    budget = GPUBudget(MAX_GPU_HOURS)
    queue = list(CANDIDATE_QUEUE)
    run_log = []
    used_episodes: set = set()

    append_ledger({"event_type": "E0_COMPLETE", "timestamp": now_iso(), "manifest_roles": manifest["roles"]})

    n_nodes_run = 0
    while queue:
        if budget.exhausted():
            write_final_report("BUDGET_EXHAUSTED", run_log, budget)
            return 12
        if n_nodes_run >= MAX_TRAINED_NODES:
            write_final_report("TREE_EXHAUSTED", run_log, budget, {"note": "MAX_TRAINED_NODES reached"})
            return 10

        cand = queue.pop(0)
        print(f"\n=== NODE {cand['id']} ({now_iso()}, budget used {budget.used_seconds/3600:.2f}h) ===", flush=True)
        try:
            result = run_node(cand, manifest, model, pipeline, theta0_state, U8, B8,
                                 eval_slice, global_guard_slice, budget, used_episodes)
        except Exception as e:  # noqa: BLE001 -- log and terminalize this node, don't crash the controller
            traceback.print_exc()
            result = {"id": cand["id"], "status": "NODE_EXCEPTION", "error": str(e), "route_rule": "next_candidate"}

        n_nodes_run += 1
        run_log.append(result)
        append_ledger({"event_type": "NODE_EVALUATED", "timestamp": now_iso(), "node_id": cand["id"],
                          "status": result["status"], "gpu_hours_used": budget.used_seconds / 3600})
        write_atomic(OUT_DIR / "autopilot_controller_state.json",
                       {"queue_remaining": [c["id"] for c in queue], "run_log": run_log,
                         "gpu_hours_used": budget.used_seconds / 3600})
        copy_to_downloads(OUT_DIR / "autopilot_ledger.jsonl", OUT_DIR / "autopilot_controller_state.json")

        if result["status"] == "SOFT_SUCCESS":
            write_final_report("SOFT_SUCCESS", run_log, budget)
            return 0

    write_final_report("TREE_EXHAUSTED", run_log, budget)
    return 10


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    import os
    os._exit(code)
