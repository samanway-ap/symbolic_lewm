"""Two-arm feasibility pilot: does the frozen action-conditioned need
curriculum (`need_curriculum`) beat matched-random sampling
(`matched_random`) at short-horizon, frozen-encoder prediction, in the
node's own predeclared need subspace N(r,a,1)?

Separate from, and does not modify, the v7 experiment tree: reuses its
frozen artifacts READ-ONLY (partial checkpoint, Lever-0 basis + U4, region
anchors, the frozen N00/N01/N02 T/V/W/N geometry, action-letter alphabet,
episode splits) via `need_controller.get_or_reconstruct_frozen_candidates`
and `need_targeted_retrieval.run_targeted_retrieval` (both already require
an exact match against Attempt 1's recorded geometry before proceeding, so
this pilot inherits that guarantee for free). Writes ONLY under the
`need_two_arm_pilot_*` namespace and never assigns/opens a confirm_A/B/C
shard. This is a feasibility signal, not a confirmatory test -- see
DECISION RULE below for exactly what a positive result does and does not
license.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

# Basic resource headroom (added after the 2026-09-05 crash: repeated
# multiprocessing pool churn during retrieval exhausted OS resources and
# froze the host). Capping BLAS/PyTorch CPU threads keeps this process from
# also competing hard for CPU against the streaming worker processes and
# whatever else is running on the machine; the model/training work itself is
# small (ViT-tiny, batch size 32) and does not need many threads.
torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.common import (  # noqa: E402
    OUT_DIR, SEED, assert_disjoint_splits, build_splits, copy_to_downloads, load_frozen_directions,
    load_partial_model, now_iso, stable_seed, write_atomic,
)
from autopilot.controller import fit_pipeline  # noqa: E402
from autopilot.dataset import ArmDataset, build_dataset_from_segments, build_replay_dataset  # noqa: E402
from autopilot.evaluate import assert_slice_nonempty, build_eval_slice, evaluate_E_W, paired_bootstrap_ci  # noqa: E402
from autopilot.need_common import AlphabetLookup, WallClockBudget, load_lever0_basis  # noqa: E402
from autopilot.need_controller import get_or_reconstruct_frozen_candidates  # noqa: E402
from autopilot.need_evaluate import build_cell_eval_slice, effective_rank, full_latent_predictions  # noqa: E402
from autopilot.need_targeted_retrieval import run_targeted_retrieval  # noqa: E402
from autopilot.train_arm import freeze_encoder, unfreeze_trainable  # noqa: E402
from oracle.lewm_g import DEVICE, HISTORY_SIZE  # noqa: E402
from training.finetune import _collate  # noqa: E402

PILOT_DIR = OUT_DIR / "need_two_arm_pilot_nodes"
MANIFEST_PATH = OUT_DIR / "need_two_arm_pilot_manifest.json"
METRICS_PATH = OUT_DIR / "need_two_arm_pilot_metrics.json"
REPORT_PATH = OUT_DIR / "need_two_arm_pilot_report.md"

MAX_WALL_HOURS = 2.0
RETRIEVAL_MAX_MINUTES = 60.0
# pilot's OWN K-cascade (distinct from the main tree's 200-threshold/200-K,
# 100-threshold/100-K): >=200 eligible segments -> K=100; else >=100 -> K=50;
# else infeasible. Same Stage 1/2 mechanism as the main tree, parameterized.
K_TARGET_THRESHOLD = 200
K_TARGET = 100
K_FALLBACK_THRESHOLD = 100
K_FALLBACK = 50
MIN_COMMON_ELIGIBLE = K_FALLBACK_THRESHOLD

N_UPDATES_DIAGNOSTIC = 200
N_UPDATES_PRIMARY = 500
SEEDS = [0, 1]
BATCH_SIZE = 32
LR = 5e-5
WEIGHT_DECAY = 1e-3
MIN_RELATIVE_GAIN = 0.02
GLOBAL_REGRESSION_MAX = 0.02
EFFRANK_DROP_MAX = 0.10
MIN_COMMON_K = 10          # below this, the two arms cannot be meaningfully compared -- infeasible
MIN_EVAL_EPISODES = 5      # fail-closed floor for eval/guard slices (P1-8)


def freeze_common_k_segments(need_segments: list[dict], random_segments: list[dict]) -> tuple[list, list, int]:
    """Bug fix, code audit 2026-09-06 (P0-4): `K` counted SELECTED SEGMENTS,
    but each arm could independently return fewer than `k_use` (the
    alignment-ratio cascade can starve short of `k_use`), so the two arms'
    datasets were not guaranteed the same cardinality -- "matched compute"
    was asserted in a comment, not enforced in code. Both arms' ALREADY-
    RANKED segment lists (highest-alignment / earliest-reservoir-index
    first) are here truncated to the same common K = min(len(need),
    len(random)), so both arms train on exactly K segments plus exactly the
    same replay examples. Returns (need_segments[:k], random_segments[:k], k)."""
    k = min(len(need_segments), len(random_segments))
    return need_segments[:k], random_segments[:k], k


def build_arm_dataset(segments: list[dict], replay_ds: ArmDataset, pipeline) -> ArmDataset:
    """Exactly `len(segments)` curriculum samples (bug fix P0-1: built from
    the EXACT selected (episode_id, t) segments, not re-windowed episode
    IDs) plus the SAME shared `replay_ds` object for both arms (bug fix
    P0-4: previously each arm built its OWN replay dataset sized to its OWN
    curriculum length with the same seed -- identical only when the
    requested sizes happened to match, which was never enforced). Passing
    one shared `ArmDataset` guarantees byte-identical replay examples."""
    curr_ds = build_dataset_from_segments(segments, pipeline)
    samples = curr_ds.samples + replay_ds.samples
    episode_ids = sorted(set(curr_ds.episode_ids) | set(replay_ds.episode_ids))
    return ArmDataset(samples=samples, pipeline=pipeline, episode_ids=episode_ids)


def assert_segments_survive(dataset: ArmDataset, frozen_segment_ids: set[tuple[int, int]], n_curriculum: int) -> None:
    """Required invariant (code audit 2026-09-06, section 9.3): every
    curriculum sample's (episode, position) must be exactly one of the
    frozen selected segments -- 100%, not merely high. Checks only the
    first `n_curriculum` samples (the curriculum portion; replay samples
    are drawn from `replay_train` and are never expected to be in
    `frozen_segment_ids`)."""
    hits = sum(1 for s in dataset.samples[:n_curriculum] if (s["episode"], s["pos"]) in frozen_segment_ids)
    if hits != n_curriculum:
        raise RuntimeError(
            f"IMPLEMENTATION_FAILURE: segment-retention invariant violated -- {hits}/{n_curriculum} "
            f"curriculum samples matched a frozen selected (episode_id, t); expected 100%")


def paired_eid_arrays(per_episode_a: dict, per_episode_b: dict) -> tuple:
    """Bug fix, code audit 2026-09-06 (P0-5): the two arms' E_N values used
    to be concatenated by ARRAY POSITION (`dict.values()` order), correct
    only if both `per_episode` dicts happen to insert episodes in identical
    order -- an unstated invariant of `evaluate_E_W`'s iteration, never
    enforced. This aligns explicitly by episode_id and fails loudly if the
    two arms' evaluated episode sets ever diverge, instead of silently
    mispairing two different episodes' errors."""
    common = sorted(set(per_episode_a) & set(per_episode_b))
    if len(common) != len(per_episode_a) or len(common) != len(per_episode_b):
        raise RuntimeError(
            f"IMPLEMENTATION_FAILURE: eval episode sets diverged between arms -- "
            f"a={sorted(per_episode_a)} b={sorted(per_episode_b)}")
    return (np.array([per_episode_a[e]["E_W"] for e in common]),
            np.array([per_episode_b[e]["E_W"] for e in common]))


def hierarchical_paired_bootstrap_ci(diffs_by_seed: dict, seed: int, n_boot: int = 1000) -> tuple:
    """Two-level (seed, then episode) paired bootstrap (code audit
    2026-09-06, P0-5): resamples SEEDS with replacement, then within each
    resampled seed resamples ITS OWN episodes with replacement, rather than
    pooling all seeds x episodes into one i.i.d. bag. Both seeds trained the
    SAME two frozen curricula end-to-end, so within a seed the episode-level
    errors share that seed's training trajectory -- pooling across seeds
    treats them as independent draws and understates variance. With only
    `len(SEEDS)==2` seeds this resampling is itself coarse (4 possible seed
    pairs); report alongside the per-seed CIs, not as a substitute for a
    larger seed count."""
    rng = np.random.default_rng(seed)
    seeds = sorted(diffs_by_seed)
    n_seeds = len(seeds)
    point = float(np.mean([diffs_by_seed[s].mean() for s in seeds]))
    boots = []
    for _ in range(n_boot):
        resampled = rng.integers(0, n_seeds, size=n_seeds)
        seed_means = []
        for si in resampled:
            d = diffs_by_seed[seeds[si]]
            seed_means.append(float(d[rng.integers(0, len(d), size=len(d))].mean()))
        boots.append(float(np.mean(seed_means)))
    boots = np.array(boots)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def run_checkpointed_training(model, theta0_state: dict, dataset: ArmDataset, seed: int,
                                 checkpoints: list[int], ckpt_dir: Path) -> dict:
    """ONE continuous training run per (arm, seed) -- exactly the four
    required (2 arms x 2 seeds) -- pausing to snapshot model state at each
    update count in `checkpoints` without resetting, so the 500-update
    primary endpoint and its 200-update diagnostic point come from the SAME
    trajectory, not two separate runs. One whole-run retry (unchanged
    scientific parameters) on OOM or any other exception, per the pilot's
    "one implementation retry" allowance."""
    def _attempt():
        model.load_state_dict(theta0_state)
        model.train()
        freeze_encoder(model)
        params = unfreeze_trainable(model)
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
        gen = torch.Generator().manual_seed(seed)
        n = len(dataset.samples)
        encoder_before = copy.deepcopy({k: v for k, v in model.state_dict().items() if k.startswith("encoder.")})

        step = 0
        results = {}
        max_updates = max(checkpoints)
        remaining = sorted(checkpoints)
        loss_val = None
        while step < max_updates:
            n_batches = max(1, n // BATCH_SIZE)
            perm = torch.randperm(n, generator=gen).tolist()
            for b in range(n_batches):
                if step >= max_updates:
                    break
                idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                if not idx:
                    continue
                batch_samples = [dataset.samples[i] for i in idx]
                batch = _collate(batch_samples, dataset.pipeline, DEVICE)
                batch["action"] = torch.nan_to_num(batch["action"], 0.0)
                info = model.encode(batch)
                emb, act_emb = info["emb"], info["act_emb"]
                ctx_emb, ctx_act = emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE]
                tgt_emb = emb[:, 1:]
                pred_emb = model.predict(ctx_emb, ctx_act)
                # Bug fix, code audit 2026-09-06 (loss dilution): only the
                # LAST predicted step corresponds to the actual selected
                # (episode_id, t) transition retrieval scored and the
                # curriculum was built on; averaging the loss over all
                # HISTORY_SIZE predicted steps mixes in unselected, purely
                # incidental context-window predictions and dilutes the
                # training signal the curriculum was meant to concentrate.
                loss = (pred_emb[:, -1] - tgt_emb[:, -1]).pow(2).mean()
                if torch.isnan(loss):
                    raise FloatingPointError("NaN loss")
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_val = float(loss.item())
                step += 1
                while remaining and step >= remaining[0]:
                    cp = remaining.pop(0)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"seed{seed}_step{cp}.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    results[cp] = {"step": step, "loss": loss_val, "checkpoint": str(ckpt_path)}

        encoder_after = {k: v for k, v in model.state_dict().items() if k.startswith("encoder.")}
        for k in encoder_before:
            if not torch.equal(encoder_before[k], encoder_after[k]):
                raise RuntimeError(f"IMPLEMENTATION_FAILURE: encoder tensor {k} changed despite freeze")
        return results

    try:
        return _attempt()
    except (torch.cuda.OutOfMemoryError, FloatingPointError) as e:  # noqa: BLE001
        print(f"    [retry] {type(e).__name__}: {e} -- one whole-run retry, unchanged parameters", flush=True)
        torch.cuda.empty_cache()
        return _attempt()


def _eval_all(model, eval_slice: dict, guard_slice: dict, pipeline, B_N, W_basis, V_basis, U8, B8) -> dict:
    D = B_N.shape[1]
    model.eval()
    ev_n = evaluate_E_W(model, eval_slice, pipeline, B_N, U8, B8)
    ev_full = evaluate_E_W(model, eval_slice, pipeline, np.eye(D), U8, B8)
    ev_w = evaluate_E_W(model, eval_slice, pipeline, W_basis, U8, B8)
    ev_v = evaluate_E_W(model, eval_slice, pipeline, V_basis, U8, B8) if V_basis.shape[0] > 0 else None
    gg_full = evaluate_E_W(model, guard_slice, pipeline, np.eye(D), U8, B8)
    return {
        "E_N": np.array([v["E_W"] for v in ev_n["per_episode"].values()]) if ev_n["per_episode"] else np.zeros(1),
        # Bug fix, code audit 2026-09-06 (P0-5): keep the eid-keyed dict too,
        # so cross-arm comparisons can pair explicitly by episode_id instead
        # of trusting `dict.values()` insertion order to match between calls.
        "E_N_per_episode": ev_n["per_episode"],
        "E_all": np.array([v["E_all"] for v in ev_full["per_episode"].values()]) if ev_full["per_episode"] else np.zeros(1),
        "E_W": np.array([v["E_W"] for v in ev_w["per_episode"].values()]) if ev_w["per_episode"] else np.zeros(1),
        "E_V": (np.array([v["E_W"] for v in ev_v["per_episode"].values()]) if ev_v and ev_v["per_episode"]
                  else np.zeros(1)),
        "guard_E_all": float(np.mean([v["E_all"] for v in gg_full["per_episode"].values()])) if gg_full["per_episode"] else 0.0,
        "guard_effrank": effective_rank(full_latent_predictions(model, guard_slice, pipeline)),
    }


def write_pilot_report(status: str, extra: dict, wall: WallClockBudget) -> None:
    payload = {"status": status, "timestamp": now_iso(), "wall_hours_used": wall.elapsed_hours(),
                 "wall_hours_budget": MAX_WALL_HOURS, **extra}
    write_atomic(METRICS_PATH, payload)
    md = [f"# Two-arm feasibility pilot", "", f"**Status:** {status}", "",
           f"Wall-clock used: {payload['wall_hours_used']:.2f}h / {MAX_WALL_HOURS}h", ""]
    if "reason" in extra:
        md += [f"Reason: {extra['reason']}", ""]
    REPORT_PATH.write_text("\n".join(md))
    copy_to_downloads(METRICS_PATH, REPORT_PATH, MANIFEST_PATH)
    print(f"\n=== PILOT FINAL: {status} ({wall.elapsed_hours():.2f}h) ===", flush=True)


def main() -> int:
    wall = WallClockBudget(MAX_WALL_HOURS)
    print(f"=== need_two_arm_pilot INIT ({now_iso()}) ===", flush=True)
    manifest = build_splits()
    assert_disjoint_splits(manifest)
    lever0_basis = load_lever0_basis()
    alphabet = AlphabetLookup()
    model = load_partial_model()
    theta0_state = copy.deepcopy(model.state_dict())
    # Bug fix, code audit 2026-09-06 (P0-6): the converter used to be fit on
    # route_val + replay_train, i.e. partly on the very data the pilot later
    # evaluates on (`route_val` backs `eval_slice`) -- any route_val-specific
    # action-statistics leak into the (frozen, reused-everywhere) pipeline
    # before evaluation even begins. Fit on replay_train only.
    pipeline = fit_pipeline(manifest["episode_ids"]["replay_train"])

    print("=== reusing frozen E0 geometry (persisted cache, or verified reconstruction) ===", flush=True)
    build_result = get_or_reconstruct_frozen_candidates(manifest, lever0_basis, model, pipeline, alphabet, wall)
    if build_result["status"] != "OK":
        write_pilot_report("RETRIEVAL_INFEASIBLE", {"reason": f"E0 build/verification failed: {build_result['status']}"}, wall)
        return 1
    candidates = build_result["candidates"]
    anchors = build_result["anchors"]
    print(f"  frozen candidates (order N00->N01->N02): "
          f"{[(c['id'], c['region'], c['action']) for c in candidates]}", flush=True)

    print(f"=== targeted retrieval (pilot cascade: >={K_TARGET_THRESHOLD}->K={K_TARGET}, "
          f">={K_FALLBACK_THRESHOLD}->K={K_FALLBACK}), {RETRIEVAL_MAX_MINUTES:.0f}-min cap ===", flush=True)
    curricula_by_cell, retrieval_stats = run_targeted_retrieval(
        candidates, manifest["episode_ids"]["retrieval_pool"], anchors, lever0_basis, pipeline, model, alphabet,
        seed=SEED, max_minutes=RETRIEVAL_MAX_MINUTES, min_common_eligible=MIN_COMMON_ELIGIBLE,
        k_target_threshold=K_TARGET_THRESHOLD, k_target=K_TARGET, k_fallback=K_FALLBACK)

    chosen = None
    for cand in candidates:   # frozen order: first retrieval-feasible cell wins, not best-scoring
        if curricula_by_cell.get((cand["region"], cand["action"])) is not None:
            chosen = cand
            break
    if chosen is None:
        write_pilot_report("RETRIEVAL_INFEASIBLE",
                              {"candidates_tried": [c["id"] for c in candidates], "retrieval_stats": retrieval_stats},
                              wall)
        return 1

    cell = (chosen["region"], chosen["action"])
    curricula = curricula_by_cell[cell]
    need_curric, random_curric = curricula["conditional_need"], curricula["random_traj"]

    # Bug fix, code audit 2026-09-06 (P0-4): freeze both arms to the SAME
    # common K = min(selected segments) instead of trusting each arm's own
    # (possibly-starved) `attrition["segments_selected"]` count to already
    # match -- they were never asserted equal.
    need_segments, random_segments, k_use = freeze_common_k_segments(
        need_curric["segments"], random_curric["segments"])
    if k_use < MIN_COMMON_K:
        write_pilot_report(
            "RETRIEVAL_INFEASIBLE",
            {"reason": f"common K={k_use} < MIN_COMMON_K={MIN_COMMON_K} after matching arm cardinality",
             "need_segments_available": len(need_curric["segments"]),
             "random_segments_available": len(random_curric["segments"])},
            wall)
        return 1
    print(f"  chosen: {chosen['id']} (region={chosen['region']}, action={chosen['action']}); "
          f"need_curriculum={need_curric['attrition']}; matched_random={random_curric['attrition']}; "
          f"common K used (after matching)={k_use}", flush=True)

    need_segment_ids = {(int(s["eid"]), int(s["t"])) for s in need_segments}
    random_segment_ids = {(int(s["eid"]), int(s["t"])) for s in random_segments}
    need_episode_ids = sorted({eid for eid, _ in need_segment_ids})
    random_episode_ids = sorted({eid for eid, _ in random_segment_ids})

    write_atomic(MANIFEST_PATH, {
        "timestamp": now_iso(), "chosen_candidate": chosen["id"], "region": chosen["region"],
        "action": chosen["action"], "candidates_considered_order": [c["id"] for c in candidates],
        "k_cascade": {"k_target_threshold": K_TARGET_THRESHOLD, "k_target": K_TARGET,
                        "k_fallback_threshold": K_FALLBACK_THRESHOLD, "k_fallback": K_FALLBACK,
                        "min_common_eligible": MIN_COMMON_ELIGIBLE},
        "retrieval_stats": retrieval_stats, "k_use_common": k_use,
        "need_curriculum": {"segments": sorted(need_segment_ids), "attrition": need_curric["attrition"]},
        "matched_random": {"segments": sorted(random_segment_ids), "attrition": random_curric["attrition"]},
        "segment_overlap": len(need_segment_ids & random_segment_ids),
        "episode_overlap": len(set(need_episode_ids) & set(random_episode_ids)),
    })

    eval_slice = build_cell_eval_slice(manifest["episode_ids"]["route_val"], chosen["region"], chosen["action"],
                                          anchors, lever0_basis, pipeline, alphabet, seed=SEED + 1)
    guard_ids = manifest["episode_ids"]["global_guard"][:80]
    guard_slice = build_eval_slice(guard_ids, n_per_episode=4, seed=SEED + 2)
    try:
        assert_slice_nonempty(eval_slice, "cell eval_slice", min_episodes=MIN_EVAL_EPISODES)
        assert_slice_nonempty(guard_slice, "global guard_slice", min_episodes=MIN_EVAL_EPISODES)
    except RuntimeError as e:
        write_pilot_report("RETRIEVAL_INFEASIBLE", {"reason": str(e)}, wall)
        return 1
    print(f"  eval_slice: {len(eval_slice['samples'])} samples; guard_slice: {len(guard_slice['samples'])} samples",
          flush=True)

    # ONE shared replay dataset, sized to the common K, reused byte-for-byte
    # by both arms (bug fix P0-4).
    replay_seed = SEED + 100
    replay_ds = build_replay_dataset(manifest["episode_ids"]["replay_train"], k_use, pipeline, seed=replay_seed)
    need_ds = build_arm_dataset(need_segments, replay_ds, pipeline)
    random_ds = build_arm_dataset(random_segments, replay_ds, pipeline)
    assert_segments_survive(need_ds, need_segment_ids, len(need_segments))
    assert_segments_survive(random_ds, random_segment_ids, len(random_segments))
    print(f"  need_curriculum dataset: {len(need_ds.samples)} samples "
          f"({len(need_segments)} curriculum + {len(replay_ds.samples)} shared replay); "
          f"matched_random dataset: {len(random_ds.samples)} samples "
          f"({len(random_segments)} curriculum + {len(replay_ds.samples)} shared replay)", flush=True)

    U8, B8 = load_frozen_directions()
    geom = chosen["geometry"]
    B_N, W_basis, V_basis = geom["B_N"], geom["W_basis"], geom["V_basis"]

    model.load_state_dict(theta0_state)
    pre = _eval_all(model, eval_slice, guard_slice, pipeline, B_N, W_basis, V_basis, U8, B8)
    pre_E_N = float(np.mean(pre["E_N"]))
    print(f"  pre-training (theta0): E_N={pre_E_N:.4f} E_all={np.mean(pre['E_all']):.4f} "
          f"guard_E_all={pre['guard_E_all']:.4f} guard_effrank={pre['guard_effrank']:.3f}", flush=True)

    node_dir = PILOT_DIR / chosen["id"]
    per_seed = {"need_curriculum": {}, "matched_random": {}}
    for seed in SEEDS:
        for arm_name, ds in (("need_curriculum", need_ds), ("matched_random", random_ds)):
            print(f"  training arm={arm_name} seed={seed} ({N_UPDATES_PRIMARY} updates, "
                  f"diagnostic at {N_UPDATES_DIAGNOSTIC})...", flush=True)
            ckpt_dir = node_dir / arm_name
            cps = run_checkpointed_training(model, theta0_state, ds, seed,
                                                [N_UPDATES_DIAGNOSTIC, N_UPDATES_PRIMARY], ckpt_dir)

            model.load_state_dict(torch.load(cps[N_UPDATES_PRIMARY]["checkpoint"], map_location=DEVICE))
            post = _eval_all(model, eval_slice, guard_slice, pipeline, B_N, W_basis, V_basis, U8, B8)

            model.load_state_dict(torch.load(cps[N_UPDATES_DIAGNOSTIC]["checkpoint"], map_location=DEVICE))
            model.eval()
            diag_ev = evaluate_E_W(model, eval_slice, pipeline, B_N, U8, B8)
            diag_E_N = float(np.mean([v["E_W"] for v in diag_ev["per_episode"].values()])) if diag_ev["per_episode"] else float("nan")

            per_seed[arm_name][seed] = {**post, "diagnostic_200_E_N_mean": diag_E_N,
                                           "loss_primary": cps[N_UPDATES_PRIMARY]["loss"]}
            print(f"    arm={arm_name} seed={seed}: E_N_mean={np.mean(post['E_N']):.4f} "
                  f"(diagnostic@200={diag_E_N:.4f}) guard_E_all={post['guard_E_all']:.4f} "
                  f"guard_effrank={post['guard_effrank']:.3f} (wall {wall.elapsed_hours():.2f}h)", flush=True)
            if wall.exhausted():
                write_pilot_report("TIME_BUDGET_EXHAUSTED", {"reason": "wall-clock exhausted mid-training"}, wall)
                return 1

    print("=== computing metrics ===", flush=True)
    gains, per_seed_E_N_mean = {}, {"need_curriculum": {}, "matched_random": {}}
    for arm in ("need_curriculum", "matched_random"):
        per_seed_gain = {}
        for seed in SEEDS:
            e_n_mean = float(np.mean(per_seed[arm][seed]["E_N"]))
            per_seed_E_N_mean[arm][seed] = e_n_mean
            per_seed_gain[seed] = (pre_E_N - e_n_mean) / max(pre_E_N, 1e-8)
        gains[arm] = per_seed_gain

    seed_signs = {seed: per_seed_E_N_mean["need_curriculum"][seed] < per_seed_E_N_mean["matched_random"][seed]
                    for seed in SEEDS}
    need_wins_both = all(seed_signs.values())
    random_wins_both = not any(seed_signs.values())
    mean_G_need = float(np.mean(list(gains["need_curriculum"].values())))
    mean_G_random = float(np.mean(list(gains["matched_random"].values())))
    per_seed_delta_G = {seed: gains["need_curriculum"][seed] - gains["matched_random"][seed] for seed in SEEDS}
    mean_delta_G = float(np.mean(list(per_seed_delta_G.values())))

    # Bug fix, code audit 2026-09-06 (P0-5): explicit eid-keyed pairing per
    # seed, then a hierarchical (seed-then-episode) bootstrap instead of
    # pooling all seeds x episodes as one i.i.d. bag.
    diffs_by_seed = {}
    per_seed_ci = {}
    for s in SEEDS:
        random_arr, need_arr = paired_eid_arrays(
            per_seed["matched_random"][s]["E_N_per_episode"], per_seed["need_curriculum"][s]["E_N_per_episode"])
        diffs_by_seed[s] = random_arr - need_arr
        pm, plo, phi = paired_bootstrap_ci(random_arr, need_arr, seed=SEED + 9 + s)
        per_seed_ci[s] = {"mean": pm, "ci95": [plo, phi], "n_episodes": int(len(random_arr))}
    ci_mean, ci_lo, ci_hi = hierarchical_paired_bootstrap_ci(diffs_by_seed, seed=SEED + 9)

    need_gg_regression = [
        (per_seed["need_curriculum"][s]["guard_E_all"] - pre["guard_E_all"]) / max(pre["guard_E_all"], 1e-8)
        for s in SEEDS]
    need_effrank_drop = [
        (pre["guard_effrank"] - per_seed["need_curriculum"][s]["guard_effrank"]) / max(pre["guard_effrank"], 1e-8)
        for s in SEEDS]
    global_ok = all(r <= GLOBAL_REGRESSION_MAX for r in need_gg_regression)
    effrank_ok = all(d <= EFFRANK_DROP_MAX for d in need_effrank_drop)

    if not need_wins_both and not random_wins_both:
        label = "INCONCLUSIVE"
    elif random_wins_both:
        label = "TWO_ARM_NEGATIVE"
    else:
        criteria = {"need_wins_both_seeds": need_wins_both, "mean_G_need_ge_0.02": mean_G_need >= MIN_RELATIVE_GAIN,
                      "mean_delta_G_positive": mean_delta_G > 0, "global_regression_ok": global_ok,
                      "effrank_ok": effrank_ok}
        if all(criteria.values()):
            label = "STRONG_TWO_ARM_SIGNAL" if ci_lo > 0 else "TWO_ARM_FEASIBILITY_SIGNAL"
        else:
            label = "BELOW_THRESHOLD"

    metrics = {
        "status": label, "timestamp": now_iso(), "wall_hours_used": wall.elapsed_hours(),
        "wall_hours_budget": MAX_WALL_HOURS, "chosen_candidate": chosen["id"], "region": chosen["region"],
        "action": chosen["action"], "k_use": k_use, "pre_E_N": pre_E_N, "pre_E_all": float(np.mean(pre["E_all"])),
        "pre_guard_E_all": pre["guard_E_all"], "pre_guard_effrank": pre["guard_effrank"],
        "gains_G": gains, "mean_G_need": mean_G_need, "mean_G_random": mean_G_random,
        "per_seed_delta_G": per_seed_delta_G, "mean_delta_G": mean_delta_G,
        "seed_signs_need_beats_random": seed_signs,
        # Renamed from `paired_advantage_random_minus_need` (code audit
        # 2026-09-06, P0-5): now a hierarchical (seed-then-episode) bootstrap
        # over EXPLICITLY eid-paired per-seed differences, not a pooled
        # i.i.d. bootstrap over positionally-concatenated arrays. Per-seed
        # CIs are reported alongside since n_seeds=2 makes the hierarchical
        # resample itself coarse.
        "paired_advantage_random_minus_need_hierarchical": {"mean": ci_mean, "ci95": [ci_lo, ci_hi]},
        "paired_advantage_random_minus_need_per_seed": per_seed_ci,
        "need_global_regression_per_seed": need_gg_regression, "need_effrank_drop_per_seed": need_effrank_drop,
        "global_regression_ok": global_ok, "effrank_ok": effrank_ok,
        "segment_overlap": len(need_segment_ids & random_segment_ids),
        "episode_overlap": len(set(need_episode_ids) & set(random_episode_ids)),
        "per_seed_detail": {
            arm: {str(s): {"E_N_mean": float(np.mean(per_seed[arm][s]["E_N"])),
                             "E_all_mean": float(np.mean(per_seed[arm][s]["E_all"])),
                             "E_W_mean": float(np.mean(per_seed[arm][s]["E_W"])),
                             "E_V_mean": float(np.mean(per_seed[arm][s]["E_V"])),
                             "diagnostic_200_E_N_mean": per_seed[arm][s]["diagnostic_200_E_N_mean"],
                             "guard_E_all": per_seed[arm][s]["guard_E_all"],
                             "guard_effrank": per_seed[arm][s]["guard_effrank"]}
                    for s in SEEDS}
            for arm in ("need_curriculum", "matched_random")},
        "not_claimed": ["validated bisimulation", "FSM correctness", "encoder representation expansion",
                          "general capability improvement", "a confirmatory (vs. feasibility) result"],
    }
    write_atomic(METRICS_PATH, metrics)

    md = [f"# Two-arm feasibility pilot", "", f"**Result: {label}**", "",
           f"Candidate: {chosen['id']} (region={chosen['region']}, action={chosen['action']}), K={k_use}", "",
           f"mean G(need_curriculum) = {mean_G_need:.4f}  |  mean G(matched_random) = {mean_G_random:.4f}  |  "
           f"mean delta_G = {mean_delta_G:.4f}", "",
           f"Paired advantage (random - need), hierarchical seed-then-episode bootstrap "
           f"(n_seeds={len(SEEDS)}, coarse): mean={ci_mean:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]",
           "", f"Per-seed paired advantage: {per_seed_ci}", "",
           "", f"Seed signs (need beats random): {seed_signs}", "",
           f"Global-latent regression per seed: {need_gg_regression} (ok={global_ok}); "
           f"effective-rank drop per seed: {need_effrank_drop} (ok={effrank_ok})", "",
           "## Not claimed", ""] + [f"- {x}" for x in metrics["not_claimed"]]
    REPORT_PATH.write_text("\n".join(md))
    copy_to_downloads(METRICS_PATH, REPORT_PATH, MANIFEST_PATH)
    print(f"\n=== PILOT FINAL: {label} ({wall.elapsed_hours():.2f}h) ===", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    import os
    os._exit(code)
