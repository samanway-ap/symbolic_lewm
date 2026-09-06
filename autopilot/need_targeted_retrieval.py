"""v7 SS4/SS5 corrected retrieval: a targeted, time-capped pass over the
ENTIRE frozen `retrieval_pool`, replacing both Attempt 1's unregistered
500-episode subsample AND the (measured, infeasible: ~7.3h) idea of decoding
full video windows for the whole pool. See artifacts/preregistration.md for
the full correction history.

Two stages:
  Stage 1 -- build a complete INVERTED ACTION-LETTER INDEX over the full
             pool using ONLY local action data (parquet reads, no video):
             every (episode, position) occurrence of a needed action letter,
             across every episode in the pool, at its full local length (not
             capped to the first N_POSITIONS positions the rest of this
             package otherwise uses). Cheap: no video decode at all.
  Stage 2 -- process those occurrences in a DETERMINISTIC hash order, up to
             a predeclared wall-clock cap, decoding only the small
             (HISTORY_SIZE+1)-frame window immediately around each
             occurrence (not the whole episode) via `stream_episode_frames`'
             arbitrary start_frame support. Every decoded occurrence updates
             bounded per-cell, per-arm selection structures (a size-K
             min-heap by alignment score for `conditional_need`/`global_W`/
             `random_dir`, a size-K min-heap by residual magnitude for
             `error_only`, and a deterministic reservoir sample for
             `random_traj`) so memory and per-item work stay bounded
             regardless of how large the full occurrence index turns out to
             be, and so a wall-clock cutoff mid-pass still yields a
             deterministic, order-independent-of-luck partial result rather
             than an arbitrary truncation.

K / STARVED semantics are UNCHANGED from the v7 text (K=200 segments,
falling back to K=100 segments if the alignment cascade at rho>=0.50 then
0.35 can't fill it; STARVED if fewer than 100 common-eligible segments were
found for a cell) -- counted in SEGMENTS throughout, per the K-unit
correction (see preregistration.md).

Eligibility-filter adaptation (values unchanged, mechanism documented):
`ACTION_FLOOR` is checked against the full episode's own raw actions (Stage
1 already reads them locally, no extra cost). `PATH_FLOOR` was calibrated
against a full N_POSITIONS(=20)-position window; decoding only
(HISTORY_SIZE+1)=4 frames per occurrence means the SAME absolute floor
would compare a much shorter path against a floor sized for a much longer
one, so it is applied as a per-step RATE (PATH_FLOOR / (N_POSITIONS - 1))
against the small window's own per-step rate -- identical to the original
absolute check when the window size IS N_POSITIONS, and the same declared
constant, just scale-adapted to the smaller decode window this correction
requires.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.need_geometry import N_POSITIONS, assign_region, random_subspace_inside  # noqa: E402
from oracle.droid_actions import load_actions_for_episodes  # noqa: E402
from oracle.droid_streaming import stream_requests_hardened  # noqa: E402
from oracle.lewm_g import DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, encode_pixel_windows_batch  # noqa: E402
from retrieval.geometric import action_magnitude, path_length  # noqa: E402

PROJECTION_RATIO_ORDER = (0.50, 0.35)
K_TARGET = 200                    # segments, per v7 SS4's own text (unit-corrected)
K_FALLBACK = 100                  # segments
MIN_COMMON_ELIGIBLE = 100         # segments
PATH_FLOOR = 5.0                  # unchanged constant, calibrated against an N_POSITIONS-length window
ACTION_FLOOR = 0.05               # unchanged constant; scale-invariant (mean, not sum), no adaptation needed
PATH_FLOOR_PER_STEP = PATH_FLOOR / (N_POSITIONS - 1)
MAX_RETRIEVAL_MINUTES = 75.0
DECODE_BATCH = 256
MAX_DECODE_WORKERS = 32
EPS = 1e-8

_counter = itertools.count()


def build_inverted_action_index(pool_episode_ids: list[int], needed_letters: set, pipeline,
                                    alphabet, action_chunk: int = 5000) -> tuple[list, dict, dict]:
    """Stage 1: LOCAL-ONLY (no video). Returns (occurrences, raw_actions_by_eid,
    episode_action_magnitude): occurrences is every (eid, t, letter) where
    letter in needed_letters, searched over each episode's FULL local action
    length (not capped at N_POSITIONS). raw_actions_by_eid / the magnitude
    dict are cached here (Stage 1 already paid to read them) so Stage 2
    never re-reads action parquet data."""
    from oracle.droid_actions import _episode_to_file
    ep2file = _episode_to_file()
    sorted_ids = sorted(pool_episode_ids, key=lambda e: ep2file.get(e, ""))

    occurrences: list[tuple[int, int, str]] = []
    raw_actions_by_eid: dict[int, np.ndarray] = {}
    episode_action_magnitude: dict[int, float] = {}

    for i in range(0, len(sorted_ids), action_chunk):
        chunk = sorted_ids[i:i + action_chunk]
        actions = load_actions_for_episodes(chunk, max_frames=None)
        for eid, raw in actions.items():
            n_positions_ep = raw.shape[0] // FRAMESKIP
            if n_positions_ep <= HISTORY_SIZE:
                continue
            raw_full = raw[:n_positions_ep * FRAMESKIP].reshape(n_positions_ep, FRAMESKIP, RAW_ACTION_DIM)
            raw_actions_by_eid[eid] = raw_full
            episode_action_magnitude[eid] = action_magnitude(raw_full)
            for t in range(HISTORY_SIZE - 1, n_positions_ep - 1):
                seg = pipeline.to_convention(raw_full[t].astype(np.float64))
                letter = alphabet.letter(seg)
                if letter in needed_letters:
                    occurrences.append((int(eid), int(t), letter))
        print(f"    Stage 1: {min(i + action_chunk, len(sorted_ids))}/{len(sorted_ids)} episodes indexed, "
              f"{len(occurrences)} occurrences so far", flush=True)
    return occurrences, raw_actions_by_eid, episode_action_magnitude


def deterministic_order(occurrences: list, seed: int) -> list:
    def h(o):
        return hashlib.sha256(f"{seed}:{o[0]}:{o[1]}".encode()).hexdigest()
    return sorted(occurrences, key=h)


def _heap_push_bounded(heap: list, key: float, record: dict, K: int) -> None:
    c = next(_counter)
    if len(heap) < K:
        heapq.heappush(heap, (key, c, record))
    elif key > heap[0][0]:
        heapq.heapreplace(heap, (key, c, record))


def _reservoir_update(reservoir: list, n_seen: int, record: dict, K: int, rng: np.random.Generator) -> int:
    n_seen += 1
    if len(reservoir) < K:
        reservoir.append(record)
    else:
        j = int(rng.integers(0, n_seen))
        if j < K:
            reservoir[j] = record
    return n_seen


def _rho(record: dict, basis: np.ndarray) -> float:
    if basis.shape[0] == 0 or record["delta_norm"] < EPS:
        return 0.0
    proj = basis.T @ (basis @ record["delta"])
    return float(np.linalg.norm(proj) / (record["delta_norm"] + EPS))


_DEFAULT_CAMERA = "observation.images.exterior_1_left"


def _decode_batch(items: list, max_workers: int = MAX_DECODE_WORKERS) -> dict:
    """items: [(eid, t, letter), ...]. Decodes ONLY the small
    (HISTORY_SIZE+1)-frame window around each occurrence (arbitrary
    start_frame, not the whole episode). Process-based (via
    stream_requests_hardened), not thread-based: a bad remote source can
    otherwise block a thread pool for hours instead of failing fast -- see
    that function's docstring (need-autopilot v7 Attempt 2, 2026-09-04)."""
    requests = [((eid, t, letter), eid, _DEFAULT_CAMERA, (t - HISTORY_SIZE + 1) * FRAMESKIP, HISTORY_SIZE + 1,
                 FRAMESKIP) for (eid, t, letter) in items]
    results, _errors = stream_requests_hardened(requests, max_workers=max_workers, per_task_timeout_s=20.0)
    return {k: v for k, v in results.items() if v.shape[0] == HISTORY_SIZE + 1}


def run_targeted_retrieval(candidates: list[dict], pool_episode_ids: list[int], anchors: np.ndarray,
                              lever0_basis: dict, pipeline, model, alphabet, seed: int,
                              max_minutes: float = MAX_RETRIEVAL_MINUTES,
                              min_common_eligible: int = MIN_COMMON_ELIGIBLE,
                              k_target_threshold: int = K_TARGET, k_target: int = K_TARGET,
                              k_fallback: int = K_FALLBACK) -> tuple[dict, dict]:
    """Returns (curricula_by_cell, stats). curricula_by_cell[(region, action)]
    is either None (STARVED) or {arm: {episode_ids, attrition}} for all six
    arms EXCEPT `continue` (built separately, unchanged, from replay_train).

    K-cascade is parameterized (not hardcoded to the v7 experiment's own
    K=200-threshold-implies-K=200 coincidence) so callers with a different
    declared cascade -- e.g. the two-arm pilot's "K=100 once >=200 eligible,
    else K=50 once >=100 eligible" -- can reuse this exact Stage 1/2
    mechanism instead of a second implementation: STARVED if fewer than
    `min_common_eligible` segments are found; `k_target` once at least
    `k_target_threshold` are found; `k_fallback` otherwise."""
    needed_cells = {(c["region"], c["action"]) for c in candidates}
    needed_letters = {a for (_, a) in needed_cells}
    cand_by_cell = {(c["region"], c["action"]): c for c in candidates}

    print(f"  Stage 1: inverted action-letter index over the FULL retrieval_pool "
          f"({len(pool_episode_ids)} episodes; letters={sorted(needed_letters)}, full local episode length, "
          f"no video)...", flush=True)
    t_stage1 = time.monotonic()
    occurrences, raw_actions_by_eid, ep_action_mag = build_inverted_action_index(
        pool_episode_ids, needed_letters, pipeline, alphabet)
    stage1_minutes = (time.monotonic() - t_stage1) / 60
    print(f"  Stage 1 done in {stage1_minutes:.1f} min: {len(occurrences)} total "
          f"(episode, position) occurrences of a needed letter across the full pool", flush=True)

    ordered = deterministic_order(occurrences, seed)

    heap_cap = max(k_target_threshold, k_target, k_fallback)
    state = {}
    for cell, cand in cand_by_cell.items():
        geom, gg = cand["geometry"], cand["global_geometry"]
        d_amb = geom["B_N"].shape[1]
        W_all = gg["W_all_basis"] if gg and not gg.get("empty") else np.zeros((0, d_amb))
        rand_seed = seed + (hash(cell) % 1_000_000) + 7
        rand_basis = random_subspace_inside(geom["W_basis"], geom["dim_N"], seed=rand_seed)
        state[cell] = {
            "W_all": W_all, "rand_basis": rand_basis,
            "heap_cond": [], "heap_gw": [], "heap_rand": [], "heap_err": [],
            "reservoir_traj": [], "n_seen_traj": 0, "n_common_eligible": 0,
            "rng_traj": np.random.default_rng(rand_seed + 4),
        }

    t0 = time.monotonic()
    max_seconds = max_minutes * 60
    n_processed = 0
    cutoff_hit = False
    print(f"  Stage 2: targeted decode, deterministic order, {max_minutes:.0f}-minute cap "
          f"({len(ordered)} occurrences queued)...", flush=True)
    for i in range(0, len(ordered), DECODE_BATCH):
        if time.monotonic() - t0 >= max_seconds:
            cutoff_hit = True
            break
        batch = ordered[i:i + DECODE_BATCH]
        decoded = _decode_batch(batch)
        valid = [k for k in batch if k in decoded]
        if valid:
            pixel_stack = np.stack([decoded[k] for k in valid])
            emb_stack = encode_pixel_windows_batch(pixel_stack).numpy()   # (B, HISTORY_SIZE+1, D)
            for k_idx, (eid, t, letter) in enumerate(valid):
                emb_window = emb_stack[k_idx]                              # (HISTORY_SIZE+1, D)
                z0, z_next = emb_window[HISTORY_SIZE - 1], emb_window[HISTORY_SIZE]
                region = int(assign_region(z0[None, :], anchors, lever0_basis)[0])
                cell = (region, letter)
                if cell not in state:
                    continue
                mean_step = path_length(emb_window) / max(1, emb_window.shape[0] - 1)
                if mean_step < PATH_FLOOR_PER_STEP or ep_action_mag.get(eid, 0.0) < ACTION_FLOOR:
                    continue

                raw_full = raw_actions_by_eid[eid]
                ctx_raw = raw_full[t - HISTORY_SIZE + 1:t + 1]
                act_norm = pipeline(ctx_raw.reshape(-1, RAW_ACTION_DIM))
                act_flat = act_norm.reshape(1, HISTORY_SIZE, FRAMESKIP * RAW_ACTION_DIM)
                a = torch.from_numpy(act_flat).float().to(DEVICE)
                emb_ctx = torch.from_numpy(emb_window[:HISTORY_SIZE]).float()
                with torch.no_grad():
                    act_emb = model.action_encoder(a)[0]
                    pred = model.predict(emb_ctx.unsqueeze(0).to(DEVICE), act_emb.unsqueeze(0).to(DEVICE))
                    z_pred = pred[0, -1].cpu().numpy()

                st = state[cell]
                st["n_common_eligible"] += 1
                delta = z_next - z0
                record = {"eid": eid, "t": t, "delta": delta, "delta_norm": float(np.linalg.norm(delta)),
                          "residual": z_pred - z_next}
                _heap_push_bounded(st["heap_cond"], _rho(record, cand_by_cell[cell]["geometry"]["B_N"]), record, heap_cap)
                _heap_push_bounded(st["heap_gw"], _rho(record, st["W_all"]), record, heap_cap)
                _heap_push_bounded(st["heap_rand"], _rho(record, st["rand_basis"]), record, heap_cap)
                _heap_push_bounded(st["heap_err"], float(np.linalg.norm(record["residual"])), record, heap_cap)
                st["n_seen_traj"] = _reservoir_update(st["reservoir_traj"], st["n_seen_traj"], record,
                                                          heap_cap, st["rng_traj"])
        n_processed += len(batch)
        if (i // DECODE_BATCH) % 10 == 0:
            print(f"    Stage 2: {n_processed}/{len(ordered)} occurrences processed "
                  f"({(time.monotonic() - t0) / 60:.1f}/{max_minutes:.0f} min)", flush=True)

    elapsed_min = (time.monotonic() - t0) / 60
    print(f"  Stage 2 done: processed {n_processed}/{len(ordered)} occurrences in {elapsed_min:.1f} min "
          f"({'WALL-CLOCK CUTOFF HIT' if cutoff_hit else 'exhausted the full index before the cap'})", flush=True)

    results = {}
    for cell, st in state.items():
        n_common = st["n_common_eligible"]
        if n_common < min_common_eligible:
            print(f"  [{cell}] common eligible segments: {n_common} < {min_common_eligible} -> STARVED", flush=True)
            results[cell] = None
            continue
        k_use = k_target if n_common >= k_target_threshold else k_fallback

        curricula = {}
        for arm, heap in (("conditional_need", st["heap_cond"]), ("global_W", st["heap_gw"]),
                            ("random_dir", st["heap_rand"])):
            items_sorted = sorted(heap, key=lambda x: -x[0])
            n50 = sum(1 for r, _, _ in items_sorted if r >= PROJECTION_RATIO_ORDER[0])
            ratio_used = PROJECTION_RATIO_ORDER[0] if n50 >= k_use else PROJECTION_RATIO_ORDER[1]
            chosen = [rec for r, _, rec in items_sorted if r >= ratio_used][:k_use]
            episode_ids = sorted(set(rec["eid"] for rec in chosen))
            curricula[arm] = {"episode_ids": episode_ids,
                                "attrition": {"segments_selected": len(chosen), "ratio_used": ratio_used,
                                                "episodes_selected": len(episode_ids),
                                                "common_eligible_segments": n_common}}

        items_sorted = sorted(st["heap_err"], key=lambda x: -x[0])[:k_use]
        chosen = [rec for _, _, rec in items_sorted]
        episode_ids = sorted(set(rec["eid"] for rec in chosen))
        curricula["error_only"] = {"episode_ids": episode_ids,
                                      "attrition": {"segments_selected": len(chosen),
                                                      "episodes_selected": len(episode_ids),
                                                      "common_eligible_segments": n_common}}

        chosen = st["reservoir_traj"][:k_use]
        episode_ids = sorted(set(rec["eid"] for rec in chosen))
        curricula["random_traj"] = {"episode_ids": episode_ids,
                                       "attrition": {"segments_selected": len(chosen),
                                                       "episodes_selected": len(episode_ids),
                                                       "common_eligible_segments": n_common}}

        for arm, c in curricula.items():
            print(f"  [{cell}] arm={arm}: {c['attrition']}", flush=True)
        results[cell] = None if any(not c["episode_ids"] for c in curricula.values()) else curricula

    stats = {"n_occurrences_total": len(ordered), "n_processed": n_processed,
             "stage1_minutes": stage1_minutes, "stage2_minutes": elapsed_min, "cutoff_hit": cutoff_hit}
    return results, stats
