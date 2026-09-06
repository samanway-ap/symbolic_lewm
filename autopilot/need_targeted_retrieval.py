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
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.common import stable_seed  # noqa: E402
from autopilot.need_geometry import N_POSITIONS, assign_region, random_subspace_inside  # noqa: E402
from oracle.droid_actions import load_actions_for_episodes  # noqa: E402
from oracle.droid_streaming import stream_requests_hardened  # noqa: E402
from oracle.lewm_g import (  # noqa: E402
    DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, advance_aligned, encode_pixel_windows_batch,
)
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

# Bug fix, code audit 2026-09-06 (TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md
# Change F): Stage 2 used to process occurrences, IN DETERMINISTIC ORDER,
# until a WALL-CLOCK cap was hit -- a legitimate-looking partial result
# whose actual sample size depended on machine speed/load that day, not a
# declared constant. It now processes a FIXED, deterministic SHA-ordered
# PREFIX of at most this many occurrences; `max_minutes` becomes a pure
# safety ABORT (raises IMPLEMENTATION_FAILURE) if even that fixed, bounded
# amount of work stalls, never a silent truncation rule.
N_OCCURRENCES_CAP = 10_000
# Change F item 6: predeclared, applied IDENTICALLY to every arm, so no arm
# (need_curriculum in particular) can end up mostly repeated windows from a
# handful of long episodes. 1 is the document's own stated preference.
PER_EPISODE_CAP = 1
# Generous memory ceiling on retained eligible candidates per cell -- well
# above any k_use (<=200) even after per-episode-cap attrition; NOT a
# selection parameter (selection is exact-k_use, enforced below).
MAX_RETAINED_PER_CELL = 800

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


def _segment_hash(eid: int, t: int, region: int, letter: str, pixels: np.ndarray, raw_action: np.ndarray) -> str:
    """Change F item 5: a content hash over every field that identifies a
    selected segment (episode, position, region, action letter, the exact
    decoded pixels, and the exact raw actions) -- persisted in every
    curriculum/manifest so a later run can verify byte-for-byte that "the
    same segment" really is the same, not just an (eid, t) coincidence."""
    h = hashlib.sha256()
    h.update(f"{eid}:{t}:{region}:{letter}".encode())
    h.update(np.ascontiguousarray(pixels).tobytes())
    h.update(np.ascontiguousarray(raw_action).tobytes())
    return h.hexdigest()


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
                              k_fallback: int = K_FALLBACK, n_occurrences_cap: int = N_OCCURRENCES_CAP,
                              per_episode_cap: int = PER_EPISODE_CAP,
                              require_exact_k: bool = True) -> tuple[dict, dict]:
    """Returns (curricula_by_cell, stats). curricula_by_cell[(region, action)]
    is either None (STARVED, or exact-K infeasible -- see below) or
    {arm: {episode_ids, segments, attrition}} for all five retrieved arms
    (`continue` is built separately, unchanged, from replay_train).

    K-cascade is parameterized (not hardcoded to the v7 experiment's own
    K=200-threshold-implies-K=200 coincidence) so callers with a different
    declared cascade -- e.g. the two-arm pilot's "K=100 once >=200 eligible,
    else K=50 once >=100 eligible" -- can reuse this exact Stage 1/2
    mechanism instead of a second implementation: STARVED if fewer than
    `min_common_eligible` segments are found; `k_target` once at least
    `k_target_threshold` are found; `k_fallback` otherwise.

    Bug fixes, code audit 2026-09-06 (TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md
    Change F):
      1. Stage 2 now processes a FIXED, deterministic SHA-ordered prefix of
         at most `n_occurrences_cap` occurrences (not however many a WALL-
         CLOCK cap happened to allow that run) -- `max_minutes` is now a
         pure safety ABORT (raises) if even that fixed, bounded amount of
         work stalls, never a silent sampling truncation.
      2. `random_traj` is selected the SAME way as every other arm now
         (SHA/insertion order stands in for "uniform", since occurrences
         are already processed in a content-blind SHA order) -- no more
         probabilistic reservoir sampling, which existed specifically to
         handle an unknown-length, wall-clock-truncated stream that no
         longer exists.
      3. A predeclared PER-EPISODE CAP (`per_episode_cap`) is applied
         IDENTICALLY to every arm's final selection, so no arm can end up
         mostly repeated windows from a handful of long episodes.
      4. Every arm's `attrition.segments_selected` must equal the declared
         `k_use` EXACTLY -- a cell that cannot fill every arm to exactly
         k_use (e.g. the ratio-filtered arms coming up short after the
         per-episode cap) is now STARVED as a whole, never silently
         returned with fewer segments than declared."""
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

    ordered = deterministic_order(occurrences, seed)[:n_occurrences_cap]

    heap_cap = MAX_RETAINED_PER_CELL
    state = {}
    for cell, cand in cand_by_cell.items():
        geom, gg = cand["geometry"], cand["global_geometry"]
        d_amb = geom["B_N"].shape[1]
        W_all = gg["W_all_basis"] if gg and not gg.get("empty") else np.zeros((0, d_amb))
        rand_seed = stable_seed(seed, cell, "rand_basis")
        rand_basis = random_subspace_inside(geom["W_basis"], geom["dim_N"], seed=rand_seed)
        state[cell] = {
            "W_all": W_all, "rand_basis": rand_basis,
            "heap_cond": [], "heap_gw": [], "heap_rand": [], "heap_err": [],
            "traj_pool": [], "n_common_eligible": 0,
        }

    t0 = time.monotonic()
    max_seconds = max_minutes * 60
    n_processed = 0
    print(f"  Stage 2: targeted decode, FIXED SHA-ordered prefix of {len(ordered)} occurrences "
          f"(cap={n_occurrences_cap}); {max_minutes:.0f}-minute wall-clock is a SAFETY ABORT, not a sampling "
          f"rule...", flush=True)
    for i in range(0, len(ordered), DECODE_BATCH):
        if time.monotonic() - t0 >= max_seconds:
            raise RuntimeError(
                f"IMPLEMENTATION_FAILURE: retrieval Stage 2 exceeded its {max_minutes:.0f}-minute wall-clock "
                f"SAFETY ABORT after {n_processed}/{len(ordered)} of the fixed occurrence prefix -- this is an "
                f"abort condition (Change F), not a sampling rule; investigate why decoding stalled instead of "
                f"resuming with a smaller/partial result")
        batch = ordered[i:i + DECODE_BATCH]
        decoded = _decode_batch(batch)
        valid = [k for k in batch if k in decoded]
        if valid:
            pixel_stack = np.stack([decoded[k] for k in valid])
            # Bug fix (Change A): thread the SAME model this function receives
            # (typically epoch-7) instead of the epoch-20 default.
            emb_stack = encode_pixel_windows_batch(pixel_stack, model=model).numpy()   # (B, HISTORY_SIZE+1, D)
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
                # Bug fix (Change B): H-1 preceding actions + a_t supplied
                # separately to advance_aligned -- the SAME model input
                # tensors need_geometry.py's scan_transitions constructs for
                # its own direct residual computation (Check 1).
                hist_raw = raw_full[t - HISTORY_SIZE + 1:t]
                hist_norm = pipeline(hist_raw.reshape(-1, RAW_ACTION_DIM))
                hist_flat = hist_norm.reshape(1, HISTORY_SIZE - 1, FRAMESKIP * RAW_ACTION_DIM)
                a_t_raw_converted = pipeline.to_convention(raw_full[t].astype(np.float64))
                a_t_normed = pipeline.normalizer(a_t_raw_converted).reshape(1, -1)
                emb_ctx = torch.from_numpy(emb_window[:HISTORY_SIZE]).float().unsqueeze(0)
                with torch.no_grad():
                    act_emb_hist = model.action_encoder(torch.from_numpy(hist_flat).float().to(DEVICE))
                    a_t_emb = model.action_encoder(
                        torch.from_numpy(a_t_normed).float().unsqueeze(0).to(DEVICE))[:, 0]
                    _new_emb, _new_hist, pred = advance_aligned(
                        model, emb_ctx.to(DEVICE), act_emb_hist, a_t_emb)
                    z_pred = pred[0, 0].cpu().numpy()

                st = state[cell]
                st["n_common_eligible"] += 1
                delta = z_next - z0
                # Preserve the EXACT decoded window this occurrence was scored from (bug fix,
                # code audit 2026-09-06, P0-1): downstream curriculum construction used to keep
                # only `eid`, discarding `t` -- dataset construction then reloaded a DIFFERENT,
                # randomly-windowed slice of the episode's first 20 positions, so the model never
                # actually trained on the segment retrieval selected. Storing the pixels/raw_action
                # already decoded here means the training dataset builder needs no second decode
                # pass and is structurally unable to substitute a different position.
                pixels = decoded[(eid, t, letter)].copy()
                raw_action = raw_full[t - HISTORY_SIZE + 1:t + 2].copy()
                record = {"eid": eid, "t": t, "region": region, "letter": letter,
                          "delta": delta, "delta_norm": float(np.linalg.norm(delta)),
                          "residual": z_pred - z_next, "pixels": pixels, "raw_action": raw_action,
                          "content_hash": _segment_hash(eid, t, region, letter, pixels, raw_action)}
                _heap_push_bounded(st["heap_cond"], _rho(record, cand_by_cell[cell]["geometry"]["B_N"]), record, heap_cap)
                _heap_push_bounded(st["heap_gw"], _rho(record, st["W_all"]), record, heap_cap)
                _heap_push_bounded(st["heap_rand"], _rho(record, st["rand_basis"]), record, heap_cap)
                _heap_push_bounded(st["heap_err"], float(np.linalg.norm(record["residual"])), record, heap_cap)
                # random_traj: SHA/insertion order stands in for "uniform" (Change F item 2) --
                # simply keep the first `heap_cap` eligible occurrences, no score involved.
                if len(st["traj_pool"]) < heap_cap:
                    st["traj_pool"].append(record)
        n_processed += len(batch)
        if (i // DECODE_BATCH) % 10 == 0:
            print(f"    Stage 2: {n_processed}/{len(ordered)} occurrences processed "
                  f"({(time.monotonic() - t0) / 60:.1f}/{max_minutes:.0f} min)", flush=True)

    elapsed_min = (time.monotonic() - t0) / 60
    print(f"  Stage 2 done: processed {n_processed}/{len(ordered)} occurrences (the full fixed prefix) "
          f"in {elapsed_min:.1f} min", flush=True)

    def _select_capped(items_with_score: list, k_use: int, ratio_used: float | None = None) -> list:
        """`items_with_score`: [(score, record), ...] already in the desired
        PRIORITY order (score-descending for ranked arms; SHA/insertion
        order for random_traj, where every score is a 0.0 placeholder).
        Applies the per-episode cap greedily in that order, additionally
        requiring `score >= ratio_used` when given. Returns at most k_use
        (score, record) pairs; the caller checks the count == k_use."""
        chosen, per_eid = [], Counter()
        for score, rec in items_with_score:
            if ratio_used is not None and score < ratio_used:
                continue
            if per_eid[rec["eid"]] >= per_episode_cap:
                continue
            chosen.append((score, rec))
            per_eid[rec["eid"]] += 1
            if len(chosen) >= k_use:
                break
        return chosen

    def _segments(chosen_with_score: list) -> list:
        # The atomic, authoritative unit each arm's dataset is built from
        # (bug fix P0-1): eid/t identify the EXACT selected transition;
        # pixels/raw_action are the SAME decoded window scored above.
        # region/letter/score/content_hash preserved per Change F item 5.
        return [{"eid": rec["eid"], "t": rec["t"], "region": rec["region"], "letter": rec["letter"],
                   "score": float(score), "content_hash": rec["content_hash"],
                   "pixels": rec["pixels"], "raw_action": rec["raw_action"]}
                  for score, rec in chosen_with_score]

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
            items_sorted = sorted(((r, rec) for r, _, rec in heap), key=lambda x: -x[0])
            n50 = sum(1 for r, _ in items_sorted if r >= PROJECTION_RATIO_ORDER[0])
            ratio_used = PROJECTION_RATIO_ORDER[0] if n50 >= k_use else PROJECTION_RATIO_ORDER[1]
            chosen = _select_capped(items_sorted, k_use, ratio_used=ratio_used)
            episode_ids = sorted(set(rec["eid"] for _, rec in chosen))
            curricula[arm] = {"episode_ids": episode_ids, "segments": _segments(chosen),
                                "attrition": {"segments_selected": len(chosen), "ratio_used": ratio_used,
                                                "episodes_selected": len(episode_ids), "k_declared": k_use,
                                                "per_episode_cap": per_episode_cap,
                                                "common_eligible_segments": n_common}}

        items_sorted = sorted(((r, rec) for r, _, rec in st["heap_err"]), key=lambda x: -x[0])
        chosen = _select_capped(items_sorted, k_use)
        episode_ids = sorted(set(rec["eid"] for _, rec in chosen))
        curricula["error_only"] = {"episode_ids": episode_ids, "segments": _segments(chosen),
                                      "attrition": {"segments_selected": len(chosen),
                                                      "episodes_selected": len(episode_ids), "k_declared": k_use,
                                                      "per_episode_cap": per_episode_cap,
                                                      "common_eligible_segments": n_common}}

        # random_traj: SHA/insertion order (uniform, content-blind), never score-sorted.
        items_ordered = [(0.0, rec) for rec in st["traj_pool"]]
        chosen = _select_capped(items_ordered, k_use)
        episode_ids = sorted(set(rec["eid"] for _, rec in chosen))
        curricula["random_traj"] = {"episode_ids": episode_ids, "segments": _segments(chosen),
                                       "attrition": {"segments_selected": len(chosen),
                                                       "episodes_selected": len(episode_ids), "k_declared": k_use,
                                                       "per_episode_cap": per_episode_cap,
                                                       "common_eligible_segments": n_common},
                                       # Exposed for callers that need MULTIPLE independent matched-random
                                       # samples from the SAME eligible pool without re-running Stage 1/2
                                       # (two-arm pilot section 4: 5 paired repetitions, each with an
                                       # independently sampled matched_random curriculum) -- see
                                       # `resample_random_traj_curriculum` below. Not consumed by the
                                       # six-arm tree; harmless additive key.
                                       "_pool_for_resampling": st["traj_pool"]}

        for arm, c in curricula.items():
            print(f"  [{cell}] arm={arm}: {c['attrition']}", flush=True)
        # Change F items 2-4 (`require_exact_k=True`, the default -- used by
        # the two-arm pilot): every arm must reach EXACTLY k_use segments, no
        # silent shrink to whatever survived the ratio/cap filters. The v6/v7
        # six-arm tree's own call site passes `require_exact_k=False` to keep
        # its existing (more lenient: non-empty is enough) starvation
        # tolerance completely unchanged -- this function's algorithm is
        # shared infrastructure (see Change A's own precedent of editing
        # need_geometry.py/need_evaluate.py in place), but its STARVATION
        # STRINGENCY is a scientific-comparison-specific choice this task
        # does not license changing for the six-arm tree.
        if require_exact_k:
            infeasible = any(c["attrition"]["segments_selected"] != k_use for c in curricula.values())
        else:
            infeasible = any(not c["episode_ids"] for c in curricula.values())
        results[cell] = None if infeasible else curricula

    stats = {"n_occurrences_total": len(ordered), "n_processed": n_processed,
             "stage1_minutes": stage1_minutes, "stage2_minutes": elapsed_min,
             "n_occurrences_cap": n_occurrences_cap, "per_episode_cap": per_episode_cap}
    return results, stats


def resample_random_traj_curriculum(traj_pool: list[dict], k_use: int, seed: int,
                                        per_episode_cap: int = PER_EPISODE_CAP) -> dict:
    """Draws ONE independent, seeded, per-episode-capped uniform sample of
    exactly `k_use` segments from `traj_pool` (a cell's raw eligible-segment
    pool, from `curricula["random_traj"]["_pool_for_resampling"]`) -- WITHOUT
    re-running Stage 1/2. For the two-arm experiment's 5 paired repetitions
    (TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md section 4: "an independently
    sampled matched-random curriculum" per pair, while `need_curriculum`
    stays the fixed deterministic top-K). A seeded permutation of the pool
    stands in for "uniformly sampled" (the pool itself has no informative
    order -- it was populated in a content-blind SHA order); the per-episode
    cap is applied identically to every other arm's selection. Returns the
    same `{episode_ids, segments, attrition}` shape `run_targeted_retrieval`
    returns for its own arms; raises if fewer than k_use segments survive
    the cap (no silent shrink, matching `require_exact_k=True`)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(traj_pool))
    chosen, per_eid = [], Counter()
    for idx in perm:
        rec = traj_pool[int(idx)]
        if per_eid[rec["eid"]] >= per_episode_cap:
            continue
        chosen.append(rec)
        per_eid[rec["eid"]] += 1
        if len(chosen) >= k_use:
            break
    if len(chosen) != k_use:
        raise RuntimeError(
            f"IMPLEMENTATION_FAILURE: resample_random_traj_curriculum could only draw {len(chosen)}/{k_use} "
            f"segments from a pool of {len(traj_pool)} under per_episode_cap={per_episode_cap} (seed={seed})")
    segments = [{"eid": rec["eid"], "t": rec["t"], "region": rec["region"], "letter": rec["letter"],
                   "score": 0.0, "content_hash": rec["content_hash"],
                   "pixels": rec["pixels"], "raw_action": rec["raw_action"]} for rec in chosen]
    episode_ids = sorted(set(rec["eid"] for rec in chosen))
    return {"episode_ids": episode_ids, "segments": segments,
              "attrition": {"segments_selected": len(chosen), "episodes_selected": len(episode_ids),
                              "k_declared": k_use, "per_episode_cap": per_episode_cap,
                              "pool_size": len(traj_pool), "resample_seed": seed}}
