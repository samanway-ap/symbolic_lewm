"""Lever 1 -- fixed-direction offset-feasibility sweep (v5 doc,
EXPERIMENTS_orthogonal_expansion_1.md SS4 "Lever 1"). Question: do any of the
8 *existing* Lever-0 residual directions admit a state-only binary offset
that is simultaneously temporally nontrivial and predictively robust? Tests
threshold placement before reopening direction search.

Frozen: residual projection, directions u_1..u_8 (Lever 0's own, unchanged),
checkpoint, action letters, data split, state-window definition. ONLY b_i
changes. No FSM extraction. No training.

Efficiency note driving the whole design: b_i does not affect the
CONTINUOUS score s_i(z) = u_i.z (only the offset subtracted from it, i.e.
which side of a threshold z falls on). So the expensive part -- one
teacher-forced model step per real (episode, t) pair -- is done ONCE,
unconditional on any candidate offset; every candidate offset is then a
cheap numpy threshold over the SAME precomputed continuous scores.

Selection/evaluation separation: everything in `sweep_offsets_for_direction`
and `select_offset` runs on TRAIN only. HELDOUT is touched exactly once,
after the offsets are frozen, in `evaluate_on_heldout`.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a0_label_repair.horizon_decomposition import real_window_state
from a0_label_repair.horizon_decomposition_b import bootstrap_ci
from corpus.latent_corpus import Corpus, build_corpus, test_ids, train_ids
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, get_model
from oracle.production_oracle import _step_fn_from_convention, load_alphabet_symbols

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
N_DIRECTIONS = 8
N_CANDIDATES_QUANTILE = 61       # quantile-spaced candidates over the training score range
SWITCH_RATE_BAND = (0.02, 0.25)
MIN_BLOCK_DETERMINISM = 0.70
MIN_OCCUPANCY = 20
MAX_BOUNDARY_BAND_FOR_FEASIBILITY = 0.5   # a candidate this dominated by boundary noise is excluded
ERROR_BAND_K = 1.0


def build_teacher_forced_dataset(corpus: Corpus, actions: dict, e_lookup: dict, pipeline, step_fn, model, U8: np.ndarray):
    """One teacher-forced model step per eligible (episode, t): real window
    at t -> real action a_t -> predicted z_hat_{t+1}. Returns per-direction
    continuous scores (NOT yet offset -- that's u_i . z, offset subtracted
    later per candidate), plus the real action symbol and episode id for
    each sample (needed for block determinism)."""
    n_pos = corpus.n_positions
    s_pred, s_real, s_start, syms, eids = [], [], [], [], []
    for i in range(corpus.n_episodes):
        eid = int(corpus.episode_ids[i])
        if eid not in actions:
            continue
        names = corpus.symbol_seq(i)   # names[t] drives real z_t -> z_{t+1}, len == n_pos-1
        for t in range(HISTORY_SIZE - 1, min(n_pos - 1, len(names))):
            ctx_start = t - HISTORY_SIZE + 1
            w = real_window_state(corpus, e_lookup, actions, eid, ctx_start, pipeline, model)
            sym = names[t]
            w2 = step_fn(w, sym)
            z_pred = w2.emb[-1].numpy()
            z_real = corpus.latents[i, t + 1]
            z_start = corpus.latents[i, t]
            s_pred.append(z_pred @ U8.T)
            s_real.append(z_real @ U8.T)
            s_start.append(z_start @ U8.T)
            syms.append(sym)
            eids.append(eid)
    return {"s_pred": np.array(s_pred), "s_real": np.array(s_real), "s_start": np.array(s_start),
              "syms": np.array(syms), "eids": np.array(eids)}


def full_sequence_scores(corpus: Corpus, U8: np.ndarray) -> np.ndarray:
    """(n_ep, n_pos, 8) continuous score, all real positions -- for
    occupancy, switch rate, and dwell/jitter on FULL trajectories."""
    return corpus.latents @ U8.T


def dwell_jitter(labels_2d: np.ndarray) -> dict:
    """labels_2d: (n_ep, n_pos) binary. Run-length of consecutive equal
    labels within each episode."""
    runs = []
    for row in labels_2d:
        changes = np.flatnonzero(np.diff(row) != 0)
        bounds = np.concatenate([[-1], changes, [len(row) - 1]])
        runs.extend((bounds[1:] - bounds[:-1]).tolist())
    runs = np.array(runs)
    return {"mean_run_length": float(runs.mean()), "median_run_length": float(np.median(runs)),
              "min_run_length": int(runs.min()), "n_runs": int(len(runs))}


def block_determinism_single_bit(bit_seq: np.ndarray, sym_seq: np.ndarray, min_support: int = 3) -> dict:
    """bit_seq: (n_ep, n_pos) binary. sym_seq: (n_ep, n_pos-1) action symbol
    per transition. (bit, action) -> next-bit determinism, transition-weighted."""
    pair_succ = defaultdict(list)
    for ep_bits, ep_syms in zip(bit_seq, sym_seq):
        n = min(len(ep_bits) - 1, len(ep_syms))
        for t in range(n):
            pair_succ[(int(ep_bits[t]), ep_syms[t])].append(int(ep_bits[t + 1]))
    n_trans, n_match, ent_sum = 0, 0, 0.0
    for succs in pair_succ.values():
        n = len(succs)
        if n < min_support:
            continue
        vals, counts = np.unique(succs, return_counts=True)
        p = counts / counts.sum()
        n_trans += n
        n_match += int(counts.max())
        ent_sum += float(-(p * np.log2(p)).sum()) * n
    return {"transition_weighted_determinism": n_match / n_trans if n_trans else None,
              "mean_successor_entropy": ent_sum / n_trans if n_trans else None, "n_transitions_supported": n_trans}


def evaluate_candidate(b: float, direction_idx: int, tf: dict, full_scores: np.ndarray, corpus: Corpus,
                          eps_std: float, sym_seq: np.ndarray) -> dict:
    s_pred, s_real, s_start = tf["s_pred"][:, direction_idx], tf["s_real"][:, direction_idx], tf["s_start"][:, direction_idx]
    bit_pred, bit_real, bit_start = (s_pred > b).astype(int), (s_real > b).astype(int), (s_start > b).astype(int)

    full_col = full_scores[:, :, direction_idx]
    full_bits = (full_col > b).astype(int)
    dj = dwell_jitter(full_bits)
    switches = (full_bits[:, 1:] != full_bits[:, :-1])
    switch_rate = float(switches.mean())

    pos_occ = int((full_col > b).sum())
    neg_occ = int((full_col <= b).sum())
    occupancy_ok = min(pos_occ, neg_occ) >= MIN_OCCUPANCY

    band_frac = float((np.abs(s_real - b) < ERROR_BAND_K * eps_std).mean())

    fidelity = float((bit_pred == bit_real).mean())
    persistence = float((bit_start == bit_real).mean())
    margin = fidelity - persistence

    actual_switch = bit_real != bit_start
    predicted_switch = bit_pred != bit_start
    n_switch = int(actual_switch.sum())
    n_nonswitch = int((~actual_switch).sum())
    recall = float((actual_switch & predicted_switch).sum() / n_switch) if n_switch else None
    fsr = float((~actual_switch & predicted_switch).sum() / n_nonswitch) if n_nonswitch else None
    specificity = 1 - fsr if fsr is not None else None
    balanced_acc = (recall + specificity) / 2 if (recall is not None and specificity is not None) else None

    det = block_determinism_single_bit(full_bits, sym_seq)

    return {
        "offset": float(b), "switch_rate": switch_rate, "in_band": SWITCH_RATE_BAND[0] <= switch_rate <= SWITCH_RATE_BAND[1],
        "occupancy_pos": pos_occ, "occupancy_neg": neg_occ, "occupancy_ok": occupancy_ok,
        "dwell_jitter": dj, "boundary_band_fraction": band_frac,
        "one_letter_fidelity": fidelity, "matched_start_persistence": persistence, "margin": margin,
        "switch_recall": recall, "false_switch_rate": fsr, "balanced_accuracy": balanced_acc,
        "n_switch_events": n_switch, "n_nonswitch_events": n_nonswitch,
        "block_determinism": det["transition_weighted_determinism"], "successor_entropy": det["mean_successor_entropy"],
        "n_transitions_supported": det["n_transitions_supported"],
    }


def find_density_valleys(scores_1d: np.ndarray, n_valleys: int = 5) -> list[float]:
    hist, edges = np.histogram(scores_1d, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    valleys = []
    for i in range(2, len(hist) - 2):
        if hist[i] < hist[i - 1] and hist[i] < hist[i + 1] and hist[i] < np.median(hist):
            valleys.append((hist[i], centers[i]))
    valleys.sort()
    return [float(c) for _, c in valleys[:n_valleys]]


def select_offset(candidates: list[dict]) -> dict | None:
    """PREDECLARED selection rule, applied to TRAIN-only candidate metrics:
    among candidates with switch rate in-band AND occupancy>=20 on both
    sides AND boundary-band fraction <= 0.5 (not overwhelmingly dominated by
    noise), pick the one maximising one-letter margin over matched
    start-persistence. A direction with no such candidate is infeasible."""
    feasible = [c for c in candidates if c["in_band"] and c["occupancy_ok"]
                 and c["boundary_band_fraction"] <= MAX_BOUNDARY_BAND_FOR_FEASIBILITY]
    if not feasible:
        return None
    return max(feasible, key=lambda c: c["margin"])


def main():
    rng = np.random.default_rng(SEED)
    train = build_corpus(train_ids(), 60, "train")
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    e_lookup_train = {int(train.episode_ids[i]): i for i in range(train.n_episodes)}
    e_lookup_held = {int(heldout.episode_ids[i]): i for i in range(heldout.n_episodes)}

    tp = np.load(OUT_DIR / "tessellation_params_residualized.npz")
    U8 = tp["U_m8"]     # directions FROZEN, offsets discarded -- Lever 1 re-derives b_i

    fit_ids = sorted(rng.choice(heldout.episode_ids, size=min(200, len(heldout.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    pipeline = ActionPipeline.fit(np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0))
    symbols_dict = load_alphabet_symbols(ALPHABET_PATH)
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(w, s):
        return step_fn_raw(w, symbols_dict[s])

    model = get_model()

    print("=== building TRAIN teacher-forced dataset (one step per real transition) ===", flush=True)
    n_raw_train = train.n_positions * FRAMESKIP
    actions_train = load_actions_for_episodes(train.episode_ids.tolist(), max_frames=n_raw_train)
    tf_train = build_teacher_forced_dataset(train, actions_train, e_lookup_train, pipeline, step_fn, model, U8)
    print(f"  {len(tf_train['s_pred'])} teacher-forced pairs built", flush=True)

    eps_train = tf_train["s_pred"] - tf_train["s_real"]     # (n, 8)
    eps_std_per_dir = eps_train.std(axis=0)
    print(f"  epsilon std per direction: {eps_std_per_dir}", flush=True)

    full_scores_train = full_sequence_scores(train, U8)     # (n_ep, n_pos, 8)
    sym_seq_train = np.stack([train.symbol_seq(i) for i in range(train.n_episodes)])

    print("\n=== sweeping candidate offsets per direction (TRAIN only) ===", flush=True)
    report = {"eps_std_per_direction": eps_std_per_dir.tolist(), "config": {
        "n_candidates_quantile": N_CANDIDATES_QUANTILE, "switch_rate_band": list(SWITCH_RATE_BAND),
        "min_occupancy": MIN_OCCUPANCY, "max_boundary_band_for_feasibility": MAX_BOUNDARY_BAND_FOR_FEASIBILITY,
        "error_band_k": ERROR_BAND_K, "selection_rule": ("switch_rate in-band AND occupancy>=20 both sides AND "
                                                              "boundary_band_fraction<=0.5 -> max margin over "
                                                              "matched start-persistence, TRAIN only"),
    }, "per_direction": {}}

    selected_offsets = {}
    for di in range(N_DIRECTIONS):
        pop = full_scores_train[:, :, di].flatten()
        quantile_candidates = np.quantile(pop, np.linspace(0.02, 0.98, N_CANDIDATES_QUANTILE)).tolist()
        valley_candidates = find_density_valleys(pop)
        all_candidates = sorted(set(quantile_candidates + valley_candidates))

        results = [evaluate_candidate(b, di, tf_train, full_scores_train, train, eps_std_per_dir[di], sym_seq_train)
                     for b in all_candidates]
        chosen = select_offset(results)
        n_feasible = sum(1 for c in results if c["in_band"] and c["occupancy_ok"]
                           and c["boundary_band_fraction"] <= MAX_BOUNDARY_BAND_FOR_FEASIBILITY)
        report["per_direction"][f"cell_h{di}"] = {
            "n_candidates_tested": len(all_candidates), "n_feasible": n_feasible,
            "selected": chosen, "all_candidates": results,
        }
        if chosen:
            selected_offsets[di] = chosen["offset"]
            print(f"  direction {di}: {n_feasible}/{len(all_candidates)} feasible, "
                  f"selected offset={chosen['offset']:.4f} switch_rate={chosen['switch_rate']:.4f} "
                  f"margin={chosen['margin']:+.4f} band_frac={chosen['boundary_band_fraction']:.3f}", flush=True)
        else:
            print(f"  direction {di}: 0/{len(all_candidates)} feasible -- NO ADMISSIBLE OFFSET ON TRAIN", flush=True)

    print(f"\n{len(selected_offsets)}/8 directions have a train-feasible offset: "
          f"{sorted(selected_offsets.keys())}", flush=True)

    # ================= evaluate ONCE on heldout, frozen offsets only =================
    print("\n=== evaluating frozen offsets ONCE on HELDOUT ===", flush=True)
    n_raw_held = heldout.n_positions * FRAMESKIP
    actions_held = load_actions_for_episodes(heldout.episode_ids.tolist(), max_frames=n_raw_held)
    tf_held = build_teacher_forced_dataset(heldout, actions_held, e_lookup_held, pipeline, step_fn, model, U8)
    full_scores_held = full_sequence_scores(heldout, U8)
    sym_seq_held = np.stack([heldout.symbol_seq(i) for i in range(heldout.n_episodes)])
    print(f"  {len(tf_held['s_pred'])} held-out teacher-forced pairs built", flush=True)

    held_out = {}
    ordered_dirs = sorted(selected_offsets.keys())
    joint_pass_prefix = []
    for di in ordered_dirs:
        b = selected_offsets[di]
        cand = evaluate_candidate(b, di, tf_held, full_scores_held, heldout, eps_std_per_dir[di], sym_seq_held)
        # margin CI via episode-level bootstrap on the pair dataset
        eids = tf_held["eids"]
        s_pred, s_real, s_start = tf_held["s_pred"][:, di], tf_held["s_real"][:, di], tf_held["s_start"][:, di]
        bit_pred, bit_real, bit_start = (s_pred > b).astype(int), (s_real > b).astype(int), (s_start > b).astype(int)
        fid_match, pers_match = (bit_pred == bit_real).astype(float), (bit_start == bit_real).astype(float)
        by_eid_margin = defaultdict(list)
        for i, eid in enumerate(eids):
            by_eid_margin[eid].append(fid_match[i] - pers_match[i])
        margin_vals = np.array([np.mean(v) for v in by_eid_margin.values()])
        mean_m, lo_m, hi_m = bootstrap_ci(margin_vals, SEED + di)

        admissible = (cand["in_band"] and cand["block_determinism"] is not None
                       and cand["block_determinism"] >= MIN_BLOCK_DETERMINISM
                       and lo_m > 0 and cand["occupancy_ok"])
        held_out[f"cell_h{di}"] = {**cand, "margin_ci95": [lo_m, hi_m], "admissible": admissible}
        joint_pass_prefix.append(admissible)
        print(f"  cell_h{di}: switch_rate={cand['switch_rate']:.4f} block_det={cand['block_determinism']} "
              f"margin={cand['margin']:+.4f} ci95=[{lo_m:+.4f},{hi_m:+.4f}] admissible={admissible}", flush=True)

    largest_admissible_prefix = 0
    for i, ok in enumerate(joint_pass_prefix):
        if ok:
            largest_admissible_prefix = i + 1
        else:
            break
    report["heldout_evaluation"] = {
        "per_direction": held_out, "ordered_directions": ordered_dirs,
        "largest_admissible_nested_prefix": largest_admissible_prefix,
        "note": "prefix counts directions in `ordered_directions` order, all must be admissible up to that point",
    }
    print(f"\n  largest admissible nested prefix: {largest_admissible_prefix} of {len(ordered_dirs)} feasible directions",
          flush=True)

    (OUT_DIR / "lever1_offset_sweep_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote artifacts/lever1_offset_sweep_report.json", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
