"""A0.1 -- horizon/error decomposition, per user spec (2026-09-02), run in
response to the Lever-0 result: switch rate moved cleanly into band, but
end-to-end FSM-vs-real fidelity got WORSE, not better. The old single-number
fidelity conflates three things: (1) symbolic-extraction error, (2) open-loop
rollout error, (3) inadequacy of phi itself. This separates them.

Frozen inputs, no refitting: `tessellation_params_residualized.npz` (Lever 0,
untouched). One FSM extraction per m (SepSeq -- a validated runtime fix, not
a new tuning exercise), then everything below reuses those four machines and
cached/rolled-out latents. No training. No Ladder A/B/C. No new hyperplanes.

Six quantities per horizon h in {1,2,4,8,12,16,21} (all read off ONE rollout
per start point/word, at all seven checkpoints, not seven separate rollouts):

  F_TF            -- teacher-forced one-step: phi(ghat(z_t,a_t)) == phi(z_{t+1}).
                     Equal to F_model_real at h=1 by construction (same start
                     points, same rollout); reported explicitly under its own
                     name because the decision logic treats it as a distinct
                     read.
  F_model_real(h) -- phi(ghat(z_t, u_t:t+h)) == phi(z_{t+h}), real start point
                     z_t, REAL action word, REAL endpoint. No FSM involved --
                     isolates rollout drift from extraction error.
  F_FSM_model(h)  -- M(u) == phi(ghat(z0,u)) on held-out RANDOM words (reset
                     symbol + random real-alphabet letters) -- the machine
                     vs. the function it was actually fit to. Isolates
                     extraction error from rollout-vs-real error.
  F_FSM_real(h)   -- M(u) == phi(z_real_endpoint), the ORIGINAL end-to-end
                     quantity (Lever-0's own comparison), now sliced by
                     horizon instead of averaged over a whole episode.
  P_start_persist(h) -- phi(z_t) == phi(z_{t+h}), copy the INITIAL label
                     forward h steps. The matched-information baseline for
                     an open-loop h-step predictor (F_model_real, F_FSM_real).
  P_closed_loop_persist -- the ORIGINAL persistence baseline (copy the
                     immediately-preceding REAL label; a 1-step quantity,
                     reported as a constant reference line, matched to
                     F_FSM_real per the pre-existing convention).

Bootstrap CIs at the EPISODE level (resample episodes with replacement,
recompute the per-horizon mean each time) -- consecutive within-episode
transitions are strongly dependent.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.walk import choose_reset, reset_reference
from corpus.latent_corpus import Corpus, build_corpus, test_ids
from e0_degeneracy.baselines import switch_persistence_majority
from learning.extract import extract
from oracle.droid_actions import load_actions_for_episodes
from oracle.latent_oracle import LatentOracle
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM
from oracle.production_oracle import (
    _step_fn_from_convention, alphabet_symbol_names, build_reset_windows,
    load_alphabet_symbols, reset_symbol_names,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
M_VALUES = (2, 4, 6, 8)
HORIZONS = [1, 2, 4, 8, 12, 16, 21]
H_MAX = 21
N_START_POINTS = 355          # one per heldout episode where possible
N_FSM_MODEL_WORDS = 150       # held-out random words for F_FSM_model
N_BOOTSTRAP = 500


def label8(z: np.ndarray, U8: np.ndarray, B8: np.ndarray) -> np.ndarray:
    return ((z @ U8.T) > B8).astype(np.int8)


def bootstrap_ci_episode(per_episode_vals: np.ndarray, seed: int, n_boot: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """per_episode_vals: (n_ep,) one scalar per episode (already averaged
    within-episode). Returns (mean, ci_lo, ci_hi) resampling EPISODES."""
    rng = np.random.default_rng(seed)
    n = len(per_episode_vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = per_episode_vals[idx].mean()
    return float(per_episode_vals.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def real_window_state(heldout, e_lookup, actions, eid, ctx_start, pipeline, model):
    import torch
    from oracle.lewm_g import DEVICE, LeWMWindowState
    n_pos = heldout.n_positions
    raw_full = actions[eid][:n_pos * FRAMESKIP].reshape(n_pos, FRAMESKIP, RAW_ACTION_DIM)
    raw_ctx = raw_full[ctx_start:ctx_start + HISTORY_SIZE]
    act_norm = pipeline(raw_ctx.reshape(-1, RAW_ACTION_DIM))
    act_flat = act_norm.reshape(1, HISTORY_SIZE, FRAMESKIP * RAW_ACTION_DIM)
    a = torch.from_numpy(act_flat).float().to(DEVICE)
    with torch.no_grad():
        act_emb = model.action_encoder(a)[0].cpu()
    emb = torch.from_numpy(heldout.latents[e_lookup[eid], ctx_start:ctx_start + HISTORY_SIZE]).float()
    return LeWMWindowState(emb=emb, act_emb=act_emb)


def machine_output(machine, word: list[str]) -> tuple:
    machine.reset_to_initial()
    out = None
    for s in word:
        out = machine.step(s)
    return out


def main():
    rng = np.random.default_rng(SEED)
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    n_pos = heldout.n_positions
    e_lookup = {int(heldout.episode_ids[i]): i for i in range(heldout.n_episodes)}

    tp = np.load(OUT_DIR / "tessellation_params_residualized.npz")
    U8, B8 = tp["U_m8"], tp["B_m8"]     # FROZEN -- Lever 0, no refitting

    fit_ids = sorted(rng.choice(heldout.episode_ids, size=min(200, len(heldout.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    pipeline = ActionPipeline.fit(np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0))
    reset_windows = build_reset_windows(pipeline)
    reset_names, reset_Z = reset_reference(reset_windows)
    symbols_dict = load_alphabet_symbols(ALPHABET_PATH)
    action_names = alphabet_symbol_names(ALPHABET_PATH)
    reset_names_all = reset_symbol_names()
    full_alphabet = reset_names_all + action_names
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(w, s):
        return step_fn_raw(w, symbols_dict[s])

    from oracle.lewm_g import get_model
    model = get_model()

    # ---------------- extract one FSM per m (frozen residualized tessellation, SepSeq) ----------------
    print("=== extracting FSMs (frozen Lever-0 hyperplanes, separation_rule=SepSeq) ===", flush=True)
    machines = {}
    MACHINES_DIR = OUT_DIR / "a0_horizon_machines"
    MACHINES_DIR.mkdir(exist_ok=True)
    for m in M_VALUES:
        def label_fn(w, m=m):
            z = w.emb[-1].numpy()
            return tuple(int((float(z @ U8[i]) > float(B8[i]))) for i in range(m))
        subset = [f"h{i}" for i in range(m)]
        oracle = LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=reset_windows)
        res = extract(subset, oracle, full_alphabet, max_learning_rounds=10, max_states_hard=2000,
                        separation_rule="SepSeq")
        print(f"  m={m}: status={res.status} n_states={res.n_states}", flush=True)
        machines[m] = res.machine
        with open(MACHINES_DIR / f"m{m}.pkl", "wb") as f:
            pickle.dump(res.machine, f)

    # ================= A / E / D(start-persist): real start points, real actions, real endpoints =================
    print("\n=== A/E/D: rolling real trajectories with REAL actions from REAL start points ===", flush=True)
    n_raw_needed = n_pos * FRAMESKIP
    all_eids = heldout.episode_ids.tolist()
    actions = load_actions_for_episodes(all_eids, max_frames=n_raw_needed)

    # per-episode: pred_labels8[h_idx] (8,), real_labels8[h_idx] (8,), start_label8 (8,)
    pred_by_h = {h: [] for h in HORIZONS}
    real_by_h = {h: [] for h in HORIZONS}
    start_label_list = []
    ep_used = []

    pos_lo, pos_hi = HISTORY_SIZE - 1, n_pos - 1 - H_MAX
    n_done = 0
    for eid in all_eids:
        eid = int(eid)
        if eid not in actions or actions[eid].shape[0] < n_raw_needed:
            continue
        if pos_hi < pos_lo:
            continue
        t = int(rng.integers(pos_lo, pos_hi + 1))
        names = heldout.symbol_seq(e_lookup[eid])
        if len(names) < t + H_MAX:
            continue
        word = names[t:t + H_MAX]

        state = real_window_state(heldout, e_lookup, actions, eid, t - HISTORY_SIZE + 1, pipeline, model)
        z_start = state.emb[-1].numpy()
        start_label = label8(z_start, U8, B8)

        real_endpoints = {h: heldout.latents[e_lookup[eid], t + h] for h in HORIZONS}

        w = state
        for step_idx, sym in enumerate(word, start=1):
            w = step_fn(w, sym)
            if step_idx in pred_by_h:
                z_pred = w.emb[-1].numpy()
                pred_by_h[step_idx].append(label8(z_pred, U8, B8))
                real_by_h[step_idx].append(label8(real_endpoints[step_idx], U8, B8))
        start_label_list.append(start_label)
        ep_used.append(eid)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{N_START_POINTS} start points rolled", flush=True)
        if n_done >= N_START_POINTS:
            break
    print(f"  total start points used: {n_done}", flush=True)

    start_label_arr = np.array(start_label_list)   # (n_ep, 8)

    # ================= B: F_FSM_model on held-out random words (reset + real-alphabet letters) =================
    print("\n=== B: F_FSM_model on held-out random words ===", flush=True)
    fsm_model_pred_by_h = {h: [] for h in HORIZONS}   # phi(ghat(z0,u)) at h, per word -- (8,)
    fsm_model_machine_by_h = {m: {h: [] for h in HORIZONS} for m in M_VALUES}   # M(u) per m, per h, per word
    import random as _random
    _random.seed(SEED + 999)
    for wi in range(N_FSM_MODEL_WORDS):
        reset_sym = _random.choice(reset_names_all)
        letters = [_random.choice(action_names) for _ in range(H_MAX)]

        w = reset_windows[reset_sym]
        for step_idx, sym in enumerate(letters, start=1):
            w = step_fn(w, sym)
            if step_idx in fsm_model_pred_by_h:
                z_pred = w.emb[-1].numpy()
                fsm_model_pred_by_h[step_idx].append(label8(z_pred, U8, B8))

        for m in M_VALUES:
            machine = machines[m]
            machine.reset_to_initial()
            machine.step(reset_sym)
            for step_idx, sym in enumerate(letters, start=1):
                out = machine.step(sym)
                if step_idx in fsm_model_machine_by_h[m]:
                    fsm_model_machine_by_h[m][step_idx].append(out)
        if (wi + 1) % 50 == 0:
            print(f"  {wi + 1}/{N_FSM_MODEL_WORDS} words done", flush=True)

    # ================= C / D(closed-loop persist): FSM vs REAL trajectory, reset-aligned, per episode =================
    print("\n=== C: F_FSM_real + closed-loop persistence, reset-aligned per episode ===", flush=True)

    def per_episode_full(machine, m, corpus: Corpus) -> tuple[list, list]:
        """Returns (outs_per_ep, reals_per_ep): lists over episodes of
        (n_steps, m) int arrays -- machine output bits vs real label bits,
        position-aligned, starting right after the reset."""
        outs_all, reals_all = [], []
        for i in range(corpus.latents.shape[0]):
            Z = corpus.latents[i]
            names = corpus.symbol_seq(i)
            reset_sym = choose_reset(Z[HISTORY_SIZE - 1], reset_names, reset_Z)
            machine.reset_to_initial()
            outs = [machine.step(reset_sym)]
            for t in range(min(len(names), Z.shape[0] - 1)):
                outs.append(machine.step(names[t]))
            n_cmp = min(len(outs) - 1, Z.shape[0] - HISTORY_SIZE)
            if n_cmp < 1:
                continue
            out_arr = np.array([outs[t + 1][:m] for t in range(n_cmp)], dtype=np.int8)
            real_arr = np.array([label8(Z[HISTORY_SIZE + t], U8, B8)[:m] for t in range(n_cmp)], dtype=np.int8)
            outs_all.append(out_arr)
            reals_all.append(real_arr)
        return outs_all, reals_all

    fsm_real_by_m = {}
    closed_persist_by_m = {}
    for m in M_VALUES:
        outs_all, reals_all = per_episode_full(machines[m], m, heldout)
        fsm_real_by_m[m] = (outs_all, reals_all)
        print(f"  m={m}: {len(outs_all)} episodes with a reset-aligned real comparison", flush=True)

    # ============================================================================================
    print("\n=== aggregating everything ===", flush=True)
    report = {"horizons": HORIZONS, "m_values": list(M_VALUES), "config": {
        "n_start_points": n_done, "n_fsm_model_words": N_FSM_MODEL_WORDS, "n_bootstrap": N_BOOTSTRAP,
        "tessellation_params": "tessellation_params_residualized.npz (Lever 0, frozen)",
    }}
    machine_states = {str(m): machines[m].size for m in M_VALUES}
    report["machine_n_states"] = machine_states

    for m in M_VALUES:
        m_report = {"n_states": machine_states[str(m)], "per_bit": {}, "joint": {}, "closed_loop_persistence": {}}

        # --- F_model_real, P_start_persist (A, D) -- per bit + joint, per horizon ---
        for h in HORIZONS:
            pred_h = np.array(pred_by_h[h])[:, :m]   # (n_ep, m)
            real_h = np.array(real_by_h[h])[:, :m]
            start_h = start_label_arr[:, :m]

            model_real_match = (pred_h == real_h)                  # (n_ep, m)
            start_persist_match = (start_h == real_h)               # (n_ep, m)
            model_real_joint = model_real_match.all(axis=1).astype(float)
            start_persist_joint = start_persist_match.all(axis=1).astype(float)

            for bi in range(m):
                key = f"cell_h{bi}"
                m_report["per_bit"].setdefault(key, {}).setdefault(str(h), {})
                mean_mr, lo_mr, hi_mr = bootstrap_ci_episode(model_real_match[:, bi].astype(float), SEED + h * 7 + bi)
                mean_sp, lo_sp, hi_sp = bootstrap_ci_episode(start_persist_match[:, bi].astype(float), SEED + h * 11 + bi)
                m_report["per_bit"][key][str(h)]["F_model_real"] = {"mean": mean_mr, "ci95": [lo_mr, hi_mr]}
                m_report["per_bit"][key][str(h)]["P_start_persist"] = {"mean": mean_sp, "ci95": [lo_sp, hi_sp]}
                m_report["per_bit"][key][str(h)]["margin_model_real_vs_start_persist"] = mean_mr - mean_sp
                if h == 1:
                    m_report["per_bit"][key]["F_TF"] = {"mean": mean_mr, "ci95": [lo_mr, hi_mr],
                                                            "note": "equals F_model_real at h=1"}

            mean_mrj, lo_mrj, hi_mrj = bootstrap_ci_episode(model_real_joint, SEED + h * 13)
            mean_spj, lo_spj, hi_spj = bootstrap_ci_episode(start_persist_joint, SEED + h * 17)
            m_report["joint"].setdefault(str(h), {})
            m_report["joint"][str(h)]["F_model_real"] = {"mean": mean_mrj, "ci95": [lo_mrj, hi_mrj]}
            m_report["joint"][str(h)]["P_start_persist"] = {"mean": mean_spj, "ci95": [lo_spj, hi_spj]}
            m_report["joint"][str(h)]["margin_model_real_vs_start_persist"] = mean_mrj - mean_spj
            if h == 1:
                m_report["joint"]["F_TF"] = {"mean": mean_mrj, "ci95": [lo_mrj, hi_mrj]}

        # --- F_FSM_model (B) -- per bit + joint, per horizon ---
        for h in HORIZONS:
            model_pred_h = np.array(fsm_model_pred_by_h[h])[:, :m]         # (n_words, m)
            fsm_out_h = np.array(fsm_model_machine_by_h[m][h])[:, :m]       # (n_words, m)
            match = (model_pred_h == fsm_out_h)
            joint = match.all(axis=1).astype(float)
            for bi in range(m):
                key = f"cell_h{bi}"
                mean_b, lo_b, hi_b = bootstrap_ci_episode(match[:, bi].astype(float), SEED + h * 19 + bi)
                m_report["per_bit"][key][str(h)]["F_FSM_model"] = {"mean": mean_b, "ci95": [lo_b, hi_b]}
            mean_j, lo_j, hi_j = bootstrap_ci_episode(joint, SEED + h * 23)
            m_report["joint"][str(h)]["F_FSM_model"] = {"mean": mean_j, "ci95": [lo_j, hi_j]}

        # --- F_FSM_real (C) + closed-loop persistence (D) -- per bit + joint, per horizon ---
        outs_all, reals_all = fsm_real_by_m[m]
        n_ep_c = len(outs_all)
        for h in HORIZONS:
            per_ep_match = np.full((n_ep_c, m), np.nan)
            per_ep_joint = np.full(n_ep_c, np.nan)
            for i, (out_arr, real_arr) in enumerate(zip(outs_all, reals_all)):
                if out_arr.shape[0] < h:
                    continue
                per_ep_match[i] = (out_arr[h - 1] == real_arr[h - 1]).astype(float)
                per_ep_joint[i] = float(np.all(out_arr[h - 1] == real_arr[h - 1]))
            valid = ~np.isnan(per_ep_joint)
            for bi in range(m):
                key = f"cell_h{bi}"
                vb = ~np.isnan(per_ep_match[:, bi])
                mean_c, lo_c, hi_c = bootstrap_ci_episode(per_ep_match[vb, bi], SEED + h * 29 + bi)
                m_report["per_bit"][key][str(h)]["F_FSM_real"] = {"mean": mean_c, "ci95": [lo_c, hi_c], "n_episodes": int(vb.sum())}
            mean_cj, lo_cj, hi_cj = bootstrap_ci_episode(per_ep_joint[valid], SEED + h * 31)
            m_report["joint"][str(h)]["F_FSM_real"] = {"mean": mean_cj, "ci95": [lo_cj, hi_cj], "n_episodes": int(valid.sum())}

        # closed-loop persistence -- constant, 1-step, matched to F_FSM_real per the original convention
        for bi in range(m):
            key = f"cell_h{bi}"
            seqs = [np.array([r[bi] for r in real_arr]) for real_arr in reals_all if len(real_arr) > 1]
            modal_len = max(set(len(s) for s in seqs), key=[len(s) for s in seqs].count) if seqs else 0
            labels = np.stack([s for s in seqs if len(s) == modal_len]) if seqs else np.zeros((0, 0))
            spm = switch_persistence_majority(labels) if labels.size else {"switch_rate": None, "persistence_baseline": None,
                                                                               "majority_baseline": None, "p_label_1": None, "n_observations": 0}
            m_report["closed_loop_persistence"][key] = spm
        m_report["closed_loop_persistence"]["joint_note"] = ("closed-loop persistence is reported per-bit only, "
                                                                 "as in the original convention; a joint version "
                                                                 "was never defined in prior entries")

        report[str(m)] = m_report
        print(f"  m={m} aggregated.", flush=True)

    (OUT_DIR / "a0_horizon_decomposition_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote artifacts/a0_horizon_decomposition_report.json", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
