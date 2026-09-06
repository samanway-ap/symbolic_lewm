"""A0.1b -- same-sample link decomposition + re-grounding + continuous h=1
diagnostics, per user corrections to A0.1 (2026-09-02). Everything frozen:
same Lever-0 tessellation, same PICKLED machines from A0.1 (no re-extraction,
no refitting, no training).

A0.1's F_model_real and F_FSM_real were computed on DIFFERENT populations
(arbitrary real start points vs. episode-beginning reset-aligned walks), so
their gap was consistent with, but did not DEMONSTRATE, reset-substitution
error. This fixes that: for the SAME (episode, t) samples, roll BOTH the true
window z_t and the canonical-reset substitute r_t = choose_reset(z_t) forward
through the SAME real action word, and step the SAME frozen machine on r_t
with that REAL word (not a random one) -- separating:

  A_h: phi(ghat(z_t,u))  vs phi(z_t+h)        true-start model error
  B_h: phi(ghat(r_t,u))  vs phi(z_t+h)        reset-substituted model error
  C_h: phi(ghat(r_t,u))  vs phi(ghat(z_t,u))  reset-substitution's effect on
                                                 the MODEL alone (no ground truth)
  D_h: M(r_t,u)          vs phi(ghat(r_t,u))  extraction fidelity on the REAL
                                                 word distribution (not random)
  E_h: M(r_t,u)          vs phi(z_t+h)        full end-to-end, same sample as A-D

Plus h=0: P[phi(r_t)=phi(z_t)] and continuous ||r_t - z_t||, both ambient and
per tessellation coordinate -- does reset substitution already fail before
any rollout at all.

Plus the omitted re-grounding sweep (r in {1,2,4,8,inf}; inf reuses the A_h
rollout, already computed) and continuous h=1 diagnostics (correlation/MAE/
RMSE of the raw score, boundary-distance-stratified accuracy, switch-event
recall/false-switch rate/balanced accuracy/confusion matrix).

Bootstrap: episode-level for everything tied to real episodes (A-E,
re-grounding). The OLD F_FSM_model (random reset + random words, no episode
association) is NOT recomputed here -- it is read back from A0.1's own report
and re-labelled as word-level-bootstrap, not episode-level, per correction.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a0_label_repair.horizon_decomposition import real_window_state
from atlas.walk import choose_reset, reset_reference
from corpus.latent_corpus import build_corpus, test_ids
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, get_model
from oracle.production_oracle import _step_fn_from_convention, build_reset_windows, load_alphabet_symbols

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
MACHINES_DIR = OUT_DIR / "a0_horizon_machines"
SEED = 3072
M_VALUES = (2, 4, 6, 8)
HORIZONS = [1, 2, 4, 8, 12, 16, 21]
H_MAX = 21
N_STARTS_PER_EPISODE = 5
REGROUND_RS = [1, 2, 4, 8]     # inf reuses the A_h (z_t) rollout
N_BOOTSTRAP = 500


def label8(z, U8, B8):
    return ((z @ U8.T) > B8).astype(np.int8)


def score8(z, U8, B8):
    return z @ U8.T - B8


_KEY_SEED_OFFSET = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}


def bootstrap_ci(per_group_vals: np.ndarray, seed: int, n_boot: int = N_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    n = len(per_group_vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = per_group_vals[idx].mean()
    return float(per_group_vals.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    rng = np.random.default_rng(SEED)
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    n_pos = heldout.n_positions
    e_lookup = {int(heldout.episode_ids[i]): i for i in range(heldout.n_episodes)}

    tp = np.load(OUT_DIR / "tessellation_params_residualized.npz")
    U8, B8 = tp["U_m8"], tp["B_m8"]

    fit_ids = sorted(rng.choice(heldout.episode_ids, size=min(200, len(heldout.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    pipeline = ActionPipeline.fit(np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0))
    reset_windows = build_reset_windows(pipeline)
    reset_names, reset_Z = reset_reference(reset_windows)
    symbols_dict = load_alphabet_symbols(ALPHABET_PATH)
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(w, s):
        return step_fn_raw(w, symbols_dict[s])

    model = get_model()

    print("=== loading FROZEN machines from A0.1 (no re-extraction) ===", flush=True)
    machines = {}
    for m in M_VALUES:
        with open(MACHINES_DIR / f"m{m}.pkl", "rb") as f:
            machines[m] = pickle.load(f)
        print(f"  m={m}: n_states={machines[m].size}", flush=True)

    n_raw_needed = n_pos * FRAMESKIP
    all_eids = heldout.episode_ids.tolist()
    actions = load_actions_for_episodes(all_eids, max_frames=n_raw_needed)
    pos_lo, pos_hi = HISTORY_SIZE - 1, n_pos - 1 - H_MAX

    # ================= build the shared start-point pool =================
    print(f"\n=== building start-point pool ({N_STARTS_PER_EPISODE}/episode) ===", flush=True)
    starts = []   # list of dicts: eid, t, word (list[str]), z_t, r_name, d_r_z
    for eid in all_eids:
        eid = int(eid)
        if eid not in actions or actions[eid].shape[0] < n_raw_needed:
            continue
        if pos_hi < pos_lo:
            continue
        names = heldout.symbol_seq(e_lookup[eid])
        chosen_ts = rng.choice(range(pos_lo, pos_hi + 1), size=min(N_STARTS_PER_EPISODE, pos_hi - pos_lo + 1), replace=False)
        for t in chosen_ts:
            t = int(t)
            if len(names) < t + H_MAX:
                continue
            z_t = heldout.latents[e_lookup[eid], t]
            r_name = choose_reset(z_t, reset_names, reset_Z)
            d_r_z = float(np.linalg.norm(reset_Z[reset_names.index(r_name)] - z_t))
            starts.append({"eid": eid, "t": t, "word": names[t:t + H_MAX], "z_t": z_t, "r_name": r_name, "d_r_z": d_r_z})
    print(f"  total start points: {len(starts)} across {len(set(s['eid'] for s in starts))} episodes", flush=True)

    # ================= A/B/C/D/E link decomposition =================
    print("\n=== rolling z_t and r_t through the SAME real action word ===", flush=True)
    recs = {h: [] for h in HORIZONS}   # each entry: dict with eid, all quantities at this h
    h0_recs = []
    for si, s in enumerate(starts):
        eid, t, word, z_t, r_name = s["eid"], s["t"], s["word"], s["z_t"], s["r_name"]
        ctx_start = t - HISTORY_SIZE + 1
        state_true = real_window_state(heldout, e_lookup, actions, eid, ctx_start, pipeline, model)
        state_reset = reset_windows[r_name]

        z_t_used = state_true.emb[-1].numpy()   # sanity: should equal z_t up to float noise
        h0_recs.append({"eid": eid, "label_match": bool(np.array_equal(label8(reset_Z[reset_names.index(r_name)], U8, B8),
                                                                            label8(z_t_used, U8, B8))),
                          "d_r_z": s["d_r_z"], "score_gap": (score8(reset_Z[reset_names.index(r_name)], U8, B8)
                                                                - score8(z_t_used, U8, B8)).tolist()})

        w_true, w_reset = state_true, state_reset
        machine_states = {m: None for m in M_VALUES}
        for m in M_VALUES:
            machines[m].reset_to_initial()
            machines[m].step(r_name)

        real_endpoints = {h: heldout.latents[e_lookup[eid], t + h] for h in HORIZONS}

        for step_idx, sym in enumerate(word, start=1):
            w_true = step_fn(w_true, sym)
            w_reset = step_fn(w_reset, sym)
            machine_out = {m: machines[m].step(sym) for m in M_VALUES}
            if step_idx in recs:
                z_pred_true = w_true.emb[-1].numpy()
                z_pred_reset = w_reset.emb[-1].numpy()
                z_real = real_endpoints[step_idx]
                recs[step_idx].append({
                    "eid": eid,
                    "label_true": label8(z_pred_true, U8, B8), "label_reset": label8(z_pred_reset, U8, B8),
                    "label_real": label8(z_real, U8, B8),
                    "score_true": score8(z_pred_true, U8, B8), "score_real": score8(z_real, U8, B8),
                    "machine_out": machine_out,
                    "label_start": label8(z_t_used, U8, B8),
                })
        if (si + 1) % 300 == 0:
            print(f"  {si + 1}/{len(starts)} start points rolled", flush=True)
    print(f"  total: {len(starts)} start points rolled", flush=True)

    # ================= re-grounding sweep =================
    print(f"\n=== re-grounding sweep, r in {REGROUND_RS} (+inf reuses A_h) ===", flush=True)
    reground_recs = {r: {h: [] for h in HORIZONS} for r in REGROUND_RS}
    for si, s in enumerate(starts):
        eid, t, word = s["eid"], s["t"], s["word"]
        ctx_start = t - HISTORY_SIZE + 1
        real_endpoints = {h: heldout.latents[e_lookup[eid], t + h] for h in HORIZONS}
        for r in REGROUND_RS:
            w = real_window_state(heldout, e_lookup, actions, eid, ctx_start, pipeline, model)
            for step_idx, sym in enumerate(word, start=1):
                w = step_fn(w, sym)
                if step_idx % r == 0:
                    new_ctx_start = (t + step_idx) - HISTORY_SIZE + 1
                    w = real_window_state(heldout, e_lookup, actions, eid, new_ctx_start, pipeline, model)
                if step_idx in reground_recs[r]:
                    z_pred = w.emb[-1].numpy()
                    z_real = real_endpoints[step_idx]
                    reground_recs[r][step_idx].append({
                        "eid": eid, "match": label8(z_pred, U8, B8) == label8(z_real, U8, B8),
                    })
        if (si + 1) % 300 == 0:
            print(f"  {si + 1}/{len(starts)} start points re-grounded", flush=True)

    # ================= aggregate: A-E per m, per bit + joint =================
    print("\n=== aggregating A-E (link decomposition) ===", flush=True)

    def agg_by_episode(vals_by_eid: dict) -> np.ndarray:
        return np.array([np.mean(v) for v in vals_by_eid.values()])

    report = {"horizons": HORIZONS, "m_values": list(M_VALUES), "n_start_points": len(starts),
                "n_episodes": len(set(s["eid"] for s in starts)), "reground_rs": REGROUND_RS + ["inf"]}

    # h=0
    from collections import defaultdict
    h0_by_eid_joint = defaultdict(list)
    h0_by_eid_perbit = {bi: defaultdict(list) for bi in range(8)}
    d_r_z_all = []
    score_gap_all = []
    for r in h0_recs:
        h0_by_eid_joint[r["eid"]].append(float(r["label_match"]))
        d_r_z_all.append(r["d_r_z"])
        score_gap_all.append(r["score_gap"])
    mean_h0, lo_h0, hi_h0 = bootstrap_ci(agg_by_episode(h0_by_eid_joint), SEED + 1)
    score_gap_arr = np.array(score_gap_all)
    report["h0"] = {
        "P_reset_matches_start_joint8": {"mean": mean_h0, "ci95": [lo_h0, hi_h0]},
        "mean_d_r_z_ambient": float(np.mean(d_r_z_all)), "std_d_r_z_ambient": float(np.std(d_r_z_all)),
        "mean_abs_score_gap_per_bit": score_gap_arr.__abs__().mean(axis=0).tolist(),
        "n": len(h0_recs),
    }
    print(f"  h=0: P[phi(r_t)=phi(z_t)] joint(8 bits)={mean_h0:.3f} ci95=[{lo_h0:.3f},{hi_h0:.3f}]  "
          f"mean||r_t-z_t||={np.mean(d_r_z_all):.3f}", flush=True)

    for m in M_VALUES:
        m_report = {"n_states": machines[m].size, "per_bit": {}, "joint": {}}
        for h in HORIZONS:
            data = recs[h]
            by_eid = {"A": defaultdict(list), "B": defaultdict(list), "C": defaultdict(list),
                        "D": defaultdict(list), "E": defaultdict(list)}
            by_eid_bits = {k: {bi: defaultdict(list) for bi in range(m)} for k in by_eid}
            for r in data:
                eid = r["eid"]
                lt, lr, lreal = r["label_true"][:m], r["label_reset"][:m], r["label_real"][:m]
                mo = np.array(r["machine_out"][m][:m])
                match_A = (lt == lreal); match_B = (lr == lreal); match_C = (lr == lt)
                match_D = (mo == lr); match_E = (mo == lreal)
                by_eid["A"][eid].append(float(match_A.all())); by_eid["B"][eid].append(float(match_B.all()))
                by_eid["C"][eid].append(float(match_C.all())); by_eid["D"][eid].append(float(match_D.all()))
                by_eid["E"][eid].append(float(match_E.all()))
                for bi in range(m):
                    by_eid_bits["A"][bi][eid].append(float(match_A[bi])); by_eid_bits["B"][bi][eid].append(float(match_B[bi]))
                    by_eid_bits["C"][bi][eid].append(float(match_C[bi])); by_eid_bits["D"][bi][eid].append(float(match_D[bi]))
                    by_eid_bits["E"][bi][eid].append(float(match_E[bi]))

            m_report["joint"][str(h)] = {}
            for k in ("A", "B", "C", "D", "E"):
                mean_v, lo_v, hi_v = bootstrap_ci(agg_by_episode(by_eid[k]), SEED + h * 3 + _KEY_SEED_OFFSET[k])
                m_report["joint"][str(h)][k] = {"mean": mean_v, "ci95": [lo_v, hi_v]}
            for bi in range(m):
                key = f"cell_h{bi}"
                m_report["per_bit"].setdefault(key, {})[str(h)] = {}
                for k in ("A", "B", "C", "D", "E"):
                    mean_v, lo_v, hi_v = bootstrap_ci(agg_by_episode(by_eid_bits[k][bi]), SEED + h * 5 + bi * 7 + _KEY_SEED_OFFSET[k])
                    m_report["per_bit"][key][str(h)][k] = {"mean": mean_v, "ci95": [lo_v, hi_v]}
        report[str(m)] = m_report
        print(f"  m={m} A-E aggregated.", flush=True)

    # ================= aggregate: re-grounding =================
    print("\n=== aggregating re-grounding sweep ===", flush=True)
    report["regrounding"] = {}
    for r in REGROUND_RS:
        report["regrounding"][str(r)] = {}
        for h in HORIZONS:
            by_eid = defaultdict(list)
            for rec in reground_recs[r][h]:
                by_eid[rec["eid"]].append(float(rec["match"].all()))
            mean_v, lo_v, hi_v = bootstrap_ci(agg_by_episode(by_eid), SEED + h * 11 + r * 13)
            report["regrounding"][str(r)][str(h)] = {"mean": mean_v, "ci95": [lo_v, hi_v]}
        print(f"  r={r}: " + ", ".join(f"h{h}={report['regrounding'][str(r)][str(h)]['mean']:.3f}" for h in HORIZONS),
              flush=True)

    # ================= continuous h=1 diagnostics (per bit, m=8 superset) =================
    print("\n=== continuous h=1 diagnostics ===", flush=True)
    data1 = recs[1]
    cont = {}
    for bi in range(8):
        s_pred = np.array([r["score_true"][bi] for r in data1])
        s_real = np.array([r["score_real"][bi] for r in data1])
        start_bit = np.array([r["label_start"][bi] for r in data1])
        real_bit = np.array([1 if r["label_real"][bi] else 0 for r in data1])
        pred_bit = np.array([1 if r["label_true"][bi] else 0 for r in data1])
        eids = np.array([r["eid"] for r in data1])

        corr = float(np.corrcoef(s_pred, s_real)[0, 1])
        mae = float(np.mean(np.abs(s_pred - s_real)))
        rmse = float(np.sqrt(np.mean((s_pred - s_real) ** 2)))

        dist_from_boundary = np.abs(s_real)
        bins = np.quantile(dist_from_boundary, [0, 0.25, 0.5, 0.75, 1.0])
        strat = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (dist_from_boundary >= lo) & (dist_from_boundary <= hi)
            acc = float((pred_bit[mask] == real_bit[mask]).mean()) if mask.sum() else None
            strat.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()), "accuracy": acc})

        actual_switch = (real_bit != start_bit)
        predicted_switch = (pred_bit != start_bit)
        n_switch = int(actual_switch.sum())
        n_nonswitch = int((~actual_switch).sum())
        n_ep_switch = len(set(eids[actual_switch].tolist()))
        tp = int((actual_switch & predicted_switch).sum())
        fn = int((actual_switch & ~predicted_switch).sum())
        fp = int((~actual_switch & predicted_switch).sum())
        tn = int((~actual_switch & ~predicted_switch).sum())
        recall = tp / n_switch if n_switch else None
        false_switch_rate = fp / n_nonswitch if n_nonswitch else None
        specificity = tn / n_nonswitch if n_nonswitch else None
        balanced_acc = (recall + specificity) / 2 if (recall is not None and specificity is not None) else None

        cont[f"cell_h{bi}"] = {
            "correlation": corr, "mae": mae, "rmse": rmse,
            "accuracy_by_boundary_distance_quartile": strat,
            "n_switch_events": n_switch, "n_nonswitch_events": n_nonswitch,
            "n_episodes_with_switch": n_ep_switch,
            "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "recall_given_switch": recall, "false_switch_rate_given_nonswitch": false_switch_rate,
            "specificity": specificity, "balanced_accuracy": balanced_acc,
            "underpowered_switch_analysis": n_switch < 30,
        }
        print(f"  {f'cell_h{bi}'}: corr={corr:.3f} mae={mae:.3f} rmse={rmse:.3f} "
              f"n_switch={n_switch} (underpowered={n_switch < 30}) recall={recall} fsr={false_switch_rate}",
              flush=True)
    report["continuous_h1_diagnostics"] = cont

    (OUT_DIR / "a0_horizon_decomposition_b_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote artifacts/a0_horizon_decomposition_b_report.json", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
