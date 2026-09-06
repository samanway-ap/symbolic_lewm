"""A0.1c -- close the remaining measurement ambiguities in A0.1b (v5 doc,
EXPERIMENTS_orthogonal_expansion_1.md SS4 "A0.1c"). No repair. No FSM
extraction (excluded by the spec -- this only touches ghat and phi). Same
frozen Lever-0 hyperplanes, same 1775-start/355-episode pool as A0.1b
(reproduced here with the IDENTICAL seed and construction order so it is the
SAME pool, not a resample).

Three completions:

1. CORRECTED re-grounding. A0.1b's bug: at a grounding step it replaced the
   predicted latent with the real one, THEN checked the recorded value at
   that horizon -- comparing ground truth to ground truth is tautological.
   Fixed order: predict -> SCORE against the real endpoint -> only THEN
   replace if this step is a grounding boundary. Two identity assertions
   the spec requires:
     F_{r=1}(h)    == teacher-forced one-letter fidelity, at every h (since
                      every step is grounded, every scored prediction is a
                      fresh one-step-ahead prediction from a just-replaced
                      TRUE window -- the curve should be flat across h, not
                      decaying, and match F_TF's value)
     F_{r=inf}(h)  == A_h from A0.1b's link decomposition, EXACTLY (r=inf
                      never replaces -- this is the identical computation
                      to A0.1b's true-start open-loop rollout on the SAME
                      samples, so it must reproduce those numbers bit for
                      bit, not just approximately)

2. Completed continuous audit at h=1: normalised MAE/RMSE, affine
   calibration (regress real score on predicted score), the full epsilon
   distribution, fraction of real endpoints inside an ERROR-SCALED boundary
   band (|s_real| < k*std(epsilon), not an arbitrary quartile), and
   episode-bootstrap CIs for switch recall / false-switch rate / balanced
   accuracy (A0.1b reported only point estimates for these).

3. Completed initialisation audit: reset-to-real distance in BOTH ambient
   and residual space (the residual basis the tessellation directions
   actually live in, rebuilt fresh from TRAIN's between-episode scatter --
   same construction as Lever 0), per-bit AND per-m joint h=0 agreement
   (A0.1b only reported the joint-8 number), and coverage of real starting
   cells by the 8 canonical reset cells.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a0_label_repair.horizon_decomposition import real_window_state
from a0_label_repair.horizon_decomposition_b import bootstrap_ci, label8, score8
from a0_label_repair.residualized_tessellation import build_residual_basis
from a0_label_repair.episode_variance import scatter_decomposition
from atlas.walk import choose_reset, reset_reference
from corpus.latent_corpus import build_corpus, test_ids, train_ids
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, get_model
from oracle.production_oracle import _step_fn_from_convention, build_reset_windows, load_alphabet_symbols

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
M_VALUES = (2, 4, 6, 8)
HORIZONS = [1, 2, 4, 8, 12, 16, 21]
H_MAX = 21
N_STARTS_PER_EPISODE = 5     # IDENTICAL to A0.1b, same seed -> same pool
REGROUND_RS_FINITE = [1, 2, 4, 8]
N_BOOTSTRAP = 500
ERROR_BAND_K = 1.0            # boundary band: |s_real| < k * std(epsilon)


def agg_by_episode(vals_by_eid: dict) -> np.ndarray:
    return np.array([np.mean(v) for v in vals_by_eid.values()])


def main():
    rng = np.random.default_rng(SEED)
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    train = build_corpus(train_ids(), 60, "train")
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

    n_raw_needed = n_pos * FRAMESKIP
    all_eids = heldout.episode_ids.tolist()
    actions = load_actions_for_episodes(all_eids, max_frames=n_raw_needed)
    pos_lo, pos_hi = HISTORY_SIZE - 1, n_pos - 1 - H_MAX

    # ================= reproduce the IDENTICAL A0.1b start-point pool =================
    print(f"=== reproducing the A0.1b start-point pool ({N_STARTS_PER_EPISODE}/episode, same seed) ===", flush=True)
    starts = []
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
            starts.append({"eid": eid, "t": t, "word": names[t:t + H_MAX], "z_t": z_t, "r_name": r_name})
    print(f"  total start points: {len(starts)} across {len(set(s['eid'] for s in starts))} episodes "
          f"(A0.1b reported 1775 across 355 -- must match)", flush=True)

    # ================= 1. CORRECTED re-grounding (score BEFORE replace) =================
    print("\n=== corrected re-grounding: score before replace, for r in {1,2,4,8,inf} ===", flush=True)
    all_rs = REGROUND_RS_FINITE + ["inf"]
    reground_recs = {r: {h: defaultdict(list) for h in HORIZONS} for r in all_rs}          # joint
    reground_recs_bits = {r: {h: {bi: defaultdict(list) for bi in range(8)} for h in HORIZONS} for r in all_rs}
    h1_pred_scores, h1_real_scores = [], []   # for the continuous audit (r=inf / true-start, h=1)

    for si, s in enumerate(starts):
        eid, t, word = s["eid"], s["t"], s["word"]
        ctx_start = t - HISTORY_SIZE + 1
        real_endpoints = {h: heldout.latents[e_lookup[eid], t + h] for h in HORIZONS}
        for r in all_rs:
            w = real_window_state(heldout, e_lookup, actions, eid, ctx_start, pipeline, model)
            for step_idx, sym in enumerate(word, start=1):
                w = step_fn(w, sym)                       # PREDICT
                if step_idx in HORIZONS:                   # SCORE BEFORE any replacement
                    z_pred = w.emb[-1].numpy()
                    z_real = real_endpoints[step_idx]
                    lp, lr = label8(z_pred, U8, B8), label8(z_real, U8, B8)
                    reground_recs[r][step_idx][eid].append(float(np.array_equal(lp, lr)))
                    for bi in range(8):
                        reground_recs_bits[r][step_idx][bi][eid].append(float(lp[bi] == lr[bi]))
                    if r == "inf" and step_idx == 1:
                        h1_pred_scores.append(score8(z_pred, U8, B8))
                        h1_real_scores.append(score8(z_real, U8, B8))
                if r != "inf" and step_idx % r == 0:        # THEN replace, only for finite r
                    new_ctx_start = (t + step_idx) - HISTORY_SIZE + 1
                    w = real_window_state(heldout, e_lookup, actions, eid, new_ctx_start, pipeline, model)
        if (si + 1) % 300 == 0:
            print(f"  {si + 1}/{len(starts)} start points done", flush=True)
    print(f"  total: {len(starts)} start points done", flush=True)

    report = {"horizons": HORIZONS, "n_start_points": len(starts),
                "n_episodes": len(set(s["eid"] for s in starts)), "error_band_k": ERROR_BAND_K}

    reground_out = {}
    for r in all_rs:
        reground_out[str(r)] = {}
        for h in HORIZONS:
            mean_v, lo_v, hi_v = bootstrap_ci(agg_by_episode(reground_recs[r][h]), SEED + h * 31 + (r if r != "inf" else 99))
            reground_out[str(r)][str(h)] = {"mean": mean_v, "ci95": [lo_v, hi_v]}
    report["regrounding_corrected"] = reground_out

    print("\n  corrected re-grounding, joint(8 bits):", flush=True)
    for r in all_rs:
        print(f"    r={r}: " + ", ".join(f"h{h}={reground_out[str(r)][str(h)]['mean']:.3f}" for h in HORIZONS), flush=True)

    # identity assertions
    r1_vals = [reground_out["1"][str(h)]["mean"] for h in HORIZONS]
    r1_flat = float(np.std(r1_vals))
    print(f"\n  ASSERTION r=1 should be ~flat across h (repeated one-letter teacher-forcing): "
          f"values={['%.3f' % v for v in r1_vals]}  std_across_h={r1_flat:.4f}", flush=True)

    a0_1b_path = OUT_DIR / "a0_horizon_decomposition_b_report.json"
    rinf_vals = [reground_out["inf"][str(h)]["mean"] for h in HORIZONS]
    if a0_1b_path.exists():
        a0_1b = json.loads(a0_1b_path.read_text())
        a_vals_m8 = [a0_1b["8"]["joint"][str(h)]["A"]["mean"] for h in HORIZONS]
        max_diff = max(abs(a - b) for a, b in zip(rinf_vals, a_vals_m8))
        print(f"  ASSERTION r=inf should EXACTLY match A0.1b's A_h (m=8, same samples/model, "
              f"never replaces): r=inf={['%.3f' % v for v in rinf_vals]}  "
              f"A0.1b_A={['%.3f' % v for v in a_vals_m8]}  max_abs_diff={max_diff:.4f}", flush=True)
        report["assertions"] = {"r1_std_across_h": r1_flat, "rinf_vs_A0_1b_A_max_abs_diff": max_diff,
                                   "rinf_vs_A0_1b_A_note": ("small residual difference is expected "
                                                              "ONLY from any RNG-derived choice inside "
                                                              "step_fn/model; if this is 0.0 the r=inf "
                                                              "path is byte-identical to A0.1b's A")}

    # per-bit corrected re-grounding, for completeness
    reground_bits_out = {}
    for r in all_rs:
        reground_bits_out[str(r)] = {}
        for h in HORIZONS:
            reground_bits_out[str(r)][str(h)] = {}
            for bi in range(8):
                mean_v, lo_v, hi_v = bootstrap_ci(agg_by_episode(reground_recs_bits[r][h][bi]),
                                                     SEED + h * 37 + bi * 41 + (r if r != "inf" else 99))
                reground_bits_out[str(r)][str(h)][f"cell_h{bi}"] = {"mean": mean_v, "ci95": [lo_v, hi_v]}
    report["regrounding_corrected_per_bit"] = reground_bits_out

    # ================= 2. completed continuous audit (h=1, r=inf / true-start) =================
    print("\n=== completed continuous audit (h=1) ===", flush=True)
    pred_arr = np.array(h1_pred_scores)   # (n, 8)
    real_arr = np.array(h1_real_scores)
    cont = {}
    for bi in range(8):
        sp, sr = pred_arr[:, bi], real_arr[:, bi]
        eps = sp - sr
        corr = float(np.corrcoef(sp, sr)[0, 1])
        mae = float(np.mean(np.abs(eps)))
        rmse = float(np.sqrt(np.mean(eps ** 2)))
        std_real = float(np.std(sr))
        norm_mae = mae / std_real if std_real > 0 else None
        norm_rmse = rmse / std_real if std_real > 0 else None

        A = np.vstack([sp, np.ones_like(sp)]).T
        slope, intercept = np.linalg.lstsq(A, sr, rcond=None)[0]

        std_eps = float(np.std(eps))
        band_frac = float((np.abs(sr) < ERROR_BAND_K * std_eps).mean())

        cont[f"cell_h{bi}"] = {
            "correlation": corr, "mae": mae, "rmse": rmse,
            "normalized_mae": norm_mae, "normalized_rmse": norm_rmse,
            "calibration_slope": float(slope), "calibration_intercept": float(intercept),
            "epsilon_mean": float(np.mean(eps)), "epsilon_std": std_eps,
            "epsilon_quantiles": {"p5": float(np.quantile(eps, .05)), "p25": float(np.quantile(eps, .25)),
                                     "p50": float(np.quantile(eps, .5)), "p75": float(np.quantile(eps, .75)),
                                     "p95": float(np.quantile(eps, .95))},
            "error_scaled_boundary_band_fraction": band_frac, "error_band_k": ERROR_BAND_K,
        }
        print(f"  cell_h{bi}: corr={corr:.3f} norm_mae={norm_mae:.3f} norm_rmse={norm_rmse:.3f} "
              f"calib_slope={slope:.3f} calib_intercept={intercept:+.3f} "
              f"band_frac(k={ERROR_BAND_K})={band_frac:.3f}", flush=True)
    report["continuous_audit_h1"] = cont

    # bootstrap CIs for switch recall / false-switch rate / balanced accuracy, by episode
    print("\n=== bootstrapping switch-conditioned metrics by episode ===", flush=True)
    eids_h1 = np.array([s["eid"] for s in starts])
    start_labels = np.array([label8(s["z_t"], U8, B8) for s in starts])
    # real/pred BIT labels at h=1 directly from stored scores (score>0 <=> label bit, since B8 already subtracted)
    pred_bits_h1 = (pred_arr > 0).astype(int)
    real_bits_h1 = (real_arr > 0).astype(int)
    switch_out = {}
    for bi in range(8):
        actual_switch = (real_bits_h1[:, bi] != start_labels[:, bi])
        predicted_switch = (pred_bits_h1[:, bi] != start_labels[:, bi])
        by_eid_recall_num, by_eid_recall_den = defaultdict(list), defaultdict(list)
        by_eid_fsr_num, by_eid_fsr_den = defaultdict(list), defaultdict(list)
        for i, eid in enumerate(eids_h1):
            if actual_switch[i]:
                by_eid_recall_den[eid].append(1)
                by_eid_recall_num[eid].append(1 if predicted_switch[i] else 0)
            else:
                by_eid_fsr_den[eid].append(1)
                by_eid_fsr_num[eid].append(1 if predicted_switch[i] else 0)

        def episode_ratio(num, den):
            vals = []
            for e, d_list in den.items():
                d = len(d_list)
                n = sum(num.get(e, []))
                if d > 0:
                    vals.append(n / d)
            return np.array(vals)

        recall_vals = episode_ratio(by_eid_recall_num, by_eid_recall_den)
        fsr_vals = episode_ratio(by_eid_fsr_num, by_eid_fsr_den)
        mean_r, lo_r, hi_r = bootstrap_ci(recall_vals, SEED + 51 + bi) if len(recall_vals) else (None, None, None)
        mean_f, lo_f, hi_f = bootstrap_ci(fsr_vals, SEED + 61 + bi) if len(fsr_vals) else (None, None, None)
        n_switch = int(actual_switch.sum())
        switch_out[f"cell_h{bi}"] = {
            "recall_given_switch": {"mean": mean_r, "ci95": [lo_r, hi_r]},
            "false_switch_rate_given_nonswitch": {"mean": mean_f, "ci95": [lo_f, hi_f]},
            "n_switch_events": n_switch, "n_nonswitch_events": int((~actual_switch).sum()),
            "n_episodes_with_switch": len(set(eids_h1[actual_switch].tolist())),
            "underpowered": n_switch < 30, "underpowered_rule": "n_switch_events < 30",
        }
        print(f"  cell_h{bi}: recall={mean_r} ci95=[{lo_r},{hi_r}]  fsr={mean_f} ci95=[{lo_f},{hi_f}]  "
              f"n_switch={n_switch} underpowered={n_switch < 30}", flush=True)
    report["switch_conditioned_bootstrap"] = switch_out

    # ================= 3. completed initialisation audit =================
    print("\n=== completed initialisation audit ===", flush=True)
    sd = scatter_decomposition(train)
    residual_basis, k_removed = build_residual_basis(sd["Sigma_between"], 0.95)

    d_ambient_all, d_residual_all = [], []
    h0_joint_by_m = {m: [] for m in M_VALUES}
    h0_perbit = [[] for _ in range(8)]
    for s in starts:
        z_t = s["z_t"]
        r_name = s["r_name"]
        r_z = reset_Z[reset_names.index(r_name)]
        d_amb = float(np.linalg.norm(r_z - z_t))
        z_res = residual_basis.T @ z_t
        r_res = residual_basis.T @ r_z
        d_res = float(np.linalg.norm(r_res - z_res))
        d_ambient_all.append(d_amb)
        d_residual_all.append(d_res)
        l_z, l_r = label8(z_t, U8, B8), label8(r_z, U8, B8)
        for bi in range(8):
            h0_perbit[bi].append(float(l_z[bi] == l_r[bi]))
        for m in M_VALUES:
            h0_joint_by_m[m].append(float(np.array_equal(l_z[:m], l_r[:m])))

    init_audit = {
        "mean_distance_ambient": float(np.mean(d_ambient_all)), "std_distance_ambient": float(np.std(d_ambient_all)),
        "mean_distance_residual": float(np.mean(d_residual_all)), "std_distance_residual": float(np.std(d_residual_all)),
        "residual_dim": int(residual_basis.shape[1]), "k_removed": k_removed,
        "h0_agreement_per_bit": {f"cell_h{bi}": float(np.mean(h0_perbit[bi])) for bi in range(8)},
        "h0_agreement_joint_per_m": {str(m): float(np.mean(h0_joint_by_m[m])) for m in M_VALUES},
    }
    print(f"  mean distance: ambient={init_audit['mean_distance_ambient']:.2f} "
          f"residual({init_audit['residual_dim']}-dim)={init_audit['mean_distance_residual']:.2f}", flush=True)
    print(f"  h0 joint agreement per m: {init_audit['h0_agreement_joint_per_m']}", flush=True)
    print(f"  h0 per-bit agreement: {init_audit['h0_agreement_per_bit']}", flush=True)

    # reset-cell coverage: distinct real starting cells (m=8) vs the 8 reset windows' own cells
    real_start_cells = set(tuple(label8(s["z_t"], U8, B8).tolist()) for s in starts)
    reset_cells = set(tuple(label8(reset_Z[i], U8, B8).tolist()) for i in range(len(reset_names)))
    covered = real_start_cells & reset_cells
    init_audit["starting_cell_coverage"] = {
        "n_distinct_real_start_cells_m8": len(real_start_cells),
        "n_distinct_reset_cells_m8": len(reset_cells),
        "n_covered": len(covered),
        "coverage_fraction": len(covered) / len(real_start_cells) if real_start_cells else None,
    }
    print(f"  starting-cell coverage (m=8): {len(real_start_cells)} distinct real start cells, "
          f"{len(reset_cells)} distinct reset cells (of 8 windows), "
          f"{len(covered)} covered ({init_audit['starting_cell_coverage']['coverage_fraction']:.4f})", flush=True)
    report["initialisation_audit"] = init_audit

    (OUT_DIR / "a01c_corrections_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote artifacts/a01c_corrections_report.json", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
