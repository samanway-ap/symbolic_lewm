"""E1 driver: intrinsic dimension (E1.1), local effective rank (E1.2), and
per-base-point expansion space V/T/W (E1.3), on the m=2 tessellation machine
-- the ONLY level satisfying the Moore counting bound (E0b.2's fresh
extraction: m=2 6 states >= 4 cells OK; m=4 11<16, m=6 26<57, m=8 23<123 all
VIOLATED). Horizon restricted to <=8 letters per E0b.1's routed "shorten the
oracle horizon" decision (nn-distance-to-real-data at ell=8 is only 1.46x
its ell=1 value, vs 3.80x at ell=24).

Applies EXPERIMENTS_orthogonal_expansion.md v2's E1 sizing routing table on
dim(W)/dim(T), not a gate -- every outcome here leads to a named next step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import build_corpus, test_ids
from e1_geometry.expansion_space import base_point_state, compute_V_T_W, moore_bound_check
from e1_geometry.intrinsic_dimension import estimate_intrinsic_dimension
from e1_geometry.local_rank import local_rank_distribution
from learning.extract import extract
from oracle.droid_actions import load_actions_for_episodes
from oracle.latent_oracle import LatentOracle
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM
from oracle.production_oracle import (
    _step_fn_from_convention, alphabet_symbol_names, build_reset_windows,
    load_alphabet_symbols, reset_symbol_names,
)
from tessellation.hyperplanes import Tessellation

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
N_BASE_POINTS = 50
K_LOCAL = 256
MAX_SUFFIX_LEN = 8      # E0b.1's routed horizon restriction
BASE_POINT_POS_RANGE = (3, 15)   # near-manifold region per E0b.1


def main():
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    train = build_corpus([], 60, "train")   # cached from Phase T
    combined = np.concatenate([train.flat_latents(), heldout.flat_latents()], axis=0)

    print("=== E1.1: global intrinsic dimension ===", flush=True)
    dim_est = estimate_intrinsic_dimension(combined, seed=SEED)
    print(f"  ambient_dim={dim_est['ambient_dim']} n_points={dim_est['n_points_used']} "
          f"TwoNN={dim_est['twonn']['d_hat']:.2f} Levina-Bickel={dim_est['levina_bickel']['d_hat']:.2f}",
          flush=True)

    print("\n=== extracting m=2 tessellation machine (only level satisfying Moore bound) ===", flush=True)
    tp = np.load(OUT_DIR / "tessellation_params.npz")
    tess = Tessellation(U=tp["U_m2"], B=tp["B_m2"], m=2, occupied={}, seed=SEED)
    tess_meta = json.loads((OUT_DIR / "tessellation.json").read_text())
    occ_m2 = [r["n_cells_occupied"] for r in tess_meta["sweep"] if r["m"] == 2][0]

    rng = np.random.default_rng(SEED)
    fit_ids = sorted(rng.choice(heldout.episode_ids, size=min(200, len(heldout.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    pipeline = ActionPipeline.fit(np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0))
    reset_windows = build_reset_windows(pipeline)
    from atlas.walk import reset_reference
    reset_names, reset_Z = reset_reference(reset_windows)
    symbols_dict = load_alphabet_symbols(ALPHABET_PATH)
    full_alphabet = reset_symbol_names() + alphabet_symbol_names(ALPHABET_PATH)
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(w, s):
        return step_fn_raw(w, symbols_dict[s])

    def label_fn(w):
        z = w.emb[-1].numpy()
        return tuple(int((float(z @ tess.U[i]) > float(tess.B[i]))) for i in range(tess.m))

    subset_names = [f"cell_h{i}" for i in range(tess.m)]
    oracle = LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=reset_windows)
    res = extract(subset_names, oracle, full_alphabet, max_learning_rounds=10, max_states_hard=2000)
    print(f"  status={res.status} n_states={res.n_states}", flush=True)
    mb = moore_bound_check(res.n_states, occ_m2)
    print(f"  Moore bound: {mb['note']}", flush=True)
    if not mb["moore_bound_ok"]:
        print("  (proceeding with E1 geometry anyway -- it is a sizing measurement of the LATENT "
              "SPACE, not a claim that this machine is a good model)", flush=True)
    machine = res.machine

    print(f"\n=== E1.2/E1.3: {N_BASE_POINTS} base points, local rank + V/T/W (max_suffix_len={MAX_SUFFIX_LEN}) ===",
          flush=True)
    n_pos = heldout.latents.shape[1]
    pos_lo, pos_hi = BASE_POINT_POS_RANGE
    pos_hi = min(pos_hi, n_pos - 1)
    ep_idx = rng.choice(heldout.n_episodes, size=N_BASE_POINTS, replace=True)
    pos_idx = rng.integers(pos_lo, pos_hi + 1, size=N_BASE_POINTS)

    real_flat = heldout.flat_latents()
    base_points = np.stack([heldout.latents[e, p] for e, p in zip(ep_idx, pos_idx)])
    rank_dist = local_rank_distribution(base_points, real_flat, k=K_LOCAL)
    print(f"  local effective rank: mean={rank_dist['mean']:.2f} median={rank_dist['median']:.1f} "
          f"min={rank_dist['min']} max={rank_dist['max']} (ambient={rank_dist['ambient_dim']}, "
          f"ratio={rank_dist['mean_ratio_to_ambient']:.3f})", flush=True)

    n_raw_needed = n_pos * FRAMESKIP
    actions = load_actions_for_episodes(heldout.episode_ids.tolist(), max_frames=n_raw_needed)

    import torch
    from oracle.lewm_g import DEVICE as _MODEL_DEVICE, LeWMWindowState, RAW_ACTION_DIM as _RAD, get_model as _gm

    e_lookup = {int(heldout.episode_ids[i]): i for i in range(heldout.n_episodes)}

    def real_window_state(eid: int, ctx_start: int) -> "LeWMWindowState":
        """A genuinely real (emb, act_emb) window: emb comes straight from the
        cached corpus (which WAS built via encode_pixel_windows_batch, so
        these ARE the real encoder outputs, just cached to avoid re-streaming
        pixels); act_emb is freshly computed from the corresponding REAL raw
        action context via the model's own action_encoder -- no pixels
        needed for that half."""
        model = _gm()
        raw_full = actions[eid][:n_raw_needed].reshape(n_pos, FRAMESKIP, _RAD)
        raw_ctx = raw_full[ctx_start:ctx_start + HISTORY_SIZE]
        act_norm = pipeline(raw_ctx.reshape(-1, _RAD))
        act_flat = act_norm.reshape(1, HISTORY_SIZE, FRAMESKIP * _RAD)
        a = torch.from_numpy(act_flat).float().to(_MODEL_DEVICE)
        with torch.no_grad():
            act_emb = model.action_encoder(a)[0].cpu()
        emb = torch.from_numpy(heldout.latents[e_lookup[eid], ctx_start:ctx_start + HISTORY_SIZE]).float()
        return LeWMWindowState(emb=emb, act_emb=act_emb)

    vtw_results = []
    for bi, (e, p) in enumerate(zip(ep_idx, pos_idx)):
        eid = int(heldout.episode_ids[e])
        if eid not in actions or actions[eid].shape[0] < n_raw_needed:
            continue
        ctx_start = max(0, p - HISTORY_SIZE + 1)
        if ctx_start + HISTORY_SIZE > n_pos:
            continue
        first_z = heldout.latents[e, HISTORY_SIZE - 1]
        real_syms = heldout.symbol_seq(e)[:ctx_start]
        q0 = base_point_state(machine, first_z, reset_names, reset_Z, real_syms)

        state0 = real_window_state(eid, ctx_start)
        vtw = compute_V_T_W(
            machine, tess, base_pixels=None, base_raw_action=None, pipeline=pipeline,
            real_latents_for_T=real_flat, symbols_dict=symbols_dict,
            full_action_alphabet=full_alphabet, q0=q0, max_suffix_len=MAX_SUFFIX_LEN,
            k_local=K_LOCAL, precomputed_state0=state0,
        )
        vtw_results.append({"episode": eid, "pos": int(p), "q0": q0,
                               "dim_V": vtw["dim_V"], "dim_T": vtw["dim_T"], "dim_W": vtw["dim_W"],
                               "W_ratio": vtw["W_ratio"], "n_pairs_used": vtw["n_pairs_used"]})

    print(f"  computed V/T/W for {len(vtw_results)}/{N_BASE_POINTS} base points", flush=True)
    q0_counts = {}
    for v in vtw_results:
        q0_counts[v["q0"]] = q0_counts.get(v["q0"], 0) + 1
    print(f"  base-point state distribution: {q0_counts}", flush=True)

    if vtw_results:
        w_ratios = np.array([v["W_ratio"] for v in vtw_results])
        print(f"  dim(W)/dim(T): mean={w_ratios.mean():.3f} median={np.median(w_ratios):.3f} "
              f"min={w_ratios.min():.3f} max={w_ratios.max():.3f}", flush=True)
        med = float(np.median(w_ratios))
        if med > 0.4:
            e1_route = "ample room; E3 at one base point is a fair test"
        elif med > 0.15:
            e1_route = "E3 at three base points, pooled"
        elif w_ratios.max() > 0:
            e1_route = ("room is thin; run E3 at highest-ratio base points only and pre-register a null is "
                          "expected; also check whether V is large because the machine over-resolves (E1'')")
        else:
            e1_route = "dim(W)=0 everywhere -- route to label-function work (E0b's last row), not E3"
    else:
        w_ratios = np.array([])
        e1_route = "no base points succeeded -- see per-base-point errors, cannot route yet"
    print(f"  E1 ROUTE: {e1_route}", flush=True)

    out = {
        "intrinsic_dimension": dim_est,
        "moore_bound_m2": mb,
        "local_rank_distribution": rank_dist,
        "base_point_states": q0_counts,
        "max_suffix_len_used": MAX_SUFFIX_LEN,
        "vtw_per_base_point": vtw_results,
        "w_ratio_summary": {"mean": float(w_ratios.mean()) if len(w_ratios) else None,
                              "median": float(np.median(w_ratios)) if len(w_ratios) else None,
                              "min": float(w_ratios.min()) if len(w_ratios) else None,
                              "max": float(w_ratios.max()) if len(w_ratios) else None},
        "route": e1_route,
    }
    (OUT_DIR / "e1_report.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote e1_report.json", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
