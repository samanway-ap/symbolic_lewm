"""Trajectory-level stability check (chosen over cProfile / learner-swap in
response to the E0c timing-probe finding: `preregistration.md`'s "E0c timing
probe" entry + addendum). The static boundary-density addendum found no
excess crowding at the m=2 hyperplanes -- that weakens but does not eliminate
hypothesis (b), and cannot distinguish it from hypothesis (a) at all, because
it never looks at DYNAMICS. This script does.

Question: given two REAL windows that are "the same" (very close together in
latent space, from two different real episodes -- not the same episode at
two times, which would trivially converge), and the SAME synthetic action
word applied to both, how often does the m=2 tessellation label end up
DIFFERENT after L letters, as a function of L and the pair's INITIAL
distance d0? Two readings of the same experiment:

  - If mismatch rate stays small and grows roughly proportionally with d0
    (i.e. nearby pairs at large L still mostly agree), the dynamics are
    comparatively stable/Lipschitz -- hypothesis (a), genuine memory, is the
    more likely explanation for the round-5 state explosion in the timing
    probe: the system really does have that much distinguishable structure.
  - If mismatch rate saturates towards the "two independent draws" ceiling
    (~0.5 for a roughly-balanced median-split bit) EVEN FOR THE CLOSEST
    pairs once L is more than a few letters, initial closeness stops
    mattering almost immediately -- the model's own rollout is sensitive
    enough to initial conditions that no finite automaton, however large,
    can cleanly summarize it. That would explain runaway state growth
    without invoking either "real memory" or "static boundary noise": it is
    dynamical, not structural or numerical.

Reuses e1_geometry/run_e1.py's `real_window_state` recipe verbatim (real
cached corpus embeddings for `emb`, real raw actions run through the model's
own action_encoder for `act_emb` -- no pixels needed) and
oracle/production_oracle.py's `_step_fn_from_convention` to advance both
windows of a pair through an IDENTICAL synthetic word, one letter at a time,
recording the m=2 label and pairwise latent distance after every prefix
length -- so one rollout per pair covers every L up to MAX_LEN at once,
mirroring E0's own nested-prefix convention.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import build_corpus, test_ids
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import (
    ActionPipeline, DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM,
    LeWMWindowState, get_model,
)
from oracle.production_oracle import (
    _step_fn_from_convention, alphabet_symbol_names, load_alphabet_symbols,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
SEED = 3072
POS_RANGE = (3, 15)          # same near-manifold restriction as e1_geometry/run_e1.py
POS_PER_EPISODE = 5          # multiple positions/episode -- one point/episode (first attempt)
                              # gave a pool too sparse to find genuine near-duplicates: min
                              # pairwise d0 across 220*219/2 pairs was 0.49, but even the
                              # bottom-10%-by-COUNT decile averaged d0~11.6 -- already close to
                              # the ~13.2 overall-spread scale, i.e. NOT near-duplicates at all.
N_CLOSE_PAIRS = 150          # globally closest pairs (by absolute rank, not decile-of-value)
N_CONTROL_PAIRS = 150        # pairs drawn from near the MEDIAN pairwise distance, for contrast
MAX_LEN = 16                 # letters; matches the oracle's own min/max_walk_len range
N_WORDS_PER_PAIR = 3         # independent random words per pair, averaged


def real_window_state(heldout, e_lookup, actions, eid, ctx_start, pipeline, model):
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


def main():
    rng = np.random.default_rng(SEED)
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    n_pos = heldout.n_positions
    e_lookup = {int(heldout.episode_ids[i]): i for i in range(heldout.n_episodes)}

    fit_ids = sorted(rng.choice(heldout.episode_ids, size=min(200, len(heldout.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    pipeline = ActionPipeline.fit(np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0))
    symbols_dict = load_alphabet_symbols(ALPHABET_PATH)
    action_names = alphabet_symbol_names(ALPHABET_PATH)   # NO reset symbols -- continuing, not resetting
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(w, s):
        return step_fn_raw(w, symbols_dict[s])

    model = get_model()
    tp = np.load(OUT_DIR / "tessellation_params.npz")
    U2, B2 = tp["U_m2"], tp["B_m2"]

    def label(z: np.ndarray) -> tuple:
        return tuple(int((float(z @ U2[i]) > float(B2[i]))) for i in range(2))

    print("=== building candidate pool of real windows (multiple positions/episode) ===", flush=True)
    pos_lo, pos_hi = POS_RANGE
    all_positions = list(range(pos_lo, pos_hi + 1))
    pool_eids = heldout.episode_ids   # use every heldout episode, not a subsample
    n_raw_needed = n_pos * FRAMESKIP
    actions = load_actions_for_episodes(pool_eids.tolist(), max_frames=n_raw_needed)

    pool = []
    for eid in pool_eids:
        eid = int(eid)
        if eid not in actions or actions[eid].shape[0] < n_raw_needed:
            continue
        chosen_pos = rng.choice(all_positions, size=min(POS_PER_EPISODE, len(all_positions)), replace=False)
        for p in chosen_pos:
            p = int(p)
            ctx_start = max(0, p - HISTORY_SIZE + 1)
            if ctx_start + HISTORY_SIZE > n_pos:
                continue
            z0 = heldout.latents[e_lookup[eid], HISTORY_SIZE - 1]
            state0 = real_window_state(heldout, e_lookup, actions, eid, ctx_start, pipeline, model)
            pool.append({"eid": eid, "pos": p, "z0": z0, "state0": state0})
    print(f"  pool size: {len(pool)} (from {len(set(pt['eid'] for pt in pool))} distinct episodes)", flush=True)

    print("=== selecting globally-closest pairs + a median-distance control group ===", flush=True)
    Z = np.stack([pt["z0"] for pt in pool])
    eids_arr = np.array([pt["eid"] for pt in pool])
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    iu, ju = np.triu_indices(len(pool), k=1)
    same_ep = eids_arr[iu] == eids_arr[ju]
    d0_all = D[iu, ju]
    d0_cross = np.where(same_ep, np.inf, d0_all)   # exclude same-episode pairs entirely

    order = np.argsort(d0_cross)
    close_idx = order[:N_CLOSE_PAIRS]
    median_rank = np.sum(np.isfinite(d0_cross)) // 2
    control_idx = order[median_rank - N_CONTROL_PAIRS // 2: median_rank + N_CONTROL_PAIRS // 2]

    pairs = []
    for group, idxs in (("close", close_idx), ("control", control_idx)):
        for k in idxs:
            pairs.append((iu[k], ju[k], float(d0_cross[k]), group))
    close_ds = [p[2] for p in pairs if p[3] == "close"]
    ctrl_ds = [p[2] for p in pairs if p[3] == "control"]
    print(f"  {len(close_ds)} close pairs: d0 in [{min(close_ds):.3f}, {max(close_ds):.3f}], "
          f"median={np.median(close_ds):.3f}", flush=True)
    print(f"  {len(ctrl_ds)} control pairs: d0 in [{min(ctrl_ds):.3f}, {max(ctrl_ds):.3f}], "
          f"median={np.median(ctrl_ds):.3f}", flush=True)

    print(f"\n=== rolling {len(pairs)} pairs x {N_WORDS_PER_PAIR} words x {MAX_LEN} letters "
          f"(plus L=0, static, no action applied) ===", flush=True)
    records = []
    for pi, (i_idx, j_idx, d0, group) in enumerate(pairs):
        a, b = pool[i_idx], pool[j_idx]
        for w in range(N_WORDS_PER_PAIR):
            word = [action_names[k] for k in rng.integers(0, len(action_names), size=MAX_LEN)]
            wa, wb = a["state0"], b["state0"]
            za0, zb0 = wa.emb[-1].numpy(), wb.emb[-1].numpy()
            la0, lb0 = label(za0), label(zb0)
            records.append({
                "pair": pi, "group": group, "d0": d0, "word_idx": w, "L": 0,
                "d_final": float(np.linalg.norm(za0 - zb0)),
                "bit0_mismatch": int(la0[0] != lb0[0]), "bit1_mismatch": int(la0[1] != lb0[1]),
                "any_mismatch": int(la0 != lb0),
            })
            for L, sym in enumerate(word, start=1):
                wa = step_fn(wa, sym)
                wb = step_fn(wb, sym)
                za, zb = wa.emb[-1].numpy(), wb.emb[-1].numpy()
                la, lb = label(za), label(zb)
                records.append({
                    "pair": pi, "group": group, "d0": d0, "word_idx": w, "L": L,
                    "d_final": float(np.linalg.norm(za - zb)),
                    "bit0_mismatch": int(la[0] != lb[0]), "bit1_mismatch": int(la[1] != lb[1]),
                    "any_mismatch": int(la != lb),
                })
        if (pi + 1) % 30 == 0:
            print(f"  {pi + 1}/{len(pairs)} pairs done", flush=True)

    print("\n=== aggregating: mismatch rate by (group, L) ===", flush=True)
    import collections
    agg = collections.defaultdict(lambda: {"n": 0, "any": 0, "b0": 0, "b1": 0, "d0_sum": 0.0, "dfinal_sum": 0.0})
    for r in records:
        key = (r["group"], r["L"])
        a = agg[key]
        a["n"] += 1
        a["any"] += r["any_mismatch"]
        a["b0"] += r["bit0_mismatch"]
        a["b1"] += r["bit1_mismatch"]
        a["d0_sum"] += r["d0"]
        a["dfinal_sum"] += r["d_final"]

    report_rows = []
    for (group, L), a in sorted(agg.items()):
        row = {
            "group": group, "L": L, "n": a["n"],
            "mean_d0": a["d0_sum"] / a["n"], "mean_d_final": a["dfinal_sum"] / a["n"],
            "p_any_mismatch": a["any"] / a["n"], "p_bit0_mismatch": a["b0"] / a["n"],
            "p_bit1_mismatch": a["b1"] / a["n"],
        }
        report_rows.append(row)

    for group in ("close", "control"):
        rows_g = [r for r in report_rows if r["group"] == group]
        rows_g.sort(key=lambda r: r["L"])
        print(f"\n  group={group} (mean_d0={rows_g[0]['mean_d0']:.3f}):", flush=True)
        for r in rows_g:
            print(f"    L={r['L']:2d}  mean_d_final={r['mean_d_final']:.3f}  "
                  f"P(mismatch)={r['p_any_mismatch']:.3f}  P(bit0)={r['p_bit0_mismatch']:.3f}  "
                  f"P(bit1)={r['p_bit1_mismatch']:.3f}", flush=True)

    out = {"config": {"pos_range": POS_RANGE, "max_len": MAX_LEN, "n_words_per_pair": N_WORDS_PER_PAIR,
                        "n_pairs": len(pairs), "n_pool": len(pool),
                        "n_close_pairs": len(close_ds), "n_control_pairs": len(ctrl_ds)},
            "per_group_per_L": report_rows}
    (OUT_DIR / "e0c_trajectory_stability_report.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote e0c_trajectory_stability_report.json", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
