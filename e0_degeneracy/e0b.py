"""E0b -- two disambiguations required before E1, per EXPERIMENTS_orthogonal_
expansion.md v2 SS3 E0b. Reuses the cached corpora built in the tessellation
phase; no new streaming.

E0b.1: is E0's variance GROWTH healthy (near-manifold, actions genuinely
diversifying outcomes) or divergent (error compounding, off-manifold drift)?
Two measurements at matched horizons:
  (a) variance of REAL held-out latents at the same step-from-reset ell
  (b) mean distance from each rolled-out latent to its NEAREST real latent

E0b.2: per-bit tessellation margin over persistence, WITH bootstrap 95% CIs,
at every swept m -- "all bits beat baseline" (E0.3) was a bare boolean, not
an effect size. Requires actually re-extracting each m's diagnostic FSM (T2)
to get PER-EPISODE agreement arrays for pairing, since gate/acceptance.py's
`evaluate()` only returns an aggregate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.walk import choose_reset, reset_reference
from local_import import load_local
from oracle.lewm_g import HISTORY_SIZE
from tessellation.hyperplanes import Tessellation

_stats = load_local("symbolic_eval_stats", "eval/stats.py")
bootstrap_paired_diff_ci = _stats.bootstrap_paired_diff_ci

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


# ---------------- E0b.1 ----------------
def real_variance_at_matched_horizons(heldout_latents: np.ndarray, ell_values: list[int]) -> list[float]:
    """heldout_latents: (n_ep, n_pos, D). Variance of REAL latents at position
    (HISTORY_SIZE-1+ell) across episodes -- the natural real-data benchmark
    for 'how spread out is data ell steps from a reset-like start'."""
    out = []
    n_pos = heldout_latents.shape[1]
    for ell in ell_values:
        pos = HISTORY_SIZE - 1 + ell
        if pos >= n_pos:
            out.append(None)
            continue
        pts = heldout_latents[:, pos, :]
        out.append(float(pts.var(axis=0).sum()))
    return out


def nearest_real_distance(trace: np.ndarray, real_latents_flat: np.ndarray, chunk: int = 500) -> list[float]:
    """trace: (n_words, max_len, D). real_latents_flat: (M, D). Returns, per
    ell, the mean over words of the distance to the NEAREST real latent --
    growing with ell means rollouts drift off the data manifold."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_t = torch.from_numpy(real_latents_flat).float().to(device)
    n_words, max_len, D = trace.shape
    out = []
    for ell in range(max_len):
        pts = torch.from_numpy(trace[:, ell, :]).float().to(device)
        mins = []
        for i in range(0, pts.shape[0], chunk):
            d = torch.cdist(pts[i:i + chunk], real_t)
            mins.append(d.min(dim=1).values.cpu())
        out.append(float(torch.cat(mins).mean().item()))
    return out


# ---------------- E0b.2 ----------------
def per_episode_bit_agreement(machine, corpus, bit_fns: list, reset_names, reset_Z) -> tuple[np.ndarray, np.ndarray]:
    """Returns (fidelity, persistence): each (n_bits, n_episodes), paired by
    episode index so a bootstrap CI on the per-bit margin is well-defined."""
    n_ep = corpus.latents.shape[0]
    n_bits = len(bit_fns)
    fidelity = np.full((n_bits, n_ep), np.nan)
    persistence = np.full((n_bits, n_ep), np.nan)
    for i in range(n_ep):
        Z = corpus.latents[i]
        names = corpus.symbol_seq(i)
        reset_sym = choose_reset(Z[HISTORY_SIZE - 1], reset_names, reset_Z)
        machine.reset_to_initial()
        outs = [machine.step(reset_sym)]
        for t in range(min(len(names), Z.shape[0] - 1)):
            outs.append(machine.step(names[t]))
        n_cmp = min(len(outs) - 1, Z.shape[0] - HISTORY_SIZE)
        if n_cmp < 2:
            continue
        for b, fn in enumerate(bit_fns):
            reals = [fn(Z[HISTORY_SIZE + t]) for t in range(n_cmp)]
            preds = [outs[t + 1][b] for t in range(n_cmp)]
            fidelity[b, i] = np.mean([p == r for p, r in zip(preds, reals)])
            persistence[b, i] = np.mean([reals[t] == reals[t - 1] for t in range(1, n_cmp)])
    return fidelity, persistence


def tessellation_margins_with_ci(corpus, tess_params_path: Path,
                                    m_values=(2, 4, 6, 8), max_states=2000,
                                    separation_rule: str = "ADS") -> dict:
    from learning.extract import extract
    from oracle.latent_oracle import LatentOracle
    from oracle.lewm_g import ActionPipeline, FRAMESKIP, get_model
    from oracle.production_oracle import _step_fn_from_convention, alphabet_symbol_names, reset_symbol_names
    from oracle.droid_actions import load_actions_for_episodes

    from oracle.production_oracle import build_reset_windows, load_alphabet_symbols

    rng = np.random.default_rng(3072)
    fit_ids = sorted(rng.choice(corpus.episode_ids, size=min(200, len(corpus.episode_ids)), replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    pipeline = ActionPipeline.fit(np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0))
    full_alphabet = reset_symbol_names() + alphabet_symbol_names(OUT_DIR / "alphabet_trainfit.json")

    tp = np.load(tess_params_path)
    reset_windows = build_reset_windows(pipeline)
    reset_names, reset_Z = reset_reference(reset_windows)
    symbols = load_alphabet_symbols(OUT_DIR / "alphabet_trainfit.json")
    step_fn_raw = _step_fn_from_convention(pipeline)

    def real_step_fn(w, s):
        return step_fn_raw(w, symbols[s])

    out = {}
    for m in m_values:
        t = Tessellation(U=tp[f"U_m{m}"], B=tp[f"B_m{m}"], m=m, occupied={}, seed=3072)
        bit_fns = [(lambda z, i=i, t=t: int((float(z @ t.U[i]) > float(t.B[i])))) for i in range(m)]
        subset = [f"h{i}" for i in range(m)]

        def label_fn(w, bit_fns=bit_fns):
            z = w.emb[-1].numpy()
            return tuple(f(z) for f in bit_fns)

        oracle = LatentOracle(step_fn=real_step_fn, label_fn=label_fn, resets=reset_windows)
        res = extract(subset, oracle, full_alphabet, max_learning_rounds=10, max_states_hard=max_states,
                        separation_rule=separation_rule)
        print(f"  m={m}: extraction status={res.status} n_states={res.n_states}", flush=True)
        if res.status != "converged" or res.machine is None:
            out[str(m)] = {"error": f"extraction {res.status}"}
            continue

        fidelity, persistence = per_episode_bit_agreement(res.machine, corpus, bit_fns, reset_names, reset_Z)
        bits_out = {}
        for i in range(m):
            valid = ~np.isnan(fidelity[i]) & ~np.isnan(persistence[i])
            f_vals, p_vals = fidelity[i, valid].tolist(), persistence[i, valid].tolist()
            d, lo, hi = bootstrap_paired_diff_ci(f_vals, p_vals, seed=3072 + i)
            bits_out[f"h{i}"] = {
                "mean_fidelity": float(np.mean(f_vals)), "mean_persistence": float(np.mean(p_vals)),
                "margin_mean_diff": d, "margin_ci95": [lo, hi],
                "beats_persistence_ci_excludes_0": lo > 0, "n_episodes": int(valid.sum()),
            }
        out[str(m)] = {"n_states": res.n_states, "bits": bits_out}
    return out
