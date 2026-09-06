"""Lever 0 (new, per user direction, ahead of the v4 doc's levers 1-4, which
both push switch rate further down and are therefore the wrong direction for
a label function that fails LOW): project out the between-episode subspace
found by episode_variance.py's scatter decomposition, and draw new
tessellation directions confined to the orthogonal complement, so phi stays
a fixed linear map of z (no episode identity needed at inference time -- it
never leaves the eigenbasis of a matrix computed once, offline, on train).

Sigma_between is symmetric PSD, so its eigenvectors are exactly orthonormal
and span all of R^D: keeping the eigenvectors with the SMALLEST eigenvalues
(after dropping the top-k that carry `energy` fraction of Sigma_between's
own trace) gives an orthonormal basis for a subspace that is, by
construction, orthogonal to the removed between-episode directions. New
directions are drawn as random combinations of ONLY that residual basis, so
u_i . z is automatically blind to the episode-mean component of z -- no
explicit projection of z is needed at label time, only at direction-design
time (train, offline, once).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a0_label_repair.episode_variance import scatter_decomposition
from corpus.latent_corpus import build_corpus, test_ids, train_ids
from e0_degeneracy.baselines import switch_persistence_majority

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SEED = 3072
ENERGY_TO_REMOVE = 0.95     # fraction of Sigma_between's OWN energy to project out
M_MAX = 8


def build_residual_basis(Sigma_between: np.ndarray, energy: float) -> tuple[np.ndarray, int]:
    eigvals, eigvecs = np.linalg.eigh(Sigma_between)      # ascending
    order = np.argsort(eigvals)[::-1]                       # descending
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    eigvals = np.clip(eigvals, 0, None)
    cum = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
    k = int(np.searchsorted(cum, energy) + 1)                 # dims to REMOVE
    residual_basis = eigvecs[:, k:]                            # (D, D-k), orthonormal, orthogonal to top-k
    return residual_basis, k


def draw_directions_in_complement(residual_basis: np.ndarray, m_max: int, seed: int) -> np.ndarray:
    D, d_complement = residual_basis.shape
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(m_max, d_complement))
    coeffs /= np.linalg.norm(coeffs, axis=1, keepdims=True)
    U = coeffs @ residual_basis.T          # (m_max, D), each row a unit vector confined to the complement
    U /= np.linalg.norm(U, axis=1, keepdims=True)   # renormalise (residual_basis columns are orthonormal, should be ~1 already)
    return U


def fit_offsets(U: np.ndarray, latents_flat: np.ndarray) -> np.ndarray:
    scores = latents_flat @ U.T   # (N, m_max)
    return np.median(scores, axis=0)


def label_sequences_from_UB(latents: np.ndarray, u: np.ndarray, b: float) -> np.ndarray:
    n_ep, n_pos, _ = latents.shape
    scores = latents @ u
    return (scores > b).astype(np.int8)


def main():
    train = build_corpus(train_ids(), 60, "train")
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=False)

    print("=== computing between-episode scatter on TRAIN (fit split) ===", flush=True)
    sd = scatter_decomposition(train)
    Sb, Sw = sd["Sigma_between"], sd["Sigma_within"]
    D = Sb.shape[0]
    residual_basis, k_removed = build_residual_basis(Sb, ENERGY_TO_REMOVE)
    print(f"  removed top {k_removed} between-episode eigenvectors "
          f"({ENERGY_TO_REMOVE * 100:.0f}% of Sigma_between's energy); "
          f"residual complement dim = {D - k_removed}", flush=True)

    U8 = draw_directions_in_complement(residual_basis, M_MAX, SEED)
    train_flat = train.flat_latents()
    B8 = fit_offsets(U8, train_flat)

    # sanity check: these directions really are ~orthogonal to Sigma_between's removed subspace
    eta_new = []
    for i in range(M_MAX):
        u = U8[i]
        b_ = float(u @ Sb @ u)
        w_ = float(u @ Sw @ u)
        eta_new.append(b_ / (b_ + w_) if (b_ + w_) > 0 else 0.0)
    print(f"  new directions' eta_sq (should be near the RESIDUAL floor, far below the "
          f"~0.93 measured for the original/random directions): {['%.4f' % e for e in eta_new]}", flush=True)

    params = {}
    for m in (2, 4, 6, 8):
        params[f"U_m{m}"] = U8[:m]
        params[f"B_m{m}"] = B8[:m]
    out_path = OUT_DIR / "tessellation_params_residualized.npz"
    np.savez(out_path, **params, k_removed=k_removed, energy_removed=ENERGY_TO_REMOVE, seed=SEED)
    print(f"  wrote {out_path.name}", flush=True)

    print("\n=== re-measuring switch rate + determinism (incl. determinism-given-switch) ===", flush=True)
    report = {"k_removed": k_removed, "energy_removed": ENERGY_TO_REMOVE, "eta_new_directions": eta_new, "by_m": {}}
    for m in (2, 4, 6, 8):
        U, B = U8[:m], B8[:m]
        bits_out = {}
        for i in range(m):
            labels_train = label_sequences_from_UB(train.latents, U[i], B[i])
            labels_held = label_sequences_from_UB(heldout.latents, U[i], B[i])
            spm_t = switch_persistence_majority(labels_train)
            spm_h = switch_persistence_majority(labels_held)
            bits_out[f"cell_h{i}"] = {
                "switch_rate_train": spm_t["switch_rate"], "switch_rate_heldout": spm_h["switch_rate"],
                "in_band_train": 0.02 <= spm_t["switch_rate"] <= 0.25,
                "in_band_heldout": 0.02 <= spm_h["switch_rate"] <= 0.25,
            }
        sr_t = [v["switch_rate_train"] for v in bits_out.values()]
        sr_h = [v["switch_rate_heldout"] for v in bits_out.values()]
        print(f"  m={m}: switch_rate train={['%.4f' % s for s in sr_t]} "
              f"heldout={['%.4f' % s for s in sr_h]}", flush=True)
        report["by_m"][str(m)] = {"bits": bits_out}

    (OUT_DIR / "a0_residualized_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote artifacts/a0_residualized_report.json", flush=True)


if __name__ == "__main__":
    main()
