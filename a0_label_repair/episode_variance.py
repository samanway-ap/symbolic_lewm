"""Sharper test of the mixture-of-episodes/scene-confound hypothesis: decompose
the variance of z into between-episode and within-episode components (an
ANOVA/MANOVA-style scatter decomposition), rather than eyeballing "which side
of the median" for individual points. All on cached latents+episode_ids, no
model calls.

For group means m_e (e = episode) and grand mean m:
  Sigma_between = (1/N) * sum_e n_e (m_e - m)(m_e - m)^T      -- a (D,D) matrix
  Sigma_within  = (1/N) * sum_e sum_t (z_et - m_e)(z_et - m_e)^T
  Sigma_total   = Sigma_between + Sigma_within   (exact, by the ANOVA identity)

Multivariate eta^2 (fraction of total variance explained by episode identity):
  eta_sq_trace = trace(Sigma_between) / trace(Sigma_total)

Per fixed direction u (e.g. the project's own tessellation hyperplane normals):
  eta_sq(u) = (u^T Sigma_between u) / (u^T Sigma_total u)

If the tessellation directions have eta_sq(u) far above what random directions
get, they are preferentially aligned with "which episode/scene," not with
within-episode dynamics -- direct evidence for the scene-confound explanation
of a uniformly-low switch rate (labels rarely change because the episode,
not the moment-to-moment state, is what the direction mostly separates).

Also PCA's Sigma_between itself to find how many leading directions carry
most of the between-episode variance -- sizes the subspace lever 0 needs to
project out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import Corpus, build_corpus, test_ids, train_ids

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SEED = 3072


def scatter_decomposition(corpus: Corpus) -> dict:
    """Returns Sigma_between, Sigma_within (both (D,D)), grand_mean, per-episode
    means and counts -- everything downstream needs is derivable from these."""
    n_ep, n_pos, D = corpus.latents.shape
    flat = corpus.latents.reshape(n_ep, n_pos, D)
    ep_means = flat.mean(axis=1)                      # (n_ep, D)
    n_e = np.full(n_ep, n_pos, dtype=np.float64)
    N = n_ep * n_pos
    grand_mean = flat.reshape(-1, D).mean(axis=0)      # (D,)

    dev_between = ep_means - grand_mean[None, :]        # (n_ep, D)
    Sigma_between = (dev_between * n_e[:, None]).T @ dev_between / N

    dev_within = flat - ep_means[:, None, :]             # (n_ep, n_pos, D)
    dw = dev_within.reshape(-1, D)
    Sigma_within = dw.T @ dw / N

    return {"Sigma_between": Sigma_between, "Sigma_within": Sigma_within,
            "grand_mean": grand_mean, "ep_means": ep_means, "n_ep": n_ep, "n_pos": n_pos}


def eta_sq_direction(u: np.ndarray, Sigma_between: np.ndarray, Sigma_within: np.ndarray) -> float:
    u = u / np.linalg.norm(u)
    b = float(u @ Sigma_between @ u)
    w = float(u @ Sigma_within @ u)
    return b / (b + w) if (b + w) > 0 else 0.0


def main():
    rng = np.random.default_rng(SEED)
    train = build_corpus(train_ids(), 60, "train")
    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=False)
    tp = np.load(OUT_DIR / "tessellation_params.npz")
    U8 = tp["U_m8"]   # the fixed u_1..u_8 sequence; lower m levels are prefixes of this

    report = {}
    for split_name, corpus in (("train", train), ("heldout", heldout)):
        sd = scatter_decomposition(corpus)
        Sb, Sw = sd["Sigma_between"], sd["Sigma_within"]
        D = Sb.shape[0]

        eta_global = float(np.trace(Sb) / np.trace(Sb + Sw))

        eta_tess = [eta_sq_direction(U8[i], Sb, Sw) for i in range(8)]

        n_random = 200
        random_dirs = rng.normal(size=(n_random, D))
        random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
        eta_random = np.array([eta_sq_direction(u, Sb, Sw) for u in random_dirs])

        eigvals_between, eigvecs_between = np.linalg.eigh(Sb)
        order = np.argsort(eigvals_between)[::-1]
        eigvals_between = eigvals_between[order]
        eigvecs_between = eigvecs_between[:, order]
        eigvals_between = np.clip(eigvals_between, 0, None)
        cum_energy = np.cumsum(eigvals_between) / max(eigvals_between.sum(), 1e-12)
        dims_for_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
        dims_for_95 = int(np.searchsorted(cum_energy, 0.95) + 1)

        print(f"=== split={split_name} (n_ep={sd['n_ep']}, n_pos={sd['n_pos']}) ===", flush=True)
        print(f"  GLOBAL eta_sq (trace ratio, all {D} ambient dims): {eta_global:.4f}", flush=True)
        print(f"  tessellation directions u_1..u_8 eta_sq: "
              f"{['%.4f' % e for e in eta_tess]}", flush=True)
        print(f"  random directions (n={n_random}) eta_sq: mean={eta_random.mean():.4f} "
              f"median={np.median(eta_random):.4f} p90={np.quantile(eta_random, 0.9):.4f} "
              f"max={eta_random.max():.4f}", flush=True)
        print(f"  Sigma_between eigenspectrum: dims for 90% energy={dims_for_90}, "
              f"95%={dims_for_95} (out of {D})", flush=True)

        report[split_name] = {
            "eta_global_trace_ratio": eta_global,
            "eta_tessellation_directions": eta_tess,
            "eta_random_directions": {"mean": float(eta_random.mean()), "median": float(np.median(eta_random)),
                                         "p90": float(np.quantile(eta_random, 0.9)), "max": float(eta_random.max()),
                                         "n": n_random},
            "between_episode_subspace_dims_90pct": dims_for_90,
            "between_episode_subspace_dims_95pct": dims_for_95,
        }
        if split_name == "heldout":
            np.savez(OUT_DIR / "a0_episode_subspace_heldout.npz",
                       eigvecs_between=eigvecs_between, eigvals_between=eigvals_between,
                       grand_mean=sd["grand_mean"])

    (OUT_DIR / "a0_episode_variance_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote artifacts/a0_episode_variance_report.json and "
          "artifacts/a0_episode_subspace_heldout.npz", flush=True)


if __name__ == "__main__":
    main()
