"""E0b driver: two disambiguations required before E1 (EXPERIMENTS_orthogonal_
expansion.md v2, SS3 E0b). Reuses the cached corpus + rollout probe; no new
video streaming. Applies the ROUTING table (not a gate) to decide how E1
proceeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import build_corpus, test_ids
from e0_degeneracy.e0b import nearest_real_distance, real_variance_at_matched_horizons, tessellation_margins_with_ci
from e0_degeneracy.rollout_collapse import run as rollout_run

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    print("=== E0b.1: variance growth -- healthy or divergent? ===", flush=True)
    result = rollout_run()   # same z0=r0, 500 words x 24 steps -- recomputed, cheap (no streaming)
    trace = result["trace"]
    ell_values = list(range(1, trace.shape[1] + 1))

    heldout = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    real_var = real_variance_at_matched_horizons(heldout.latents, ell_values)
    model_var_by_ell = trace[:, :, :].var(axis=0).sum(axis=1).tolist()   # (max_len,)

    real_flat = heldout.flat_latents()
    nn_dist = nearest_real_distance(trace, real_flat)

    print("  ell : model_var : real_var(matched horizon) : nearest_real_dist", flush=True)
    for i in [0, 3, 7, 11, 15, 19, 23]:
        rv = real_var[i]
        print(f"    {ell_values[i]:3d} : {model_var_by_ell[i]:8.4f} : "
              f"{('n/a' if rv is None else f'{rv:8.4f}')} : {nn_dist[i]:8.4f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(ell_values, model_var_by_ell, "o-", label="model rollout variance (500 words, z0=r0)")
    valid = [(e, v) for e, v in zip(ell_values, real_var) if v is not None]
    axes[0].plot([e for e, v in valid], [v for e, v in valid], "s--", label="real held-out variance (matched horizon)")
    axes[0].set_xlabel("word length ell"); axes[0].set_ylabel("total variance")
    axes[0].set_title("E0b.1(a): model vs real variance"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ell_values, nn_dist, "o-", color="tab:red")
    axes[1].set_xlabel("word length ell"); axes[1].set_ylabel("mean dist to nearest REAL latent")
    axes[1].set_title("E0b.1(b): distance to data manifold vs length"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "e0b1_manifold_check.png", dpi=140)
    print("  saved e0b1_manifold_check.png", flush=True)

    nn_growth_ratio = nn_dist[-1] / max(nn_dist[0], 1e-9)
    var_ratio_at_max = model_var_by_ell[-1] / real_var[-1] if real_var[-1] else float("nan")
    # Operational thresholds, stated here (not tuned after seeing the numbers
    # beyond this first look): "healthy" if the rollout stays within an order
    # of magnitude of the real-data benchmark on both measures.
    healthy = (nn_growth_ratio < 3.0) and (0.1 <= var_ratio_at_max <= 10.0)
    e0b1 = {
        "ell": ell_values, "model_variance": model_var_by_ell, "real_variance_matched": real_var,
        "nearest_real_distance": nn_dist, "nn_growth_ratio_ell1_to_ell24": nn_growth_ratio,
        "model_to_real_variance_ratio_at_ell24": var_ratio_at_max,
        "route": "near_manifold_healthy" if healthy else "off_manifold_drift",
    }
    print(f"\n  nn_dist growth ratio (ell24/ell1) = {nn_growth_ratio:.3f}", flush=True)
    print(f"  model/real variance ratio at ell24 = {var_ratio_at_max:.3f}", flush=True)
    print(f"  E0b.1 ROUTE: {e0b1['route']}", flush=True)

    print("\n=== E0b.2: tessellation per-bit margins with bootstrap 95% CIs ===", flush=True)
    margins = tessellation_margins_with_ci(heldout, OUT_DIR / "tessellation_params.npz")
    all_margins = []
    for m, mres in margins.items():
        if "error" in mres:
            print(f"  m={m}: {mres['error']}", flush=True)
            continue
        for bit, b in mres["bits"].items():
            all_margins.append(b["margin_mean_diff"])
            print(f"  m={m} {bit}: fidelity={b['mean_fidelity']:.4f} persistence={b['mean_persistence']:.4f} "
                  f"margin={b['margin_mean_diff']:+.4f} CI=[{b['margin_ci95'][0]:+.4f},{b['margin_ci95'][1]:+.4f}] "
                  f"excludes_0={b['beats_persistence_ci_excludes_0']}", flush=True)

    mean_margin = float(np.mean(all_margins)) if all_margins else float("nan")
    n_clearly_positive = sum(
        1 for m in margins.values() if "bits" in m for b in m["bits"].values()
        if b["beats_persistence_ci_excludes_0"] and b["margin_mean_diff"] > 0)
    n_total_bits = sum(len(m["bits"]) for m in margins.values() if "bits" in m)
    frac_clearly_positive = n_clearly_positive / n_total_bits if n_total_bits else 0.0

    if mean_margin > 0.02 and frac_clearly_positive > 0.5:
        e0b2_route = "margins_clearly_positive"
    elif mean_margin > 0 and frac_clearly_positive > 0.0:
        e0b2_route = "margins_small_but_positive"
    else:
        e0b2_route = "margins_near_zero"
    print(f"\n  mean margin across all bits = {mean_margin:+.4f}; "
          f"{n_clearly_positive}/{n_total_bits} bits clearly positive (CI excludes 0, margin>0)", flush=True)
    print(f"  E0b.2 ROUTE: {e0b2_route}", flush=True)

    # ---------------- combined routing decision ----------------
    if e0b1["route"] == "near_manifold_healthy" and e0b2_route == "margins_clearly_positive":
        combined_route = "E1 as written"
    elif e0b1["route"] == "off_manifold_drift":
        combined_route = "E1 runs, but restrict base points to near-manifold region and shorten oracle horizon"
    elif e0b2_route == "margins_small_but_positive":
        combined_route = "E1 runs; raise E3 seed count from 3 to 5"
    else:
        combined_route = "E1prime: measure geometry anyway (cheap, reusable), redirect effort to the label function"

    print(f"\n=== E0b COMBINED ROUTE: {combined_route} ===", flush=True)

    out = {"e0b1": e0b1, "e0b2": margins, "e0b2_summary": {
        "mean_margin": mean_margin, "n_clearly_positive": n_clearly_positive, "n_total_bits": n_total_bits,
        "route": e0b2_route,
    }, "combined_route": combined_route}
    (OUT_DIR / "e0b_report.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote e0b_report.json + e0b1_manifold_check.png", flush=True)


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
