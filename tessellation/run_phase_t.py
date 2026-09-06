"""Phase T driver: build the shared latent corpora (train + held-out), fit
the hyperplane tessellation sweep, report occupancy, pick best_m by the rule
fixed in preregistration.md, and fit the matched equal-mass Voronoi
comparison. Writes artifacts/tessellation.json + tessellation_params.npz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.latent_corpus import build_corpus, test_ids, train_ids
from tessellation.hyperplanes import fit_tessellation, fit_voronoi, sweep_report

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SEED = 3072
N_TRAIN_EPISODES = 320
N_TRAIN_POSITIONS = 60
N_TEST_EPISODES = 600
N_TEST_POSITIONS = 80


def main():
    rng = np.random.default_rng(SEED)
    tr_all = train_ids()
    tr_sample = sorted(rng.choice(tr_all, size=min(N_TRAIN_EPISODES, len(tr_all)), replace=False).tolist())
    te_all = test_ids()
    te_sample = sorted(rng.choice(te_all, size=min(N_TEST_EPISODES, len(te_all)), replace=False).tolist())

    print("=== Phase T: building shared corpora (streaming; cached for all later phases) ===", flush=True)
    train_corpus = build_corpus(tr_sample, N_TRAIN_POSITIONS, "train", keep_raw_actions=False)
    test_corpus = build_corpus(te_sample, N_TEST_POSITIONS, "heldout", keep_raw_actions=True)

    Z_train = train_corpus.flat_latents()
    Z_test = test_corpus.flat_latents()
    print(f"\ntrain latents: {Z_train.shape} | held-out latents: {Z_test.shape}", flush=True)

    print("\n=== Phase T: hyperplane tessellation sweep ===", flush=True)
    results, tess, best_m, rule = sweep_report(Z_train, m_values=(2, 4, 6, 8), seed=SEED)
    for r in results:
        print(f"  m={r['m']}: {r['n_cells_occupied']}/{r['n_cells_nominal']} cells occupied "
              f"(>=20 train latents), occupied mass {r['occupied_mass_frac']:.3f}, "
              f"occupancy min/med/max = {r['min_occupancy']}/{r['median_occupancy']}/{r['max_occupancy']}",
              flush=True)
    print(f"\nbest_m = {best_m}  [{rule}]", flush=True)

    best = tess[best_m]
    # held-out occupancy at best_m -- what Phase R's eligibility floor sees
    test_cell_ids = best.cell_ids(Z_test)
    uniq, counts = np.unique(test_cell_ids, return_counts=True)
    heldout_occ = {int(c): int(n) for c, n in zip(uniq, counts)}

    print("\n=== Phase T secondary: equal-mass Voronoi (matched cell count) ===", flush=True)
    vor = fit_voronoi(Z_train, n_centroids=len(best.occupied), seed=SEED)
    print(f"  {len(vor.occupied)} occupied Voronoi cells (matched target {len(best.occupied)}), "
          f"occupancy min/med/max = {min(vor.occupied.values())}/"
          f"{int(np.median(list(vor.occupied.values())))}/{max(vor.occupied.values())}", flush=True)

    out = {
        "seed": SEED,
        "train_corpus": {"n_episodes": int(train_corpus.n_episodes), "n_positions": int(train_corpus.n_positions),
                           "n_latents": int(Z_train.shape[0])},
        "heldout_corpus": {"n_episodes": int(test_corpus.n_episodes), "n_positions": int(test_corpus.n_positions),
                             "n_latents": int(Z_test.shape[0])},
        "min_train_per_cell": 20,
        "sweep": results,
        "best_m": best_m,
        "best_m_rule": rule,
        "best_m_occupied_cells": {str(k): v for k, v in sorted(best.occupied.items())},
        "best_m_heldout_occupancy": {str(k): v for k, v in sorted(heldout_occ.items())},
        "voronoi_secondary": {
            "n_centroids": int(vor.centroids.shape[0]),
            "n_occupied": len(vor.occupied),
            "min_occupancy": int(min(vor.occupied.values())) if vor.occupied else 0,
            "median_occupancy": int(np.median(list(vor.occupied.values()))) if vor.occupied else 0,
            "max_occupancy": int(max(vor.occupied.values())) if vor.occupied else 0,
        },
    }
    (OUT_DIR / "tessellation.json").write_text(json.dumps(out, indent=2))
    np.savez(OUT_DIR / "tessellation_params.npz",
             **{f"U_m{m}": tess[m].U for m in tess}, **{f"B_m{m}": tess[m].B for m in tess},
             voronoi_centroids=vor.centroids, best_m=np.array([best_m]))
    print("\nwrote tessellation.json + tessellation_params.npz", flush=True)


if __name__ == "__main__":
    main()
    import sys as _s, os as _o
    _s.stdout.flush()
    _o._exit(0)
