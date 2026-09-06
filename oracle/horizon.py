"""Phase A.4-A.5: measure L_max(k) for k in {2,4,8,16} on M held-out drawer
episodes, pick k* = argmax_k L_max(k). Kill Criterion A: if
max_k L_max(k) < 10 letters, stop.

One rollout at the model's native step granularity (FRAMESKIP=2 raw frames)
using the REAL recorded actions serves every k in the sweep -- the
trajectory doesn't depend on k (no alphabet is used here, only real
actions); k only changes which positions along that one trace are scored
for agreement. See lewm_g.rollout_batch's docstring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.droid_streaming import stream_many_windows
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import (
    ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM,
    encode_action_embeddings_batch, encode_pixel_windows_batch, rollout_batch,
)

M_HELD_OUT = 200
MAX_RAW_HORIZON = 96   # raw frames beyond the initial history window
K_SWEEP = [2, 4, 8, 16]
FIDELITY_THRESH = 0.85
N_PROVISIONAL_CLUSTERS = 12
SEED = 3072
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def largest_L_meeting_threshold(agreement_per_letter: np.ndarray, thresh: float) -> int:
    """Largest L (1-indexed count of letters) such that the mean agreement
    over the FIRST L letters >= thresh. 0 if even L=1 fails."""
    cummean = np.cumsum(agreement_per_letter) / (np.arange(len(agreement_per_letter)) + 1)
    ok = np.where(cummean >= thresh)[0]
    return int(ok.max()) + 1 if len(ok) else 0


def main():
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    split = json.loads((OUT_DIR / "episode_split.json").read_text())

    # Redone against the canonical test-20% split (episode_split.json) --
    # the original run sampled from the whole family and 171/200 of those
    # episodes turned out to land in the train partition once the split
    # was frozen. L_max doesn't fit anything, so that wasn't a leakage bug
    # per se, but every held-out reference from here on uses this one file.
    pool = sorted(set(split["test_episode_ids"]) - reset_ids)
    rng = np.random.default_rng(SEED)
    held_out_ids = sorted(rng.choice(pool, size=min(M_HELD_OUT, len(pool)), replace=False).tolist())
    print(f"held-out sample (test-20% split): {len(held_out_ids)} episodes", flush=True)

    n_positions = MAX_RAW_HORIZON // FRAMESKIP + HISTORY_SIZE  # subsampled obs positions needed
    n_raw_actions_needed = n_positions * FRAMESKIP

    print("streaming real frames...", flush=True)
    windows = stream_many_windows(held_out_ids, num_frames=n_positions, frameskip=FRAMESKIP, max_workers=16)
    print(f"streamed {len(windows)}/{len(held_out_ids)} OK", flush=True)

    print("loading real actions...", flush=True)
    actions = load_actions_for_episodes(held_out_ids, max_frames=n_raw_actions_needed)

    usable_ids = sorted(
        eid for eid in windows
        if eid in actions and actions[eid].shape[0] >= n_raw_actions_needed
    )
    print(f"usable (frames+actions both OK): {len(usable_ids)}", flush=True)
    if len(usable_ids) < 50:
        raise RuntimeError(f"too few usable held-out episodes ({len(usable_ids)}) to measure L_max")

    pixel_windows = np.stack([windows[eid] for eid in usable_ids])          # (N, n_positions, h, w, 3)
    raw_actions = np.stack([actions[eid][:n_raw_actions_needed] for eid in usable_ids])  # (N, n_raw, 7)

    # M1's action-convention guardrail: convert+normalize via ActionPipeline
    # (droid_100-range conversion, then z-score), not a bare z-score fit --
    # ActionNormalizer.fit no longer exists post-refactor (see lewm_g.py).
    pipeline = ActionPipeline.fit(raw_actions.reshape(-1, RAW_ACTION_DIM))

    print("encoding real latent trace...", flush=True)
    real_trace = encode_pixel_windows_batch(pixel_windows)  # (N, n_positions, D)

    init_emb = real_trace[:, :HISTORY_SIZE]  # (N, HISTORY_SIZE, D)
    init_actions = raw_actions[:, :HISTORY_SIZE * FRAMESKIP].reshape(len(usable_ids), HISTORY_SIZE, FRAMESKIP, RAW_ACTION_DIM)
    init_act_emb = encode_action_embeddings_batch(init_actions, pipeline)  # (N, HISTORY_SIZE, D)

    n_model_steps_total = n_positions - HISTORY_SIZE
    future_actions = raw_actions[:, HISTORY_SIZE * FRAMESKIP:].reshape(
        len(usable_ids), n_model_steps_total, FRAMESKIP, RAW_ACTION_DIM
    )

    print(f"rolling out model trace ({n_model_steps_total} model steps)...", flush=True)
    model_trace = rollout_batch(init_emb, init_act_emb, future_actions, pipeline)  # (N, n_model_steps_total, D)
    real_future = real_trace[:, HISTORY_SIZE:HISTORY_SIZE + n_model_steps_total]      # (N, n_model_steps_total, D)

    # provisional predicate: k-means over ALL positions of the real trace
    # (explicitly a throwaway dynamics-measurement tool, per the plan's own
    # allowance -- NOT Phase C's real predicate discovery, and unlike Phase
    # C this is fit on raw z directly since it is discarded after this
    # measurement and never becomes part of the symbolic model itself)
    from sklearn.cluster import KMeans
    flat_real = real_trace.reshape(-1, real_trace.shape[-1]).numpy()
    km = KMeans(n_clusters=N_PROVISIONAL_CLUSTERS, random_state=SEED, n_init=10).fit(flat_real)

    labels_real = km.predict(real_future.reshape(-1, real_future.shape[-1]).numpy()).reshape(len(usable_ids), n_model_steps_total)
    labels_model = km.predict(model_trace.reshape(-1, model_trace.shape[-1]).numpy()).reshape(len(usable_ids), n_model_steps_total)
    agreement_per_model_step = (labels_real == labels_model).mean(axis=0)  # (n_model_steps_total,)

    results = {"k_sweep": {}, "n_usable_episodes": len(usable_ids), "n_model_steps_total": n_model_steps_total}
    for k in K_SWEEP:
        stride = k // FRAMESKIP
        max_letters = n_model_steps_total // stride
        letter_agreement = np.array([
            agreement_per_model_step[l * stride - 1] for l in range(1, max_letters + 1)
        ])
        L_max_k = largest_L_meeting_threshold(letter_agreement, FIDELITY_THRESH)
        results["k_sweep"][str(k)] = {
            "L_max_letters": L_max_k,
            "raw_horizon": L_max_k * k,
            "agreement_curve": letter_agreement.tolist(),
        }
        print(f"k={k:2d}: L_max={L_max_k} letters (raw horizon {L_max_k*k} frames)", flush=True)

    k_star = max(K_SWEEP, key=lambda k: results["k_sweep"][str(k)]["L_max_letters"])
    best_L_max = results["k_sweep"][str(k_star)]["L_max_letters"]
    results["k_star"] = k_star
    results["L_max_k_star"] = best_L_max
    results["kill_criterion_A_triggered"] = best_L_max < 10

    (OUT_DIR / "horizon_sweep.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v["L_max_letters"] for k, v in results["k_sweep"].items()}, indent=2))
    print(f"k* = {k_star}, L_max(k*) = {best_L_max}")
    print(f"KILL CRITERION A TRIGGERED: {results['kill_criterion_A_triggered']}")

    _plot(results)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in K_SWEEP:
        curve = results["k_sweep"][str(k)]["agreement_curve"]
        cummean = np.cumsum(curve) / (np.arange(len(curve)) + 1)
        letters = np.arange(1, len(curve) + 1)
        ax.plot(letters, cummean, marker="o", markersize=3, label=f"k={k} (L_max={results['k_sweep'][str(k)]['L_max_letters']})")
    ax.axhline(FIDELITY_THRESH, color="k", linestyle="--", label=f"threshold={FIDELITY_THRESH}")
    ax.set_xlabel("letters (L)")
    ax.set_ylabel("mean predicate-trace agreement over first L letters")
    ax.set_title("Phase A.4-A.5: horizon sweep (drawer_open_close family)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "horizon_sweep.png", dpi=140)
    print("saved", OUT_DIR / "horizon_sweep.png")


if __name__ == "__main__":
    main()
