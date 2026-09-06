"""§16 'Metric non-gameability', run against Phase B.5's plan-ranking MRR
before trusting Kill Criterion B's result: feed the metric a deliberately
DEGENERATE predictor (every candidate's predicted final latent replaced by
a near-constant point close to the origin, scaled 0.1x, independent of
which action sequence produced it) and check MRR does not improve over the
honest model. If it does, the honest metric is *worse than a lazy
constant predictor* at this task -- the metric itself is unreliable, not
merely "the alphabet is inexpressive".

Reuses the exact same held-out episodes/candidates as validate.py (same
seeds) so this is a direct, matched comparison, not a fresh random draw.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphabet.segments import segment_feature
from alphabet.validate import (
    H_LETTERS, K_STAR, N_HELD_OUT, N_NEGATIVES, SEED, _target_raw_frame,
    _stream_window_and_target, load_alphabet, quantize_sequence, rollout_final_latent,
)
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import (
    ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM,
    encode_action_embeddings_batch, encode_pixel_windows_batch,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def mrr_from_latents(true_idx: int, latents: list[torch.Tensor], zg: torch.Tensor) -> float:
    dists = [float(torch.norm(l - zg).item()) for l in latents]
    rank = 1 + sorted(dists).index(dists[true_idx])
    return 1.0 / rank


def main():
    ids = json.loads((OUT_DIR / "scope_episode_ids.json").read_text())
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    drawer_ids = set(ids["primary"]["episode_ids"])
    alphabet, feats, segs, syms = load_alphabet()

    pool = sorted(drawer_ids - reset_ids)
    rng = np.random.default_rng(SEED + 1)
    target_frame = _target_raw_frame()
    n_raw_needed = target_frame + 1

    held_out_ids = sorted(rng.choice(pool, size=min(N_HELD_OUT * 2, len(pool)), replace=False).tolist())
    actions_all = load_actions_for_episodes(held_out_ids, max_frames=n_raw_needed)
    usable_ids = [e for e in held_out_ids if e in actions_all and actions_all[e].shape[0] >= n_raw_needed][:N_HELD_OUT]

    windows, targets = {}, {}
    with ThreadPoolExecutor(max_workers=16) as pool_exec:
        futs = {pool_exec.submit(_stream_window_and_target, eid, target_frame): eid for eid in usable_ids}
        for fut in as_completed(futs):
            eid = futs[fut]
            try:
                w, t = fut.result()
                windows[eid], targets[eid] = w, t
            except Exception:
                pass

    final_ids = sorted(set(windows) & set(targets))
    print(f"episodes: {len(final_ids)}", flush=True)

    all_raw = np.concatenate([actions_all[e][:n_raw_needed] for e in final_ids], axis=0)
    pipeline = ActionPipeline.fit(all_raw)

    pixel_windows = np.stack([windows[e] for e in final_ids])
    init_emb_all = encode_pixel_windows_batch(pixel_windows)
    init_action_windows = np.stack([
        actions_all[e][:HISTORY_SIZE * FRAMESKIP].reshape(HISTORY_SIZE, FRAMESKIP, RAW_ACTION_DIM)
        for e in final_ids
    ])
    init_act_emb_all = encode_action_embeddings_batch(init_action_windows, pipeline)
    target_pixels = np.stack([targets[e][None] for e in final_ids])
    zg_all = encode_pixel_windows_batch(target_pixels)[:, 0]

    n_raw_future = target_frame - (HISTORY_SIZE - 1) * FRAMESKIP
    rng2 = np.random.default_rng(SEED + 2)

    honest_u, degenerate_u, honest_r, degenerate_r = [], [], [], []

    for i, eid in enumerate(final_ids):
        start = (HISTORY_SIZE - 1) * FRAMESKIP
        raw_future = actions_all[eid][start:start + n_raw_future]
        conv_future = pipeline.to_convention(raw_future.astype(np.float64))

        # ---- unrestricted candidates (identical construction to validate.py,
        # post-fix: other-episode negatives only, see validate.py's comment) ----
        candidates_u = [conv_future]
        other_eids = [e for e in final_ids if e != eid]
        for oe in rng2.choice(other_eids, size=min(N_NEGATIVES, len(other_eids)), replace=False):
            raw_o = actions_all[oe][start:start + n_raw_future]
            candidates_u.append(pipeline.to_convention(raw_o.astype(np.float64)))

        lat_u = [rollout_final_latent(init_emb_all[i], init_act_emb_all[i], c, pipeline) for c in candidates_u]
        honest_u.append(mrr_from_latents(0, lat_u, zg_all[i]))
        degenerate_lat_u = [l * 0.1 for l in lat_u]  # lazy predictor: everything collapses toward origin
        degenerate_u.append(mrr_from_latents(0, degenerate_lat_u, zg_all[i]))

        # ---- restricted candidates ----
        quantized_true = quantize_sequence(conv_future, K_STAR, feats, segs)
        n_chunks = n_raw_future // K_STAR
        candidates_r = [quantized_true]
        for _ in range(N_NEGATIVES):
            chunk_ids = rng2.integers(0, len(syms), size=n_chunks)
            candidates_r.append(np.concatenate([segs[c] for c in chunk_ids], axis=0))

        lat_r = [rollout_final_latent(init_emb_all[i], init_act_emb_all[i], c, pipeline) for c in candidates_r]
        honest_r.append(mrr_from_latents(0, lat_r, zg_all[i]))
        degenerate_lat_r = [l * 0.1 for l in lat_r]
        degenerate_r.append(mrr_from_latents(0, degenerate_lat_r, zg_all[i]))

    results = {
        "honest_mrr_unrestricted": float(np.mean(honest_u)),
        "degenerate_mrr_unrestricted": float(np.mean(degenerate_u)),
        "honest_mrr_restricted": float(np.mean(honest_r)),
        "degenerate_mrr_restricted": float(np.mean(degenerate_r)),
        "chance_mrr_16_candidates": float(sum(1.0 / r for r in range(1, 17)) / 16),
        "metric_broken_unrestricted": float(np.mean(degenerate_u)) > float(np.mean(honest_u)),
        "metric_broken_restricted": float(np.mean(degenerate_r)) > float(np.mean(honest_r)),
    }
    (OUT_DIR / "metric_sanity_check.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
