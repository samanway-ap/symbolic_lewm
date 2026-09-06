"""Phase B.4-B.5: coverage + plan-ranking expressiveness (offline, no
environment -- v2 §5.5 replaces CEM-restricted-vs-unrestricted with MRR).

Kill Criterion B: restricted MRR < 60% of unrestricted MRR -> the alphabet
cannot express the task; an automaton over an inexpressive alphabet is a
model of nothing.

All segments/medoids are in droid_100 CONVENTION (never z-scored) per the
action-convention guardrail; z-scoring is added only where the pipeline is
called at the action_encoder boundary (inside lewm_g's rollout helpers).
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
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_episode_frames
from oracle.lewm_g import (
    ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM,
    encode_action_embeddings_batch, encode_pixel_windows_batch, get_model,
)

K_STAR = 4
H_LETTERS = 8            # horizon for B.5, safely under L_max(k*=4)=17
N_HELD_OUT = 150
N_NEGATIVES = 15          # 5 other-episode + 5 time-shifted + 5 noise-perturbed
SEED = 3072
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _target_raw_frame() -> int:
    return (HISTORY_SIZE - 1) * FRAMESKIP + H_LETTERS * K_STAR


def load_alphabet():
    alphabet = json.loads((OUT_DIR / "alphabet.json").read_text())
    feats, segs, syms = [], [], []
    for s in alphabet["symbols"]:
        seg = np.array(s["medoid_segment"])  # (k,7) droid_100 convention
        feats.append(segment_feature(seg))
        segs.append(seg)
        syms.append(s["symbol"])
    return alphabet, np.stack(feats), np.stack(segs), syms


def quantize_sequence(raw_seq_converted: np.ndarray, k: int, feats: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """raw_seq_converted: (n_raw, 7) droid_100 convention -> concatenation
    of nearest-medoid k-length segments, chunk by chunk."""
    n_chunks = raw_seq_converted.shape[0] // k
    out = []
    for c in range(n_chunks):
        chunk = raw_seq_converted[c * k:(c + 1) * k]
        f = segment_feature(chunk)
        idx = int(np.argmin(np.linalg.norm(feats - f, axis=1)))
        out.append(segs[idx])
    return np.concatenate(out, axis=0)


def _stream_window_and_target(episode_id, target_frame):
    window = stream_episode_frames(episode_id, num_frames=HISTORY_SIZE, frameskip=FRAMESKIP)
    target = stream_episode_frames(episode_id, start_frame=target_frame, num_frames=1)
    return window, target[0]


def rollout_final_latent(init_emb, init_act_emb, raw_action_seq_droid100, pipeline: ActionPipeline) -> torch.Tensor:
    """raw_action_seq_droid100: (n_raw, 7) already in droid_100 convention
    (already-quantized/perturbed/etc candidates) -- rolls model forward and
    returns the FINAL predicted latent, (D,)."""
    model = get_model()
    n_raw = raw_action_seq_droid100.shape[0]
    n_steps = n_raw // FRAMESKIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = init_emb.unsqueeze(0).to(device)
    act_emb = init_act_emb.unsqueeze(0).to(device)
    with torch.no_grad():
        for s in range(n_steps):
            chunk = raw_action_seq_droid100[s * FRAMESKIP:(s + 1) * FRAMESKIP]
            normed = pipeline.normalizer(chunk).reshape(1, -1)
            a = torch.from_numpy(normed).float().unsqueeze(0).to(device)
            new_act_emb = model.action_encoder(a)[:, 0]
            pred = model.predict(emb, act_emb)[:, -1:]
            emb = torch.cat([emb[:, 1:], pred], dim=1)
            act_emb = torch.cat([act_emb[:, 1:], new_act_emb.unsqueeze(1)], dim=1)
    return emb[0, -1].cpu()


def main():
    ids = json.loads((OUT_DIR / "scope_episode_ids.json").read_text())
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    drawer_ids = set(ids["primary"]["episode_ids"])
    alphabet, feats, segs, syms = load_alphabet()
    print(f"alphabet: {len(syms)} symbols", flush=True)

    pool = sorted(drawer_ids - reset_ids)
    rng = np.random.default_rng(SEED + 1)  # different seed from A.4's held-out sample
    target_frame = _target_raw_frame()
    n_raw_needed = target_frame + 1
    print(f"target raw frame: {target_frame} (H={H_LETTERS} letters, k*={K_STAR})", flush=True)

    held_out_ids = sorted(rng.choice(pool, size=min(N_HELD_OUT * 2, len(pool)), replace=False).tolist())
    actions_all = load_actions_for_episodes(held_out_ids, max_frames=n_raw_needed)
    usable_ids = [e for e in held_out_ids if e in actions_all and actions_all[e].shape[0] >= n_raw_needed]
    usable_ids = usable_ids[:N_HELD_OUT]
    print(f"usable held-out episodes (enough raw actions): {len(usable_ids)}", flush=True)

    print("streaming window+target frames...", flush=True)
    windows, targets = {}, {}
    with ThreadPoolExecutor(max_workers=16) as pool_exec:
        futs = {pool_exec.submit(_stream_window_and_target, eid, target_frame): eid for eid in usable_ids}
        for fut in as_completed(futs):
            eid = futs[fut]
            try:
                w, t = fut.result()
                windows[eid], targets[eid] = w, t
            except Exception as e:
                print(f"  episode {eid} stream failed: {e!r}", file=sys.stderr)

    final_ids = sorted(set(windows) & set(targets))
    print(f"final usable episodes: {len(final_ids)}", flush=True)
    if len(final_ids) < 30:
        raise RuntimeError("too few usable held-out episodes for B.4/B.5")

    all_raw = np.concatenate([actions_all[e][:n_raw_needed] for e in final_ids], axis=0)
    pipeline = ActionPipeline.fit(all_raw)

    pixel_windows = np.stack([windows[e] for e in final_ids])
    init_emb_all = encode_pixel_windows_batch(pixel_windows)  # (N,HISTORY_SIZE,D)
    init_action_windows = np.stack([
        actions_all[e][:HISTORY_SIZE * FRAMESKIP].reshape(HISTORY_SIZE, FRAMESKIP, RAW_ACTION_DIM)
        for e in final_ids
    ])
    init_act_emb_all = encode_action_embeddings_batch(init_action_windows, pipeline)

    target_pixels = np.stack([targets[e][None] for e in final_ids])  # (N,1,h,w,3)
    zg_all = encode_pixel_windows_batch(target_pixels)[:, 0]  # (N,D)

    n_raw_future = target_frame - (HISTORY_SIZE - 1) * FRAMESKIP  # raw frames from window-end to target
    assert n_raw_future == H_LETTERS * K_STAR

    # --- B.4 coverage ---
    print("B.4: coverage (endpoint latent error, true vs quantized)...", flush=True)
    endpoint_errors = []
    for i, eid in enumerate(final_ids):
        # init window covers raw frames 0..(HISTORY_SIZE-1)*FRAMESKIP; the
        # future action sequence starts right after that.
        start = (HISTORY_SIZE - 1) * FRAMESKIP
        raw_future = actions_all[eid][start:start + n_raw_future]
        conv_future = pipeline.to_convention(raw_future.astype(np.float64))
        quantized = quantize_sequence(conv_future, K_STAR, feats, segs)

        z_true = rollout_final_latent(init_emb_all[i], init_act_emb_all[i], conv_future, pipeline)
        z_quant = rollout_final_latent(init_emb_all[i], init_act_emb_all[i], quantized, pipeline)
        endpoint_errors.append(float(torch.norm(z_true - z_quant).item()))
    endpoint_errors = np.array(endpoint_errors)
    print(f"  endpoint error: mean={endpoint_errors.mean():.4f} median={np.median(endpoint_errors):.4f} "
          f"p90={np.percentile(endpoint_errors,90):.4f}", flush=True)

    # --- B.5 plan-ranking expressiveness ---
    print("B.5: plan-ranking MRR (unrestricted vs restricted)...", flush=True)
    rng2 = np.random.default_rng(SEED + 2)
    mrr_unrestricted, mrr_restricted = [], []

    for i, eid in enumerate(final_ids):
        start = (HISTORY_SIZE - 1) * FRAMESKIP
        raw_future = actions_all[eid][start:start + n_raw_future]
        conv_future = pipeline.to_convention(raw_future.astype(np.float64))

        # ---- unrestricted ----
        # Other-episode negatives ONLY (genuinely different trajectories).
        # An earlier version mixed in time-shifted-same-episode and small
        # (std=0.15) noise-perturbed negatives; both turned out to be near-
        # duplicates of the true sequence (same task, same workspace, or a
        # barely-perturbed copy), diluting the ranking signal enough that
        # the metric FAILED its own non-gameability check (a degenerate
        # 0.1x-collapsed predictor scored BETTER than the honest model --
        # see metric_sanity_check.py). Dropped in favour of N_NEGATIVES
        # other-episode negatives, matching the restricted side's candidate
        # count. Re-verified against metric_sanity_check.py after this fix.
        candidates_u = [conv_future]
        other_eids = [e for e in final_ids if e != eid]
        for oe in rng2.choice(other_eids, size=min(N_NEGATIVES, len(other_eids)), replace=False):
            raw_o = actions_all[oe][start:start + n_raw_future]
            candidates_u.append(pipeline.to_convention(raw_o.astype(np.float64)))

        dists_u = [float(torch.norm(rollout_final_latent(init_emb_all[i], init_act_emb_all[i], c, pipeline) - zg_all[i]).item())
                   for c in candidates_u]
        rank_u = 1 + sorted(dists_u).index(dists_u[0])  # position of the true (index 0) among sorted distances
        mrr_unrestricted.append(1.0 / rank_u)

        # ---- restricted (alphabet-expressible only) ----
        quantized_true = quantize_sequence(conv_future, K_STAR, feats, segs)
        n_chunks = n_raw_future // K_STAR
        candidates_r = [quantized_true]
        for _ in range(N_NEGATIVES):
            chunk_ids = rng2.integers(0, len(syms), size=n_chunks)
            seq = np.concatenate([segs[c] for c in chunk_ids], axis=0)
            candidates_r.append(seq)

        dists_r = [float(torch.norm(rollout_final_latent(init_emb_all[i], init_act_emb_all[i], c, pipeline) - zg_all[i]).item())
                   for c in candidates_r]
        rank_r = 1 + sorted(dists_r).index(dists_r[0])
        mrr_restricted.append(1.0 / rank_r)

    mrr_u = float(np.mean(mrr_unrestricted))
    mrr_r = float(np.mean(mrr_restricted))
    ratio = mrr_r / mrr_u if mrr_u > 0 else 0.0
    kill_b = ratio < 0.6

    results = {
        "n_held_out": len(final_ids),
        "H_letters": H_LETTERS,
        "k_star": K_STAR,
        "n_negatives": N_NEGATIVES,
        "coverage_endpoint_error": {
            "mean": float(endpoint_errors.mean()), "median": float(np.median(endpoint_errors)),
            "p90": float(np.percentile(endpoint_errors, 90)), "max": float(endpoint_errors.max()),
        },
        "mrr_unrestricted": mrr_u,
        "mrr_restricted": mrr_r,
        "restricted_over_unrestricted": ratio,
        "kill_criterion_B_triggered": kill_b,
    }
    (OUT_DIR / "alphabet_validation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"KILL CRITERION B TRIGGERED: {kill_b}")


if __name__ == "__main__":
    main()
