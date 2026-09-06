"""Phase B.4-B.5 + §16 non-gameability + bootstrap CIs, run for TWO
conditions through one shared code path (avoids the construction-mismatch
risk that produced a real bug the first time around -- see M2_report.md):

  - "insample": the ORIGINAL setup, kept for contrast -- codebook fit on
    the whole family, held-out episodes drawn from the whole family too
    (minus reset medoids). Not a real train/test split.
  - "heldout": codebook fit on the canonical TRAIN 80% only
    (alphabet_trainfit.json), held-out episodes drawn ONLY from the
    canonical TEST 20% (episode_split.json). This is the one Kill
    Criterion B is actually decided on.

Per-episode honest/degenerate MRR arrays are retained (not just the mean)
so a paired bootstrap CI on (honest - degenerate) can be reported for both
conditions, for both unrestricted and restricted -- a CI containing zero
means "passes non-gameability" is NOT supported by the point estimate alone.
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

K_STAR = 2  # corrected M1 value on the held-out split; see M1_report.md's correction section
H_LETTERS = 8
N_HELD_OUT = 150
N_NEGATIVES = 15
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _target_raw_frame() -> int:
    return (HISTORY_SIZE - 1) * FRAMESKIP + H_LETTERS * K_STAR


def load_alphabet(path: Path):
    alphabet = json.loads(path.read_text())
    feats, segs, syms = [], [], []
    for s in alphabet["symbols"]:
        seg = np.array(s["medoid_segment"])
        feats.append(segment_feature(seg))
        segs.append(seg)
        syms.append(s["symbol"])
    return alphabet, np.stack(feats), np.stack(segs), syms


def quantize_sequence(raw_seq_converted: np.ndarray, k: int, feats: np.ndarray, segs: np.ndarray) -> np.ndarray:
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
    """raw_action_seq_droid100 must already be in droid_100 convention --
    only z-scoring (pipeline.normalizer) is applied here."""
    model = get_model()
    n_steps = raw_action_seq_droid100.shape[0] // FRAMESKIP
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


def bootstrap_paired_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def run_condition(label: str, episode_pool: list[int], alphabet_path: Path, base_seed: int):
    print(f"=== condition: {label} (alphabet={alphabet_path.name}, pool={len(episode_pool)}) ===", flush=True)
    alphabet, feats, segs, syms = load_alphabet(alphabet_path)

    target_frame = _target_raw_frame()
    n_raw_needed = target_frame + 1
    rng = np.random.default_rng(base_seed + 1)

    held_out_ids = sorted(rng.choice(episode_pool, size=min(N_HELD_OUT * 2, len(episode_pool)), replace=False).tolist())
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
            except Exception as e:
                print(f"  episode {eid} stream failed: {e!r}", file=sys.stderr)

    final_ids = sorted(set(windows) & set(targets))
    print(f"  usable episodes: {len(final_ids)}", flush=True)
    if len(final_ids) < 30:
        raise RuntimeError(f"[{label}] too few usable held-out episodes")

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
    rng2 = np.random.default_rng(base_seed + 2)

    endpoint_errors = []
    honest_u, degenerate_u, honest_r, degenerate_r = [], [], [], []

    for i, eid in enumerate(final_ids):
        start = (HISTORY_SIZE - 1) * FRAMESKIP
        raw_future = actions_all[eid][start:start + n_raw_future]
        conv_future = pipeline.to_convention(raw_future.astype(np.float64))
        quantized_true = quantize_sequence(conv_future, K_STAR, feats, segs)

        # B.4 coverage
        z_true = rollout_final_latent(init_emb_all[i], init_act_emb_all[i], conv_future, pipeline)
        z_quant = rollout_final_latent(init_emb_all[i], init_act_emb_all[i], quantized_true, pipeline)
        endpoint_errors.append(float(torch.norm(z_true - z_quant).item()))

        # B.5 unrestricted: other-episode negatives only (see M2_report.md
        # for why time-shifted/noise-perturbed negatives were dropped)
        candidates_u = [conv_future]
        other_eids = [e for e in final_ids if e != eid]
        for oe in rng2.choice(other_eids, size=min(N_NEGATIVES, len(other_eids)), replace=False):
            raw_o = actions_all[oe][start:start + n_raw_future]
            candidates_u.append(pipeline.to_convention(raw_o.astype(np.float64)))
        lat_u = [rollout_final_latent(init_emb_all[i], init_act_emb_all[i], c, pipeline) for c in candidates_u]
        dists_u = [float(torch.norm(l - zg_all[i]).item()) for l in lat_u]
        honest_u.append(1.0 / (1 + sorted(dists_u).index(dists_u[0])))
        dists_u_deg = [float(torch.norm(l * 0.1 - zg_all[i]).item()) for l in lat_u]
        degenerate_u.append(1.0 / (1 + sorted(dists_u_deg).index(dists_u_deg[0])))

        # B.5 restricted
        n_chunks = n_raw_future // K_STAR
        candidates_r = [quantized_true]
        for _ in range(N_NEGATIVES):
            chunk_ids = rng2.integers(0, len(syms), size=n_chunks)
            candidates_r.append(np.concatenate([segs[c] for c in chunk_ids], axis=0))
        lat_r = [rollout_final_latent(init_emb_all[i], init_act_emb_all[i], c, pipeline) for c in candidates_r]
        dists_r = [float(torch.norm(l - zg_all[i]).item()) for l in lat_r]
        honest_r.append(1.0 / (1 + sorted(dists_r).index(dists_r[0])))
        dists_r_deg = [float(torch.norm(l * 0.1 - zg_all[i]).item()) for l in lat_r]
        degenerate_r.append(1.0 / (1 + sorted(dists_r_deg).index(dists_r_deg[0])))

    endpoint_errors = np.array(endpoint_errors)
    honest_u, degenerate_u = np.array(honest_u), np.array(degenerate_u)
    honest_r, degenerate_r = np.array(honest_r), np.array(degenerate_r)

    result = {
        "label": label, "n_episodes": len(final_ids),
        "coverage_endpoint_error": {
            "mean": float(endpoint_errors.mean()), "median": float(np.median(endpoint_errors)),
            "p90": float(np.percentile(endpoint_errors, 90)),
        },
        "mrr_unrestricted": float(honest_u.mean()),
        "mrr_restricted": float(honest_r.mean()),
        "restricted_over_unrestricted": float(honest_r.mean() / honest_u.mean()) if honest_u.mean() > 0 else None,
        "nongameability_unrestricted": bootstrap_paired_ci(honest_u - degenerate_u, seed=base_seed + 3),
        "nongameability_restricted": bootstrap_paired_ci(honest_r - degenerate_r, seed=base_seed + 4),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    ids = json.loads((OUT_DIR / "scope_episode_ids.json").read_text())
    drawer_ids = set(ids["primary"]["episode_ids"])
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    split = json.loads((OUT_DIR / "episode_split.json").read_text())

    insample_pool = sorted(drawer_ids - reset_ids)
    heldout_pool = sorted(set(split["test_episode_ids"]) - reset_ids)

    insample = run_condition("insample", insample_pool, OUT_DIR / "alphabet.json", base_seed=3072)
    heldout = run_condition("heldout", heldout_pool, OUT_DIR / "alphabet_trainfit.json", base_seed=3172)

    kill_b_heldout = heldout["restricted_over_unrestricted"] < 0.6

    out = {"insample": insample, "heldout": heldout, "kill_criterion_B_triggered": kill_b_heldout,
           "kill_criterion_B_basis": "heldout (canonical test-20% split, train-fit codebook)"}
    (OUT_DIR / "alphabet_validation_split.json").write_text(json.dumps(out, indent=2))
    print(f"\nKILL CRITERION B (on held-out numbers): {kill_b_heldout}")


if __name__ == "__main__":
    main()
