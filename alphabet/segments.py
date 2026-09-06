"""Phase B.1-B.2: temporally-extended action segments, gripper-split first.

Codebook construction (B.1-B.3) needs ONLY actions, not video -- the
drawer family's full action data is already local (M0.5's data/-only
LeRobotDataset pull), so this samples segments from ALL 3184 episodes
cheaply. Video streaming is reserved for B.4/B.5's held-out validation.

Every raw action here is in droid_100's CONVENTION (via
`ActionPipeline.to_convention`), never z-scored -- per the action-convention
guardrail in M1_report.md, the codebook/medoids must be human-facing and
droid_100-matched; z-scoring is added only at the action_encoder boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.lewm_g import ActionPipeline

GRIPPER_PATTERNS = ("stay_open", "stay_closed", "open_to_close", "close_to_open")


def classify_gripper_pattern(gripper_vals: np.ndarray, open_thresh: float = 0.5) -> str:
    """gripper_vals: (k,) in droid_100 convention (last raw dim, [0,1])."""
    is_open = gripper_vals > open_thresh
    start_open, end_open = bool(is_open[0]), bool(is_open[-1])
    if start_open and end_open:
        return "stay_open"
    if not start_open and not end_open:
        return "stay_closed"
    if start_open and not end_open:
        return "open_to_close"
    return "close_to_open"


def segment_feature(action_window: np.ndarray) -> np.ndarray:
    """action_window: (k, 7) in droid_100 convention. Feature = flattened
    sequence + per-dim mean + cumulative displacement (sum of the 6
    non-gripper dims, a velocity-command proxy for net motion)."""
    flat = action_window.reshape(-1)
    mean = action_window.mean(0)
    disp = action_window[:, :-1].sum(0)
    return np.concatenate([flat, mean, disp]).astype(np.float32)


def sample_segments(
    episode_actions: dict[int, np.ndarray],  # eid -> (T,7) raw action.original
    k: int,
    n_samples: int,
    pipeline: ActionPipeline,
    seed: int = 3072,
) -> dict[str, dict[str, np.ndarray]]:
    """Randomly sample (episode, start) windows of length k, convert to
    droid_100 convention, classify by gripper pattern, compute features.

    Returns {pattern: {"features": (n,D), "segments": (n,k,7) [droid_100
    convention], "episode_ids": (n,), "starts": (n,)}}.
    """
    rng = np.random.default_rng(seed)
    eids = np.array(list(episode_actions.keys()))
    lengths = np.array([episode_actions[e].shape[0] for e in eids])
    valid = lengths >= k
    eids, lengths = eids[valid], lengths[valid]

    # sample proportional to (length - k + 1), i.e. uniform over all valid
    # (episode, start) pairs across the family, not uniform over episodes
    n_starts = lengths - k + 1
    probs = n_starts / n_starts.sum()
    chosen_ep_idx = rng.choice(len(eids), size=n_samples, p=probs)
    chosen_eids = eids[chosen_ep_idx]
    chosen_starts = np.array([
        rng.integers(0, n_starts[i]) for i in chosen_ep_idx
    ])

    buckets: dict[str, list] = {p: [] for p in GRIPPER_PATTERNS}
    for eid, start in zip(chosen_eids, chosen_starts):
        raw = episode_actions[eid][start:start + k]  # (k,7) action.original
        conv = pipeline.to_convention(raw.astype(np.float64))
        pattern = classify_gripper_pattern(conv[:, -1])
        buckets[pattern].append((conv, segment_feature(conv), int(eid), int(start)))

    out = {}
    for pattern, items in buckets.items():
        if not items:
            out[pattern] = {"features": np.zeros((0, 0)), "segments": np.zeros((0, k, 7)),
                             "episode_ids": np.zeros(0, dtype=int), "starts": np.zeros(0, dtype=int)}
            continue
        segs, feats, eids_, starts_ = zip(*items)
        out[pattern] = {
            "features": np.stack(feats),
            "segments": np.stack(segs),
            "episode_ids": np.array(eids_),
            "starts": np.array(starts_),
        }
    return out
