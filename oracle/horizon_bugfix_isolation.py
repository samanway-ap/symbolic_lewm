"""One-shot diagnostic: isolate the ActionPipeline bugfix's effect on
L_max from the train/held-out generalization gap's effect, by re-running
L_max(2) and L_max(4) on the ORIGINAL (pre-split) M1 episode sample --
same 200-episode draw as the first, buggy run -- but with the FIXED
action-pipeline code. Not a new measurement methodology, just holding the
episode sample constant while only the bugfix varies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.droid_streaming import stream_many_windows
from oracle.droid_actions import load_actions_for_episodes
from oracle.horizon import FIDELITY_THRESH, HISTORY_SIZE, MAX_RAW_HORIZON, M_HELD_OUT, N_PROVISIONAL_CLUSTERS, SEED, largest_L_meeting_threshold
from oracle.lewm_g import ActionPipeline, FRAMESKIP, RAW_ACTION_DIM, encode_action_embeddings_batch, encode_pixel_windows_batch, rollout_batch

K_SWEEP_ISOLATION = [2, 4]
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    ids = json.loads((OUT_DIR / "scope_episode_ids.json").read_text())
    drawer_ids = set(ids["primary"]["episode_ids"])
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))

    # EXACT same pool/sample as the ORIGINAL (first, buggy) M1 run.
    pool = sorted(drawer_ids - reset_ids)
    rng = np.random.default_rng(SEED)
    held_out_ids = sorted(rng.choice(pool, size=min(M_HELD_OUT, len(pool)), replace=False).tolist())
    print(f"ORIGINAL in-sample episode draw: {len(held_out_ids)} episodes", flush=True)

    n_positions = MAX_RAW_HORIZON // FRAMESKIP + HISTORY_SIZE
    n_raw_actions_needed = n_positions * FRAMESKIP

    windows = stream_many_windows(held_out_ids, num_frames=n_positions, frameskip=FRAMESKIP, max_workers=16)
    actions = load_actions_for_episodes(held_out_ids, max_frames=n_raw_actions_needed)
    usable_ids = sorted(eid for eid in windows if eid in actions and actions[eid].shape[0] >= n_raw_actions_needed)
    print(f"usable: {len(usable_ids)}", flush=True)

    pixel_windows = np.stack([windows[eid] for eid in usable_ids])
    raw_actions = np.stack([actions[eid][:n_raw_actions_needed] for eid in usable_ids])
    pipeline = ActionPipeline.fit(raw_actions.reshape(-1, RAW_ACTION_DIM))  # THE FIX

    real_trace = encode_pixel_windows_batch(pixel_windows)
    init_emb = real_trace[:, :HISTORY_SIZE]
    init_actions = raw_actions[:, :HISTORY_SIZE * FRAMESKIP].reshape(len(usable_ids), HISTORY_SIZE, FRAMESKIP, RAW_ACTION_DIM)
    init_act_emb = encode_action_embeddings_batch(init_actions, pipeline)

    n_model_steps_total = n_positions - HISTORY_SIZE
    future_actions = raw_actions[:, HISTORY_SIZE * FRAMESKIP:].reshape(len(usable_ids), n_model_steps_total, FRAMESKIP, RAW_ACTION_DIM)
    model_trace = rollout_batch(init_emb, init_act_emb, future_actions, pipeline)
    real_future = real_trace[:, HISTORY_SIZE:HISTORY_SIZE + n_model_steps_total]

    from sklearn.cluster import KMeans
    flat_real = real_trace.reshape(-1, real_trace.shape[-1]).numpy()
    km = KMeans(n_clusters=N_PROVISIONAL_CLUSTERS, random_state=SEED, n_init=10).fit(flat_real)
    labels_real = km.predict(real_future.reshape(-1, real_future.shape[-1]).numpy()).reshape(len(usable_ids), n_model_steps_total)
    labels_model = km.predict(model_trace.reshape(-1, model_trace.shape[-1]).numpy()).reshape(len(usable_ids), n_model_steps_total)
    agreement = (labels_real == labels_model).mean(axis=0)

    out = {}
    for k in K_SWEEP_ISOLATION:
        stride = k // FRAMESKIP
        max_letters = n_model_steps_total // stride
        letter_agreement = np.array([agreement[l * stride - 1] for l in range(1, max_letters + 1)])
        L_max_k = largest_L_meeting_threshold(letter_agreement, FIDELITY_THRESH)
        out[str(k)] = L_max_k
        print(f"k={k}: L_max={L_max_k} (fixed code, ORIGINAL in-sample episodes)", flush=True)

    (OUT_DIR / "horizon_bugfix_isolation.json").write_text(json.dumps({
        "n_usable_episodes": len(usable_ids), "L_max_by_k": out,
        "original_buggy_L_max": {"2": 36, "4": 17},
        "heldout_fixed_L_max": {"2": 21, "4": 10},
    }, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
