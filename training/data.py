"""Phase G.1/G.3/G.4 + Phase H's per-sample side table, built in ONE real
pass over a fixed sample of scoped TRAIN episodes against the FROZEN
accepted automaton (anti-feedback protocol, plan §11.1 -- extracted once,
never re-extracted mid-run). This is deliberately the SAME pass that
produces the fine-tune training samples themselves: state_id (G.1),
edge_rarity (G.3) and abstraction_failure_flag (G.4) are attached directly
to each (pixels, action) training window, index-for-index, rather than
computed separately and joined later -- sample_prio/latent_subspace's
signals must be exactly aligned with what is actually trained on.

Episode sample is disjoint from Phase E's own fit/fidelity/proxy episode
samples (different rng offset) and never touches Phase F's canonical
test-20% split (`episode_ids` passed in must be TRAIN-split only).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate.acceptance import K_STAR, load_alphabet_feats, quantize_to_symbols
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, encode_pixel_windows_batch
from oracle.production_oracle import build_reset_windows
from predicates.functions import build_predicate_fns, make_label_fn
from signals.edge_stats import compute_edge_rarity

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass
class FinetuneDataset:
    samples: list[dict]           # see build_finetune_dataset docstring for keys
    rarity_table: dict[tuple, float]
    pipeline: "ActionPipeline"
    label_fn: callable
    subset: list[str]

    def __len__(self):
        return len(self.samples)


def build_finetune_dataset(
    machine, subset: list[str], episode_ids: list[int], alphabet_path: Path,
    n_episodes: int = 150, n_positions: int = 12, seed: int = 3072,
    cache_path: Path | None = None,
) -> FinetuneDataset:
    """Each sample dict:
      pixels: (HISTORY_SIZE+1, H, W, 3) uint8 -- context window + 1 target frame
      raw_action: (HISTORY_SIZE+1, FRAMESKIP, 7) float, action.original convention
      state_id: automaton state BEFORE this transition (G.1)
      symbol: the alphabet symbol this transition applies
      abstraction_failure_flag: int 0/1 (G.4)
      edge_rarity: float, filled in after the full visitation table is built (G.3)
      episode, pos, n_pos_in_episode: provenance; pos/n_pos_in_episode is the
        normalized timestep the `time_binned_sample_prio` control arm bins
    """
    all_fns = build_predicate_fns()
    fns = [all_fns[n] for n in subset]
    label_fn = make_label_fn(subset, all_fns)

    rng = np.random.default_rng(seed + 71)  # offset from Phase E's fit/fidelity/proxy samples
    sample_ids = sorted(rng.choice(
        episode_ids, size=min(n_episodes, len(episode_ids)), replace=False,
    ).tolist())

    n_raw_needed = n_positions * FRAMESKIP

    # The streaming + action-loading pass is the only expensive, network-bound
    # part of this function and is a pure function of (sample_ids, n_positions).
    # Phase G (relevance subspace) and Phase H (training) both need the SAME
    # dataset; caching guarantees they get bit-identical inputs rather than two
    # independent HTTP-range decodes that could differ if a fetch fails.
    cached = None
    if cache_path is not None and cache_path.exists():
        blob = np.load(cache_path, allow_pickle=False)
        c_usable = blob["usable"].tolist()
        # Read each member ONCE -- see eval/plan_ranking.py's _load_or_stream for
        # why indexing blob[...] inside the comprehension is quadratic in memory.
        w_all = blob["windows"]
        a_all = blob["actions"]
        cached = {
            "usable": c_usable,
            "windows": {int(e): w_all[k] for k, e in enumerate(c_usable)},
            "actions": {int(e): a_all[k] for k, e in enumerate(c_usable)},
        }
        print(f"  finetune dataset: loaded {len(c_usable)} episodes from cache {cache_path.name}", flush=True)

    if cached is None:
        actions = load_actions_for_episodes(sample_ids, max_frames=n_raw_needed)
        usable = [e for e in sample_ids if e in actions and actions[e].shape[0] >= n_raw_needed]
        windows = stream_many_windows(usable, num_frames=n_positions, frameskip=FRAMESKIP, max_workers=16)
        usable = sorted(set(usable) & set(windows.keys()))
        if cache_path is not None:
            np.savez(
                cache_path,
                usable=np.asarray(usable, dtype=np.int64),
                windows=np.stack([windows[e] for e in usable]),
                actions=np.stack([actions[e][:n_raw_needed] for e in usable]),
            )
            print(f"  finetune dataset: wrote cache {cache_path.name}", flush=True)
    else:
        usable = cached["usable"]
        windows = cached["windows"]
        actions = cached["actions"]

    print(f"  finetune dataset: {len(usable)} usable episodes", flush=True)

    all_raw = np.concatenate([actions[e][:n_raw_needed] for e in usable], axis=0)
    pipeline = ActionPipeline.fit(all_raw)
    reset_windows = build_reset_windows(pipeline)
    alphabet_feats, alphabet_names = load_alphabet_feats(alphabet_path)

    pixel_stack = np.stack([windows[e] for e in usable])  # (N, n_positions, H, W, 3)
    real_trace = encode_pixel_windows_batch(pixel_stack).numpy()  # only for real_labels/failure-flag

    reset_names = list(reset_windows.keys())
    reset_first_z = np.stack([reset_windows[r].emb[-1].numpy() for r in reset_names])

    samples = []
    visitation: dict[tuple, int] = {}
    for i, eid in enumerate(usable):
        first_z = real_trace[i, HISTORY_SIZE - 1]
        d = np.linalg.norm(reset_first_z - first_z[None, :], axis=1)
        reset_sym = reset_names[int(d.argmin())]

        conv = pipeline.to_convention(actions[eid][:n_raw_needed].astype(np.float64))
        future_conv = conv[HISTORY_SIZE * FRAMESKIP:]
        symbols = quantize_to_symbols(future_conv, K_STAR, alphabet_feats, alphabet_names)

        real_labels = [
            tuple(f(real_trace[i, pos]) for f in fns)
            for pos in range(HISTORY_SIZE, min(n_positions, HISTORY_SIZE + len(symbols)))
        ]

        machine.reset_to_initial()
        machine.step(reset_sym)
        state_id = machine.current_state.state_id

        raw_full = actions[eid][:n_raw_needed].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM)
        pix = windows[eid]

        n_cmp = min(len(symbols), len(real_labels))
        max_t = min(n_cmp, n_positions - HISTORY_SIZE - 1)
        for t in range(max_t):
            sym = symbols[t]
            out = machine.step(sym)
            next_state_id = machine.current_state.state_id
            is_failure = int(out != real_labels[t])
            visitation[(state_id, sym)] = visitation.get((state_id, sym), 0) + 1

            ctx_start = t  # window [t, t+HISTORY_SIZE] -> context [t,t+HISTORY_SIZE-1], target t+HISTORY_SIZE
            samples.append({
                "pixels": pix[ctx_start:ctx_start + HISTORY_SIZE + 1].copy(),
                "raw_action": raw_full[ctx_start:ctx_start + HISTORY_SIZE + 1].copy(),
                "state_id": state_id, "symbol": sym,
                "abstraction_failure_flag": is_failure,
                "episode": int(eid), "pos": t, "n_pos_in_episode": max_t,
            })
            state_id = next_state_id

    rarity_table = compute_edge_rarity(visitation)
    global_rarity = float(np.median(list(rarity_table.values()))) if rarity_table else 1.0
    for s in samples:
        s["edge_rarity"] = rarity_table.get((s["state_id"], s["symbol"]), global_rarity)

    print(f"  finetune dataset: {len(samples)} samples, {len(rarity_table)} distinct (state,symbol) keys, "
          f"{sum(s['abstraction_failure_flag'] for s in samples)}/{len(samples)} flagged as abstraction failures",
          flush=True)
    return FinetuneDataset(samples=samples, rarity_table=rarity_table, pipeline=pipeline,
                             label_fn=label_fn, subset=subset)
