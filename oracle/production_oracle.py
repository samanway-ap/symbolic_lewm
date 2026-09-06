"""Phase D wiring: real LatentOracle over the trained theta_0, the k*=2
alphabet (12 action symbols + 8 reset symbols), and a CHOSEN predicate
subset (Phase E searches over subsets of the Phase C pool).

Medoid segments (alphabet.json) and reset windows are built in droid_100
CONVENTION already (see M1's guardrail) -- so stepping through them uses
`pipeline.normalizer` only (already-converted input), NOT the full
`pipeline(...)` (which would double-convert). This mirrors
alphabet/validate_split.py's `rollout_final_latent`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_episode_frames
from oracle.latent_oracle import LatentOracle
from oracle.lewm_g import (
    ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, EMBED_DIM,
    LeWMWindowState, advance, encode_initial_window, get_model,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RESET_PREFIX = "r"
ACTION_PREFIX = "a"


def _step_fn_from_convention(pipeline: ActionPipeline):
    """Like lewm_g.step_fn_factory, but the raw_action_segment argument is
    ALREADY in droid_100 convention (alphabet medoids are stored that way)
    -- only the normalizer (z-score) stage is applied, not the full
    convert+normalize pipeline."""
    model = get_model()

    def step_fn(window: LeWMWindowState, action_segment_converted: np.ndarray) -> LeWMWindowState:
        n_raw = action_segment_converted.shape[0]
        n_model_steps = n_raw // FRAMESKIP
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb = window.emb.unsqueeze(0).to(device)
        act_emb = window.act_emb.unsqueeze(0).to(device)
        with torch.no_grad():
            for s in range(n_model_steps):
                chunk = action_segment_converted[s * FRAMESKIP:(s + 1) * FRAMESKIP]
                normed = pipeline.normalizer(chunk).reshape(1, -1)
                a = torch.from_numpy(normed).float().unsqueeze(0).to(device)
                new_act_emb = model.action_encoder(a)[:, 0]
                emb, act_emb, _pred = advance(model, emb, act_emb, new_act_emb)
        return LeWMWindowState(emb=emb[0].cpu(), act_emb=act_emb[0].cpu())

    return step_fn


def build_reset_windows(pipeline: ActionPipeline) -> dict[str, LeWMWindowState]:
    reset_ids = json.loads((OUT_DIR / "reset_episode_ids.json").read_text())
    n_raw_needed = HISTORY_SIZE * FRAMESKIP
    actions = load_actions_for_episodes(reset_ids, max_frames=n_raw_needed)
    resets = {}
    for i, eid in enumerate(reset_ids):
        frames = stream_episode_frames(eid, num_frames=HISTORY_SIZE, frameskip=FRAMESKIP)
        raw = actions[eid][:n_raw_needed].reshape(HISTORY_SIZE, FRAMESKIP, RAW_ACTION_DIM)
        state = encode_initial_window(frames, raw, pipeline)
        resets[f"{RESET_PREFIX}{i}"] = state
    return resets


def load_alphabet_symbols(path: Path):
    """Returns {symbol_name: (k, 7) medoid segment in droid_100 convention}."""
    alphabet = json.loads(path.read_text())
    return {f"{ACTION_PREFIX}{s['symbol']}": np.array(s["medoid_segment"]) for s in alphabet["symbols"]}


def build_production_oracle(pipeline: ActionPipeline, label_fn, alphabet_path: Path = None) -> LatentOracle:
    alphabet_path = alphabet_path or (OUT_DIR / "alphabet_trainfit.json")
    symbols = load_alphabet_symbols(alphabet_path)
    resets = build_reset_windows(pipeline)
    step_fn_raw = _step_fn_from_convention(pipeline)

    def step_fn(window: LeWMWindowState, symbol: str) -> LeWMWindowState:
        return step_fn_raw(window, symbols[symbol])

    return LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=resets)


ALPHABET_SYMBOLS = None  # filled lazily; alphabet symbol NAMES (not resets)


def alphabet_symbol_names(alphabet_path: Path = None) -> list[str]:
    alphabet_path = alphabet_path or (OUT_DIR / "alphabet_trainfit.json")
    return list(load_alphabet_symbols(alphabet_path).keys())


def reset_symbol_names() -> list[str]:
    reset_ids = json.loads((OUT_DIR / "reset_episode_ids.json").read_text())
    return [f"{RESET_PREFIX}{i}" for i in range(len(reset_ids))]
