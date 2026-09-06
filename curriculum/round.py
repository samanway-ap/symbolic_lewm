"""Phase K -- one curriculum round for one cell.

Every round branches from the SAME frozen base checkpoint (not from the
previous round's weights) unless sequential accumulation is explicitly
requested and recorded, so rounds stay independent and a bad round cannot
silently poison every later one.

Three arms per round, all identical in optimizer steps, batch order and
step count for a given seed -- the ONLY difference is which trajectories
are mixed with the replay stream:
  targeted        replay + K trajectories retrieved for THIS cell
  control         replay only, no cell data
  matched_random  replay + K trajectories from a random UNEXPLORED cell

Three measurements per arm:
  LOCAL     plan-ranking MRR on held-out segments INSIDE the cell
  GLOBAL    plan-ranking MRR on the standard held-out set
  SYMBOLIC  abstraction-failure rate in the cell: the FROZEN machine's
            predicted next label vs phi(real next latent) on REAL held-out
            data. The machine is never re-extracted mid-loop -- re-extracting
            against the model you just trained is the self-confirming trap
            this whole protocol exists to avoid.

Plus a CONTRACTION CONTROL on LOCAL (pred_scale=0.1), because the primary
metric failed its own non-gameability check in the preceding run; if
contraction moves LOCAL as much as the intervention did, the round is
reported as confounded rather than as a result.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_import import load_local  # noqa: E402

# NOT `from eval.plan_ranking import ...` -- see local_import.py (le-wm/eval.py
# shadows this repo's eval package once oracle.lewm_g is imported).
_plan_ranking = load_local("symbolic_eval_plan_ranking", "eval/plan_ranking.py")
HORIZON_LETTERS = _plan_ranking.HORIZON_LETTERS
evaluate_plan_ranking = _plan_ranking.evaluate_plan_ranking
from oracle.lewm_g import DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, _img_transform, get_model
from tessellation.guard import assert_not_tessellation
from module import SIGReg  # le-wm

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CKPT_DIR = OUT_DIR / "curriculum_checkpoints"

LR = 5e-5
WEIGHT_DECAY = 1e-3
SIGREG_WEIGHT = 0.09
SIGREG_KWARGS = {"knots": 17, "num_proj": 1024}
BATCH_SIZE = 16
N_EPOCHS = 3
REPLAY_RATIO = 1.0     # 1:1 cell:replay by default; raised on a forgetting verdict


def _collate(samples, pipeline, device):
    tfm = _img_transform()
    B, T = len(samples), HISTORY_SIZE + 1
    pix = np.stack([s["pixels"] for s in samples])
    px = torch.from_numpy(pix).float() / 255.0
    px = px.permute(0, 1, 4, 2, 3).reshape(B * T, 3, pix.shape[2], pix.shape[3])
    px = tfm({"pixels": px})["pixels"]
    _, oh, ow = px.shape[-3:]
    px = px.reshape(B, T, 3, oh, ow).to(device)
    raw = np.stack([s["raw_action"] for s in samples])
    conv = pipeline(raw.reshape(-1, RAW_ACTION_DIM)).reshape(B, T, FRAMESKIP * RAW_ACTION_DIM)
    return {"pixels": px, "action": torch.from_numpy(conv).float().to(device)}


def build_mixture(cell_samples: list[dict], replay_samples: list[dict], ratio: float, seed: int):
    """Cell data + replay of the original mix. With ratio=1.0 the replay
    stream is matched 1:1 in sample count to the cell stream."""
    rng = np.random.default_rng(seed + 555)
    n_replay = int(round(len(cell_samples) * ratio)) if cell_samples else len(replay_samples)
    n_replay = min(n_replay, len(replay_samples))
    idx = rng.choice(len(replay_samples), size=n_replay, replace=False) if n_replay else []
    return list(cell_samples) + [replay_samples[i] for i in idx]


def finetune(samples: list[dict], pipeline, base_state: dict, seed: int,
               n_epochs: int = N_EPOCHS, batch_size: int = BATCH_SIZE) -> dict:
    model = get_model()
    model.load_state_dict(base_state)      # ALWAYS branch from the frozen base
    model.requires_grad_(True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sigreg = SIGReg(**SIGREG_KWARGS).to(DEVICE)

    n = len(samples)
    n_batches = max(1, n // batch_size)
    gen = torch.Generator().manual_seed(seed)   # same seed -> identical order across arms
    for epoch in range(n_epochs):
        perm = torch.randperm(n, generator=gen).tolist()
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if not idx:
                continue
            batch = _collate([samples[i] for i in idx], pipeline, DEVICE)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            info = model.encode(batch)
            emb, act_emb = info["emb"], info["act_emb"]
            pred = model.predict(emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE])
            loss = (pred - emb[:, 1:]).pow(2).mean() + SIGREG_WEIGHT * sigreg(emb.transpose(0, 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    state = copy.deepcopy(model.state_dict())
    model.eval()
    return state


def symbolic_failure_rate(model_state, machine, subset, local_episode_ids, test_ids_all) -> float:
    """Abstraction-failure rate INSIDE the cell: the FROZEN machine's predicted
    next label vs phi(real next latent), on REAL held-out data.

    Latents are RE-ENCODED under `model_state` rather than read from the
    cached theta_0 corpus -- the whole question is whether fine-tuning moved
    the model's own latents toward or away from the frozen symbolic model, and
    cached latents would return a bit-identical number for every arm.
    The machine is never re-extracted (anti-feedback protocol)."""
    import torch
    from gate.acceptance import K_STAR, load_alphabet_feats, quantize_to_symbols
    from oracle.lewm_g import ActionPipeline, encode_pixel_windows_batch
    from predicates.functions import build_predicate_fns
    _load_or_stream = _plan_ranking._load_or_stream

    assert_not_tessellation(subset, "curriculum.symbolic_failure_rate")
    all_fns = build_predicate_fns()
    fns = [all_fns[n] for n in subset]

    horizon = HORIZON_LETTERS
    n_positions = HISTORY_SIZE + horizon + 1
    n_raw = n_positions * FRAMESKIP
    usable, windows, actions = _load_or_stream(test_ids_all, horizon, n_positions, n_raw)
    eps = [e for e in usable if int(e) in local_episode_ids]
    if not eps:
        return float("nan")

    model = get_model()
    model.load_state_dict(model_state)
    model.eval()
    px = np.stack([windows[e] for e in eps])
    with torch.no_grad():
        Z = encode_pixel_windows_batch(px, batch_size=8).numpy()   # re-encoded, not cached

    all_raw = np.concatenate([actions[e][:n_raw] for e in eps], axis=0)
    pipeline = ActionPipeline.fit(all_raw)
    feats, names = load_alphabet_feats(
        Path(__file__).resolve().parents[1] / "artifacts" / "alphabet_trainfit.json")

    # A word must start with a RESET symbol or the machine sinks and this
    # returns 1.0 unconditionally -- see atlas/walk.py
    from atlas.walk import reset_reference, walk
    from oracle.production_oracle import build_reset_windows
    reset_names, reset_Z = reset_reference(build_reset_windows(pipeline))

    fails = total = 0
    for i, e in enumerate(eps):
        conv = pipeline.to_convention(actions[e][:n_raw].astype(np.float64))
        syms = quantize_to_symbols(conv, K_STAR, feats, names)
        f_, t_, _, _ = walk(machine, Z[i], syms, reset_names, reset_Z, fns)
        fails += f_
        total += t_
    return fails / total if total else float("nan")


def run_round(cell_bits: str, cell_id: int, seeds: list[int], base_state: dict,
                pipeline, cell_samples, random_samples, replay_samples,
                test_ids_all: list[int], local_episode_ids: set[int],
                machine=None, subset=None, corpus=None, cell_mask_by_ep=None,
                replay_ratio: float = REPLAY_RATIO, n_local_segments: int = 120,
                n_global_segments: int = 120) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    arms = {"targeted": cell_samples, "control": [], "matched_random": random_samples}
    per_arm = {a: {"local": [], "global": [], "symbolic": [], "local_contracted": []} for a in arms}

    for seed in seeds:
        for arm, extra in arms.items():
            mix = build_mixture(extra, replay_samples, replay_ratio, seed)
            print(f"    [{cell_bits}] arm={arm} seed={seed}: {len(mix)} samples "
                  f"({len(extra)} cell + {len(mix) - len(extra)} replay)", flush=True)
            state = finetune(mix, pipeline, base_state, seed)

            loc = evaluate_plan_ranking(state, test_ids_all, n_segments=n_local_segments,
                                          horizon=HORIZON_LETTERS, restrict_episodes=local_episode_ids)
            glo = evaluate_plan_ranking(state, test_ids_all, n_segments=n_global_segments,
                                          horizon=HORIZON_LETTERS)
            loc_c = evaluate_plan_ranking(state, test_ids_all, n_segments=n_local_segments,
                                            horizon=HORIZON_LETTERS, restrict_episodes=local_episode_ids,
                                            pred_scale=0.1)
            per_arm[arm]["local"].append(loc["mrr"])
            per_arm[arm]["global"].append(glo["mrr"])
            per_arm[arm]["local_contracted"].append(loc_c["mrr"])
            if machine is not None and subset is not None:
                per_arm[arm]["symbolic"].append(
                    symbolic_failure_rate(state, machine, subset, local_episode_ids, test_ids_all))
            print(f"      LOCAL={loc['mrr']:.5f} (n={loc['n_segments']}) GLOBAL={glo['mrr']:.5f} "
                  f"contracted_LOCAL={loc_c['mrr']:.5f}"
                  + (f" SYMBOLIC_fail={per_arm[arm]['symbolic'][-1]:.4f}" if per_arm[arm]["symbolic"] else ""),
                  flush=True)
            del state
    return per_arm
