"""Phase H -- the real fine-tune.

EXPLORATORY RUN (2026-08-21): the automaton driving these signals FAILED
Phase F (coverage 83.3% vs 90%; 3-seed reproducibility FAIL). That verdict
is unamended. See preregistration.md, "Phase G/H/I EXPLORATORY RUN", for
the full disclosure and for what is and is not being claimed here.

Arms: control, sample_prio + its shuffled twin + `time_binned_sample_prio`
(the STEP 2 temporal control, training/time_binned.py), and latent_subspace
+ its shuffled twin (kept because STEP 1 measured r/d=0.083 <= 0.8, so the
subspace mask is not vacuous). The exact list is read from
artifacts/arm_decision.json rather than hardcoded. input_patch + its
shuffled twin remain cut per the earlier budget contingency -- input_patch
needs a maskable-input JEPA mask sampler and integrated-gradients patch
attribution that do not exist anywhere in this codebase or le-wm's encoder
today; building that from scratch is out of scope. See
preregistration.md / results.md for the disclosure.

Anti-feedback protocol (plan §11.1), how each requirement is met:
  - frozen theta_0, frozen automaton: `best_machine.pkl` and `theta0_state`
    are loaded/copied once, before any arm trains, and never touched again.
  - identical data, identical batch order per seed, identical step count:
    all arms iterate the SAME FinetuneDataset with a `torch.Generator`
    re-seeded to the SAME `seed` value at the start of each arm's run (a
    fresh generator per call, not a shared stateful one), so
    `torch.randperm` produces bit-identical epoch permutations across arms
    for a given seed. n_batches_per_epoch * N_EPOCHS is a shared constant.
  - identical optimiser: fresh AdamW, same lr/weight_decay, every run.
  - SIGReg(Z) unmodified: the exact `module.SIGReg` class from le-wm, same
    weight/kwargs as theta_0's own training config
    (le-wm/config/train/lewm.yaml), added to every arm's loss identically.

NOTE on `training/masked_trainer.py` + `training/arms/*.py`: those classes
are the model-agnostic ISOLATION-PROPERTY test scaffold (a dummy nn.Module
standing in for JEPA, exercised by tests/test_arm_isolation.py in the
pre-experiment phase) -- their `compute_loss(batch, signals)` contract
assumes `self.model(batch["x"]) -> pred` directly, which doesn't match
real JEPA (`predict(ctx_emb, ctx_act)` needs an already-encoded context
window, not a single tensor call). This file reimplements the SAME loss
formulas (control / latent_subspace / sample_prio) directly against real
`pred_emb`/`tgt_emb` tensors from `model.encode`+`model.predict`, rather
than routing through those classes -- same math, real shapes.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.lewm_g import (
    ActionPipeline, DEVICE, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM,
    _img_transform, get_model,
)
from module import SIGReg  # le-wm/module.py; le-wm dir is on sys.path via oracle.lewm_g
from training.time_binned import build_time_binned_signals

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CKPT_DIR = OUT_DIR / "checkpoints"

N_EPOCHS = 5
BATCH_SIZE = 32
LR = 5e-5
WEIGHT_DECAY = 1e-3
SIGREG_WEIGHT = 0.09
SIGREG_KWARGS = {"knots": 17, "num_proj": 1024}
SEEDS = [0, 1, 2]
# The arm list is NOT hardcoded here: it is read from artifacts/arm_decision.json,
# written by signals/run_relevance.py when it applied the pre-registered r/d rule
# BEFORE any training run (preregistration.md, STEP 1). Falls back to the full
# list only if that artifact is missing, which would mean STEP 1 never ran.
DEFAULT_ARM_NAMES = ["control", "sample_prio", "shuffled_sample_prio", "time_binned_sample_prio",
                     "latent_subspace", "shuffled_latent_subspace"]
ANNEAL_START, ANNEAL_END = 1.0, 0.3


def load_arm_names() -> list[str]:
    path = OUT_DIR / "arm_decision.json"
    if not path.exists():
        print("WARNING: arm_decision.json missing -- STEP 1 (r/d check) has not run", flush=True)
        return DEFAULT_ARM_NAMES
    dec = json.loads(path.read_text())
    print(f"arm list from STEP 1 decision: r/d={dec['r_over_d']:.4f} "
          f"(threshold {dec['threshold']}), latent_subspace_vacuous={dec['latent_subspace_vacuous']}",
          flush=True)
    return dec["arms"]


def cosine_anneal(step: int, total_steps: int, start: float = ANNEAL_START, end: float = ANNEAL_END) -> float:
    """1.0 -> 0.3 cosine schedule over training, shared verbatim by every
    real arm and its shuffled twin (same magnitude/sparsity trajectory,
    only the semantic content of what's being masked differs)."""
    if total_steps <= 1:
        return end
    frac = min(1.0, step / (total_steps - 1))
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * frac))


def per_sample_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean(dim=-1)  # (B,T)


def control_loss(pred_emb, tgt_emb, **_):
    return per_sample_squared_error(pred_emb, tgt_emb).mean()


def latent_subspace_loss(pred_emb, tgt_emb, P_V, beta):
    if P_V is None:
        return per_sample_squared_error(pred_emb, tgt_emb).mean()
    err = pred_emb - tgt_emb                 # (B,T,D)
    proj = err @ P_V                          # P_V is a (D,D) symmetric projector onto span(V)
    orth = err - proj
    per = proj.pow(2).mean(dim=-1) + beta * orth.pow(2).mean(dim=-1)
    return per.mean()


def sample_prio_loss(pred_emb, tgt_emb, edge_rarity, failure_flag, kappa):
    per = per_sample_squared_error(pred_emb, tgt_emb)              # (B,T)
    weight = edge_rarity.unsqueeze(1) * (1.0 + kappa * failure_flag.unsqueeze(1))  # (B,1), broadcasts over T
    return (per * weight).mean()


def _collate(samples: list[dict], pipeline: "ActionPipeline", device):
    tfm = _img_transform()
    B, T = len(samples), HISTORY_SIZE + 1
    pix = np.stack([s["pixels"] for s in samples])          # (B,T,H,W,3)
    px = torch.from_numpy(pix).float() / 255.0
    px = px.permute(0, 1, 4, 2, 3)                            # (B,T,3,H,W)
    px = px.reshape(B * T, *px.shape[2:])
    px = tfm({"pixels": px})["pixels"]
    _, oh, ow = px.shape[-3:]
    px = px.reshape(B, T, 3, oh, ow).to(device)

    raw = np.stack([s["raw_action"] for s in samples])       # (B,T,FRAMESKIP,7)
    conv = pipeline(raw.reshape(-1, RAW_ACTION_DIM)).reshape(B, T, FRAMESKIP * RAW_ACTION_DIM)
    act = torch.from_numpy(conv).float().to(device)
    return {"pixels": px, "action": act}


def build_arm_signal_configs(dataset, relevance_result: dict, shuffle_seed: int = 3072) -> dict:
    d = relevance_result["d"]
    r = relevance_result["r"]
    P_V_real = torch.tensor(relevance_result["P_V"], dtype=torch.float32, device=DEVICE) if r > 0 else None

    if r > 0:
        rng = np.random.default_rng(shuffle_seed + 302)
        G = rng.standard_normal((d, r))
        Q, _ = np.linalg.qr(G)
        P_V_shuf = torch.tensor(Q @ Q.T, dtype=torch.float32, device=DEVICE)
    else:
        P_V_shuf = None

    n = len(dataset.samples)
    rng2 = np.random.default_rng(shuffle_seed + 301)
    perm = rng2.permutation(n)
    real_rarity = np.array([s["edge_rarity"] for s in dataset.samples], dtype=np.float32)
    real_flag = np.array([s["abstraction_failure_flag"] for s in dataset.samples], dtype=np.float32)

    # STEP 2 control arm: same weight multiset, conditioned on episode phase
    # rather than on automaton state (training/time_binned.py).
    tb = build_time_binned_signals(dataset.samples, real_rarity, real_flag, seed=shuffle_seed)
    (OUT_DIR / "time_binned_signal_diagnostics.json").write_text(json.dumps(tb["diagnostics"], indent=2))
    print(f"  time_binned control: bins={tb['diagnostics']['bin_counts']} "
          f"rarity-multiset-identical={tb['diagnostics']['rarity_multiset_identical_to_real']} "
          f"NMI(state,timebin)={tb['diagnostics']['normalized_mutual_information_state_vs_timebin']:.3f}",
          flush=True)

    return {
        "P_V_real": P_V_real, "P_V_shuf": P_V_shuf,
        "real_rarity": real_rarity, "real_flag": real_flag,
        "shuf_rarity": real_rarity[perm], "shuf_flag": real_flag[perm],
        "tb_rarity": tb["rarity"], "tb_flag": tb["flag"],
    }


def run_arm(arm_name: str, seed: int, dataset, signal_cfg: dict, theta0_state: dict) -> dict:
    model = get_model()
    model.load_state_dict(theta0_state)  # reset to frozen theta_0 before EVERY run
    model.requires_grad_(True)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sigreg = SIGReg(**SIGREG_KWARGS).to(DEVICE)

    n = len(dataset.samples)
    n_batches = max(1, n // BATCH_SIZE)
    total_steps = n_batches * N_EPOCHS
    gen = torch.Generator().manual_seed(seed)  # fresh generator, same seed -> identical order across arms

    is_shuffled = arm_name.startswith("shuffled_")
    if arm_name == "shuffled_sample_prio":
        rarity, flag = signal_cfg["shuf_rarity"], signal_cfg["shuf_flag"]
    elif arm_name == "time_binned_sample_prio":
        rarity, flag = signal_cfg["tb_rarity"], signal_cfg["tb_flag"]
    else:
        rarity, flag = signal_cfg["real_rarity"], signal_cfg["real_flag"]
    if "latent_subspace" in arm_name:
        P_V = signal_cfg["P_V_shuf"] if is_shuffled else signal_cfg["P_V_real"]
    else:
        P_V = None

    step = 0
    loss_history = []
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n, generator=gen).tolist()
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch_samples = [dataset.samples[i] for i in idx]
            batch = _collate(batch_samples, dataset.pipeline, DEVICE)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

            info = model.encode(batch)
            emb, act_emb = info["emb"], info["act_emb"]
            ctx_emb, ctx_act = emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE]
            tgt_emb = emb[:, 1:]
            pred_emb = model.predict(ctx_emb, ctx_act)

            if "latent_subspace" in arm_name:
                beta = cosine_anneal(step, total_steps)
                pred_loss = latent_subspace_loss(pred_emb, tgt_emb, P_V, beta)
            elif "sample_prio" in arm_name:
                kappa = cosine_anneal(step, total_steps)
                r_b = torch.tensor(rarity[idx], dtype=torch.float32, device=DEVICE)
                f_b = torch.tensor(flag[idx], dtype=torch.float32, device=DEVICE)
                pred_loss = sample_prio_loss(pred_emb, tgt_emb, r_b, f_b, kappa)
            else:
                pred_loss = control_loss(pred_emb, tgt_emb)

            sigreg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + SIGREG_WEIGHT * sigreg_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_history.append({"step": step, "pred_loss": float(pred_loss.item()),
                                   "sigreg_loss": float(sigreg_loss.item())})
            step += 1
        print(f"  [{arm_name} seed={seed}] epoch {epoch + 1}/{N_EPOCHS} "
              f"pred_loss={loss_history[-1]['pred_loss']:.5f} sigreg_loss={loss_history[-1]['sigreg_loss']:.5f}",
              flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{arm_name}_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return {"arm": arm_name, "seed": seed, "n_steps": step,
            "final_pred_loss": loss_history[-1]["pred_loss"],
            "final_sigreg_loss": loss_history[-1]["sigreg_loss"],
            "loss_history": loss_history, "checkpoint": str(ckpt_path)}


def main():
    import pickle
    from training.data import build_finetune_dataset
    from signals.relevance_subspace import compute_relevance_subspace

    with open(OUT_DIR / "best_machine.pkl", "rb") as f:
        best = pickle.load(f)
    machine, subset = best["machine"], best["subset"]
    print(f"Phase H: fine-tuning from accepted machine |S|={len(subset)} n_states={machine.size}", flush=True)

    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    reset_ids = set(json.loads((OUT_DIR / "reset_episode_ids.json").read_text()))
    train_pool = sorted(set(split["train_episode_ids"]) - reset_ids)
    alphabet_path = OUT_DIR / "alphabet_trainfit.json"

    print("building Phase H fine-tune dataset (also serves G.1/G.3/G.4)...", flush=True)
    dataset = build_finetune_dataset(machine, subset, train_pool, alphabet_path,
                                       cache_path=OUT_DIR / "finetune_stream_cache.npz")

    # Phase G.2 already ran in STEP 1 (signals/run_relevance.py), on the SAME
    # cached dataset -- reload its saved result rather than recomputing it, so
    # the subspace the arms train against is byte-identical to the one the
    # pre-registered r/d decision was made on.
    rel_path = OUT_DIR / "relevance_subspace.npz"
    if rel_path.exists():
        blob = np.load(rel_path)
        summary = json.loads((OUT_DIR / "relevance_subspace_summary.json").read_text())
        relevance = {"P_V": blob["P_V"], "r": summary["r"], "d": summary["d"],
                       "r_over_d": summary["r_over_d"]}
        print(f"  reusing STEP 1 relevance subspace: r={relevance['r']}/{relevance['d']} "
              f"(r/d={relevance['r_over_d']:.4f})", flush=True)
    else:
        print("computing Phase G.2 relevance subspace...", flush=True)
        relevance = compute_relevance_subspace(machine, subset, dataset, alphabet_path)
        np.savez(rel_path, P_V=relevance["P_V"],
                  explained_variance_curve=np.array(relevance["explained_variance_curve"]))
        (OUT_DIR / "relevance_subspace_summary.json").write_text(json.dumps(
            {k: v for k, v in relevance.items() if k != "P_V"}, indent=2))

    signal_cfg = build_arm_signal_configs(dataset, relevance)

    model = get_model()
    theta0_state = copy.deepcopy(model.state_dict())

    arm_names = load_arm_names()
    print(f"ARMS ({len(arm_names)}): {arm_names}", flush=True)

    all_runs = []
    for seed in SEEDS:
        for arm_name in arm_names:
            print(f"\n=== training arm='{arm_name}' seed={seed} ===", flush=True)
            result = run_arm(arm_name, seed, dataset, signal_cfg, theta0_state)
            all_runs.append({k: v for k, v in result.items() if k != "loss_history"})
            (OUT_DIR / "training_runs.json").write_text(json.dumps(all_runs, indent=2))
            (OUT_DIR / f"loss_history_{arm_name}_seed{seed}.json").write_text(
                json.dumps(result["loss_history"]))

    print("\n=== Phase H complete ===", flush=True)
    print(f"wrote {len(all_runs)} checkpoints to {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    import os
    os._exit(0)
