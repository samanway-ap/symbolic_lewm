"""v6 SS4.2: train one arm/seed. Encoder frozen (`model.encoder.requires_
grad_(False)`, excluded from the optimizer's param group); predictor,
action_encoder, projector, pred_proj remain trainable ("only the predictor
moves" -- the encoder submodule specifically is what's frozen). Identical
optimizer/schedule/effective-batch across every arm in a node; only the
curriculum dataset differs, per v6 SS5.

Simplification, documented: trains on a plain per-step MSE control loss
(no SIGReg) rather than reusing training/finetune.py's SIGReg-regularized
recipe -- SIGReg shapes the EMBEDDING marginal, which is moot with a frozen
encoder, and the v6 hypothesis is about curriculum/data selection, not a
specialized loss shape. Evaluation (autopilot/evaluate.py) uses the
DIFFERENT, v6-mandated E_W projected-error metric -- training loss and
eval metric are intentionally not the same function.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.lewm_g import DEVICE, HISTORY_SIZE, _img_transform
from training.finetune import _collate

BATCH_SIZE = 32
LR = 5e-5
WEIGHT_DECAY = 1e-3


def freeze_encoder(model) -> None:
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.encoder.eval()


def unfreeze_trainable(model) -> list[torch.nn.Parameter]:
    params = []
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            params.append(p)
    return params


def run_arm(model, theta0_state: dict, dataset, n_updates: int, seed: int,
              ckpt_path: Path | None = None, oom_retry: bool = True) -> dict:
    model.load_state_dict(theta0_state)   # reset to the frozen node-start checkpoint
    model.train()
    freeze_encoder(model)
    params = unfreeze_trainable(model)

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)

    n = len(dataset.samples)
    batch_size = BATCH_SIZE
    encoder_state_before = copy.deepcopy({k: v for k, v in model.state_dict().items() if k.startswith("encoder.")})

    step = 0
    loss_history = []
    t0 = time.time()
    retried_once = False
    while step < n_updates:
        n_batches = max(1, n // batch_size)
        perm = torch.randperm(n, generator=gen).tolist()
        for b in range(n_batches):
            if step >= n_updates:
                break
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if not idx:
                continue
            batch_samples = [dataset.samples[i] for i in idx]
            try:
                batch = _collate(batch_samples, dataset.pipeline, DEVICE)
                batch["action"] = torch.nan_to_num(batch["action"], 0.0)

                info = model.encode(batch)
                emb, act_emb = info["emb"], info["act_emb"]
                ctx_emb, ctx_act = emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE]
                tgt_emb = emb[:, 1:]
                pred_emb = model.predict(ctx_emb, ctx_act)

                loss = (pred_emb - tgt_emb).pow(2).mean()
                if torch.isnan(loss):
                    raise FloatingPointError("NaN loss")

                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_history.append(float(loss.item()))
            except torch.cuda.OutOfMemoryError:
                if not oom_retry:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(4, batch_size // 2)
                print(f"    [OOM] halving batch size to {batch_size}, retrying", flush=True)
                oom_retry = False
                continue
            except FloatingPointError:
                if retried_once:
                    raise
                print("    [NaN] restoring node-start checkpoint, retrying once", flush=True)
                model.load_state_dict(theta0_state)
                freeze_encoder(model)
                retried_once = True
                continue
            step += 1

    encoder_state_after = {k: v for k, v in model.state_dict().items() if k.startswith("encoder.")}
    for k in encoder_state_before:
        if not torch.equal(encoder_state_before[k], encoder_state_after[k]):
            raise RuntimeError(f"IMPLEMENTATION_FAILURE: encoder tensor {k} changed despite freeze")

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    return {"n_steps": step, "n_samples": n, "batch_size": batch_size,
              "final_loss": loss_history[-1] if loss_history else None,
              "mean_loss_last10": float(np.mean(loss_history[-10:])) if loss_history else None,
              "wall_time_s": time.time() - t0, "checkpoint": str(ckpt_path) if ckpt_path else None}
