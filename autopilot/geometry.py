"""v6 SS4: working geometry. T (local tangent), V_proxy (true-start,
real-word score gradients along the frozen U_m prefix), W_proxy = T cap
V_proxy^perp. Substitutes FSM distinguishing-suffix gradients with
true-start local score-gradient banks, per the document's own explicit
"this substitution is explicit" note -- no FSM anywhere in this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.lewm_g import DEVICE, FRAMESKIP, RAW_ACTION_DIM
from signals.relevance_subspace import differentiable_step

PCA_ENERGY = 0.95


def score_grad_bank(model, state0, real_letters_converted: list[np.ndarray], pipeline,
                       U_m: np.ndarray) -> list[np.ndarray]:
    """One gradient v_i(z0) = grad_z0[u_i . ghat(z0, letters)] per direction
    i in 0..m-1, for the given real action letter(s) (already converted to
    droid_100 convention -- callers pass the ACTUAL word from the base
    point's own episode, never a canonical reset). Unit-normalised."""
    Um_t = torch.from_numpy(U_m).float().to(DEVICE)   # (m, D)
    emb0 = state0.emb.clone().to(DEVICE).unsqueeze(0)
    emb0.requires_grad_(True)
    act_emb0 = state0.act_emb.clone().to(DEVICE).unsqueeze(0)
    emb, act_emb = emb0, act_emb0
    for seg in real_letters_converted:
        emb, act_emb = differentiable_step(model, emb, act_emb, seg, pipeline)
    zeta_final = emb[0, -1]

    rows = []
    for i in range(U_m.shape[0]):
        score = zeta_final @ Um_t[i]
        grad = torch.autograd.grad(score, emb0, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            continue
        v = grad[0, -1].detach().cpu().numpy()
        n = np.linalg.norm(v)
        if n > 1e-8:
            rows.append(v / n)
    return rows


def local_tangent(base_z: np.ndarray, real_latents: np.ndarray, k: int = 256, energy: float = PCA_ENERGY):
    d2 = ((real_latents - base_z[None, :]) ** 2).sum(axis=1)
    idx = np.argpartition(d2, min(k, len(d2) - 1))[:k]
    neigh = real_latents[idx]
    centre = neigh.mean(axis=0)
    dev = neigh - centre
    _, s, Vt = np.linalg.svd(dev, full_matrices=False)
    cum = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
    r = int(np.searchsorted(cum, energy) + 1)
    return Vt[:max(1, r)], neigh, idx


def compute_T_V_W(model, state0, base_z: np.ndarray, real_words_converted: list[list[np.ndarray]],
                     pipeline, U_m: np.ndarray, real_latents_pool: np.ndarray,
                     k_local: int = 256, energy: float = PCA_ENERGY, tol: float = 1e-6) -> dict:
    """real_words_converted: list of real action words (each a list of
    per-letter converted segments) from the base point's OWN episode --
    typically [[a_t]] for the one-letter family, optionally
    [[a_t],[a_t,a_t+1]] once the two-letter branch is authorised."""
    grad_rows = []
    for word in real_words_converted:
        grad_rows.extend(score_grad_bank(model, state0, word, pipeline, U_m))

    d = base_z.shape[0]
    if not grad_rows:
        V_basis = np.zeros((0, d))
    else:
        G = np.stack(grad_rows)
        _, s, Vt = np.linalg.svd(G, full_matrices=False)
        cum = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
        r = int(np.searchsorted(cum, energy) + 1)
        V_basis = Vt[:max(1, r)]

    T_basis, neigh, neigh_idx = local_tangent(base_z, real_latents_pool, k_local, energy)

    if V_basis.shape[0] == 0:
        W_basis = T_basis
    else:
        proj = T_basis - (T_basis @ V_basis.T) @ V_basis
        norms = np.linalg.norm(proj, axis=1)
        keep = norms > tol
        if keep.sum() == 0:
            W_basis = np.zeros((0, d))
        else:
            Q, _ = np.linalg.qr(proj[keep].T)
            W_basis = Q.T[: keep.sum()]

    return {"dim_T": int(T_basis.shape[0]), "dim_V": int(V_basis.shape[0]), "dim_W": int(W_basis.shape[0]),
              "W_ratio": float(W_basis.shape[0] / max(1, T_basis.shape[0])),
              "T_basis": T_basis, "V_basis": V_basis, "W_basis": W_basis,
              "neigh_idx": neigh_idx, "neigh_z": neigh, "base_z": base_z}


def random_subspace_like(W_basis: np.ndarray, T_basis: np.ndarray, seed: int) -> np.ndarray:
    """`random_dir` arm: a subspace INSIDE T with the same dimensionality as
    W_basis, drawn uniformly at random -- the required equal-compute,
    dimension-matched control. Returns (dim_w, D) with orthonormal ROWS,
    matching compute_T_V_W's own basis convention (QR on the TALL D x dim_w
    matrix, then transpose -- not QR on the wide dim_w x D matrix, which
    would give a (dim_w, dim_w) Q instead of a (dim_w, D) one)."""
    rng = np.random.default_rng(seed)
    dim_w = W_basis.shape[0]
    dim_t = T_basis.shape[0]
    if dim_w == 0 or dim_t == 0:
        return np.zeros((0, T_basis.shape[1]))
    coeffs = rng.normal(size=(dim_t, dim_w))          # (dim_T, dim_w)
    raw = T_basis.T @ coeffs                            # (D, dim_T) @ (dim_T, dim_w) = (D, dim_w), tall
    Q, _ = np.linalg.qr(raw)                             # Q: (D, dim_w) orthonormal columns
    return Q.T[:dim_w]                                    # (dim_w, D) orthonormal rows
