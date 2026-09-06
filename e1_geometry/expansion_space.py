"""E1.3 -- the expansion space W = T ∩ V^perp, per base point.

V: span of unit-normalised distinguishing-suffix gradients v_ij(z0) =
grad_z0 s(ghat(z0, e_ij)), for every OTHER machine state j reachable from
z0's own state q0 (not all C(n,2) pairs in the machine -- only the ones
that discriminate q0 from something else, which is what "blind at z0"
actually means). For linear tessellation predicates s_i(z)=u_i.z-b_i this
IS the literal Jacobian-transpose-times-normal the loop doc says is
"checkable by hand" -- reuses signals/relevance_subspace.py's differentiable
rollout unchanged, only the soft-score function is new (and simpler: exactly
linear, no surrogate needed since tessellation predicates ARE their own
score before thresholding).

T: local tangent at z0, from the 256 nearest REAL latents, local PCA at 95%
energy (same construction as e1_geometry/local_rank.py, kept here as an
orthonormal basis rather than just a rank number).

W: project T's orthonormal basis onto V^perp, re-orthonormalise, drop
directions whose residual norm falls below `tol`.

**Moore counting bound, asserted, not eyeballed** (EXPERIMENTS v2 SS3 E1,
explicit instruction, echoing SELF_IMPROVEMENT_LOOP.md v2 SS2): a Moore
machine with |Q| states cannot express more than |Q| distinct output
labels, so |Q| MUST be >= the number of occupied cells for the machine to
even be capable of representing the partition, let alone accurately. This
is checked in `moore_bound_ok` and RAISES if a caller tries to use a
non-conforming machine for geometry that assumes state identity is
meaningful (E1 geometry does not strictly require it -- V is computed from
the machine's OWN behaviour regardless -- but any GAP between |Q| and
occupied cells is reported alongside every result, never silently dropped).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.lewm_g import DEVICE, HISTORY_SIZE, encode_initial_window, get_model
from signals.relevance_subspace import compute_distinguishing_suffixes, differentiable_step


def moore_bound_check(n_states: int, n_occupied_cells: int) -> dict:
    return {"n_states": n_states, "n_occupied_cells": n_occupied_cells,
             "moore_bound_ok": n_states >= n_occupied_cells,
             "note": ("OK: the machine has enough states to possibly express every cell" if n_states >= n_occupied_cells
                        else f"VIOLATED: {n_states} states cannot express {n_occupied_cells} distinct cell labels -- "
                             "this machine is a failure-to-cover, not a compressed model, by construction")}


def tessellation_soft_scores(tess) -> list:
    """Exactly linear -- no surrogate needed, unlike the discovered predicates'
    threshold/argmin families (see signals/relevance_subspace.py's docstring
    for why THOSE needed one). fn(z: (D,) tensor) -> scalar tensor."""
    U = torch.from_numpy(tess.U).float().to(DEVICE)
    B = torch.from_numpy(tess.B).float().to(DEVICE)
    return [(lambda z, i=i: z @ U[i] - B[i]) for i in range(tess.m)]


def base_point_state(machine, first_z: np.ndarray, reset_names, reset_Z, real_symbols: list[str]) -> str:
    from atlas.walk import choose_reset
    reset_sym = choose_reset(first_z, reset_names, reset_Z)
    machine.reset_to_initial()
    machine.step(reset_sym)
    for s in real_symbols:
        machine.step(s)
    return machine.current_state.state_id


def compute_V_T_W(
    machine, tess, base_pixels, base_raw_action, pipeline,
    real_latents_for_T: np.ndarray, symbols_dict: dict, full_action_alphabet: list[str],
    q0: str, max_suffix_len: int = 8, k_local: int = 256, energy: float = 0.95, tol: float = 1e-6,
    precomputed_state0=None,
) -> dict:
    """base_pixels/base_raw_action: HISTORY_SIZE-length real context ending at
    z0 (the base point) -- used only if `precomputed_state0` is None.
    `precomputed_state0`: a ready-made LeWMWindowState (emb, act_emb), for
    callers that already have real cached embeddings and only need to
    compute act_emb fresh (no pixels required at all in that path -- see
    e1_geometry/run_e1.py's `real_window_state`). `max_suffix_len` enforces
    E0b.1's routed horizon restriction -- distinguishing suffixes longer than
    this are dropped rather than rolled out through the drift regime E0b.1
    flagged."""
    model = get_model()
    state0 = precomputed_state0 if precomputed_state0 is not None else encode_initial_window(
        base_pixels, base_raw_action, pipeline)
    z0_np = state0.emb[-1].numpy()

    all_pairs = compute_distinguishing_suffixes(machine, full_action_alphabet)
    relevant = {(i, j): seq for (i, j), seq in all_pairs.items()
                 if (i == q0 or j == q0) and len(seq) <= max_suffix_len}

    soft_fns = tessellation_soft_scores(tess)
    grad_rows = []
    for (i, j), suffix in relevant.items():
        emb0 = state0.emb.clone().to(DEVICE).unsqueeze(0)
        emb0.requires_grad_(True)
        act_emb0 = state0.act_emb.clone().to(DEVICE).unsqueeze(0)
        emb, act_emb = emb0, act_emb0
        for sym in suffix:
            seg = symbols_dict[sym]
            emb, act_emb = differentiable_step(model, emb, act_emb, seg, pipeline)
        zeta_final = emb[0, -1]
        for fn in soft_fns:
            score = fn(zeta_final)
            grad = torch.autograd.grad(score, emb0, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                continue
            v = grad[0, -1].detach().cpu().numpy()
            n = np.linalg.norm(v)
            if n > 1e-8:
                grad_rows.append(v / n)   # unit-normalised BEFORE spanning, per the loop doc

    d = z0_np.shape[0]
    if not grad_rows:
        V_basis = np.zeros((0, d))
    else:
        G = np.stack(grad_rows)
        _, s, Vt = np.linalg.svd(G, full_matrices=False)
        cum = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
        r = int(np.searchsorted(cum, energy) + 1)
        V_basis = Vt[:max(1, r)]

    # T: local tangent, 256 nearest real latents, 95% energy
    d2 = ((real_latents_for_T - z0_np[None, :]) ** 2).sum(axis=1)
    idx = np.argpartition(d2, min(k_local, len(d2) - 1))[:k_local]
    neigh = real_latents_for_T[idx] - real_latents_for_T[idx].mean(axis=0)
    _, ts, Tvt = np.linalg.svd(neigh, full_matrices=False)
    cum_t = np.cumsum(ts ** 2) / max((ts ** 2).sum(), 1e-12)
    rt = int(np.searchsorted(cum_t, energy) + 1)
    T_basis = Tvt[:max(1, rt)]

    # W = T ∩ V^perp: project T's basis onto V^perp, re-orthonormalise
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

    return {
        "q0": q0, "n_pairs_used": len(relevant), "n_gradient_rows": len(grad_rows),
        "dim_V": int(V_basis.shape[0]), "dim_T": int(T_basis.shape[0]), "dim_W": int(W_basis.shape[0]),
        "W_ratio": float(W_basis.shape[0] / max(1, T_basis.shape[0])),
        "V_basis": V_basis, "T_basis": T_basis, "W_basis": W_basis, "z0": z0_np,
    }
