"""Phase G.2 -- the relevance subspace V.

Plan §10 G.2 literally: "for each state pair (i,j), TTT gives a
distinguishing suffix e_ij. Compute v_ij(zeta) = grad_zeta phi(ghat(zeta,
e_ij)) by autograd through the g-rollout." Two adaptations, both
documented here rather than silently applied:

1. This project uses run_Lsharp, not TTT (PRE_EXPERIMENT_NOTES.md) -- the
   distinguishing suffixes are recomputed directly from the accepted
   Moore machine's own transition structure via partition refinement
   (same algorithm as gate/acceptance.py's max_distinguishing_suffix_length,
   here also returning the witness suffix WORD, not just its length).

2. phi (the predicate label function) is a set of HARD threshold/argmin
   indicator functions -- gradient is zero almost everywhere, so autograd
   through phi directly is useless. Each predicate family already computes
   a continuous score before its hard cutoff (slow_feature: the linear
   projection before thresholding; vq_code/bisim_cluster: negative distance
   to the assigned code/cluster center before the argmin). This module
   reuses those SAME saved parameters to build differentiable soft-score
   surrogates and takes v_ij(zeta) = grad_zeta [soft_score(ghat(zeta,e_ij))]
   for each active predicate -- the natural continuous relaxation of phi,
   not a different quantity.

Directions are collected across sampled (pair, seed-window) probes, then
PCA'd; r is chosen by 95% explained variance (plan's own selection rule).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.lewm_g import DEVICE, EMBED_DIM, HISTORY_SIZE, advance, encode_initial_window, get_model
from oracle.production_oracle import load_alphabet_symbols

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def compute_distinguishing_suffixes(machine, alphabet: list[str]) -> dict[tuple[int, int], tuple]:
    """Shortest witness suffix WORD per distinguishable state-id pair,
    keyed by (state_id_i, state_id_j). Uses AALpy's own
    `MooreMachine.find_distinguishing_seq` (validated BFS shipped with the
    library) rather than a hand-rolled partition refinement -- same
    algorithm gate/acceptance.py's max_distinguishing_suffix_length uses,
    here keeping the actual sequence (not just its length) since G.2 needs
    to roll out along it."""
    states = list(machine.states)
    n = len(states)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            seq = machine.find_distinguishing_seq(states[i], states[j], alphabet)
            if seq is not None:
                out[(states[i].state_id, states[j].state_id)] = tuple(seq)
    return out


def build_soft_score_fns(device=DEVICE) -> dict[str, callable]:
    """Differentiable surrogates for build_predicate_fns()'s hard
    predicates, reusing the exact same saved parameters -- see module
    docstring point 2. Each fn(z: (D,) tensor, grad-enabled) -> scalar
    tensor whose SIGN matches the hard predicate's bit."""
    from predicates.discover import PsiNet, VQBottleneck

    sf_params = json.loads((OUT_DIR / "predicate_functions_slowfeature.json").read_text())
    nets = torch.load(OUT_DIR / "predicate_functions_networks.pt", map_location=device, weights_only=False)

    fns = {}
    for name, p in sf_params.items():
        mean = torch.tensor(p["mean"], dtype=torch.float32, device=device)
        w = torch.tensor(p["w"], dtype=torch.float32, device=device)
        b = float(p["b"])
        fns[f"slow_feature__{name}"] = (lambda z, mean=mean, w=w, b=b: (z - mean) @ w - b)

    if nets["vq_surviving_codes"]:
        vq = VQBottleneck(nets["input_dim"], code_dim=nets["vq_code_dim"], K=nets["vq_K"]).to(device)
        vq.load_state_dict(nets["vq_model_state"])
        vq.eval()
        for k in nets["vq_surviving_codes"]:
            def make_vq(code_idx):
                def fn(z):
                    e = vq.enc(z.unsqueeze(0))[0]
                    return -((e - vq.codebook[code_idx]) ** 2).sum()
                return fn
            fns[f"vq_code__vq{k}"] = make_vq(k)

    if nets["bisim_surviving_clusters"]:
        psi = PsiNet(nets["input_dim"]).to(device)
        psi.load_state_dict(nets["bisim_psi_state"])
        psi.eval()
        centers = torch.tensor(np.asarray(nets["bisim_centers"]), dtype=torch.float32, device=device)
        for c in nets["bisim_surviving_clusters"]:
            def make_bisim(cluster_idx):
                def fn(z):
                    p = psi(z.unsqueeze(0))[0]
                    return -((p - centers[cluster_idx]) ** 2).sum()
                return fn
            fns[f"bisim_cluster__bc{c}"] = make_bisim(c)

    return fns


def differentiable_step(model, emb, act_emb, raw_segment_converted: np.ndarray, pipeline):
    """One model step with gradients flowing back to `emb`/`act_emb`
    (unlike oracle/production_oracle.py's `_step_fn_from_convention`, which
    wraps this in torch.no_grad() for inference). `raw_segment_converted`
    is a (FRAMESKIP,7) array already in droid_100 convention (alphabet
    medoids are stored that way) -- only the z-score normalizer stage
    applies, matching the production oracle's own convention handling.
    Uses the shared `advance()` (correct action-timing order: the new
    action is appended to `act_emb` BEFORE `predict` is called) -- see its
    docstring. For the single-letter (h=1) callers in this codebase the
    prior append-after-predict bug was benign (the passed-in `act_emb`'s
    last slot already held the real current action from the caller's own
    window construction), but for multi-letter suffixes (`len(suffix) > 1`
    in `compute_relevance_subspace`) each step after the first used a
    stale action, so this fix changes results for suffix length > 1."""
    normed = pipeline.normalizer(raw_segment_converted).reshape(1, -1)
    a = torch.from_numpy(normed).float().unsqueeze(0).to(emb.device)
    new_act_emb = model.action_encoder(a)[:, 0]
    emb, act_emb, _pred = advance(model, emb, act_emb, new_act_emb)
    return emb, act_emb


def compute_relevance_subspace(
    machine, subset: list[str], dataset, alphabet_path: Path,
    samples_per_state: int = 4, max_pairs: int = 200, seed: int = 3072,
) -> dict:
    """dataset: a training.data.FinetuneDataset (duck-typed here to avoid
    a signals<->training import cycle) -- its samples supply real
    (pixels, raw_action) windows tagged with the automaton state they were
    observed in, used as the seed zeta points for each state i."""
    model = get_model()
    device = DEVICE
    soft_fns = build_soft_score_fns(device)
    active_soft_fns = [soft_fns[n] for n in subset if n in soft_fns]
    if not active_soft_fns:
        print("  G.2: no differentiable surrogate available for any active predicate -- skipping", flush=True)
        return {"r": 0, "d": EMBED_DIM, "r_over_d": 0.0, "explained_variance_curve": [],
                "P_V": np.zeros((EMBED_DIM, EMBED_DIM)), "n_gradient_rows": 0, "per_pair": []}

    symbols_dict = load_alphabet_symbols(alphabet_path)
    full_action_alphabet = list(symbols_dict.keys())

    pair_suffixes = compute_distinguishing_suffixes(machine, full_action_alphabet)
    print(f"  G.2: {len(pair_suffixes)} distinguishable state pairs with finite suffixes "
          f"(machine has {machine.size} states)", flush=True)

    by_state: dict[int, list] = {}
    for s in dataset.samples:
        by_state.setdefault(s["state_id"], []).append(s)

    rng = np.random.default_rng(seed + 191)
    pair_items = list(pair_suffixes.items())
    if len(pair_items) > max_pairs:
        chosen_idx = rng.choice(len(pair_items), size=max_pairs, replace=False)
        pair_items = [pair_items[k] for k in chosen_idx]

    grad_rows = []
    per_pair_contrib = []
    for (i, j), suffix in pair_items:
        seeds_i = by_state.get(i, [])
        if not seeds_i:
            continue
        n_take = min(samples_per_state, len(seeds_i))
        chosen = [seeds_i[k] for k in rng.choice(len(seeds_i), size=n_take, replace=False)]
        n_rows_before = len(grad_rows)
        for samp in chosen:
            state = encode_initial_window(samp["pixels"][:HISTORY_SIZE], samp["raw_action"][:HISTORY_SIZE], dataset.pipeline)
            emb0 = state.emb.clone().to(device).unsqueeze(0)
            emb0.requires_grad_(True)
            act_emb0 = state.act_emb.clone().to(device).unsqueeze(0)

            emb, act_emb = emb0, act_emb0
            for sym in suffix:
                seg = symbols_dict[sym]
                emb, act_emb = differentiable_step(model, emb, act_emb, seg, dataset.pipeline)
            zeta_final = emb[0, -1]

            for fn in active_soft_fns:
                score = fn(zeta_final)
                grad = torch.autograd.grad(score, emb0, retain_graph=True, allow_unused=True)[0]
                if grad is None:
                    continue
                v = grad[0, -1].detach().cpu().numpy()
                if np.linalg.norm(v) > 1e-8:
                    grad_rows.append(v)
        # AALpy state_ids are strings ('s0','s1',...), not ints -- keep them as-is.
        per_pair_contrib.append({"i": str(i), "j": str(j), "suffix_len": len(suffix),
                                   "n_rows": len(grad_rows) - n_rows_before})

    d = EMBED_DIM
    if not grad_rows:
        print("  G.2: no usable gradient rows -- relevance subspace degenerate, r=0", flush=True)
        return {"r": 0, "d": d, "r_over_d": 0.0, "explained_variance_curve": [],
                "P_V": np.zeros((d, d)), "n_gradient_rows": 0, "per_pair": per_pair_contrib}

    G = np.stack(grad_rows)
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    _, S, Vt = np.linalg.svd(G, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    cumulative = np.cumsum(explained)
    r = int(np.searchsorted(cumulative, 0.95) + 1)
    r = max(1, min(r, len(S)))
    V_r = Vt[:r]
    P_V = V_r.T @ V_r

    print(f"  G.2: r={r}/{d} (r/d={r / d:.3f}), 95% variance from {len(grad_rows)} gradient directions "
          f"across {len(per_pair_contrib)} pairs", flush=True)
    if r / d > 0.9:
        print("  G.2 WARNING: r ~= d -- every direction matters, latent_subspace arm is vacuous "
              "(plan §10 G.2's explicit caveat). Report this, do not silently run a no-op mask.", flush=True)

    return {
        "r": r, "d": d, "r_over_d": r / d,
        "explained_variance_curve": explained.tolist(),
        "P_V": P_V, "n_gradient_rows": len(grad_rows), "per_pair": per_pair_contrib,
    }
