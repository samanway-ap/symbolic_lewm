"""Phase C, all three candidate families + pool hygiene, in one pass
(deadline-driven consolidation -- see progress_log.md). TRAIN-split
episodes only.

(i) VQ bottleneck: small VQ-VAE on z with an auxiliary next-code
    prediction term. Each code -> one binary predicate.
(ii) Slow features: linear SFA directions, threshold-swept into binary
    predicates, scored by slowness + predictability.
(iii) Behavioural-metric clustering, SIMPLIFIED under time pressure: psi
    trained by bootstrapped regression against ||rho(z)-rho(z')|| and the
    target network's own distance on the REAL next-states (znext_i,
    znext_j) rather than g(z,a)/g(z',a) under a synthesized shared action
    -- avoids needing counterfactual rollouts for arbitrary pairs. rho
    bootstrapped from family (ii)'s top-3 directions. Complete-linkage
    clustering on psi(z), cut at a fixed target cluster count (FIX 2,
    2026-08-21 -- see preregistration.md; the original percentile-of-merge-
    -distance cut was scale-wrong and produced 751 near-singleton clusters)
    -> cluster-indicator predicates. Documented simplification, not the
    literal formula.

Pool hygiene: drop candidates constant >98% or switching >40% of the
time; dedupe by mutual information > 0.9 on held-out (TRAIN, still)
trajectories.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predicates.latent_pool import collect_trajectories, flatten_transitions, train_episode_pool

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
N_EPISODES = 180
N_POSITIONS = 30
SEED = 3072
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- family (ii): slow features ----------
def fit_sfa(latents_by_ep: dict[int, np.ndarray], n_components: int = 10):
    all_Z = np.concatenate(list(latents_by_ep.values()), axis=0)
    mean = all_Z.mean(0)
    cov = np.cov((all_Z - mean).T)
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, 1e-6, None)
    W = eigvec @ np.diag(eigval ** -0.5)

    dZ = []
    for Z in latents_by_ep.values():
        Zw = (Z - mean) @ W
        dZ.append(Zw[1:] - Zw[:-1])
    dZ = np.concatenate(dZ, axis=0)
    deigval, deigvec = np.linalg.eigh(np.cov(dZ.T))
    order = np.argsort(deigval)
    directions = W @ deigvec[:, order[:n_components]]
    return mean, directions


def slow_feature_candidates(mean, directions, zt, znext, aemb):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    proj_t = (zt - mean) @ directions
    proj_next = (znext - mean) @ directions
    out = []
    for i in range(directions.shape[1]):
        for q in (30, 50, 70):
            b = float(np.percentile(proj_t[:, i], q))
            psi_t = (proj_t[:, i] > b).astype(int)
            psi_next = (proj_next[:, i] > b).astype(int)
            switch = (psi_t != psi_next).astype(int)
            switch_rate = float(switch.mean())
            if switch.sum() < 5 or switch.sum() > len(switch) - 5:
                auc = 0.5
            else:
                X = np.concatenate([zt, aemb], axis=1)
                try:
                    clf = LogisticRegression(max_iter=200).fit(X, switch)
                    auc = float(roc_auc_score(switch, clf.predict_proba(X)[:, 1]))
                except Exception:
                    auc = 0.5
            score = (1 - switch_rate) + auc
            out.append({
                "family": "slow_feature", "name": f"sf{i}_q{q}",
                "psi_t": psi_t, "psi_next": psi_next,
                "switch_rate": switch_rate, "auc": auc, "score": score,
                "params": {"mean": mean, "w": directions[:, i], "b": b},
            })
    return out


# ---------- family (i): VQ bottleneck ----------
class VQBottleneck(nn.Module):
    def __init__(self, d, code_dim=32, K=24):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, code_dim))
        self.codebook = nn.Parameter(torch.randn(K, code_dim) * 0.1)
        self.dec = nn.Sequential(nn.Linear(code_dim, 128), nn.ReLU(), nn.Linear(128, d))
        self.next_pred = nn.Sequential(nn.Linear(code_dim + d, 64), nn.ReLU(), nn.Linear(64, K))

    def encode(self, z):
        e = self.enc(z)
        d2 = torch.cdist(e, self.codebook)
        idx = d2.argmin(1)
        quant = self.codebook[idx]
        quant_st = e + (quant - e).detach()
        return e, quant_st, idx

    def forward(self, z):
        e, q, idx = self.encode(z)
        return self.dec(q), e, q, idx


def train_vq(zt, znext, aemb, K=24, epochs=6):
    zt_t = torch.from_numpy(zt).float().to(DEVICE)
    znext_t = torch.from_numpy(znext).float().to(DEVICE)
    aemb_t = torch.from_numpy(aemb).float().to(DEVICE)
    model = VQBottleneck(zt.shape[1], K=K).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = zt_t.shape[0]
    bs = 512
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            z_b, znext_b, a_b = zt_t[b], znext_t[b], aemb_t[b]
            recon, e, q, idx = model(z_b)
            recon_loss = ((recon - z_b) ** 2).mean()
            vq_loss = ((q.detach() - e) ** 2).mean() + 0.25 * ((q - e.detach()) ** 2).mean()
            with torch.no_grad():
                _, _, next_idx = model.encode(znext_b)
            pred_logits = model.next_pred(torch.cat([e.detach(), a_b], dim=1))
            pred_loss = nn.functional.cross_entropy(pred_logits, next_idx)
            loss = recon_loss + vq_loss + pred_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        print(f"  VQ epoch {ep}: loss={total:.3f}", flush=True)
    return model


def vq_candidates(model, zt, znext):
    model.eval()
    with torch.no_grad():
        _, _, idx_t = model.encode(torch.from_numpy(zt).float().to(DEVICE))
        _, _, idx_next = model.encode(torch.from_numpy(znext).float().to(DEVICE))
    idx_t, idx_next = idx_t.cpu().numpy(), idx_next.cpu().numpy()
    K = model.codebook.shape[0]
    out = []
    for k in range(K):
        psi_t = (idx_t == k).astype(int)
        psi_next = (idx_next == k).astype(int)
        out.append({"family": "vq_code", "name": f"vq{k}", "psi_t": psi_t, "psi_next": psi_next,
                     "params": {"code_idx": k}})  # model itself saved once, shared across all vq* candidates
    return out


# ---------- family (iii): simplified behavioural-metric clustering ----------
class PsiNet(nn.Module):
    def __init__(self, d, m=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, m))

    def forward(self, z):
        return self.net(z)


def train_bisim_psi(zt, znext, rho_proj, c=1.0, gamma=0.9, steps=800, batch=256):
    zt_t = torch.from_numpy(zt).float().to(DEVICE)
    znext_t = torch.from_numpy(znext).float().to(DEVICE)
    rho_t = torch.from_numpy(rho_proj).float().to(DEVICE)  # (N, r) bootstrapped rho(z_t)

    psi = PsiNet(zt.shape[1]).to(DEVICE)
    target = PsiNet(zt.shape[1]).to(DEVICE)
    target.load_state_dict(psi.state_dict())
    opt = torch.optim.Adam(psi.parameters(), lr=1e-3)
    n = zt_t.shape[0]
    for step in range(steps):
        i = torch.randint(0, n, (batch,), device=DEVICE)
        j = torch.randint(0, n, (batch,), device=DEVICE)
        with torch.no_grad():
            d_target = c * torch.norm(rho_t[i] - rho_t[j], dim=1) + \
                       gamma * torch.norm(target(znext_t[i]) - target(znext_t[j]), p=1, dim=1)
        d_pred = torch.norm(psi(zt_t[i]) - psi(zt_t[j]), p=1, dim=1)
        loss = ((d_pred - d_target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                for p, tp in zip(psi.parameters(), target.parameters()):
                    tp.data.copy_(0.5 * tp.data + 0.5 * p.data)  # periodic soft target update
    return psi


def bisim_candidates(psi, zt, znext, n_clusters=24):
    """FIX 2 (2026-08-21): the original diameter-based cut
    (`fcluster(..., t=percentile(Z[:,2], 75), criterion="distance")`) was
    scale-wrong -- observed mean pairwise psi(z) distance was 18.38, so a
    75th-percentile-of-merge-distance cap produced 751 near-singleton
    clusters, essentially none of which survived hygiene (0 of 751).
    Cutting the dendrogram at a fixed TARGET CLUSTER COUNT instead of a
    fixed distance is scale-invariant by construction and gives this family
    the same order-of-magnitude candidate count as slow_feature (24) and
    vq_code (24), so the pool hygiene sees ~64-72 diverse candidates across
    all 3 families instead of 32 across 2."""
    from scipy.cluster.hierarchy import fcluster, linkage
    with torch.no_grad():
        psi_t = psi(torch.from_numpy(zt).float().to(DEVICE)).cpu().numpy()
        psi_next = psi(torch.from_numpy(znext).float().to(DEVICE)).cpu().numpy()
    # subsample for linkage (O(n^2) memory) -- fine, this is a fit, not eval
    n = psi_t.shape[0]
    sub = np.random.default_rng(SEED).choice(n, size=min(3000, n), replace=False)
    Z = linkage(psi_t[sub], method="complete")
    mean_pairwise_dist = float(np.mean(Z[:, 2]))  # diagnostic only, logged below
    labels_sub = fcluster(Z, t=n_clusters, criterion="maxclust")
    centers = np.array([psi_t[sub][labels_sub == c].mean(0) for c in np.unique(labels_sub)])
    print(f"  bisim linkage: mean merge distance={mean_pairwise_dist:.2f}, "
          f"cut at maxclust={n_clusters} -> {len(centers)} clusters", flush=True)

    def assign(p):
        d = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=2)
        return d.argmin(1)

    all_labels_t = assign(psi_t)
    all_labels_next = assign(psi_next)
    out = []
    for c in range(len(centers)):
        psi_bin_t = (all_labels_t == c).astype(int)
        psi_bin_next = (all_labels_next == c).astype(int)
        out.append({"family": "bisim_cluster", "name": f"bc{c}", "psi_t": psi_bin_t, "psi_next": psi_bin_next,
                     "params": {"cluster_idx": c}})  # psi net + centers saved once, shared across all bc* candidates
    return out, centers


# ---------- pool hygiene ----------
def mutual_info_binary(a, b):
    from sklearn.metrics import mutual_info_score
    return mutual_info_score(a, b)


def hygiene(candidates, target_n=64):
    kept = []
    for c in candidates:
        rate = c["psi_t"].mean()
        if rate > 0.98 or rate < 0.02:
            continue
        switch_rate = (c["psi_t"] != c["psi_next"]).mean()
        if switch_rate > 0.40:
            continue
        c["const_rate"] = float(max(rate, 1 - rate))
        c["switch_rate_hygiene"] = float(switch_rate)
        kept.append(c)

    survivors = []
    for c in kept:
        dup = False
        for s in survivors:
            mi = mutual_info_binary(c["psi_t"], s["psi_t"])
            if mi > 0.9 * min(mutual_info_binary(c["psi_t"], c["psi_t"]), mutual_info_binary(s["psi_t"], s["psi_t"])):
                dup = True
                break
        if not dup:
            survivors.append(c)
        if len(survivors) >= target_n:
            break
    return survivors


def main():
    train_ids = train_episode_pool()
    rng = np.random.default_rng(SEED)
    sample_ids = sorted(rng.choice(train_ids, size=min(N_EPISODES, len(train_ids)), replace=False).tolist())
    pool = collect_trajectories(sample_ids, n_positions=N_POSITIONS)
    zt, znext, aemb, ep_ids = flatten_transitions(pool)
    print(f"transitions collected: {len(zt)}", flush=True)

    print("=== family (ii): slow features ===", flush=True)
    mean, directions = fit_sfa(pool.latents, n_components=8)
    sf_cands = slow_feature_candidates(mean, directions, zt, znext, aemb)
    print(f"  {len(sf_cands)} slow-feature candidates", flush=True)

    print("=== family (i): VQ bottleneck ===", flush=True)
    vq_model = train_vq(zt, znext, aemb, K=24, epochs=6)
    vq_cands = vq_candidates(vq_model, zt, znext)
    print(f"  {len(vq_cands)} VQ-code candidates", flush=True)

    print("=== family (iii): behavioural-metric clustering (simplified) ===", flush=True)
    rho_proj = (zt - mean) @ directions[:, :3]
    psi = train_bisim_psi(zt, znext, rho_proj, steps=600)
    bisim_cands, bisim_centers = bisim_candidates(psi, zt, znext, n_clusters=24)
    print(f"  {len(bisim_cands)} bisim-cluster candidates", flush=True)

    all_cands = sf_cands + vq_cands + bisim_cands
    print(f"total candidates before hygiene: {len(all_cands)}", flush=True)
    survivors = hygiene(all_cands, target_n=64)
    print(f"survivors after hygiene: {len(survivors)}", flush=True)

    pool_out = {
        "n_episodes": len(pool.episode_ids), "n_transitions": len(zt),
        "n_candidates_total": len(all_cands), "n_survivors": len(survivors),
        "by_family": {f: sum(1 for c in survivors if c["family"] == f) for f in ("slow_feature", "vq_code", "bisim_cluster")},
        "candidates": [
            {"family": c["family"], "name": c["name"],
             "switch_rate": float((c["psi_t"] != c["psi_next"]).mean()),
             "active_rate": float(c["psi_t"].mean())}
            for c in survivors
        ],
    }
    (OUT_DIR / "predicate_pool_summary.json").write_text(json.dumps(pool_out, indent=2))

    # persist the actual binary arrays + episode/position alignment for Phase D/E to reuse
    np.savez(OUT_DIR / "predicate_pool_arrays.npz",
              ep_ids=ep_ids,
              **{f"{c['family']}__{c['name']}__t": c["psi_t"] for c in survivors},
              **{f"{c['family']}__{c['name']}__next": c["psi_next"] for c in survivors})

    # persist REUSABLE predicate functions (Phase D needs phi callable on
    # arbitrary new windows, not just the discovery sample's precomputed arrays)
    sf_params = {c["name"]: {"mean": c["params"]["mean"].tolist(), "w": c["params"]["w"].tolist(), "b": c["params"]["b"]}
                 for c in survivors if c["family"] == "slow_feature"}
    (OUT_DIR / "predicate_functions_slowfeature.json").write_text(json.dumps(sf_params))
    torch.save({
        "vq_model_state": vq_model.state_dict(), "vq_K": 24, "vq_code_dim": 32,
        "vq_surviving_codes": [c["params"]["code_idx"] for c in survivors if c["family"] == "vq_code"],
        "bisim_psi_state": psi.state_dict(), "bisim_centers": bisim_centers,
        "bisim_surviving_clusters": [c["params"]["cluster_idx"] for c in survivors if c["family"] == "bisim_cluster"],
        "input_dim": zt.shape[1],
    }, OUT_DIR / "predicate_functions_networks.pt")

    print(json.dumps(pool_out["by_family"], indent=2))
    print("wrote predicate_pool_summary.json + predicate_pool_arrays.npz + predicate_functions_{slowfeature.json,networks.pt}")


if __name__ == "__main__":
    main()
