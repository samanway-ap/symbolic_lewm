"""Reconstruct callable predicate functions phi_i(z) -> {0,1} from
discover.py's saved params, and combine a chosen SUBSET into one label_fn
for LatentOracle (Phase E searches over subsets of these)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predicates.discover import PsiNet, VQBottleneck

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_networks = None


def _load_networks():
    global _networks
    if _networks is None:
        _networks = torch.load(OUT_DIR / "predicate_functions_networks.pt", map_location=DEVICE, weights_only=False)
    return _networks


def all_predicate_names() -> list[str]:
    summary = json.loads((OUT_DIR / "predicate_pool_summary.json").read_text())
    return [f"{c['family']}__{c['name']}" for c in summary["candidates"]]


def build_predicate_fns() -> dict[str, callable]:
    """Returns {full_name: phi(z_numpy: (D,)) -> int in {0,1}}."""
    sf_params = json.loads((OUT_DIR / "predicate_functions_slowfeature.json").read_text())
    nets = _load_networks()

    fns = {}
    for name, p in sf_params.items():
        mean, w, b = np.array(p["mean"]), np.array(p["w"]), p["b"]
        fns[f"slow_feature__{name}"] = (lambda z, mean=mean, w=w, b=b: int((z - mean) @ w > b))

    if nets["vq_surviving_codes"]:
        vq = VQBottleneck(nets["input_dim"], code_dim=nets["vq_code_dim"], K=nets["vq_K"]).to(DEVICE)
        vq.load_state_dict(nets["vq_model_state"])
        vq.eval()
        for k in nets["vq_surviving_codes"]:
            def make_vq_fn(code_idx):
                def fn(z):
                    with torch.no_grad():
                        _, _, idx = vq.encode(torch.from_numpy(z).float().unsqueeze(0).to(DEVICE))
                    return int(idx.item() == code_idx)
                return fn
            fns[f"vq_code__vq{k}"] = make_vq_fn(k)

    if nets["bisim_surviving_clusters"]:
        psi = PsiNet(nets["input_dim"]).to(DEVICE)
        psi.load_state_dict(nets["bisim_psi_state"])
        psi.eval()
        centers = nets["bisim_centers"]
        for c in nets["bisim_surviving_clusters"]:
            def make_bisim_fn(cluster_idx):
                def fn(z):
                    with torch.no_grad():
                        p = psi(torch.from_numpy(z).float().unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                    d = np.linalg.norm(p[None, :] - centers, axis=1)
                    return int(d.argmin() == cluster_idx)
                return fn
            fns[f"bisim_cluster__bc{c}"] = make_bisim_fn(c)

    return fns


def make_label_fn(predicate_subset: list[str], all_fns: dict[str, callable]):
    """label_fn(window) -> tuple of bits, one per predicate in the subset,
    evaluated on the window's CURRENT (last) latent."""
    fns = [all_fns[name] for name in predicate_subset]

    def label_fn(window) -> tuple:
        z = window.emb[-1].numpy()
        return tuple(f(z) for f in fns)

    return label_fn
