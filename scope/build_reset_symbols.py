"""Phase A.2 reset symbols + the visual-spread diagnostic that replaces
§3.1's scene-consistency pre-check (per explicit instruction: costs nothing
extra since Phase A.2 already needs these first-frame embeddings; k-medoids
k=8; report mean distance to medoid + cluster sizes; fragmentation, if any,
surfaces at the Phase F acceptance gate, not here).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import HISTORY_SIZE, FRAMESKIP, encode_pixel_windows_batch

N_SAMPLE = 400
K = 8
SEED = 3072
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def medoid_kmeans(features: np.ndarray, k: int, seed: int):
    """k-means then snap each centroid to its nearest REAL sample (same
    medoid-not-centroid pattern as alphabet/codebook.py) -- avoids adding a
    k-medoids dependency for what is here just a diagnostic."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(features)
    medoid_idx = []
    for c in range(k):
        member_idx = np.where(km.labels_ == c)[0]
        centroid = km.cluster_centers_[c]
        d = np.linalg.norm(features[member_idx] - centroid, axis=1)
        medoid_idx.append(int(member_idx[np.argmin(d)]))
    return km.labels_, medoid_idx


def main():
    ids = json.loads((OUT_DIR / "scope_episode_ids.json").read_text())
    drawer_ids = ids["primary"]["episode_ids"]
    rng = np.random.default_rng(SEED)
    sample_ids = sorted(rng.choice(drawer_ids, size=min(N_SAMPLE, len(drawer_ids)), replace=False).tolist())
    print(f"sampling {len(sample_ids)} of {len(drawer_ids)} drawer episodes", flush=True)

    windows = stream_many_windows(sample_ids, num_frames=HISTORY_SIZE, frameskip=FRAMESKIP, max_workers=16)
    ok_ids = sorted(windows.keys())
    print(f"streamed {len(ok_ids)}/{len(sample_ids)} episode windows OK", flush=True)

    pixel_windows = np.stack([windows[eid] for eid in ok_ids])  # (N, H, h, w, 3)
    emb = encode_pixel_windows_batch(pixel_windows)             # (N, H, D)
    first_frame_emb = emb[:, 0].numpy()                          # (N, D) -- clustering feature

    labels, medoid_idx = medoid_kmeans(first_frame_emb, K, SEED)

    diagnostics = {"n_sampled": len(ok_ids), "k": K, "clusters": []}
    for c in range(K):
        member_mask = labels == c
        member_idx = np.where(member_mask)[0]
        medoid_feat = first_frame_emb[medoid_idx[c]]
        dists = np.linalg.norm(first_frame_emb[member_idx] - medoid_feat, axis=1)
        diagnostics["clusters"].append({
            "cluster": c,
            "medoid_episode_id": ok_ids[medoid_idx[c]],
            "n_members": int(member_mask.sum()),
            "mean_dist_to_medoid": float(dists.mean()),
            "max_dist_to_medoid": float(dists.max()),
        })

    overall_pairwise = np.linalg.norm(
        first_frame_emb[:, None, :] - first_frame_emb[None, :, :], axis=-1
    )
    diagnostics["overall_mean_pairwise_dist"] = float(overall_pairwise[np.triu_indices(len(ok_ids), k=1)].mean())

    (OUT_DIR / "reset_symbol_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))

    reset_episode_ids = [c["medoid_episode_id"] for c in diagnostics["clusters"]]
    (OUT_DIR / "reset_episode_ids.json").write_text(json.dumps(reset_episode_ids, indent=2))
    print("reset episode ids:", reset_episode_ids)


if __name__ == "__main__":
    main()
