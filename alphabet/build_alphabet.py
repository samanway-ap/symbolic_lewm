"""Phase B.1-B.3 driver: sample segments (all 3184 drawer episodes, actions
only, already local), gripper-split, per-group k-means -> MEDOID codebook
(never a bare centroid -- may be dynamically unrealisable), n_codes=12
total allocated proportionally to each group's pool size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphabet.segments import GRIPPER_PATTERNS, sample_segments
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import ActionPipeline

N_CODES_TOTAL = 12
N_SEGMENT_SAMPLES = 60_000
K_STAR = 2  # corrected M1 value on the held-out split; see M1_report.md's correction section
SEED = 3072
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def fit_medoids(features: np.ndarray, n_codes: int, seed: int) -> list[int]:
    """k-means then snap each centroid to its nearest REAL sample -- returns
    indices into `features` (and, by extension, the paired raw segments)."""
    from sklearn.cluster import KMeans
    if n_codes <= 0 or len(features) == 0:
        return []
    n_codes = min(n_codes, len(features))
    km = KMeans(n_clusters=n_codes, random_state=seed, n_init=10).fit(features)
    medoid_idx = []
    for c in range(n_codes):
        member_idx = np.where(km.labels_ == c)[0]
        if len(member_idx) == 0:
            continue
        centroid = km.cluster_centers_[c]
        d = np.linalg.norm(features[member_idx] - centroid, axis=1)
        medoid_idx.append(int(member_idx[np.argmin(d)]))
    return medoid_idx


def main(episode_ids=None, out_name="alphabet.json"):
    """episode_ids=None fits on the WHOLE family (the original, in-sample
    run, kept for contrast). Pass the canonical train split's ids to
    refit properly out-of-sample -- see episode_split.json / M2_report.md."""
    if episode_ids is None:
        ids = json.loads((OUT_DIR / "scope_episode_ids.json").read_text())
        episode_ids = ids["primary"]["episode_ids"]

    print(f"loading actions for {len(episode_ids)} episodes...", flush=True)
    episode_actions = load_actions_for_episodes(episode_ids)
    print(f"loaded {len(episode_actions)} episodes' actions", flush=True)

    all_raw = np.concatenate(list(episode_actions.values()), axis=0)
    pipeline = ActionPipeline.fit(all_raw)
    print("fit ActionPipeline on full family's raw actions", flush=True)

    print(f"sampling {N_SEGMENT_SAMPLES} segments (k={K_STAR})...", flush=True)
    buckets = sample_segments(episode_actions, k=K_STAR, n_samples=N_SEGMENT_SAMPLES, pipeline=pipeline, seed=SEED)
    for p in GRIPPER_PATTERNS:
        print(f"  {p}: {len(buckets[p]['features'])} segments", flush=True)

    total_segments = sum(len(buckets[p]["features"]) for p in GRIPPER_PATTERNS)
    alphabet = {"k_star": K_STAR, "n_codes_total": N_CODES_TOTAL, "n_segment_samples": N_SEGMENT_SAMPLES,
                "symbols": [], "gripper_groups": {}}

    remaining_codes = N_CODES_TOTAL
    group_order = sorted(GRIPPER_PATTERNS, key=lambda p: len(buckets[p]["features"]))
    allocations = {}
    for i, p in enumerate(group_order):
        n = len(buckets[p]["features"])
        if n == 0:
            allocations[p] = 0
            continue
        remaining_groups = len(group_order) - i
        share = max(1, round(N_CODES_TOTAL * n / total_segments))
        share = min(share, remaining_codes - (remaining_groups - 1))  # leave >=1 for each remaining nonempty group
        share = max(share, 1)
        allocations[p] = share
        remaining_codes -= share

    symbol_id = 0
    for p in GRIPPER_PATTERNS:
        n_codes_p = allocations.get(p, 0)
        feats = buckets[p]["features"]
        segs = buckets[p]["segments"]
        alphabet["gripper_groups"][p] = {"n_segments_sampled": int(len(feats)), "n_codes": n_codes_p, "symbol_ids": []}
        if n_codes_p == 0 or len(feats) == 0:
            continue
        medoid_idx = fit_medoids(feats, n_codes_p, SEED)
        for idx in medoid_idx:
            sym = f"a{symbol_id}"
            alphabet["symbols"].append({
                "symbol": sym,
                "gripper_pattern": p,
                "medoid_segment": segs[idx].tolist(),   # (k,7), droid_100 convention
                "source_episode_id": int(buckets[p]["episode_ids"][idx]),
                "source_start_frame": int(buckets[p]["starts"][idx]),
            })
            alphabet["gripper_groups"][p]["symbol_ids"].append(sym)
            symbol_id += 1

    print(f"built alphabet: {len(alphabet['symbols'])} symbols "
          f"({ {p: alphabet['gripper_groups'][p]['n_codes'] for p in GRIPPER_PATTERNS} })", flush=True)

    (OUT_DIR / out_name).write_text(json.dumps(alphabet, indent=2))
    print("wrote", OUT_DIR / out_name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "trainfit":
        split = json.loads((OUT_DIR / "episode_split.json").read_text())
        main(episode_ids=split["train_episode_ids"], out_name="alphabet_trainfit.json")
    else:
        main()
