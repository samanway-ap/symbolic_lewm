"""Load `action.original` for arbitrary DROID episodes from the locally
downloaded `data/*.parquet` files (Phase A.4-A.5 needs real per-frame
actions; the file-level "episode subset download" is already done for the
scoped drawer family -- see M0.5's LeRobotDataset(episodes=..., download_videos=False)
run). Groups episodes by which parquet file holds them so each file is read
exactly once regardless of how many requested episodes it contains.
"""
from __future__ import annotations

import functools
import glob
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_GLOB = str(
    Path.home() / ".cache" / "huggingface" / "lerobot" / "hub"
    / "datasets--lerobot--droid_1.0.1" / "snapshots" / "*" / "data" / "chunk-*" / "*.parquet"
)


@functools.lru_cache(maxsize=1)
def _episode_to_file():
    """episode_index -> local parquet path, built from the meta/episodes
    (chunk_index, file_index) columns already downloaded for M0.5, mapped
    onto the actual local data/ file paths present on disk."""
    from oracle.droid_streaming import _episodes_meta
    meta = _episodes_meta()

    data_files = sorted(glob.glob(_DATA_GLOB))
    by_chunk_file = {}
    for p in data_files:
        parts = Path(p).parts
        chunk = int(parts[-2].split("-")[1])
        file_idx = int(parts[-1].split("-")[1].split(".")[0])
        by_chunk_file[(chunk, file_idx)] = p

    mapping = {}
    for eid, row in meta.iterrows():
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if key in by_chunk_file:
            mapping[eid] = by_chunk_file[key]
    return mapping


def load_actions_for_episodes(
    episode_ids: list[int], field: str = "action.original", max_frames: int | None = None
) -> dict[int, np.ndarray]:
    """Returns {episode_id: (T, dim) float32 array}, reading each backing
    parquet file exactly once."""
    ep2file = _episode_to_file()
    by_file: dict[str, list[int]] = {}
    missing = []
    for eid in episode_ids:
        f = ep2file.get(eid)
        if f is None:
            missing.append(eid)
        else:
            by_file.setdefault(f, []).append(eid)
    if missing:
        import sys
        print(f"load_actions_for_episodes: {len(missing)} episodes have no local data file "
              f"(not yet downloaded?), e.g. {missing[:5]}", file=sys.stderr)

    out: dict[int, np.ndarray] = {}
    for f, eids in by_file.items():
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", field])
        df = df[df["episode_index"].isin(eids)]
        for eid, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            arr = np.stack(g[field].values).astype(np.float32)
            out[int(eid)] = arr[:max_frames] if max_frames else arr
    return out
