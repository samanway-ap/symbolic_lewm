"""Direct HTTP-range streaming of DROID video frames (Phase A, "stream
episodes rather than materialising the family locally where the loader
allows it").

lerobot's own `episodes=[...]` filtering downloads whole chunk FILES
(verified empirically: the drawer family's 3184 episodes touch 100% of
data/ AND 100% of videos/ chunk files, since episodes are scattered roughly
uniformly across chunks -- file-level selection buys nothing here). Full
DROID's videos/ total ~400GB; downloading whole files is not an option.

What DOES work, verified empirically (see session notes): PyAV/ffmpeg can
open the remote .mp4 URL directly and seek to an arbitrary timestamp via
HTTP range requests, decoding only the needed frames without pulling the
whole file (~3s per episode: ~1.8s container open + seek, ~1.3s decode a
handful of frames -- nowhere near the 30-90s a full ~500MB download would
take). This module builds on that: one container open + one seek per
episode, then a single contiguous forward decode of the needed frame span
(never re-seeking per frame within an episode).
"""
from __future__ import annotations

import functools
import glob
import io
import sys
from pathlib import Path

import numpy as np

_REPO_ID = "lerobot/droid_1.0.1"
_FPS = 15.0
_META_GLOB = str(
    Path.home() / ".cache" / "huggingface" / "lerobot" / "hub"
    / "datasets--lerobot--droid_1.0.1" / "snapshots" / "*" / "meta" / "episodes" / "chunk-*" / "*.parquet"
)


# Root cause of the 2026-09-05 memory-exhaustion incident (see
# artifacts/preregistration.md): of this parquet schema's 311 columns, 290
# are per-episode `stats/*` percentile columns (nested arrays -> pandas
# `object` dtype), never read by anything in this codebase, but pandas still
# materializes them as Python objects on load -- 591MB of on-disk parquet
# (compressed, columnar) exploded to ~7.1GB in memory as a result. That cost
# used to be paid ONCE per long-lived process (this function is
# `lru_cache`d, so every thread in the old single-process design shared one
# copy) and went unnoticed; it only became catastrophic once retrieval
# started using MULTIPLE separate processes for streaming concurrency (each
# paying the full 7.1GB independently -- 16 processes tried to allocate
# ~114GB and froze the host). The actual fix is here, not in how many
# processes call this: read only the columns anything actually uses.
_NEEDED_META_COLUMNS = [
    "episode_index", "length", "data/chunk_index", "data/file_index",
    "videos/observation.images.wrist_left/chunk_index", "videos/observation.images.wrist_left/file_index",
    "videos/observation.images.wrist_left/from_timestamp", "videos/observation.images.wrist_left/to_timestamp",
    "videos/observation.images.exterior_1_left/chunk_index", "videos/observation.images.exterior_1_left/file_index",
    "videos/observation.images.exterior_1_left/from_timestamp", "videos/observation.images.exterior_1_left/to_timestamp",
    "videos/observation.images.exterior_2_left/chunk_index", "videos/observation.images.exterior_2_left/file_index",
    "videos/observation.images.exterior_2_left/from_timestamp", "videos/observation.images.exterior_2_left/to_timestamp",
]


@functools.lru_cache(maxsize=1)
def _episodes_meta():
    import pandas as pd
    files = sorted(glob.glob(_META_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No local meta/episodes parquet found under {_META_GLOB} -- "
            "run the M0.5 metadata snapshot_download first."
        )
    dfs = [pd.read_parquet(f, columns=_NEEDED_META_COLUMNS) for f in files]
    df = pd.concat(dfs, ignore_index=True).set_index("episode_index", drop=False)
    return df


def _video_url(camera: str, chunk_index: int, file_index: int) -> str:
    from huggingface_hub import hf_hub_url
    cam_key = camera.replace("videos.", "") if camera.startswith("videos.") else camera
    path = f"videos/{cam_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    return hf_hub_url(_REPO_ID, path, repo_type="dataset")


def stream_episode_frames(
    episode_id: int,
    camera: str = "observation.images.exterior_1_left",
    start_frame: int = 0,
    num_frames: int | None = None,
    frameskip: int = 1,
) -> np.ndarray:
    """Decode `num_frames` frames (stride `frameskip`, starting at local
    frame `start_frame`) of `episode_id` directly from the remote mp4 via a
    single seek + contiguous decode. Returns (T, H, W, 3) uint8.

    num_frames=None decodes through the rest of the episode.
    """
    import av

    meta = _episodes_meta()
    row = meta.loc[episode_id]
    col = f"videos/{camera}"
    chunk_index = int(row[f"{col}/chunk_index"])
    file_index = int(row[f"{col}/file_index"])
    ep_from_ts = float(row[f"{col}/from_timestamp"])
    ep_to_ts = float(row[f"{col}/to_timestamp"])
    ep_length = int(row["length"])

    if num_frames is None:
        n_local_frames = (ep_length - start_frame + frameskip - 1) // frameskip
    else:
        n_local_frames = num_frames

    target_local_frames = [start_frame + i * frameskip for i in range(n_local_frames)]
    target_times = [ep_from_ts + f / _FPS for f in target_local_frames]
    if target_times and target_times[-1] > ep_to_ts + 1e-3:
        raise ValueError(
            f"episode {episode_id}: requested frame range exceeds episode length "
            f"({target_local_frames[-1]} >= {ep_length})"
        )

    url = _video_url(camera, chunk_index, file_index)
    container = av.open(url)
    try:
        stream = container.streams.video[0]
        container.seek(int(target_times[0] * av.time_base), any_frame=False, backward=True, stream=None)

        frames = []
        ti = 0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if ti >= len(target_times):
                break
            # advance ti past any targets this frame's time has already reached/passed
            if frame.time + 1e-3 >= target_times[ti]:
                frames.append(frame.to_ndarray(format="rgb24"))
                ti += 1
                while ti < len(target_times) and target_times[ti] <= frame.time + 1e-3:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    ti += 1
        return np.stack(frames[: len(target_times)]) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)
    finally:
        container.close()


def stream_first_frame(episode_id: int, camera: str = "observation.images.exterior_1_left") -> np.ndarray:
    frames = stream_episode_frames(episode_id, camera=camera, start_frame=0, num_frames=1)
    return frames[0]


_WORKER_SCRIPT = str(Path(__file__).resolve().parent / "_stream_worker_cli.py")
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
MAX_SAFE_WORKERS = 16   # a subprocess per task costs ~tens of MB, not GB -- see module docstring


def _run_one_subprocess(req: tuple, per_task_timeout_s: float):
    """req: (key, eid, camera, start_frame, num_frames, frameskip). Runs
    oracle/_stream_worker_cli.py as a genuinely separate `python` process
    (not `multiprocessing`) so it never re-imports the CALLER's heavy module
    graph -- see that script's docstring for why. `subprocess.run`'s
    `timeout` properly kills the child on expiry (unlike a stuck thread, an
    OS process actually stops), and waiting on it releases the GIL (unlike
    the in-process `av.open`/`container.decode` calls that caused the
    original hang), so plain thread-based concurrency is safe here."""
    import subprocess
    key, eid, camera, start_frame, num_frames, frameskip = req
    try:
        proc = subprocess.run(
            [sys.executable, _WORKER_SCRIPT, str(eid), camera, str(start_frame), str(num_frames), str(frameskip)],
            capture_output=True, timeout=per_task_timeout_s, cwd=_PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return key, None, proc.stderr.decode(errors="replace")[:300]
        frames = np.load(io.BytesIO(proc.stdout))
        return key, frames, None
    except subprocess.TimeoutExpired:
        return key, None, f"subprocess timed out after {per_task_timeout_s:.0f}s (stuck remote decode, killed)"
    except Exception as e:  # noqa: BLE001 -- record and continue
        return key, None, repr(e)


def stream_requests_hardened(
    requests: list[tuple],   # (key, eid, camera, start_frame, num_frames, frameskip)
    max_workers: int = 16,
    per_task_timeout_s: float = 30.0,
    batch_size: int | None = None,   # unused, kept for call-site compatibility
) -> tuple[dict, dict]:
    """Thread pool of genuinely separate OS PROCESSES (one `python` subprocess
    per request, via `_run_one_subprocess`), with a real per-task timeout.

    History (see artifacts/preregistration.md): the original thread-pool
    design assumed a stuck `av.open`/`container.decode` call against a bad
    remote source would at worst leak one idle thread while `as_completed`'s
    own timeout let the caller move on -- in practice (2026-09-04) that
    failed, since PyAV's blocking I/O did not reliably yield the GIL, so the
    timeout-checking code (main thread) couldn't run either, and streaming
    400 episodes against one corrupted video chunk took ~4 hours instead of
    ~25 minutes. The first fix (`multiprocessing.Pool`, spawned per batch)
    solved THAT but caused a worse failure the next day: Windows `spawn`
    re-imports the calling script's entire module graph in every child, and
    every caller here top-level-imports torch/transformers for its own
    purposes, so each of 6 "worker processes" independently reloaded the
    whole model stack (~7GB each) and exhausted system memory, freezing the
    host. This version launches each task as a genuinely fresh, minimal
    `python oracle/_stream_worker_cli.py` subprocess (tens of MB, no heavy
    imports) and waits on it from a thread pool -- `subprocess.wait()`
    properly releases the GIL while blocked, so threads give real
    concurrency here without inheriting av's GIL problem, and
    `subprocess.run(timeout=...)` gives an ACTUAL per-task kill, finer
    grained than the earlier whole-batch-pool termination.

    Returns (results, errors), both keyed by each request's own `key`."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed

    max_workers = min(max_workers, MAX_SAFE_WORKERS)
    results: dict = {}
    errors: dict = {}
    # generous outer bound: real enforcement is per-task via subprocess.run's own timeout
    outer_timeout = per_task_timeout_s * max(1.0, len(requests) / max(1, max_workers)) + 60.0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_run_one_subprocess, req, per_task_timeout_s): req[0] for req in requests}
        try:
            for fut in as_completed(futs, timeout=outer_timeout):
                key, frames, err = fut.result()
                if err is not None:
                    errors[key] = err
                else:
                    results[key] = frames
        except FutureTimeoutError:
            n_pending = sum(1 for f in futs if not f.done())
            for f, key in futs.items():
                if not f.done() and key not in results and key not in errors:
                    errors[key] = f"outer batch timeout ({outer_timeout:.0f}s) with {n_pending} still pending"

    return results, errors


def stream_many_windows(
    episode_ids: list[int],
    num_frames: int,
    frameskip: int = 1,
    camera: str = "observation.images.exterior_1_left",
    max_workers: int = 12,
    per_episode_timeout_s: float = 30.0,
) -> dict[int, np.ndarray]:
    """Concurrently stream a (num_frames, H, W, 3) window per episode. See
    `stream_requests_hardened` for why this is process- (not thread-) based
    and how the wall-clock cap is actually enforced."""
    requests = [(eid, eid, camera, 0, num_frames, frameskip) for eid in episode_ids]
    out, errors = stream_requests_hardened(requests, max_workers=max_workers,
                                              per_task_timeout_s=per_episode_timeout_s)
    if errors:
        import sys
        print(f"stream_many_windows: {len(errors)}/{len(episode_ids)} episodes failed, "
              f"e.g. {next(iter(errors.items()))}", file=sys.stderr)
    return out
