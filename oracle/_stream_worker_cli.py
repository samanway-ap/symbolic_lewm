"""Standalone, minimal-import CLI entry point for decoding ONE video window.
Run as a genuinely separate `python` process (via `subprocess`, NOT via
`multiprocessing`) specifically so it never re-imports whatever heavy module
graph (torch, transformers, stable_worldmodel, ...) the CALLING process
happens to have loaded.

Why this exists: `oracle.droid_streaming.stream_requests_hardened` used to
spawn workers via `multiprocessing.get_context("spawn").Pool(...)`. On
Windows, spawning re-imports the top-level `__main__` module (whatever
script was actually run) in every child process to bootstrap it -- and every
caller of this streaming code (need_controller.py, need_two_arm_pilot.py,
...) top-level-imports torch/transformers/the whole `autopilot` package for
its OWN purposes. That meant every one of the (say) 6 worker processes
independently re-loaded the entire model stack (~7GB each observed in
practice, 2026-09-05 -- see artifacts/preregistration.md), exhausting system
memory and freezing the host machine even though none of that was needed
just to decode a handful of video frames. A subprocess invocation of THIS
script starts a genuinely fresh interpreter whose only imports are numpy and
`oracle.droid_streaming` (itself lightweight at module level -- av/pandas/
huggingface_hub are imported lazily inside its functions), so each worker
costs tens of MB, not gigabytes.

Usage: python oracle/_stream_worker_cli.py <eid> <camera> <start_frame> <num_frames> <frameskip>
Writes the decoded (num_frames, H, W, 3) uint8 array to stdout in .npy
format (self-describing: dtype + shape included) on success; writes the
exception repr to stderr and exits 1 on failure.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    eid = int(sys.argv[1])
    camera = sys.argv[2]
    start_frame = int(sys.argv[3])
    num_frames = int(sys.argv[4])
    frameskip = int(sys.argv[5])

    from oracle.droid_streaming import stream_episode_frames
    frames = stream_episode_frames(eid, camera=camera, start_frame=start_frame,
                                      num_frames=num_frames, frameskip=frameskip)
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, frames)
    sys.stdout.buffer.write(buf.getvalue())
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:  # noqa: BLE001 -- report and exit nonzero; parent treats this as a failed request
        print(repr(e), file=sys.stderr)
        code = 1
    sys.exit(code)
