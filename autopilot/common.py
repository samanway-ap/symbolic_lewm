"""Shared utilities for the v6 autopilot: episode-disjoint splits, the
frozen partial checkpoint, and small helpers reused across every stage.

Scope note (stated once, applies to the whole autopilot/ package): this
implements the SCIENTIFICALLY LOAD-BEARING parts of
SELF_IMPROVEMENT_LOOP_1.md v5 / EXPERIMENTS_orthogonal_expansion_1.md v6 in
full (episode-disjoint splits with no leakage, frozen encoder, identical
optimizer/schedule/batch-order/seed across arms within a node, frozen
per-node geometry, paired episode bootstrap, equal-compute arms, a real
GPU-hour budget). It simplifies the FULL formal machinery the documents also
specify (SHA-256 node identity over a dozen fields, atomic snapshot+ledger
replay for crash recovery, OOM/NaN auto-retry with microbatch halving) to a
plain sequential controller with an append-only JSONL ledger and short
node ids -- this runs as one long-lived unattended process, not a
crash-prone distributed job, so full replay-from-ledger recovery is not
worth the added complexity it would require. Documented here rather than
left implicit.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoints import PARTIAL_EPOCH, load_epoch  # noqa: E402
from oracle.droid_streaming import _episodes_meta  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "artifacts"
AP_DIR = OUT_DIR / "autopilot_nodes"
DOWNLOADS_DIR = Path.home() / "Downloads"
SEED = 3072


def short_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def stable_seed(base: int, *parts: object) -> int:
    """Deterministic seed derivation, replacing `hash(x) % N` (bug fix, code
    audit 2026-09-06): Python intentionally salts `str`/`tuple` hashing
    per-PROCESS unless `PYTHONHASHSEED` is fixed before interpreter startup,
    so `base + hash(some_str_or_tuple) % N` -- used throughout this package
    to derive per-cell/per-arm/per-candidate seeds -- silently produced a
    DIFFERENT seed on every fresh process, breaking cross-run
    reproducibility of anything downstream (which candidate's random_dir
    basis, which reservoir sample, etc.). SHA-256 over a JSON-serialized,
    sorted representation of `(base, *parts)` is stable across processes,
    interpreters, and machines."""
    blob = json.dumps([base, *parts], sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2 ** 32)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_frozen_directions() -> tuple[np.ndarray, np.ndarray]:
    """Lever 0's U_m8/B_m8 -- frozen for the whole tree. Offsets are carried
    only as a diagnostic reference (v6's own config:
    `offsets: lever1_selected_offsets_diagnostic_only`); the continuous
    score u_i.z is what actually drives geometry and evaluation."""
    tp = np.load(OUT_DIR / "tessellation_params_residualized.npz")
    return tp["U_m8"], tp["B_m8"]


def load_partial_model():
    """The frozen partial checkpoint (epoch 7/20, ~35%) -- NOT the memoized
    `oracle.lewm_g.get_model()` singleton (that always loads epoch 20 and
    caching would make loading a second, independently-mutated copy
    impossible within one process)."""
    return load_epoch(PARTIAL_EPOCH)


def checkpoint_file_sha256(epoch: int = PARTIAL_EPOCH) -> str:
    """SHA-256 of the raw checkpoint FILE on disk (not the loaded module's
    state_dict) -- see `code_and_state_fingerprint`'s docstring (required
    change, TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md Change A/D): every new
    geometry/retrieval cache and manifest must record which checkpoint it
    was built against, so a stale cache from a different epoch can never be
    silently reused."""
    import warnings
    warnings.filterwarnings("ignore")
    from stable_worldmodel.data import get_cache_dir
    p = get_cache_dir(None, sub_folder="checkpoints") / "lewm_droid_s0" / f"weights_epoch_{epoch}.pt"
    if not p.exists():
        raise RuntimeError(f"FATAL_DATA_ERROR: checkpoint file not found at {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def module_state_hash(module) -> str:
    """SHA-256 over a `torch.nn.Module`'s own parameters AND buffers (order-
    stable via `state_dict()`'s deterministic key order) -- used both to
    fingerprint which encoder/projector weights a cache was built from, and
    (Change C) to verify after training that a supposedly-frozen module's
    weights and BatchNorm running stats genuinely did not move."""
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def git_commit_hash() -> str:
    """Best-effort; returns 'unknown' rather than raising if git is
    unavailable (e.g. a packaged deployment with no .git directory) -- a
    fingerprint field callers must still be able to compare/reject on, not
    a hard dependency on git being installed."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                                capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001 -- fingerprinting must never crash the run over a missing git binary
        return "unknown"


def build_splits(force: bool = False) -> dict:
    """Episode-disjoint splits by deterministic hash. Roles:
      replay_train    -- original training mixture, used by the `continue` arm
      retrieval_pool  -- unseen DROID episodes outside the drawer family AND
                          outside episode_split.json, source for curricula
      route_val       -- reusable, exploratory routing/successive-halving eval
      confirm_A/B/C   -- disjoint sealed shards, each consumed at most once
      global_guard    -- forgetting/collapse monitor

    replay_train is drawn from episode_split.json's train split (the SAME
    drawer-family episodes theta_partial was itself trained near -- exactly
    what "continued training" should mean). retrieval_pool/route_val/
    confirm_*/global_guard are drawn from the ~92,474-episode non-drawer
    pool, split by a deterministic hash of episode id so re-running
    `build_splits` is idempotent without needing to persist RNG state.
    """
    path = OUT_DIR / "autopilot_split_v1.json"
    if path.exists() and not force:
        return json.loads(path.read_text())

    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    drawer = set(split["train_episode_ids"]) | set(split["test_episode_ids"])
    replay_train = sorted(split["train_episode_ids"])

    meta = _episodes_meta()
    all_ids = set(int(e) for e in meta.index.tolist())
    pool_all = sorted(all_ids - drawer)

    def bucket(eid: int) -> str:
        h = int(hashlib.sha256(f"{SEED}:{eid}".encode()).hexdigest(), 16)
        r = h % 10000
        # deterministic proportions: retrieval_pool 70%, route_val 15%,
        # confirm_A/B/C 2% each (fresh, small, sealed), global_guard 9%
        if r < 7000:
            return "retrieval_pool"
        if r < 8500:
            return "route_val"
        if r < 8700:
            return "confirm_A"
        if r < 8900:
            return "confirm_B"
        if r < 9100:
            return "confirm_C"
        return "global_guard"

    roles: dict[str, list[int]] = {"retrieval_pool": [], "route_val": [], "confirm_A": [],
                                      "confirm_B": [], "confirm_C": [], "global_guard": []}
    for eid in pool_all:
        roles[bucket(eid)].append(eid)
    roles["replay_train"] = replay_train

    manifest = {"seed": SEED, "created": now_iso(), "drawer_family_size": len(drawer),
                  "pool_all_size": len(pool_all), "roles": {k: len(v) for k, v in roles.items()},
                  "episode_ids": roles}
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def assert_disjoint_splits(manifest: dict) -> None:
    roles = manifest["episode_ids"]
    seen: dict[int, str] = {}
    for role, ids in roles.items():
        for eid in ids:
            if eid in seen and seen[eid] != role:
                raise RuntimeError(f"FATAL_DATA_ERROR: episode {eid} in both {seen[eid]} and {role}")
            seen[eid] = role
    print(f"  split disjointness verified: {sum(len(v) for v in roles.values())} episodes, "
          f"{len(roles)} roles, zero overlap", flush=True)


def append_ledger(event: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "autopilot_ledger.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def copy_to_downloads(*paths: Path) -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.exists():
            import shutil
            shutil.copy2(p, DOWNLOADS_DIR / p.name)
