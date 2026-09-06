"""Shared utilities for the v6-controller/v7-experiment "need autopilot":
action-conditional need spaces N(r,a,1). Thin layer on top of autopilot/
common.py (splits, frozen checkpoint, ledger/atomic-write helpers, SEED) --
see that module's docstring for the standing simplification note, which
applies here too (plain sequential controller + append-only ledger, not the
documents' full SHA-256-node-identity/ledger-replay crash-recovery
machinery; this again runs as one long-lived unattended process, not a
crash-prone distributed job).

New pieces this module owns:
  - the Lever-0 residual basis (the (D, D-k) orthonormal complement used to
    define the "Lever-0 residualized metric" for region assignment). This
    basis was NEVER PERSISTED by a0_label_repair/residualized_tessellation.py
    -- only the drawn directions U8/B8 and k_removed were saved. It is
    reconstructed here from the already-cached artifacts/corpus_train.npz
    (same train episode ids, same SEED, same energy=0.95 -> bit-for-bit the
    same Sigma_between -> the same subspace), then cached so it is not
    recomputed on every restart.
  - the action-letter alphabet (artifacts/alphabet_trainfit.json, fit on the
    train split only, so no route_val/confirm leakage), reusing
    gate.acceptance.quantize_to_symbols verbatim -- the same nearest-medoid,
    feature-space decoding already validated for the FSM extraction pipeline.
  - need_autopilot_* output paths (kept fully separate from autopilot_* so
    the completed, immutable C00-C06 tree is never touched).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.common import (  # noqa: E402
    AP_DIR, OUT_DIR, SEED, assert_disjoint_splits, build_splits, copy_to_downloads,
    load_frozen_directions, load_partial_model, now_iso, short_hash, write_atomic,
)
from gate.acceptance import load_alphabet_feats, quantize_to_symbols  # noqa: E402

NEED_AP_DIR = OUT_DIR / "need_autopilot_nodes"
LEDGER_PATH = OUT_DIR / "need_autopilot_ledger.jsonl"
STATE_PATH = OUT_DIR / "need_autopilot_controller_state.json"
FINAL_JSON = OUT_DIR / "need_autopilot_final_report.json"
FINAL_MD = OUT_DIR / "need_autopilot_final_report.md"
RUN_MANIFEST_PATH = OUT_DIR / "need_autopilot_run_manifest.json"
REGIONS_PATH = OUT_DIR / "need_autopilot_regions.json"
CANDIDATES_PATH = OUT_DIR / "need_autopilot_candidates.json"
LEVER0_BASIS_PATH = OUT_DIR / "lever0_residual_basis.npz"
ALPHABET_PATH = OUT_DIR / "alphabet_trainfit.json"
FROZEN_GEOMETRY_PATH = OUT_DIR / "need_autopilot_frozen_geometry.npz"
TARGETED_RETRIEVAL_CACHE_PATH = OUT_DIR / "need_autopilot_targeted_retrieval.pkl"

K_STAR = 2   # == FRAMESKIP; one alphabet segment == one model-step raw action chunk

CONTROLLER_VERSION = 6
EXPERIMENT_VERSION = 7


def append_need_ledger(event: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    event = {"controller_version": CONTROLLER_VERSION, "experiment_version": EXPERIMENT_VERSION,
             "scientific_scope": "exploratory_action_conditional_predictor_adaptation",
             "soundness_status": "HELD_BY_OWNER", **event}
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def load_lever0_basis(force: bool = False) -> dict:
    """Rebuild (or load the cached rebuild of) the Lever-0 residual
    complement basis: eigenvectors of Sigma_between (computed once, offline,
    on the cached train corpus) with the TOP energy=0.95 fraction dropped.
    Adjustment logged: this basis is a mechanical reconstruction of an
    already-frozen, already-used subspace (same corpus cache, same SEED,
    same energy target as the original residualized_tessellation.py run),
    not a new scientific choice -- required because the original script only
    persisted the drawn directions U8/B8, not the basis itself."""
    if LEVER0_BASIS_PATH.exists() and not force:
        d = np.load(LEVER0_BASIS_PATH)
        return {"residual_basis": d["residual_basis"], "k_removed": int(d["k_removed"]),
                "energy_removed": float(d["energy_removed"])}

    from a0_label_repair.episode_variance import scatter_decomposition
    from a0_label_repair.residualized_tessellation import ENERGY_TO_REMOVE, build_residual_basis
    from corpus.latent_corpus import build_corpus, train_ids

    train = build_corpus(train_ids(), 60, "train")
    sd = scatter_decomposition(train)
    residual_basis, k_removed = build_residual_basis(sd["Sigma_between"], ENERGY_TO_REMOVE)

    expected = json.loads((OUT_DIR / "a0_residualized_report.json").read_text())["k_removed"]
    if k_removed != expected:
        raise RuntimeError(f"FATAL_SPEC_AMBIGUITY: reconstructed Lever-0 k_removed={k_removed} "
                            f"!= recorded {expected}; the cached corpus or SEED must have changed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(LEVER0_BASIS_PATH, residual_basis=residual_basis, k_removed=k_removed,
             energy_removed=ENERGY_TO_REMOVE)
    return {"residual_basis": residual_basis, "k_removed": k_removed, "energy_removed": ENERGY_TO_REMOVE}


def project_residualized(basis: dict, z: np.ndarray) -> np.ndarray:
    """z: (..., D) -> (..., D-k) coordinates in the Lever-0 residual
    complement. Distances/nearest-neighbours computed on this projection
    define the "Lever-0 residualized metric" the documents specify for
    region assignment."""
    return z @ basis["residual_basis"]


class AlphabetLookup:
    """Wraps gate.acceptance's validated nearest-medoid decoding. `letter(raw_seg)`
    takes one already-droid_100-converted (K_STAR, 7) action chunk -- exactly
    autopilot/base_points.py's `_real_word_converted` output for a single
    letter -- and returns its symbol name."""

    def __init__(self, path: Path = ALPHABET_PATH):
        self.path = path
        self.feats, self.names = load_alphabet_feats(path)
        self.content_hash = hashlib.sha256(np.round(self.feats, 6).tobytes()).hexdigest()[:16]

    def letter(self, raw_seg_converted: np.ndarray) -> str:
        return quantize_to_symbols(raw_seg_converted, K_STAR, self.feats, self.names)[0]

    def letters_batch(self, raw_segs_converted: np.ndarray) -> list[str]:
        """raw_segs_converted: (n, K_STAR, 7)."""
        return [quantize_to_symbols(seg, K_STAR, self.feats, self.names)[0] for seg in raw_segs_converted]


class WallClockBudget:
    """Tracks WALL-CLOCK time from controller launch (not GPU-seconds --
    v7's budget is explicitly wall-clock, unlike the completed v6 tree's
    GPU-hour budget). `initial_elapsed_hours` lets a relaunch (e.g. Attempt 2
    after Attempt 1's pre-training abort) count a prior attempt's spent time
    against the SAME cumulative five-hour budget, per owner instruction --
    backdates the clock rather than tracking two separate budgets."""

    def __init__(self, max_hours: float, initial_elapsed_hours: float = 0.0):
        self.t0 = time.monotonic() - initial_elapsed_hours * 3600
        self.max_seconds = max_hours * 3600

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def elapsed_hours(self) -> float:
        return self.elapsed() / 3600

    def remaining(self) -> float:
        return self.max_seconds - self.elapsed()

    def exhausted(self) -> bool:
        return self.elapsed() >= self.max_seconds

    def past(self, hours: float) -> bool:
        return self.elapsed_hours() >= hours


def node_id(region_id: int, action_letter: str, extra: dict) -> str:
    """Pragmatic node-identity hash (simplified from the documents' full
    dozen-field SHA-256 spec -- see module docstring): enough of the frozen,
    outcome-independent fields to make accidental collisions and accidental
    silent re-use of a different candidate's cache practically impossible."""
    payload = {"region_id": region_id, "action_letter": action_letter, "horizon": 1, **extra}
    return "N" + short_hash(payload)[:10]


def save_frozen_geometry(anchors: np.ndarray, candidates: list[dict]) -> None:
    """Persists anchors + every candidate's T/V/W/N bases to disk immediately
    after E0, so a later attempt can literally REUSE this frozen geometry
    instead of recomputing it -- Attempt 1 never did this (only summary
    scalars were saved), which is why Attempt 2 had to reconstruct-and-verify
    rather than load outright. Kept flat (one npz, prefixed by candidate id)
    since candidates have differing basis dimensions."""
    payload = {"anchors": anchors}
    for c in candidates:
        g = c["geometry"]
        for key in ("T_basis", "V_basis", "W_basis", "B_N"):
            payload[f"{c['id']}__{key}"] = g[key]
        payload[f"{c['id']}__meta"] = np.array([c["region"], c["action"], g["q"], g["eta_c"]], dtype=object)
        gg = c.get("global_geometry") or {}
        if not gg.get("empty", True):
            payload[f"{c['id']}__W_all_basis"] = gg["W_all_basis"]
            payload[f"{c['id']}__B_N_all"] = gg["B_N_all"]
    np.savez(FROZEN_GEOMETRY_PATH, **payload)


def load_frozen_geometry() -> dict | None:
    if not FROZEN_GEOMETRY_PATH.exists():
        return None
    d = np.load(FROZEN_GEOMETRY_PATH, allow_pickle=True)
    anchors = d["anchors"]
    ids = sorted({k.split("__")[0] for k in d.files if "__" in k})
    candidates = {}
    for cid in ids:
        meta = d[f"{cid}__meta"]
        geometry = {"T_basis": d[f"{cid}__T_basis"], "V_basis": d[f"{cid}__V_basis"],
                    "W_basis": d[f"{cid}__W_basis"], "B_N": d[f"{cid}__B_N"],
                    "dim_T": int(d[f"{cid}__T_basis"].shape[0]), "dim_V": int(d[f"{cid}__V_basis"].shape[0]),
                    "dim_W": int(d[f"{cid}__W_basis"].shape[0]), "dim_N": int(d[f"{cid}__B_N"].shape[0]),
                    "q": int(meta[2]), "eta_c": float(meta[3])}
        global_geometry = {"empty": True}
        if f"{cid}__W_all_basis" in d.files:
            global_geometry = {"empty": False, "W_all_basis": d[f"{cid}__W_all_basis"],
                                "B_N_all": d[f"{cid}__B_N_all"]}
        candidates[cid] = {"id": cid, "region": int(meta[0]), "action": str(meta[1]),
                           "geometry": geometry, "global_geometry": global_geometry}
    return {"anchors": anchors, "candidates": candidates}
