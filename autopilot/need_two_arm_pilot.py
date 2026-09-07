"""Two-arm feasibility pilot, v3 (W_soft): does the frozen action-conditioned
need curriculum (`need_curriculum`) beat matched-random sampling
(`matched_random`) at short-horizon, frozen-encoder(+projector) prediction,
in the node's own predeclared need subspace N(r,a,1)?

v2 (TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md, reviewed revision db57a54)
implemented Changes A-F required before this experiment was interpretable,
and ran to completion: EVERY pre-screened (region, action) cell's exact
W = T \\cap V^perp intersection came out empty (dim_V, 20-45, far exceeded
dim_T, 4-10, in all 11 cells -- a generic, expected consequence of the
corrected intersection math, not a bug), so v2 recorded a legitimate
RETRIEVAL_INFEASIBLE (`need_two_arm_v2_metrics.json`) -- preserved verbatim,
never overwritten or deleted; see `note_v2_result`.

v3 (user-directed, 2026-09-07) is a SEPARATE, ADDITIONAL experiment
revision exploring a deliberately looser W construction, W_soft (see
`autopilot.geometry.soft_tangent_min_alignment`): the bottom min(4, dim_T)
right-singular directions of `V_basis @ T_basis.T` -- the directions in T
LEAST aligned with V, rather than EXACTLY orthogonal to it. Everything else
(GRADIENT_ENERGY, M_ACTION_DIRS, the residual-energy pre-screen, candidate
count, retrieval rules/constants, two arms, five pairs, training schedule,
evaluation, guards) is UNCHANGED from v2. Reuses the fingerprinted epoch-7
Lever-0 corpus verbatim (`build_v2_lever0_and_directions`, still v2-named
since it is not itself changing); candidate geometry is invalidated and
rebuilt (GEOMETRY_SCHEMA_VERSION bump + a new fingerprint field) since the
W-computation itself changed. Writes ONLY under the `need_two_arm_v3_*`
namespace (retrieval/geometry caches, manifest, metrics, report, progress,
wall-clock state) and never assigns/opens a confirm_A/B/C shard (this is a
feasibility signal, not a confirmatory test -- see the DECISION RULE in
`main` for exactly what a positive result does and does not license).
"""
from __future__ import annotations

import copy
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Hardening after the 2026-09-07 crash: a single non-ASCII character in one
# print() call (the mathematical "cap"/intersection symbol) raised an
# UnicodeEncodeError and killed an in-progress multi-hour unattended run,
# because this Windows console's redirected stdout defaults to the cp1252
# codepage, not UTF-8. Reconfiguring stdout/stderr to UTF-8 with
# errors="replace" means any future stray non-ASCII character in a log line
# degrades to a "?" instead of ever taking the whole run down again.
# reconfigure() is only present on real TextIOWrapper streams (not always
# true under some test/CI runners), so this is a no-op there rather than a
# hard dependency.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Basic resource headroom (added after the 2026-09-05 crash: repeated
# multiprocessing pool churn during retrieval exhausted OS resources and
# froze the host). Capping BLAS/PyTorch CPU threads keeps this process from
# also competing hard for CPU against the streaming worker processes and
# whatever else is running on the machine; the model/training work itself is
# small (ViT-tiny, batch size 32) and does not need many threads.
torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopilot.common import (  # noqa: E402
    OUT_DIR, SEED, assert_disjoint_splits, build_splits, checkpoint_file_sha256, copy_to_downloads,
    git_commit_hash, load_partial_model, module_state_hash, now_iso, short_hash, stable_seed, write_atomic,
)
from autopilot.controller import fit_pipeline  # noqa: E402
from autopilot.dataset import ArmDataset, build_dataset_from_segments, build_replay_dataset  # noqa: E402
from autopilot.evaluate import assert_slice_nonempty, build_eval_slice, evaluate_E_W, paired_bootstrap_ci  # noqa: E402
from autopilot.need_common import AlphabetLookup, WallClockBudget  # noqa: E402
from autopilot.need_controller import build_regions_and_cells  # noqa: E402
from autopilot.need_evaluate import build_cell_eval_slice, effective_rank, full_latent_predictions  # noqa: E402
from autopilot.need_geometry import M_ACTION_DIRS  # noqa: E402
from autopilot.need_targeted_retrieval import (  # noqa: E402
    K_FALLBACK, K_TARGET, MIN_COMMON_ELIGIBLE, N_OCCURRENCES_CAP, PER_EPISODE_CAP,
    resample_random_traj_curriculum, run_targeted_retrieval,
)
from checkpoints import PARTIAL_EPOCH  # noqa: E402
from oracle.lewm_g import DEVICE, HISTORY_SIZE  # noqa: E402
from training.finetune import _collate  # noqa: E402

PILOT_DIR = OUT_DIR / "need_two_arm_v3_nodes"
MANIFEST_PATH = OUT_DIR / "need_two_arm_v3_manifest.json"
METRICS_PATH = OUT_DIR / "need_two_arm_v3_metrics.json"
REPORT_PATH = OUT_DIR / "need_two_arm_v3_report.md"
FROZEN_GEOMETRY_NPZ = OUT_DIR / "need_two_arm_v3_frozen_geometry.npz"
FROZEN_GEOMETRY_FINGERPRINT_JSON = OUT_DIR / "need_two_arm_v3_frozen_geometry_fingerprint.json"
# Reused verbatim from v2 -- the Lever-0/U8/B8 construction itself is NOT
# changing in this revision (only the candidate-geometry W-computation is),
# so this stays v2-named and is neither renamed nor rebuilt.
LEVER0_V2_NPZ = OUT_DIR / "need_two_arm_v2_lever0_and_directions.npz"
RETRIEVAL_CACHE_PKL = OUT_DIR / "need_two_arm_v3_retrieval_cache.pkl"
PROGRESS_PATH = OUT_DIR / "need_two_arm_v3_progress.json"
WALL_CLOCK_STATE_PATH = OUT_DIR / "need_two_arm_v3_wall_clock_state.json"

V1_METRICS_PATH = OUT_DIR / "need_two_arm_pilot_metrics.json"   # superseded, not deleted -- see mark_v1_invalidated
# v2's completed, VALID (not invalidated) RETRIEVAL_INFEASIBLE result --
# read-only reference, never written/overwritten by this (v3) script. See
# `note_v2_result`.
V2_METRICS_PATH = OUT_DIR / "need_two_arm_v2_metrics.json"

# W construction for candidate geometry: "exact" (T \\cap V^perp, v2's
# choice) or "soft" (autopilot.geometry.soft_tangent_min_alignment, v3's
# choice -- see module docstring). This is the ONLY scientific-method
# change between v2 and v3; every other predeclared constant below is
# unchanged from v2.
W_MODE = "soft"
W_SOFT_M_MAX = 4

# Bumped 3 -> 4 (v3, W_soft revision): candidate geometry now uses a
# genuinely different W-construction, so the geometry cache must be
# invalidated and rebuilt -- an explicit version bump FORCES this rather
# than relying only on the fingerprint dict happening to differ (it also
# lives under a brand new v3-named cache file, so there is no risk of
# colliding with v2's own exact-intersection cache either way).
GEOMETRY_SCHEMA_VERSION = 4

MAX_WALL_HOURS = 5.0    # was 2.0 under v1's 2-arm x 2-seed design (4 training runs);
                          # v2's 2-arm x 5-pair design needs 10 -- scaled proportionally,
                          # decided from the known compute-scaling factor, not from any outcome.
RETRIEVAL_MAX_MINUTES = 60.0   # Change F: a safety ABORT now, not a sampling truncation.
# pilot's OWN K-cascade (distinct from the main tree's 200-threshold/200-K,
# 100-threshold/100-K): >=200 eligible segments -> K=100; else >=100 -> K=50;
# else infeasible. Same Stage 1/2 mechanism as the main tree, parameterized.
K_TARGET_THRESHOLD = 200
PILOT_K_TARGET = 100
K_FALLBACK_THRESHOLD = 100
PILOT_K_FALLBACK = 50
PILOT_MIN_COMMON_ELIGIBLE = K_FALLBACK_THRESHOLD

N_UPDATES_DIAGNOSTIC = 200
N_UPDATES_PRIMARY = 500
N_PAIRS = 5                 # Change: section 4 -- 5 paired repetitions, replacing v1's 2 seeds.
PAIR_SEEDS = list(range(N_PAIRS))
BATCH_SIZE = 32
LR = 5e-5
WEIGHT_DECAY = 1e-3
GLOBAL_REGRESSION_MAX = 0.02
EFFRANK_DROP_MAX = 0.10
MIN_EVAL_EPISODES = 5      # fail-closed floor for eval/guard slices (P1-8)
MIN_PAIRS_FOR_SIGN = 4      # "at least four of five pairs" (section 4 decision rule)


def mark_v1_invalidated() -> None:
    """Change D: "Mark the prior TWO_ARM_NEGATIVE report as
    INVALIDATED_IMPLEMENTATION, without deleting or rewriting its numerical
    contents." Additive metadata patch only -- every original field is kept
    verbatim; only new `invalidated`/`invalidation_reason` keys are added
    (or refreshed, if this has already run before)."""
    if not V1_METRICS_PATH.exists():
        return
    payload = json.loads(V1_METRICS_PATH.read_text())
    payload["invalidated"] = True
    payload["invalidation_reason"] = (
        "TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md (reviewed revision db57a54): this result mixed an epoch-20 "
        "encoder with an epoch-7 predictor in several latent-encoding call sites (Change A), computed the "
        "action-timing alignment inconsistently with the upstream LeWM rollout convention (Change B), left the "
        "projector trainable despite frozen-geometry claims (Change C), and paired/aggregated statistics without "
        "the corrections in Changes D-F. Not deleted or numerically rewritten -- see need_two_arm_v2_metrics.json "
        "for the corrected implementation's result."
    )
    payload["invalidated_by"] = str(V2_METRICS_PATH.name)   # v2 is what actually superseded v1, not this (v3) run
    write_atomic(V1_METRICS_PATH, payload)
    print(f"  marked {V1_METRICS_PATH.name} INVALIDATED_IMPLEMENTATION (numerical contents preserved verbatim)",
          flush=True)


def note_v2_result() -> None:
    """v2's completed RETRIEVAL_INFEASIBLE result is a LEGITIMATE outcome
    under correct code (every pre-screened cell's exact T \\cap V^perp
    intersection came out empty), not a bug -- so unlike v1, it is NOT
    invalidated. This just reads it (read-only, never writes) so the log of
    a v3 run states plainly, at a glance, what v2 already established."""
    if not V2_METRICS_PATH.exists():
        print("  (no v2 result found on disk to reference)", flush=True)
        return
    v2 = json.loads(V2_METRICS_PATH.read_text())
    print(f"  v2 (exact intersection) result on record: status={v2.get('status')} "
          f"reason={v2.get('reason', '(none)')} -- preserved untouched, not overwritten by this v3 run", flush=True)


def compute_fingerprint(model, pipeline, alphabet: AlphabetLookup, U8: np.ndarray, residual_basis: np.ndarray,
                            manifest: dict) -> dict:
    """Change A ("a runtime assertion recording the intended checkpoint hash
    with every latent cache") / Change D ("put this fingerprint in every new
    cache and manifest; refuse cache loading when any fingerprint field
    differs or is absent"). Every field here is either a content hash of
    something that, if it changed, would silently invalidate the geometry
    this pilot depends on, or a declared selection constant.

    User-directed patch (item 6): added `u8_hash`/`residual_basis_hash` --
    U8 and the Lever-0 residual_basis are now recomputed in epoch-7
    coordinates (build_v2_lever0_and_directions), so their content must gate
    cache reuse exactly like the checkpoint/encoder/projector hashes
    already do. Also carries `w_mode`/`w_soft_m_max` (v3): a geometry cache
    built under one W-construction must never be silently reused under a
    different one, even if every other input happens to match."""
    return {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "w_mode": W_MODE, "w_soft_m_max": W_SOFT_M_MAX if W_MODE == "soft" else None,
        "code_commit": git_commit_hash(),
        "checkpoint_epoch": PARTIAL_EPOCH,
        "checkpoint_sha256": checkpoint_file_sha256(PARTIAL_EPOCH),
        "encoder_hash": module_state_hash(model.encoder),
        "projector_hash": module_state_hash(model.projector),
        "split_hash": short_hash(manifest["episode_ids"]),
        "action_pipeline_hash": short_hash({
            "converter_lo": pipeline.converter.src_lo.round(6).tolist(),
            "converter_hi": pipeline.converter.src_hi.round(6).tolist(),
            "normalizer_mean": pipeline.normalizer.mean.round(6).tolist(),
            "normalizer_std": pipeline.normalizer.std.round(6).tolist(),
        }),
        "alphabet_hash": alphabet.content_hash,
        "u8_hash": short_hash(U8.round(6).tolist()),
        "residual_basis_hash": short_hash(residual_basis.round(6).tolist()),
        "selection_constants": {
            "k_target_threshold": K_TARGET_THRESHOLD, "k_target": PILOT_K_TARGET,
            "k_fallback_threshold": K_FALLBACK_THRESHOLD, "k_fallback": PILOT_K_FALLBACK,
            "min_common_eligible": PILOT_MIN_COMMON_ELIGIBLE,
            "n_occurrences_cap": N_OCCURRENCES_CAP, "per_episode_cap": PER_EPISODE_CAP,
        },
    }


def build_v2_lever0_and_directions(model) -> dict:
    """User-directed minimal-changes patch (before launch), items 2-4: U8/B8
    (autopilot.common.load_frozen_directions -> tessellation_params_
    residualized.npz) and the Lever-0 residual_basis (autopilot.need_common.
    load_lever0_basis) were BOTH built from epoch-20-encoded latents
    (encode_pixel_windows_batch's implicit default, inside both a0_label_
    repair/residualized_tessellation.py's original construction and
    corpus.latent_corpus.build_corpus) -- the same class of coordinate
    mismatch Change A already fixed elsewhere on the v2 path, just not here.

    Mechanically reproduces the ORIGINAL construction with NO tuning: same
    fitting episodes (corpus.latent_corpus.train_ids()), same n_positions
    (60), same ENERGY_TO_REMOVE (0.95), same M_MAX (8), same SEED (3072,
    residualized_tessellation.py's own module constant) -- only the ONE
    place latents get encoded now takes the explicit epoch-7 `model`
    instead of falling through to encode_pixel_windows_batch's epoch-20
    default. Cached under a need_two_arm_v2_* name; never overwrites the
    six-arm tree's own lever0_residual_basis.npz / tessellation_params_
    residualized.npz. `k_removed` is NOT compared against a0_residualized_
    report.json's recorded value (unlike load_lever0_basis's own six-arm-
    path check) -- that record was itself computed from epoch-20 latents,
    so a different checkpoint legitimately producing a different k here is
    expected, not an error."""
    if LEVER0_V2_NPZ.exists():
        d = np.load(LEVER0_V2_NPZ)
        return {"residual_basis": d["residual_basis"], "k_removed": int(d["k_removed"]),
                  "energy_removed": float(d["energy_removed"]), "U8": d["U8"], "B8": d["B8"]}

    from a0_label_repair.episode_variance import scatter_decomposition
    from a0_label_repair.residualized_tessellation import (
        ENERGY_TO_REMOVE, M_MAX, SEED as LEVER0_SEED, build_residual_basis, draw_directions_in_complement,
        fit_offsets,
    )
    from corpus.latent_corpus import build_corpus, train_ids

    train = build_corpus(train_ids(), 60, "need_two_arm_v2_train", model=model, force=True)
    sd = scatter_decomposition(train)
    residual_basis, k_removed = build_residual_basis(sd["Sigma_between"], ENERGY_TO_REMOVE)
    U8 = draw_directions_in_complement(residual_basis, M_MAX, LEVER0_SEED)
    B8 = fit_offsets(U8, train.flat_latents())

    np.savez(LEVER0_V2_NPZ, residual_basis=residual_basis, k_removed=k_removed, energy_removed=ENERGY_TO_REMOVE,
                U8=U8, B8=B8, seed=LEVER0_SEED)
    return {"residual_basis": residual_basis, "k_removed": k_removed, "energy_removed": ENERGY_TO_REMOVE,
              "U8": U8, "B8": B8}


def save_v3_frozen_geometry(anchors: np.ndarray, candidates: list[dict], fingerprint: dict) -> None:
    payload = {"anchors": anchors}
    for c in candidates:
        g = c["geometry"]
        for key in ("T_basis", "V_basis", "W_basis", "B_N"):
            payload[f"{c['id']}__{key}"] = g[key]
        payload[f"{c['id']}__meta"] = np.array([c["region"], c["action"], g["q"], g["eta_c"]], dtype=object)
        # W_soft diagnostics (user-directed): ALL singular values, recorded
        # for inspection only -- never consumed by load_v3_frozen_geometry
        # to gate anything.
        if "w_soft_singular_values" in g:
            payload[f"{c['id']}__w_soft_singular_values"] = g["w_soft_singular_values"]
        gg = c.get("global_geometry") or {}
        if not gg.get("empty", True):
            payload[f"{c['id']}__W_all_basis"] = gg["W_all_basis"]
            payload[f"{c['id']}__B_N_all"] = gg["B_N_all"]
    np.savez(FROZEN_GEOMETRY_NPZ, **payload)
    FROZEN_GEOMETRY_FINGERPRINT_JSON.write_text(json.dumps({"fingerprint": fingerprint}, indent=2, default=str))


def load_v3_frozen_geometry() -> dict | None:
    if not (FROZEN_GEOMETRY_NPZ.exists() and FROZEN_GEOMETRY_FINGERPRINT_JSON.exists()):
        return None
    d = np.load(FROZEN_GEOMETRY_NPZ, allow_pickle=True)
    anchors = d["anchors"]
    ids = sorted({k.split("__")[0] for k in d.files if "__" in k})
    candidates = []
    for cid in ids:
        meta = d[f"{cid}__meta"]
        geometry = {"T_basis": d[f"{cid}__T_basis"], "V_basis": d[f"{cid}__V_basis"],
                    "W_basis": d[f"{cid}__W_basis"], "B_N": d[f"{cid}__B_N"],
                    "dim_T": int(d[f"{cid}__T_basis"].shape[0]), "dim_V": int(d[f"{cid}__V_basis"].shape[0]),
                    "dim_W": int(d[f"{cid}__W_basis"].shape[0]), "dim_N": int(d[f"{cid}__B_N"].shape[0]),
                    "q": int(meta[2]), "eta_c": float(meta[3])}
        if f"{cid}__w_soft_singular_values" in d.files:
            geometry["w_soft_singular_values"] = d[f"{cid}__w_soft_singular_values"]
        global_geometry = {"empty": True}
        if f"{cid}__W_all_basis" in d.files:
            global_geometry = {"empty": False, "W_all_basis": d[f"{cid}__W_all_basis"],
                                "B_N_all": d[f"{cid}__B_N_all"]}
        candidates.append({"id": cid, "region": int(meta[0]), "action": str(meta[1]),
                              "geometry": geometry, "global_geometry": global_geometry})
    fingerprint = json.loads(FROZEN_GEOMETRY_FINGERPRINT_JSON.read_text())["fingerprint"]
    return {"anchors": anchors, "candidates": candidates, "fingerprint": fingerprint}


def save_v3_retrieval_cache(fingerprint: dict, chosen_id: str, chosen_region: int, chosen_action: str,
                                curricula: dict, retrieval_stats: dict) -> None:
    """Resumability: Stage 1/2 retrieval (a full retrieval_pool scan plus
    decoding up to N_OCCURRENCES_CAP occurrences) is the single most
    expensive step after the one-time Lever-0/geometry builds -- if the
    process is interrupted partway through training and restarted, this
    lets it skip straight back to training instead of re-scanning the pool.
    Only the CHOSEN cell's curricula are persisted (not every candidate
    cell's, which would also include large per-candidate eligible-segment
    pools for cells that were never used) -- gated by the SAME `fingerprint`
    already used for the frozen-geometry cache, so it is invalidated exactly
    when the geometry/checkpoint/pipeline it depends on would be."""
    with open(RETRIEVAL_CACHE_PKL, "wb") as f:
        pickle.dump({"fingerprint": fingerprint, "chosen_id": chosen_id, "chosen_region": chosen_region,
                        "chosen_action": chosen_action, "curricula": curricula, "retrieval_stats": retrieval_stats}, f)


def load_v3_retrieval_cache(fingerprint: dict) -> dict | None:
    if not RETRIEVAL_CACHE_PKL.exists():
        return None
    with open(RETRIEVAL_CACHE_PKL, "rb") as f:
        cached = pickle.load(f)
    if cached.get("fingerprint") != fingerprint:
        return None
    return cached


def write_progress(stage: str, wall: WallClockBudget, extra: dict | None = None) -> None:
    """Resumability / "keep me posted while traveling": a small, cheap,
    always-current snapshot of which stage the run has reached, written
    after every major milestone (Lever-0 done, geometry done, retrieval
    done + chosen candidate, and after EVERY pair's training+eval
    completes with cumulative per-pair deltas so far) -- readable at any
    time without needing the process to still be alive, and copied to
    Downloads exactly like the final report already is."""
    payload = {"stage": stage, "timestamp": now_iso(), "wall_hours_used": wall.elapsed_hours(),
                 "wall_hours_budget": MAX_WALL_HOURS, **(extra or {})}
    write_atomic(PROGRESS_PATH, payload)
    write_atomic(WALL_CLOCK_STATE_PATH, {"cumulative_hours_used": wall.elapsed_hours()})
    copy_to_downloads(PROGRESS_PATH)
    print(f"  [progress] stage={stage} wall={wall.elapsed_hours():.2f}h", flush=True)


def load_cumulative_wall_hours() -> float:
    """Resumability: the wall-clock budget must persist ACROSS restarts, not
    reset to 0 every time the process relaunches -- matching this codebase's
    existing convention (autopilot.need_common.WallClockBudget's own
    docstring: "lets a relaunch... count a prior attempt's spent time
    against the SAME cumulative budget... backdates the clock rather than
    tracking two separate budgets")."""
    if WALL_CLOCK_STATE_PATH.exists():
        return float(json.loads(WALL_CLOCK_STATE_PATH.read_text()).get("cumulative_hours_used", 0.0))
    return 0.0


def get_or_build_v3_candidates(manifest: dict, lever0_basis: dict, model, pipeline, alphabet: AlphabetLookup,
                                  wall: WallClockBudget, fingerprint: dict, U4: np.ndarray) -> dict:
    """Change D (carried forward from v2, applied again for v3's own cache):
    NEVER loads v1's or v2's frozen geometry/retrieval caches (each built
    under a different W-construction / checkpoint mixture / action-timing
    convention). Rebuilds regions and candidates fresh under W_MODE every
    time the fingerprint changes; a matching-fingerprint v3 cache IS reused
    (this is expensive to rebuild -- a full replay_train/route_val scan --
    and the fingerprint already proves nothing scientifically relevant
    changed). "Select the first supported candidate in a frozen
    deterministic order; do not force the old N00 identity to survive the
    correction" (Change D) -- `build_regions_and_cells` without
    `target_cells` already does exactly this: fresh eta_c-ranked candidates
    N00/N01/N02, no comparison against any prior attempt's recorded values.

    `U4` is passed straight through to `build_regions_and_cells`, which must
    never fall back to its own `load_frozen_directions()` (epoch-20
    coordinates) on this path; `W_MODE`/`W_SOFT_M_MAX` (module-level) select
    the soft W-construction for this revision."""
    cached = load_v3_frozen_geometry()
    if cached is not None and cached["fingerprint"] == fingerprint:
        print("  reusing persisted v3 frozen geometry (fingerprint match)", flush=True)
        return {"status": "OK", "candidates": cached["candidates"], "anchors": cached["anchors"]}
    if cached is not None:
        print("  v3 frozen-geometry cache fingerprint MISMATCH -- discarding stale cache, rebuilding from scratch",
              flush=True)
    print(f"  building regions/candidates fresh under W_MODE={W_MODE!r} (no v1/v2 cache, no ATTEMPT1 "
          f"reference verification) ...", flush=True)
    result = build_regions_and_cells(manifest, lever0_basis, model, pipeline, alphabet, wall, U4=U4, w_mode=W_MODE)
    if result["status"] != "OK":
        return result
    save_v3_frozen_geometry(result["anchors"], result["candidates"], fingerprint)
    return {"status": "OK", "candidates": result["candidates"], "anchors": result["anchors"]}


def freeze_encoder_and_projector(model) -> None:
    """Change C: the shared `train_arm.freeze_encoder` (used by the six-arm
    v6/v7 trees, out of this task's scope to modify) leaves `projector`
    trainable -- but U_4/T/V/W/N are all defined in PROJECTOR-OUTPUT
    coordinates, so projector updates during an arm would move the
    coordinate system the frozen geometry was computed in. This LOCAL
    function (used only by this pilot) freezes both."""
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.projector.parameters():
        p.requires_grad_(False)
    model.encoder.eval()
    model.projector.eval()


def unfreeze_predictor_only(model) -> list[torch.nn.Parameter]:
    """Change C: "Train only predictor, action_encoder, and pred_proj."""
    params = []
    for name, p in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("projector."):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            params.append(p)
    return params


def build_arm_dataset(segments: list[dict], replay_ds: ArmDataset, pipeline) -> ArmDataset:
    """Exactly `len(segments)` curriculum samples (built from the EXACT
    selected (episode_id, t) segments, not re-windowed episode IDs) plus the
    SAME shared `replay_ds` object for both arms -- guarantees byte-identical
    replay examples (Change F)."""
    curr_ds = build_dataset_from_segments(segments, pipeline)
    samples = curr_ds.samples + replay_ds.samples
    episode_ids = sorted(set(curr_ds.episode_ids) | set(replay_ds.episode_ids))
    return ArmDataset(samples=samples, pipeline=pipeline, episode_ids=episode_ids)


def build_exact_k_replay_dataset(replay_train_ids: list[int], k_use: int, pipeline, seed: int) -> ArmDataset:
    """Change F item 3: "Build exactly K replay samples, truncate
    deterministically, and assert the count." `build_replay_dataset` targets
    `k_use` but samples whole EPISODES (8 windows each) -- ceil-dividing can
    overshoot; this truncates the resulting sample list to EXACTLY `k_use`
    (deterministic: whatever order `build_replay_dataset` produced, which is
    itself seeded) and asserts the count, rather than silently training on a
    different-sized replay set than declared."""
    ds = build_replay_dataset(replay_train_ids, k_use, pipeline, seed=seed)
    if len(ds.samples) < k_use:
        raise RuntimeError(
            f"IMPLEMENTATION_FAILURE: could only build {len(ds.samples)}/{k_use} replay samples from "
            f"{len(replay_train_ids)} replay_train episodes")
    samples = ds.samples[:k_use]
    episode_ids = sorted(set(int(s["episode"]) for s in samples))
    out = ArmDataset(samples=samples, pipeline=pipeline, episode_ids=episode_ids)
    assert len(out.samples) == k_use, "IMPLEMENTATION_FAILURE: exact-K replay truncation produced the wrong count"
    return out


def assert_segments_survive(dataset: ArmDataset, frozen_segments: list[dict], n_curriculum: int) -> None:
    """Required invariant (TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md section
    3, Check 3 / preflight): every curriculum sample must be exactly one of
    the frozen selected segments -- 100%, not merely high. Checks only the
    first `n_curriculum` samples (the curriculum portion; replay samples are
    drawn from `replay_train` and are never expected to match a frozen
    segment).

    User-directed patch (item 7): checks CONTENT (the pixels/raw_action
    bytes, via each segment's own `content_hash`), not merely (episode_id,
    t) -- an (eid, t) match alone would not catch a wiring bug that
    substituted a different decode of the same nominal position (e.g. a
    stale cache, a different camera, a re-decode under different frame
    alignment)."""
    frozen_by_key = {(int(s["eid"]), int(s["t"])): s["content_hash"] for s in frozen_segments}
    if len(frozen_by_key) != len(frozen_segments):
        raise RuntimeError("IMPLEMENTATION_FAILURE: duplicate (episode_id, t) among frozen segments")
    hits = 0
    for s in dataset.samples[:n_curriculum]:
        key = (s["episode"], s["pos"])
        expected_hash = frozen_by_key.get(key)
        if expected_hash is None:
            continue   # not even an (episode_id, t) match
        if s.get("content_hash") is None:
            raise RuntimeError(
                f"IMPLEMENTATION_FAILURE: dataset sample for {key} carries no content_hash -- "
                f"build_dataset_from_segments did not thread it through")
        if s["content_hash"] == expected_hash:
            hits += 1
    if hits != n_curriculum:
        raise RuntimeError(
            f"IMPLEMENTATION_FAILURE: segment-retention invariant violated -- {hits}/{n_curriculum} "
            f"curriculum samples matched a frozen selected segment's (episode_id, t) AND content_hash; "
            f"expected 100%")


def paired_eid_arrays(per_episode_a: dict, per_episode_b: dict) -> tuple:
    """Explicit episode_id pairing (not array-position pairing) between two
    arms' per-episode E_N dicts -- fails loudly if the two arms' evaluated
    episode sets ever diverge, instead of silently mispairing two different
    episodes' errors."""
    common = sorted(set(per_episode_a) & set(per_episode_b))
    if len(common) != len(per_episode_a) or len(common) != len(per_episode_b):
        raise RuntimeError(
            f"IMPLEMENTATION_FAILURE: eval episode sets diverged between arms -- "
            f"a={sorted(per_episode_a)} b={sorted(per_episode_b)}")
    return (np.array([per_episode_a[e]["E_W"] for e in common]),
            np.array([per_episode_b[e]["E_W"] for e in common]))


def pair_level_bootstrap_ci(deltas: list[float], seed: int, n_boot: int = 2000) -> tuple:
    """PRIMARY interval for the decision rule (section 4: "Episode-level
    resampling may be included as a secondary conditional interval, but
    must not replace the five paired experimental units") -- resamples the
    N_PAIRS delta_j values themselves, WITH replacement. With N_PAIRS=5 this
    is necessarily coarse (only a handful of distinct resamples exist);
    report alongside the raw 5 values and their sign count, not as a
    substitute for them."""
    arr = np.array(deltas)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.array([arr[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return float(arr.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def reset_all_rngs(seed: int) -> None:
    """Change E: "reset Python, NumPy, PyTorch CPU, and all CUDA RNGs to the
    pair seed; enable deterministic CUDA/PyTorch behaviour where supported."
    The LOCAL `torch.Generator` used for the batch permutation already makes
    batch ORDER identical between the two arms of a pair, but the LeWM
    predictor may use dropout, which consumes GLOBAL torch RNG state -- if
    that state has drifted differently between the two arms (e.g. from
    unrelated calls elsewhere in the process), their dropout masks would
    differ even with an identical permutation. Resetting every RNG this
    process could plausibly touch, right before each arm, removes that
    degree of freedom entirely."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: this codebase calls into a third-party pretrained ViT/JEPA
    # stack that may include ops without a deterministic CUDA kernel: fail
    # LOUDLY via a printed warning rather than crashing the whole run over an
    # op we don't control and cannot swap out.
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_checkpointed_training(model, theta0_state: dict, dataset: ArmDataset, seed: int,
                                 checkpoints: list[int], ckpt_dir: Path) -> dict:
    """ONE continuous training run per (arm, pair) -- pausing to snapshot
    model state at each update count in `checkpoints` without resetting, so
    the 500-update primary endpoint and its 200-update diagnostic point come
    from the SAME trajectory, not two separate runs (the 200-update value is
    logged as a diagnostic only -- it never routes or stops training, per
    section 4). Change E: full RNG reset (not just the local permutation
    generator) at the start of every attempt, and no per-arm batch-size
    change -- OOM triggers one whole-run retry at the SAME batch size;
    a second failure propagates as an implementation failure rather than
    silently shrinking the batch for only one arm of a pair."""
    def _attempt():
        reset_all_rngs(seed)
        model.load_state_dict(theta0_state)
        model.train()
        freeze_encoder_and_projector(model)
        params = unfreeze_predictor_only(model)
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
        gen = torch.Generator().manual_seed(seed)
        n = len(dataset.samples)
        frozen_before = copy.deepcopy({k: v for k, v in model.state_dict().items()
                                          if k.startswith("encoder.") or k.startswith("projector.")})

        step = 0
        results = {}
        max_updates = max(checkpoints)
        remaining = sorted(checkpoints)
        loss_val = None
        while step < max_updates:
            n_batches = max(1, n // BATCH_SIZE)
            perm = torch.randperm(n, generator=gen).tolist()
            for b in range(n_batches):
                if step >= max_updates:
                    break
                idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                if not idx:
                    continue
                batch_samples = [dataset.samples[i] for i in idx]
                batch = _collate(batch_samples, dataset.pipeline, DEVICE)
                batch["action"] = torch.nan_to_num(batch["action"], 0.0)
                info = model.encode(batch)
                emb, act_emb = info["emb"], info["act_emb"]
                ctx_emb, ctx_act = emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE]
                tgt_emb = emb[:, 1:]
                pred_emb = model.predict(ctx_emb, ctx_act)
                # Only the LAST predicted step corresponds to the actual
                # selected (episode_id, t) transition retrieval scored and
                # the curriculum was built on; averaging over all HISTORY_SIZE
                # predicted steps would mix in unselected, incidental
                # context-window predictions and dilute the training signal.
                loss = (pred_emb[:, -1] - tgt_emb[:, -1]).pow(2).mean()
                if torch.isnan(loss):
                    raise FloatingPointError("NaN loss")
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_val = float(loss.item())
                step += 1
                while remaining and step >= remaining[0]:
                    cp = remaining.pop(0)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"seed{seed}_step{cp}.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    results[cp] = {"step": step, "loss": loss_val, "checkpoint": str(ckpt_path)}

        frozen_after = {k: v for k, v in model.state_dict().items()
                           if k.startswith("encoder.") or k.startswith("projector.")}
        for k in frozen_before:
            if not torch.equal(frozen_before[k], frozen_after[k]):
                raise RuntimeError(f"IMPLEMENTATION_FAILURE: frozen tensor {k} changed despite freeze "
                                     f"(Change C requires encoder AND projector frozen)")
        return results

    try:
        return _attempt()
    except (torch.cuda.OutOfMemoryError, FloatingPointError) as e:  # noqa: BLE001
        print(f"    [retry] {type(e).__name__}: {e} -- one whole-run retry, SAME batch size, unchanged "
              f"parameters (Change E: no per-arm batch-size change)", flush=True)
        torch.cuda.empty_cache()
        return _attempt()


def _eval_all(model, eval_slice: dict, guard_slice: dict, pipeline, B_N, W_basis, V_basis, U8, B8) -> dict:
    D = B_N.shape[1]
    model.eval()
    ev_n = evaluate_E_W(model, eval_slice, pipeline, B_N, U8, B8)
    ev_full = evaluate_E_W(model, eval_slice, pipeline, np.eye(D), U8, B8)
    ev_w = evaluate_E_W(model, eval_slice, pipeline, W_basis, U8, B8)
    ev_v = evaluate_E_W(model, eval_slice, pipeline, V_basis, U8, B8) if V_basis.shape[0] > 0 else None
    gg_full = evaluate_E_W(model, guard_slice, pipeline, np.eye(D), U8, B8)
    return {
        "E_N": np.array([v["E_W"] for v in ev_n["per_episode"].values()]) if ev_n["per_episode"] else np.zeros(1),
        "E_N_per_episode": ev_n["per_episode"],
        "E_all": np.array([v["E_all"] for v in ev_full["per_episode"].values()]) if ev_full["per_episode"] else np.zeros(1),
        "E_W": np.array([v["E_W"] for v in ev_w["per_episode"].values()]) if ev_w["per_episode"] else np.zeros(1),
        "E_V": (np.array([v["E_W"] for v in ev_v["per_episode"].values()]) if ev_v and ev_v["per_episode"]
                  else np.zeros(1)),
        "guard_E_all": float(np.mean([v["E_all"] for v in gg_full["per_episode"].values()])) if gg_full["per_episode"] else 0.0,
        "guard_effrank": effective_rank(full_latent_predictions(model, guard_slice, pipeline)),
    }


def write_pilot_report(status: str, extra: dict, wall: WallClockBudget) -> None:
    payload = {"status": status, "timestamp": now_iso(), "wall_hours_used": wall.elapsed_hours(),
                 "wall_hours_budget": MAX_WALL_HOURS, **extra}
    write_atomic(METRICS_PATH, payload)
    md = [f"# Two-arm feasibility pilot (v3, W_soft)", "", f"**Status:** {status}", "",
           f"Wall-clock used: {payload['wall_hours_used']:.2f}h / {MAX_WALL_HOURS}h", ""]
    if "reason" in extra:
        md += [f"Reason: {extra['reason']}", ""]
    REPORT_PATH.write_text("\n".join(md))
    copy_to_downloads(METRICS_PATH, REPORT_PATH, MANIFEST_PATH)
    print(f"\n=== PILOT v3 FINAL: {status} ({wall.elapsed_hours():.2f}h) ===", flush=True)


def main() -> int:
    mark_v1_invalidated()
    note_v2_result()
    initial_elapsed = load_cumulative_wall_hours()
    wall = WallClockBudget(MAX_WALL_HOURS, initial_elapsed_hours=initial_elapsed)
    print(f"=== need_two_arm_pilot v3 (W_soft) INIT ({now_iso()}) ===", flush=True)
    if initial_elapsed > 0:
        print(f"  resuming: {initial_elapsed:.2f}h already charged against the {MAX_WALL_HOURS}h budget from a "
              f"prior (interrupted) attempt", flush=True)
    manifest = build_splits()
    assert_disjoint_splits(manifest)

    # User-directed patch (minimal changes before launch), item 1: load epoch
    # 7 FIRST -- every downstream geometry step (Lever-0 basis, U8/B8, region
    # anchors, retrieval, evaluation) now depends on this exact model object.
    model = load_partial_model()   # epoch 7 -- see checkpoints.PARTIAL_EPOCH
    theta0_state = copy.deepcopy(model.state_dict())
    alphabet = AlphabetLookup()
    # Fit on replay_train only (Change P0-6, carried forward): route_val backs
    # eval_slice, so fitting on it too would leak eval-set action statistics
    # into the (frozen, reused-everywhere) pipeline before evaluation begins.
    pipeline = fit_pipeline(manifest["episode_ids"]["replay_train"])

    # Items 2-4: re-encode the ORIGINAL fixed Lever-0 fitting episodes with
    # the explicit epoch-7 model and mechanically recompute residual_basis/
    # U8/B8 using the original parameters and seeds (no tuning); item 5:
    # save under need_two_arm_v2_* names, never touching the six-arm tree's
    # own lever0_residual_basis.npz / tessellation_params_residualized.npz.
    print("=== recomputing Lever-0 residual basis + U8/B8 in epoch-7 coordinates (v2) ===", flush=True)
    v2_lever0 = build_v2_lever0_and_directions(model)
    lever0_basis = {"residual_basis": v2_lever0["residual_basis"], "k_removed": v2_lever0["k_removed"],
                       "energy_removed": v2_lever0["energy_removed"]}
    U8, B8 = v2_lever0["U8"], v2_lever0["B8"]
    U4 = U8[:M_ACTION_DIRS]
    print(f"  v2 Lever-0: k_removed={v2_lever0['k_removed']} (epoch-7 coordinates; NOT compared against the "
          f"epoch-20 a0_residualized_report.json record -- a different checkpoint's latents legitimately give "
          f"a different k)", flush=True)
    write_progress("lever0_done", wall, {"k_removed": v2_lever0["k_removed"]})

    fingerprint = compute_fingerprint(model, pipeline, alphabet, U8, v2_lever0["residual_basis"], manifest)
    print(f"  fingerprint: checkpoint_sha256={fingerprint['checkpoint_sha256'][:16]}... "
          f"encoder_hash={fingerprint['encoder_hash'][:16]}... u8_hash={fingerprint['u8_hash'][:16]}... "
          f"residual_basis_hash={fingerprint['residual_basis_hash'][:16]}... "
          f"code_commit={fingerprint['code_commit'][:12]}", flush=True)

    print(f"=== building v3 region/candidate geometry (epoch-7 coordinates, W_MODE={W_MODE!r}, "
          f"no v1/v2 cache) ===", flush=True)
    # U4 passed explicitly -- build_regions_and_cells must never fall
    # through to its own load_frozen_directions() (epoch-20) in this path.
    build_result = get_or_build_v3_candidates(manifest, lever0_basis, model, pipeline, alphabet, wall, fingerprint,
                                                   U4=U4)
    if build_result["status"] != "OK":
        write_pilot_report("RETRIEVAL_INFEASIBLE", {"reason": f"E0 build failed: {build_result['status']}"}, wall)
        return 1
    candidates = build_result["candidates"]
    anchors = build_result["anchors"]
    print(f"  candidates (frozen deterministic order, by eta_c descending -- NOT forced to match any prior "
          f"attempt's N00 identity): {[(c['id'], c['region'], c['action']) for c in candidates]}", flush=True)
    write_progress("geometry_done", wall, {"candidates": [c["id"] for c in candidates]})

    # Resumability: the retrieval cache is gated by the SAME fingerprint as
    # the frozen-geometry cache -- a hit skips the (expensive) Stage 1/2
    # scan entirely and goes straight to the already-chosen cell's curricula.
    cached_retrieval = load_v3_retrieval_cache(fingerprint)
    if cached_retrieval is not None:
        print(f"  reusing persisted v3 retrieval cache (fingerprint match): chosen={cached_retrieval['chosen_id']}",
              flush=True)
        chosen = next(c for c in candidates if c["id"] == cached_retrieval["chosen_id"])
        curricula = cached_retrieval["curricula"]
        retrieval_stats = cached_retrieval["retrieval_stats"]
    else:
        print(f"=== targeted retrieval (pilot cascade: >={K_TARGET_THRESHOLD}->K={PILOT_K_TARGET}, "
              f">={K_FALLBACK_THRESHOLD}->K={PILOT_K_FALLBACK}), fixed {N_OCCURRENCES_CAP}-occurrence "
              f"SHA-ordered prefix, {RETRIEVAL_MAX_MINUTES:.0f}-min SAFETY ABORT, per-episode "
              f"cap={PER_EPISODE_CAP} ===", flush=True)
        curricula_by_cell, retrieval_stats = run_targeted_retrieval(
            candidates, manifest["episode_ids"]["retrieval_pool"], anchors, lever0_basis, pipeline, model, alphabet,
            seed=SEED, max_minutes=RETRIEVAL_MAX_MINUTES, min_common_eligible=PILOT_MIN_COMMON_ELIGIBLE,
            k_target_threshold=K_TARGET_THRESHOLD, k_target=PILOT_K_TARGET, k_fallback=PILOT_K_FALLBACK,
            n_occurrences_cap=N_OCCURRENCES_CAP, per_episode_cap=PER_EPISODE_CAP, require_exact_k=True)

        chosen = None
        for cand in candidates:   # frozen order: first retrieval-feasible cell wins, not best-scoring
            if curricula_by_cell.get((cand["region"], cand["action"])) is not None:
                chosen = cand
                break
        if chosen is None:
            write_pilot_report("RETRIEVAL_INFEASIBLE",
                                  {"candidates_tried": [c["id"] for c in candidates],
                                    "retrieval_stats": retrieval_stats}, wall)
            return 1
        curricula = curricula_by_cell[(chosen["region"], chosen["action"])]
        save_v3_retrieval_cache(fingerprint, chosen["id"], chosen["region"], chosen["action"], curricula,
                                    retrieval_stats)

    need_curric, random_curric = curricula["conditional_need"], curricula["random_traj"]
    write_progress("retrieval_done", wall, {"chosen_candidate": chosen["id"], "region": chosen["region"],
                                                "action": chosen["action"]})
    # run_targeted_retrieval(require_exact_k=True) already guarantees every
    # arm has EXACTLY k_use segments (Change F items 2-4) -- no post-hoc
    # min()-truncation ("MIN_COMMON_K") needed or permitted here anymore.
    need_segments = need_curric["segments"]
    k_use = len(need_segments)
    assert len(random_curric["segments"]) == k_use, (
        "IMPLEMENTATION_FAILURE: require_exact_k=True did not produce matched arm cardinality")
    traj_pool = random_curric["_pool_for_resampling"]
    chosen_geom = chosen["geometry"]
    print(f"  chosen: {chosen['id']} (region={chosen['region']}, action={chosen['action']}); K={k_use}; "
          f"need_curriculum={need_curric['attrition']}; random_traj eligible pool={len(traj_pool)}", flush=True)
    print(f"  chosen geometry: w_mode={chosen_geom.get('w_mode')} dim_T={chosen_geom['dim_T']} "
          f"dim_V={chosen_geom['dim_V']} dim_W={chosen_geom['dim_W']} q={chosen_geom['q']} "
          f"eta_c={chosen_geom['eta_c']:.4f}"
          + (f" w_soft_singular_values={np.round(chosen_geom['w_soft_singular_values'], 4).tolist()}"
               if "w_soft_singular_values" in chosen_geom else ""), flush=True)

    write_atomic(MANIFEST_PATH, {
        "timestamp": now_iso(), "fingerprint": fingerprint,
        "chosen_candidate": chosen["id"], "region": chosen["region"], "action": chosen["action"],
        "candidates_considered_order": [c["id"] for c in candidates],
        # User-directed: geometry diagnostics for the chosen candidate --
        # dim_T/dim_V/dim_W plus, in soft mode, EVERY singular value of
        # C = V_basis @ T_basis.T (never used to gate feasibility, recorded
        # for inspection only).
        "chosen_geometry_diagnostics": {
            "w_mode": chosen_geom.get("w_mode"), "dim_T": chosen_geom["dim_T"], "dim_V": chosen_geom["dim_V"],
            "dim_W": chosen_geom["dim_W"], "q": chosen_geom["q"], "eta_c": chosen_geom["eta_c"],
            "w_soft_singular_values": (chosen_geom["w_soft_singular_values"].tolist()
                                          if "w_soft_singular_values" in chosen_geom else None),
        },
        "k_cascade": {"k_target_threshold": K_TARGET_THRESHOLD, "k_target": PILOT_K_TARGET,
                        "k_fallback_threshold": K_FALLBACK_THRESHOLD, "k_fallback": PILOT_K_FALLBACK,
                        "min_common_eligible": PILOT_MIN_COMMON_ELIGIBLE, "per_episode_cap": PER_EPISODE_CAP,
                        "n_occurrences_cap": N_OCCURRENCES_CAP},
        "retrieval_stats": retrieval_stats, "k_use": k_use,
        "need_curriculum": {"segments": [{"eid": s["eid"], "t": s["t"], "region": s["region"], "letter": s["letter"],
                                              "score": s["score"], "content_hash": s["content_hash"]}
                                             for s in need_segments],
                              "attrition": need_curric["attrition"]},
        "random_traj_pool_size": len(traj_pool),
        "n_pairs": N_PAIRS, "pair_seeds": PAIR_SEEDS,
    })

    eval_slice = build_cell_eval_slice(manifest["episode_ids"]["route_val"], chosen["region"], chosen["action"],
                                          anchors, lever0_basis, pipeline, alphabet, seed=SEED + 1, model=model)
    guard_ids = manifest["episode_ids"]["global_guard"][:80]
    guard_slice = build_eval_slice(guard_ids, n_per_episode=4, seed=SEED + 2)
    try:
        assert_slice_nonempty(eval_slice, "cell eval_slice", min_episodes=MIN_EVAL_EPISODES)
        assert_slice_nonempty(guard_slice, "global guard_slice", min_episodes=MIN_EVAL_EPISODES)
    except RuntimeError as e:
        write_pilot_report("RETRIEVAL_INFEASIBLE", {"reason": str(e)}, wall)
        return 1
    print(f"  eval_slice: {len(eval_slice['samples'])} samples; guard_slice: {len(guard_slice['samples'])} samples",
          flush=True)

    # ONE shared replay dataset, sized to EXACTLY k_use, reused byte-for-byte
    # by both arms across ALL 5 pairs (Change F item 3).
    replay_seed = SEED + 100
    replay_ds = build_exact_k_replay_dataset(manifest["episode_ids"]["replay_train"], k_use, pipeline,
                                                 seed=replay_seed)
    need_ds = build_arm_dataset(need_segments, replay_ds, pipeline)
    assert_segments_survive(need_ds, need_segments, len(need_segments))
    print(f"  need_curriculum dataset: {len(need_ds.samples)} samples "
          f"({len(need_segments)} curriculum + {len(replay_ds.samples)} shared replay, fixed across all "
          f"{N_PAIRS} pairs)", flush=True)

    geom = chosen["geometry"]
    B_N, W_basis, V_basis = geom["B_N"], geom["W_basis"], geom["V_basis"]

    model.load_state_dict(theta0_state)
    pre = _eval_all(model, eval_slice, guard_slice, pipeline, B_N, W_basis, V_basis, U8, B8)
    pre_E_N = float(np.mean(pre["E_N"]))
    print(f"  pre-training (theta0): E_N={pre_E_N:.4f} E_all={np.mean(pre['E_all']):.4f} "
          f"guard_E_all={pre['guard_E_all']:.4f} guard_effrank={pre['guard_effrank']:.3f}", flush=True)

    node_dir = PILOT_DIR / chosen["id"]
    per_pair = {"need_curriculum": {}, "matched_random": {}}
    random_curricula_by_pair = {}
    for pair_idx in PAIR_SEEDS:
        # Section 4: "an independently sampled matched-random curriculum" per
        # pair, while need_curriculum's segments stay fixed across all pairs.
        resample_seed = stable_seed(SEED, "resample", pair_idx)
        random_curric_j = resample_random_traj_curriculum(traj_pool, k_use, seed=resample_seed,
                                                               per_episode_cap=PER_EPISODE_CAP)
        random_curricula_by_pair[pair_idx] = random_curric_j
        random_ds_j = build_arm_dataset(random_curric_j["segments"], replay_ds, pipeline)
        assert_segments_survive(random_ds_j, random_curric_j["segments"], k_use)

        for arm_name, ds in (("need_curriculum", need_ds), ("matched_random", random_ds_j)):
            ckpt_dir = node_dir / arm_name / f"pair{pair_idx}"
            primary_ckpt = ckpt_dir / f"seed{PAIR_SEEDS[pair_idx]}_step{N_UPDATES_PRIMARY}.pt"
            diag_ckpt = ckpt_dir / f"seed{PAIR_SEEDS[pair_idx]}_step{N_UPDATES_DIAGNOSTIC}.pt"
            # Resumability: if a prior (interrupted) run already produced
            # BOTH checkpoints for this exact (pair, arm), reuse them instead
            # of retraining -- the curriculum/replay/seed/RNG-reset are all
            # fully deterministic (Check 4), so a retrained run would give an
            # identical result anyway; this only saves the wasted recompute.
            # A run that died BEFORE reaching the 200-update diagnostic
            # checkpoint leaves neither file, so at most 200 updates of work
            # are ever lost to an interruption.
            if primary_ckpt.exists() and diag_ckpt.exists():
                print(f"  pair={pair_idx} arm={arm_name}: RESUMING -- both checkpoints already exist, "
                      f"skipping training", flush=True)
                cps = {N_UPDATES_DIAGNOSTIC: {"checkpoint": str(diag_ckpt), "loss": None},
                         N_UPDATES_PRIMARY: {"checkpoint": str(primary_ckpt), "loss": None}}
            else:
                print(f"  pair={pair_idx} training arm={arm_name} seed={PAIR_SEEDS[pair_idx]} "
                      f"({N_UPDATES_PRIMARY} updates, diagnostic at {N_UPDATES_DIAGNOSTIC})...", flush=True)
                cps = run_checkpointed_training(model, theta0_state, ds, PAIR_SEEDS[pair_idx],
                                                    [N_UPDATES_DIAGNOSTIC, N_UPDATES_PRIMARY], ckpt_dir)

            model.load_state_dict(torch.load(cps[N_UPDATES_PRIMARY]["checkpoint"], map_location=DEVICE))
            post = _eval_all(model, eval_slice, guard_slice, pipeline, B_N, W_basis, V_basis, U8, B8)

            model.load_state_dict(torch.load(cps[N_UPDATES_DIAGNOSTIC]["checkpoint"], map_location=DEVICE))
            model.eval()
            diag_ev = evaluate_E_W(model, eval_slice, pipeline, B_N, U8, B8)
            diag_E_N = float(np.mean([v["E_W"] for v in diag_ev["per_episode"].values()])) if diag_ev["per_episode"] else float("nan")

            per_pair[arm_name][pair_idx] = {**post, "diagnostic_200_E_N_mean": diag_E_N,
                                                "loss_primary": cps[N_UPDATES_PRIMARY]["loss"]}
            print(f"    pair={pair_idx} arm={arm_name}: E_N_mean={np.mean(post['E_N']):.4f} "
                  f"(diagnostic@200={diag_E_N:.4f}) guard_E_all={post['guard_E_all']:.4f} "
                  f"guard_effrank={post['guard_effrank']:.3f} (wall {wall.elapsed_hours():.2f}h)", flush=True)
            if wall.exhausted():
                write_pilot_report("TIME_BUDGET_EXHAUSTED",
                                      {"reason": f"wall-clock exhausted mid-training at pair={pair_idx} "
                                                   f"arm={arm_name}"}, wall)
                return 1

        # Resumability / "keep me posted": a running tally after every pair,
        # readable even if the process is later interrupted or the machine
        # is unreachable -- includes the per-pair-so-far E_N means and the
        # running gain delta for every COMPLETED pair.
        pairs_done_so_far = [p for p in PAIR_SEEDS if p in per_pair["need_curriculum"] and p in per_pair["matched_random"]]
        write_progress("pair_completed", wall, {
            "pair_just_completed": pair_idx, "pairs_done": pairs_done_so_far, "n_pairs_total": N_PAIRS,
            "pre_E_N": pre_E_N,
            "per_pair_E_N_mean_so_far": {
                arm: {str(p): float(np.mean(per_pair[arm][p]["E_N"])) for p in pairs_done_so_far}
                for arm in ("need_curriculum", "matched_random")},
        })

    print("=== computing metrics ===", flush=True)
    gains, per_pair_E_N_mean = {}, {"need_curriculum": {}, "matched_random": {}}
    for arm in ("need_curriculum", "matched_random"):
        pair_gain = {}
        for j in PAIR_SEEDS:
            e_n_mean = float(np.mean(per_pair[arm][j]["E_N"]))
            per_pair_E_N_mean[arm][j] = e_n_mean
            pair_gain[j] = (pre_E_N - e_n_mean) / max(pre_E_N, 1e-8)
        gains[arm] = pair_gain

    deltas = [gains["need_curriculum"][j] - gains["matched_random"][j] for j in PAIR_SEEDS]
    mean_delta = float(np.mean(deltas))
    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    ci_mean, ci_lo, ci_hi = pair_level_bootstrap_ci(deltas, seed=SEED + 9)

    # Secondary, episode-level (not pair-level) interval -- explicit eid
    # pairing per pair, then a hierarchical (pair-then-episode) bootstrap.
    # Reported alongside, never in place of the primary 5-pair-level CI
    # above (section 4: "must not replace the five paired experimental units").
    diffs_by_pair, per_pair_ci = {}, {}
    for j in PAIR_SEEDS:
        random_arr, need_arr = paired_eid_arrays(
            per_pair["matched_random"][j]["E_N_per_episode"], per_pair["need_curriculum"][j]["E_N_per_episode"])
        diffs_by_pair[j] = random_arr - need_arr
        pm, plo, phi = paired_bootstrap_ci(random_arr, need_arr, seed=SEED + 90 + j)
        per_pair_ci[j] = {"mean": pm, "ci95": [plo, phi], "n_episodes": int(len(random_arr))}

    need_gg_regression = [
        (per_pair["need_curriculum"][j]["guard_E_all"] - pre["guard_E_all"]) / max(pre["guard_E_all"], 1e-8)
        for j in PAIR_SEEDS]
    need_effrank_drop = [
        (pre["guard_effrank"] - per_pair["need_curriculum"][j]["guard_effrank"]) / max(pre["guard_effrank"], 1e-8)
        for j in PAIR_SEEDS]
    global_ok = all(r <= GLOBAL_REGRESSION_MAX for r in need_gg_regression)
    effrank_ok = all(d <= EFFRANK_DROP_MAX for d in need_effrank_drop)
    guard_ok = global_ok and effrank_ok

    # DECISION RULE (TWO_ARM_EXPERIMENT_REQUIRED_CHANGES.md section 4) --
    # this replaces v1's STRONG/FEASIBILITY/BELOW_THRESHOLD tiering entirely.
    if mean_delta > 0 and n_positive >= MIN_PAIRS_FOR_SIGN and guard_ok:
        label = "TWO_ARM_POSITIVE"
    elif mean_delta < 0 and n_negative >= MIN_PAIRS_FOR_SIGN and guard_ok:
        label = "TWO_ARM_NEGATIVE"
    else:
        label = "INCONCLUSIVE"

    metrics = {
        "status": label, "timestamp": now_iso(), "wall_hours_used": wall.elapsed_hours(),
        "wall_hours_budget": MAX_WALL_HOURS, "chosen_candidate": chosen["id"], "region": chosen["region"],
        "action": chosen["action"], "k_use": k_use, "n_pairs": N_PAIRS,
        "pre_E_N": pre_E_N, "pre_E_all": float(np.mean(pre["E_all"])),
        "pre_guard_E_all": pre["guard_E_all"], "pre_guard_effrank": pre["guard_effrank"],
        "gains_G": gains, "deltas_per_pair": {str(j): d for j, d in zip(PAIR_SEEDS, deltas)},
        "mean_delta": mean_delta, "n_positive": n_positive, "n_negative": n_negative,
        "pair_level_bootstrap_ci_primary": {"mean": ci_mean, "ci95": [ci_lo, ci_hi]},
        "episode_level_hierarchical_ci_secondary": per_pair_ci,
        "need_global_regression_per_pair": need_gg_regression, "need_effrank_drop_per_pair": need_effrank_drop,
        "global_regression_ok": global_ok, "effrank_ok": effrank_ok, "guard_ok": guard_ok,
        "fingerprint": fingerprint,
        "per_pair_detail": {
            arm: {str(j): {"E_N_mean": float(np.mean(per_pair[arm][j]["E_N"])),
                             "E_all_mean": float(np.mean(per_pair[arm][j]["E_all"])),
                             "E_W_mean": float(np.mean(per_pair[arm][j]["E_W"])),
                             "E_V_mean": float(np.mean(per_pair[arm][j]["E_V"])),
                             "diagnostic_200_E_N_mean": per_pair[arm][j]["diagnostic_200_E_N_mean"],
                             "guard_E_all": per_pair[arm][j]["guard_E_all"],
                             "guard_effrank": per_pair[arm][j]["guard_effrank"]}
                    for j in PAIR_SEEDS}
            for arm in ("need_curriculum", "matched_random")},
        "random_traj_curricula_by_pair": {
            str(j): {"segments_selected": random_curricula_by_pair[j]["attrition"]["segments_selected"],
                       "episode_ids": random_curricula_by_pair[j]["episode_ids"]}
            for j in PAIR_SEEDS},
        "not_claimed": ["validated bisimulation", "FSM correctness", "encoder representation expansion",
                          "general capability improvement", "a confirmatory (vs. feasibility) result",
                          "a universal claim that the method does or does not beat random curriculum selection"],
    }
    write_atomic(METRICS_PATH, metrics)

    md = [f"# Two-arm feasibility pilot (v3, W_soft)", "", f"**Result: {label}**", "",
           f"Candidate: {chosen['id']} (region={chosen['region']}, action={chosen['action']}), K={k_use}, "
           f"N_PAIRS={N_PAIRS}", "",
           f"Per-pair Delta_j (need gain - random gain): {deltas}", "",
           f"mean Delta = {mean_delta:.4f}  |  positive pairs = {n_positive}/{N_PAIRS}  |  "
           f"negative pairs = {n_negative}/{N_PAIRS}", "",
           f"Primary pair-level bootstrap CI (N_PAIRS={N_PAIRS}, coarse): "
           f"mean={ci_mean:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]", "",
           f"Secondary episode-level per-pair CIs: {per_pair_ci}", "",
           f"Global-latent regression per pair: {need_gg_regression} (ok={global_ok}); "
           f"effective-rank drop per pair: {need_effrank_drop} (ok={effrank_ok})", "",
           "## Not claimed", ""] + [f"- {x}" for x in metrics["not_claimed"]]
    REPORT_PATH.write_text("\n".join(md))
    copy_to_downloads(METRICS_PATH, REPORT_PATH, MANIFEST_PATH)
    print(f"\n=== PILOT v3 FINAL: {label} ({wall.elapsed_hours():.2f}h) ===", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    import os
    os._exit(code)
