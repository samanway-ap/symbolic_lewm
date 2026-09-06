"""Phases R + G + K driver: build the atlas over occupied cells, retrieve
unseen trajectories per cell, run curriculum rounds, and write verdicts.

Runs only if Phase S's gate passed -- checked here rather than assumed, so
this file cannot be run out of order by accident.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.atlas import (
    MIN_CANDIDATE_TRAJECTORIES, MIN_HELDOUT_SEGMENTS, bit_informativeness,
    coherence_fraction, load_atlas, merge_underpowered, new_record, save_atlas, seed_weights,
)
from atlas.walk import episode_state, reset_reference, walk
from oracle.production_oracle import build_reset_windows
from corpus.latent_corpus import build_corpus, test_ids
from curriculum.decide import apply_multiplicity, bootstrap_pvalue, decide
from curriculum.round import run_round
from local_import import load_local  # noqa: E402
HORIZON_LETTERS = load_local("symbolic_eval_plan_ranking", "eval/plan_ranking.py").HORIZON_LETTERS
from oracle.droid_actions import load_actions_for_episodes
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, get_model
from predicates.functions import build_predicate_fns
from retrieval.geometric import (
    ACTION_MAG_FLOOR_PCT, PATH_LENGTH_FLOOR_PCT, greedy_dispersion,
    screen_episode, write_retrieval_report,
)
from tessellation.guard import assert_not_tessellation
from tessellation.hyperplanes import Tessellation

import os

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SEED = 3072
SEEDS = [0, 1, 2]
K_TRAJECTORIES = 10
MAX_CELLS_TO_TEST = int(os.environ.get("RGK_MAX_CELLS", "3"))
RETRIEVAL_POOL_SCAN = int(os.environ.get("RGK_POOL_SCAN", "260"))
RETRIEVAL_POSITIONS = 40

# Granularity override. The PRE-REGISTERED primary is best_m from Phase T
# (=8). At m=8 there are 123 occupied cells and geometric retrieval is starved
# at the containment filter (measured: 2/255 unseen episodes qualify), so no
# region reaches a curriculum round. Setting RGK_M runs the SAME machinery at a
# coarser granularity as a clearly-labelled SUPPLEMENTARY analysis -- it does
# not replace or amend the pre-registered primary result.
M_OVERRIDE = os.environ.get("RGK_M")
RESULTS_SUFFIX = os.environ.get("RGK_SUFFIX", "")


def unseen_pool(exclude: set[int], n: int, seed: int = SEED) -> list[int]:
    """DROID episodes outside the drawer family AND outside episode_split."""
    from oracle.droid_streaming import _episodes_meta
    meta = _episodes_meta()
    all_ids = meta["episode_index"].to_numpy()
    pool = np.setdiff1d(all_ids, np.fromiter(exclude, dtype=all_ids.dtype, count=len(exclude)))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())


def samples_from_pixels(pixels, raw, n_positions):
    """Slice one streamed episode into (context+target) training windows."""
    out = []
    for t in range(0, n_positions - HISTORY_SIZE - 1):
        out.append({"pixels": pixels[t:t + HISTORY_SIZE + 1].copy(),
                     "raw_action": raw[t:t + HISTORY_SIZE + 1].copy()})
    return out


def build_training_samples(episode_ids: list[int], n_positions: int, label: str) -> list[dict]:
    """The corpus deliberately discards pixels (a full pixel stack is tens of
    GB), but fine-tuning needs them -- so re-stream just the handful of
    SELECTED episodes. K is ~10 per cell, so this is seconds, not hours."""
    from oracle.droid_streaming import stream_many_windows
    if not episode_ids:
        return []
    n_raw = n_positions * FRAMESKIP
    actions = load_actions_for_episodes(episode_ids, max_frames=n_raw)
    ok = [e for e in episode_ids if e in actions and actions[e].shape[0] >= n_raw]
    windows = stream_many_windows(ok, num_frames=n_positions, frameskip=FRAMESKIP,
                                    max_workers=8, per_episode_timeout_s=90.0)
    out = []
    for e in sorted(set(ok) & set(windows.keys())):
        if windows[e].shape[0] != n_positions:
            continue
        raw = actions[e][:n_raw].reshape(n_positions, FRAMESKIP, RAW_ACTION_DIM).astype(np.float32)
        out += samples_from_pixels(windows[e], raw, n_positions)
    print(f"    [{label}] {len(out)} training windows from {len(episode_ids)} episodes", flush=True)
    return out


def main():
    spec = json.loads((OUT_DIR / "spectral_report.json").read_text())
    if not spec.get("gate", {}).get("passes"):
        print("Phase S gate did NOT pass -- refusing to run R/G/K.", flush=True)
        return
    print("Phase S gate passed; running R -> G -> K.", flush=True)

    tess_meta = json.loads((OUT_DIR / "tessellation.json").read_text())
    tess_params = np.load(OUT_DIR / "tessellation_params.npz")
    best_m = int(tess_meta["best_m"])
    if M_OVERRIDE:
        best_m = int(M_OVERRIDE)
        print(f"*** SUPPLEMENTARY RUN at m={best_m} (pre-registered primary is "
              f"m={int(tess_meta['best_m'])}); labelled as such in the report. ***", flush=True)
    tess = Tessellation(U=tess_params[f"U_m{best_m}"], B=tess_params[f"B_m{best_m}"],
                          m=best_m, occupied={}, seed=SEED)

    corpus = build_corpus(test_ids(), 80, "heldout", keep_raw_actions=True)
    with open(OUT_DIR / "best_machine.pkl", "rb") as f:
        best = pickle.load(f)
    machine, subset = best["machine"], best["subset"]
    assert_not_tessellation(subset, "phase_rgk.main")

    # ---------- Phase R: eligibility, merging, coherence ----------
    Z = corpus.flat_latents()
    cell_ids_flat = tess.cell_ids(Z).reshape(corpus.latents.shape[0], corpus.latents.shape[1])
    ep_primary_cell = np.array([np.bincount(row).argmax() for row in cell_ids_flat])
    uniq, counts = np.unique(ep_primary_cell, return_counts=True)
    heldout_segments = {int(c): int(n) for c, n in zip(uniq, counts)}
    print(f"\nPhase R: {len(heldout_segments)} cells hold held-out episodes at m={best_m}", flush=True)

    info = bit_informativeness(tess.bits(Z))
    groups, merged_counts, merge_log = merge_underpowered(heldout_segments, best_m,
                                                            bit_informativeness=info)
    print(f"  after merging below the {MIN_HELDOUT_SEGMENTS}-segment floor: {len(merged_counts)} regions "
          f"({len(merge_log)} merges)", flush=True)

    atlas = load_atlas()
    fns = [build_predicate_fns()[n] for n in subset]

    # reset reference: a word must START with a reset symbol or the machine
    # falls into the sink and every downstream diagnostic becomes 1.0 -- see
    # atlas/walk.py
    rng0 = np.random.default_rng(SEED)
    fit0 = sorted(rng0.choice(corpus.episode_ids, size=min(200, len(corpus.episode_ids)),
                                replace=False).tolist())
    fa0 = load_actions_for_episodes(fit0, max_frames=40)
    pipeline0 = ActionPipeline.fit(np.concatenate([v for v in fa0.values() if len(v) > 0], axis=0))
    reset_windows = build_reset_windows(pipeline0)
    reset_names, reset_Z = reset_reference(reset_windows)

    for rep, members in groups.items():
        bits = format(rep, f"0{best_m}b")
        rec = atlas["cells"].get(bits) or new_record(rep, bits, merged_counts[rep])
        rec["merge_history"] = [m for m in merge_log if m["into"] == bits]
        member_eps = np.isin(ep_primary_cell, members)
        rec["eligibility_counts"] = {"heldout_segments": int(merged_counts[rep]),
                                       "min_required": MIN_HELDOUT_SEGMENTS}
        # behavioural coherence under the FROZEN discovered machine
        states = [episode_state(machine, corpus.latents[i], reset_names, reset_Z)
                  for i in np.flatnonzero(member_eps)[:200]]
        frac, ok = coherence_fraction(np.array(states), machine.size)
        rec["coherence_fraction"] = frac
        if not ok:
            rec["status"] = "BLOCKED_INCOHERENT"
            rec["block_reason"] = (f"modal-state fraction {frac:.3f} does not clear chance "
                                     f"(1/{machine.size}) by the required margin")
        atlas["cells"][bits] = rec
    save_atlas(atlas)

    # ---------- abstraction-failure rate per cell (seed weighting) ----------
    fail_by_cell = {}
    for rep, members in groups.items():
        member_eps = np.flatnonzero(np.isin(ep_primary_cell, members))[:120]
        fails = total = 0
        for i in member_eps:
            f_, t_, _, _ = walk(machine, corpus.latents[i], corpus.symbol_seq(i),
                                  reset_names, reset_Z, fns)
            fails += f_
            total += t_
        fail_by_cell[rep] = fails / total if total else 0.0
    print(f"  abstraction-failure rate by region: "
          f"{ {format(k, f'0{best_m}b'): round(v, 4) for k, v in fail_by_cell.items()} }", flush=True)

    eligible = [r for r in groups
                if atlas["cells"][format(r, f"0{best_m}b")]["status"] == "UNEXPLORED"]
    if not eligible:
        print("no eligible UNEXPLORED regions after coherence screening; nothing to test.", flush=True)
        (OUT_DIR / f"curriculum_results{RESULTS_SUFFIX}.json").write_text(json.dumps(
            {"tested": [], "note": "no eligible regions"}, indent=2))
        return

    w = seed_weights(fail_by_cell, eligible)
    rng = np.random.default_rng(SEED)
    n_test = min(MAX_CELLS_TO_TEST, len(eligible))
    chosen_cells = rng.choice(eligible, size=n_test, replace=False, p=w).tolist()
    print(f"  seeded {n_test} region(s) by abstraction-failure weight: "
          f"{[format(c, f'0{best_m}b') for c in chosen_cells]}", flush=True)

    # ---------- base state + replay pool ----------
    model = get_model()
    base_state = copy.deepcopy(model.state_dict())
    rng2 = np.random.default_rng(SEED)
    fit_ids = sorted(rng2.choice(corpus.episode_ids, size=min(200, len(corpus.episode_ids)),
                                   replace=False).tolist())
    fit_actions = load_actions_for_episodes(fit_ids, max_frames=40)
    all_raw = np.concatenate([v for v in fit_actions.values() if len(v) > 0], axis=0)
    pipeline = ActionPipeline.fit(all_raw)

    # Replay stream = the ORIGINAL training mix (drawer-family TRAIN split),
    # re-streamed for pixels. Held out from every evaluation set.
    from corpus.latent_corpus import train_ids as _train_ids
    replay_eps = sorted(np.random.default_rng(SEED + 88).choice(
        _train_ids(), size=12, replace=False).tolist())
    replay_samples = build_training_samples(replay_eps, RETRIEVAL_POSITIONS, "replay")
    results = []
    exclude = set(int(e) for e in corpus.episode_ids) | set(json.loads(
        (OUT_DIR / "scope_episode_ids.json").read_text())["primary"]["episode_ids"])

    for rep in chosen_cells:
        bits = format(rep, f"0{best_m}b")
        print(f"\n=== Phase G: retrieval for region {bits} ===", flush=True)
        pool = unseen_pool(exclude, RETRIEVAL_POOL_SCAN, seed=SEED + rep)
        scan = build_corpus(pool, RETRIEVAL_POSITIONS, f"unseen_{bits}", keep_raw_actions=True)
        attr = {"considered": len(pool), "encoded": int(scan.n_episodes)}

        plens, amags, screens = [], [], []
        for i in range(scan.n_episodes):
            s = screen_episode(scan.latents[i], scan.raw_actions[i], tess, rep, -np.inf, -np.inf)
            screens.append(s)
            plens.append(s["path_length"])
            amags.append(s["action_magnitude"])
        path_floor = float(np.percentile(plens, PATH_LENGTH_FLOOR_PCT))
        act_floor = float(np.percentile(amags, ACTION_MAG_FLOOR_PCT))

        used = set()
        for r in atlas["cells"].values():
            used |= set(r.get("trajectory_ids_used", []))

        c_ok = [i for i, s in enumerate(screens) if s["containment_ok"]]
        d_ok = [i for i in c_ok if screens[i]["dwell_ok"]]
        n_ok = [i for i in d_ok if screens[i]["path_length"] >= path_floor
                and screens[i]["action_magnitude"] >= act_floor]
        v_ok = [i for i in n_ok if int(scan.episode_ids[i]) not in used]
        attr.update({"containment": len(c_ok), "dwell": len(d_ok), "non_degenerate": len(n_ok),
                       "novelty": len(v_ok), "rejected_by_non_degeneracy": len(d_ok) - len(n_ok)})
        print(f"  attrition: {attr}", flush=True)

        if len(v_ok) < 1:
            rec = atlas["cells"][bits]
            rec["status"] = "BLOCKED_UNDERPOWERED"
            rec["block_reason"] = f"retrieval returned {len(v_ok)} qualifying unseen trajectories"
            rec["eligibility_counts"]["candidate_trajectories"] = len(v_ok)
            rec["eligibility_counts"]["min_required_candidates"] = MIN_CANDIDATE_TRAJECTORIES
            save_atlas(atlas)
            write_retrieval_report(bits, attr, [], "No qualifying trajectories; region not testable.")
            results.append({"cell": bits, "verdict": "BLOCKED_UNDERPOWERED",
                              "reason": rec["block_reason"], "attrition": attr})
            continue

        reps = np.stack([scan.latents[i].mean(0) for i in v_ok])
        pick = greedy_dispersion(reps, K_TRAJECTORIES, seed=SEED)
        sel_idx = [v_ok[p] for p in pick]
        sel_eps = [int(scan.episode_ids[i]) for i in sel_idx]
        write_retrieval_report(bits, attr, sel_eps)
        print(f"  selected {len(sel_eps)} trajectories by dispersion", flush=True)

        # ---------- Phase K: rounds for this region ----------
        cell_samples = build_training_samples(sel_eps, RETRIEVAL_POSITIONS, f"cell {bits}")
        if not cell_samples:
            rec = atlas["cells"][bits]
            rec["status"] = "BLOCKED_UNDERPOWERED"
            rec["block_reason"] = "selected trajectories failed to re-stream for training"
            save_atlas(atlas)
            results.append({"cell": bits, "verdict": "BLOCKED_UNDERPOWERED",
                              "reason": rec["block_reason"], "attrition": attr})
            continue

        # matched_random: same K, drawn from a DIFFERENT unexplored region
        others = [c for c in eligible if c != rep]
        rand_samples = []
        rand_eps = []
        if others:
            other = int(np.random.default_rng(SEED + rep).choice(others))
            o_ok = [i for i in range(scan.n_episodes)
                    if screen_episode(scan.latents[i], scan.raw_actions[i], tess, other,
                                        path_floor, act_floor)["containment_ok"]]
            rand_eps = [int(scan.episode_ids[i]) for i in o_ok[:K_TRAJECTORIES]]
            rand_samples = build_training_samples(rand_eps, RETRIEVAL_POSITIONS,
                                                    f"matched_random ({format(other, f'0{best_m}b')})")

        local_eps = set(int(corpus.episode_ids[i]) for i in np.flatnonzero(np.isin(ep_primary_cell, groups[rep])))
        cell_mask = (cell_ids_flat[:, :-1] == rep)
        print(f"\n=== Phase K: rounds for region {bits} ({len(local_eps)} local held-out episodes) ===", flush=True)
        per_arm = run_round(bits, rep, SEEDS, base_state, pipeline, cell_samples, rand_samples,
                              replay_samples, test_ids(), local_eps,
                              machine=machine, subset=subset, corpus=corpus, cell_mask_by_ep=cell_mask)

        dec = decide(per_arm["targeted"]["local"], per_arm["control"]["local"],
                       per_arm["targeted"]["global"], per_arm["control"]["global"])
        dec["local_pvalue"] = bootstrap_pvalue(per_arm["targeted"]["local"], per_arm["control"]["local"])
        dec["matched_random_vs_control"] = float(
            np.mean(per_arm["matched_random"]["local"]) - np.mean(per_arm["control"]["local"]))
        dec["targeted_vs_control_point"] = float(
            np.mean(per_arm["targeted"]["local"]) - np.mean(per_arm["control"]["local"]))
        dec["contraction_control_delta"] = float(
            np.mean(per_arm["targeted"]["local_contracted"]) - np.mean(per_arm["targeted"]["local"]))
        dec["confounded_by_contraction"] = bool(
            abs(dec["contraction_control_delta"]) >= abs(dec["targeted_vs_control_point"]))
        if dec["matched_random_vs_control"] > 0 and dec["targeted_vs_control_point"] <= 0:
            dec["TARGETING_SUSPECT"] = ("matched_random improved LOCAL while targeted selection did not -- "
                                          "the geometric targeting is doing nothing here and the whole "
                                          "retrieval mechanism is suspect.")
            print(f"  !! {dec['TARGETING_SUSPECT']}", flush=True)

        rec = atlas["cells"][bits]
        rec["rounds_attempted"] += 1
        rec["trajectory_ids_used"] = sorted(set(rec["trajectory_ids_used"]) | set(sel_eps))
        rec["metric_history"].append({
            "round": rec["rounds_attempted"],
            "local": float(np.mean(per_arm["targeted"]["local"])),
            "global": float(np.mean(per_arm["targeted"]["global"])),
            "symbolic": float(np.mean(per_arm["targeted"]["symbolic"])) if per_arm["targeted"]["symbolic"] else None,
            "ci_low": dec["local_ci95"][0], "ci_high": dec["local_ci95"][1],
        })
        if dec["verdict"] != "REDO_FORGETTING":
            rec["status"] = dec["verdict"]
            rec["block_reason"] = dec["reason"] if dec["verdict"].startswith("BLOCKED") else None
        rec["eligibility_counts"]["candidate_trajectories"] = len(v_ok)
        save_atlas(atlas)

        print(f"  VERDICT {bits}: {dec['verdict']} -- {dec['reason']}", flush=True)
        results.append({"cell": bits, "attrition": attr, "selected_episodes": sel_eps,
                          "matched_random_episodes": rand_eps, "per_arm": per_arm, **dec})

    tested = [r for r in results if "local_ci95" in r]
    if tested:
        apply_multiplicity(tested)
        for r in tested:
            bits = r["cell"]
            if r.get("verdict_bh_adjusted") and r["verdict_bh_adjusted"] != r.get("verdict"):
                atlas["cells"][bits]["status"] = r["verdict_bh_adjusted"]
                atlas["cells"][bits]["block_reason"] = r.get("bh_note")
        print(f"\nBenjamini-Hochberg applied across {len(tested)} region tests", flush=True)

    (OUT_DIR / f"curriculum_results{RESULTS_SUFFIX}.json").write_text(json.dumps(results, indent=2, default=float))
    save_atlas(atlas)
    print("\nwrote curriculum_results.json + atlas.json", flush=True)


if __name__ == "__main__":
    main()
    import sys as _s, os as _o
    _s.stdout.flush()
    _o._exit(0)
