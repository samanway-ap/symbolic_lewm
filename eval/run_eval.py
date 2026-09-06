"""Phase I driver: primary (plan_ranking MRR/top1/top5), secondary
(reachability AUC), statistics (paired bootstrap CI vs control AND vs
shuffled twin), and the §16 non-gameability check, run once at the end on
actual post-training checkpoints. All evaluation is against the canonical
TEST-20% split (episode_split.json) -- never the TRAIN episodes Phase H
trained on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.nongameability import run_nongameability_check
from eval.plan_ranking import HORIZON_LETTERS, N_EVAL_SEGMENTS, evaluate_plan_ranking
from eval.reachability import evaluate_reachability
from eval.stats import compare_arm

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CKPT_DIR = OUT_DIR / "checkpoints"

REAL_ARMS = {"latent_subspace": "shuffled_latent_subspace", "sample_prio": "shuffled_sample_prio"}
CONTROL = "control"
SEEDS = [0, 1, 2]

# Exploratory run's STEP 2: sample_prio must ALSO be compared against the
# time-binned control, not just against control and its shuffled twin
# (preregistration.md). Reported whether or not it is favourable.
EXTRA_BASELINES = {"sample_prio": {"time_binned": "time_binned_sample_prio"}}
# Arms that are themselves controls get their own vs-control CI reported too,
# so the writeup can state the temporal arm's own effect, not only the contrast.
CONTROL_ARMS_TO_REPORT = ["time_binned_sample_prio", "shuffled_sample_prio",
                          "shuffled_latent_subspace"]


def main():
    runs = json.loads((OUT_DIR / "training_runs.json").read_text())
    split = json.loads((OUT_DIR / "episode_split.json").read_text())
    test_ids = split["test_episode_ids"]

    mrr_by = {}
    top1_by = {}
    top5_by = {}
    auc_by = {}
    per_run_detail = []

    for run in runs:
        arm, seed, ckpt_path = run["arm"], run["seed"], run["checkpoint"]
        print(f"\n=== evaluating arm={arm} seed={seed} ===", flush=True)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        pr = evaluate_plan_ranking(state, test_ids, n_segments=N_EVAL_SEGMENTS, horizon=HORIZON_LETTERS)
        rc = evaluate_reachability(state, test_ids, n_segments=200, horizon=HORIZON_LETTERS)
        print(f"  plan_ranking: mrr={pr['mrr']:.4f} top1={pr['top1_acc']:.3f} top5={pr['top5_acc']:.3f} "
              f"| reachability_auc={rc['auc']:.4f}", flush=True)

        mrr_by[(arm, seed)] = pr["mrr"]
        top1_by[(arm, seed)] = pr["top1_acc"]
        top5_by[(arm, seed)] = pr["top5_acc"]
        auc_by[(arm, seed)] = rc["auc"]
        per_run_detail.append({"arm": arm, "seed": seed, **pr, "reachability_auc": rc["auc"]})

        (OUT_DIR / "eval_per_run.json").write_text(json.dumps(per_run_detail, indent=2))

    comparisons = {"mrr": [], "top1_acc": [], "top5_acc": [], "reachability_auc": []}
    for metric_name, table in [("mrr", mrr_by), ("top1_acc", top1_by), ("top5_acc", top5_by),
                                  ("reachability_auc", auc_by)]:
        for real_arm, shuf_arm in REAL_ARMS.items():
            comparisons[metric_name].append(compare_arm(
                table, real_arm, control=CONTROL, shuffled=shuf_arm,
                extra_baselines=EXTRA_BASELINES.get(real_arm)))
        for ctrl_arm in CONTROL_ARMS_TO_REPORT:
            if any(a == ctrl_arm for (a, _) in table):
                comparisons[metric_name].append(compare_arm(table, ctrl_arm, control=CONTROL))

    print("\n=== §16 metric non-gameability check (once, on post-training checkpoints) ===", flush=True)
    ng_checkpoints = {}
    for run in runs:
        if run["seed"] == SEEDS[0]:  # one seed's worth of every arm -- "once", per instruction
            ng_checkpoints[run["arm"]] = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    nongameability = run_nongameability_check(ng_checkpoints, test_ids, n_segments=150, horizon=HORIZON_LETTERS)

    out = {
        "per_run": per_run_detail,
        "comparisons": comparisons,
        "nongameability": nongameability,
        "config": {"n_eval_segments": N_EVAL_SEGMENTS, "horizon": HORIZON_LETTERS, "seeds": SEEDS,
                     "real_arms_and_shuffled_twins": REAL_ARMS},
    }
    (OUT_DIR / "eval_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote eval_results.json", flush=True)


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    import os
    os._exit(0)
