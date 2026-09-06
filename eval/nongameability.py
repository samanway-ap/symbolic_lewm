"""Plan §16 "Metric non-gameability" test, applied once at the end (per
this rerun's explicit instruction) to actual POST-TRAINING checkpoints,
not just theta_0: feed the plan-ranking metric a deliberately contracted
model (scale ALL predicted/candidate latents by 0.1) and confirm MRR does
NOT improve. If it does, the metric itself is broken -- fix it before
trusting any Phase I result. Bootstrap CI on (MRR_contracted - MRR_real)
across the checkpoints tested; the check passes iff the CI does not
exclude/lie above 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.plan_ranking import evaluate_plan_ranking
from eval.stats import bootstrap_paired_diff_ci

CONTRACTION_SCALE = 0.1


def run_nongameability_check(checkpoints: dict[str, dict], episode_ids: list[int],
                                n_segments: int = 150, horizon: int = 10, seed: int = 3072) -> dict:
    """checkpoints: {label: model_state_dict}. Runs plan-ranking MRR at
    pred_scale=1.0 (real) and pred_scale=0.1 (deliberately contracted) for
    each, paired by checkpoint label."""
    real_mrrs, contracted_mrrs, per_ckpt = [], [], {}
    for label, state in checkpoints.items():
        real = evaluate_plan_ranking(state, episode_ids, n_segments=n_segments, horizon=horizon,
                                        seed=seed, pred_scale=1.0)
        contracted = evaluate_plan_ranking(state, episode_ids, n_segments=n_segments, horizon=horizon,
                                              seed=seed, pred_scale=CONTRACTION_SCALE)
        real_mrrs.append(real["mrr"])
        contracted_mrrs.append(contracted["mrr"])
        per_ckpt[label] = {"real_mrr": real["mrr"], "contracted_mrr": contracted["mrr"]}
        print(f"  [nongameability] {label}: real_mrr={real['mrr']:.4f} "
              f"contracted(scale={CONTRACTION_SCALE})_mrr={contracted['mrr']:.4f}", flush=True)

    mean_diff, lo, hi = bootstrap_paired_diff_ci(contracted_mrrs, real_mrrs, seed=seed)
    metric_is_sound = hi <= 0 or lo <= 0 <= hi  # CI does not lie strictly above 0
    print(f"  [nongameability] contracted-minus-real MRR: mean={mean_diff:.4f} 95% CI=[{lo:.4f},{hi:.4f}] "
          f"-> metric {'SOUND' if metric_is_sound else 'BROKEN -- DO NOT TRUST PHASE I RESULTS'}", flush=True)
    return {
        "per_checkpoint": per_ckpt,
        "contracted_minus_real_mrr": {"mean_diff": mean_diff, "ci95": [lo, hi]},
        "metric_is_sound": metric_is_sound,
    }
