"""Phase I.4 -- statistics. 3 seeds per arm (budget cut from the plan's 5,
per preregistration.md's deadline-driven scope cuts), paired BY SEED
against both `control` and the arm's matched shuffled twin. Report mean
paired difference with bootstrap 95% CI, never a point estimate alone
(plan §12.4). The shuffled arm is the interpretive key (§12.4): if a real
arm beats control but not its shuffled twin, non-uniform masking helped
and the symbolic content contributed nothing -- report that prominently,
never bury it.
"""
from __future__ import annotations

import numpy as np


def bootstrap_paired_diff_ci(values_a: list[float], values_b: list[float],
                                n_boot: int = 10000, seed: int = 3072, alpha: float = 0.05):
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    diffs = a - b
    n = len(diffs)
    mean_diff = float(diffs.mean()) if n else float("nan")
    if n == 0:
        return mean_diff, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[boot_idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean_diff, float(lo), float(hi)


def compare_arm(metric_by_arm_seed: dict[tuple[str, int], float], arm: str,
                  control: str = "control", shuffled: str | None = None,
                  extra_baselines: dict[str, str] | None = None) -> dict:
    """`extra_baselines`: {comparison_label: baseline_arm_name} -- used for the
    exploratory run's third required comparison, `sample_prio` against
    `time_binned_sample_prio` (preregistration.md STEP 2). Every comparison is
    paired BY SEED and reported with its bootstrap 95% CI; a comparison counts
    as won only if the CI excludes zero."""
    seeds = sorted({s for (a, s) in metric_by_arm_seed if a == arm})
    arm_vals = [metric_by_arm_seed[(arm, s)] for s in seeds]
    ctrl_vals = [metric_by_arm_seed[(control, s)] for s in seeds]
    out = {"arm": arm, "seeds": seeds, "values": arm_vals, "control_values": ctrl_vals}
    d, lo, hi = bootstrap_paired_diff_ci(arm_vals, ctrl_vals)
    out["vs_control"] = {"mean_diff": d, "ci95": [lo, hi], "beats_control": lo > 0}
    if shuffled is not None and (shuffled, seeds[0] if seeds else None) in metric_by_arm_seed:
        shuf_vals = [metric_by_arm_seed[(shuffled, s)] for s in seeds]
        out["shuffled_values"] = shuf_vals
        d2, lo2, hi2 = bootstrap_paired_diff_ci(arm_vals, shuf_vals)
        out["vs_shuffled"] = {"mean_diff": d2, "ci95": [lo2, hi2], "beats_shuffled": lo2 > 0}
    for label, base_arm in (extra_baselines or {}).items():
        if (base_arm, seeds[0] if seeds else None) not in metric_by_arm_seed:
            continue
        base_vals = [metric_by_arm_seed[(base_arm, s)] for s in seeds]
        d3, lo3, hi3 = bootstrap_paired_diff_ci(arm_vals, base_vals)
        out[f"vs_{label}"] = {"baseline_arm": base_arm, "baseline_values": base_vals,
                                "mean_diff": d3, "ci95": [lo3, hi3], "beats_baseline": lo3 > 0}
    return out
