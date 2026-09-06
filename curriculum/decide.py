"""Phase K decision rule + multiplicity control.

The single most important property of this file: **"no improvement" is NOT
"homogeneous"**. Accepting the null on a wide confidence interval would
silently delete territory from the atlas forever -- a cell wrongly marked
BLOCKED_HOMOGENEOUS is never revisited, so a merely underpowered test
becomes a permanent false negative. Homogeneity may therefore ONLY be
declared by an EQUIVALENCE result: the whole CI must fit inside
[-delta_neg, +delta_neg]. A CI too wide to fit inside the band is
BLOCKED_UNDERPOWERED and stays revisitable, whatever its point estimate.

Thresholds (delta_min = delta_neg = 0.01 MRR) are pre-registered in
preregistration.md and are NOT re-derived here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_import import load_local  # noqa: E402

# NOT `from eval.stats import ...`: le-wm/eval.py shadows this repo's eval
# package once oracle.lewm_g puts le-wm on sys.path. See local_import.py.
bootstrap_paired_diff_ci = load_local("symbolic_eval_stats", "eval/stats.py").bootstrap_paired_diff_ci

DELTA_MIN = 0.01      # improvement threshold (pre-registered)
DELTA_NEG = 0.01      # equivalence band half-width (pre-registered)
GLOBAL_REGRESSION_TOLERANCE = 0.005


def decide(local_arm: list[float], local_control: list[float],
             global_arm: list[float], global_control: list[float],
             delta_min: float = DELTA_MIN, delta_neg: float = DELTA_NEG,
             global_tol: float = GLOBAL_REGRESSION_TOLERANCE, seed: int = 3072) -> dict:
    d, lo, hi = bootstrap_paired_diff_ci(local_arm, local_control, seed=seed)
    gd, glo, ghi = bootstrap_paired_diff_ci(global_arm, global_control, seed=seed + 1)
    half_width = (hi - lo) / 2.0

    out = {
        "local_mean_diff": d, "local_ci95": [lo, hi], "local_ci_half_width": half_width,
        "global_mean_diff": gd, "global_ci95": [glo, ghi],
        "delta_min": delta_min, "delta_neg": delta_neg,
    }

    # Forgetting is a TRAINING artifact, not a property of the cell -- it must
    # not produce a cell verdict at all, or the atlas records a fact about the
    # replay ratio as though it were a fact about the region.
    if gd < -global_tol:
        out["verdict"] = "REDO_FORGETTING"
        out["reason"] = (f"GLOBAL regressed {gd:.5f} (beyond -{global_tol}); raise replay ratio and "
                           "redo the round. No cell verdict written.")
        return out

    if lo > 0 and lo > delta_min:
        out["verdict"] = "IMPROVED"
        out["reason"] = f"CI excludes 0 and lower bound {lo:.5f} > delta_min {delta_min}"
    elif lo >= -delta_neg and hi <= delta_neg:
        out["verdict"] = "BLOCKED_HOMOGENEOUS"
        out["reason"] = (f"equivalence: whole CI [{lo:.5f},{hi:.5f}] inside +/-{delta_neg}. "
                           "This is the ONLY valid route to a homogeneity claim.")
    elif half_width > delta_neg:
        out["verdict"] = "BLOCKED_UNDERPOWERED"
        out["reason"] = (f"CI half-width {half_width:.5f} > delta_neg {delta_neg}: the test cannot "
                           "distinguish improvement from equivalence. NOT homogeneous; revisit later.")
    else:
        out["verdict"] = "BLOCKED_UNDERPOWERED"
        out["reason"] = (f"CI [{lo:.5f},{hi:.5f}] neither clears delta_min nor fits the equivalence "
                           "band; inconclusive, revisit later.")
    return out


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.flatnonzero(passed)
    cutoff = k.max() + 1 if k.size else 0
    out = np.zeros(n, dtype=bool)
    if cutoff:
        out[order[:cutoff]] = True
    return out.tolist()


def bootstrap_pvalue(arm: list[float], control: list[float], n_boot: int = 10000, seed: int = 3072) -> float:
    """Two-sided bootstrap p-value for mean paired difference == 0, obtained by
    inverting the same percentile bootstrap the CIs use (so p and CI can never
    disagree about whether 0 is inside)."""
    a, b = np.asarray(arm, float), np.asarray(control, float)
    diffs = a - b
    n = len(diffs)
    if n == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    frac_le = float((boot <= 0).mean())
    return float(min(1.0, 2 * min(frac_le, 1 - frac_le)))


def apply_multiplicity(cell_results: list[dict], alpha: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg across all cell tests in a sweep. RAW intervals are
    preserved untouched alongside the adjusted decision -- reporting only
    adjusted values would hide how much of a verdict is multiplicity
    correction versus signal."""
    pvals = [c.get("local_pvalue", 1.0) for c in cell_results]
    flags = benjamini_hochberg(pvals, alpha=alpha)
    for c, f in zip(cell_results, flags):
        c["bh_significant"] = bool(f)
        c["bh_alpha"] = alpha
        if c.get("verdict") == "IMPROVED" and not f:
            c["verdict_bh_adjusted"] = "BLOCKED_UNDERPOWERED"
            c["bh_note"] = ("raw CI cleared delta_min but did not survive Benjamini-Hochberg "
                              "across the sweep; downgraded, not reported as an improvement.")
        else:
            c["verdict_bh_adjusted"] = c.get("verdict")
    return cell_results
