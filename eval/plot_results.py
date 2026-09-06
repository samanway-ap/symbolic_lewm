"""Phase I figure: every arm, every pre-registered comparison, with its paired
bootstrap 95% CI. Deliberately NOT a "best result" chart -- the exploratory
run's whole interpretive burden is that `sample_prio` must be read against
THREE baselines at once (control, its shuffled twin, and the time-binned
control), so all three are drawn side by side. Comparisons that fail are drawn
exactly as prominently as ones that pass.

Two context marks are non-negotiable on this figure, because without them a
"win" is trivially misread:
  1. the CHANCE line for MRR over (k_negatives+1) candidates, and
  2. the section-16 non-gameability verdict, which on this run FAILED.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PRIMARY = "mrr"

ARM_ORDER = ["control", "sample_prio", "shuffled_sample_prio", "time_binned_sample_prio",
             "latent_subspace", "shuffled_latent_subspace"]
NICE = {
    "vs_control": "vs control",
    "vs_shuffled": "vs shuffled twin",
    "vs_time_binned": "vs time-binned control",
}


def main():
    res = json.loads((OUT_DIR / "eval_results.json").read_text())
    per_run = res["per_run"]
    comps = res["comparisons"][PRIMARY]

    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in per_run)]
    by_arm = {a: [r[PRIMARY] for r in per_run if r["arm"] == a] for a in arms}

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(16, 6.8), gridspec_kw={"width_ratios": [1.0, 1.3]})

    # ---- left: absolute primary metric per arm, per seed -------------------
    xs = np.arange(len(arms))
    means = [float(np.mean(by_arm[a])) for a in arms]
    colors = ["#7f7f7f" if a == "control" else
              ("#d62728" if a.startswith("shuffled_") else
               ("#ff7f0e" if a.startswith("time_binned") else "#1f77b4")) for a in arms]
    ax0.bar(xs, means, color=colors, alpha=0.82, width=0.62, zorder=2)
    for i, a in enumerate(arms):
        ax0.scatter([i] * len(by_arm[a]), by_arm[a], color="k", s=26, zorder=4, alpha=0.85)

    n_cand = int(per_run[0].get("k_negatives", 63)) + 1
    chance = float(np.sum(1.0 / np.arange(1, n_cand + 1)) / n_cand)
    ax0.axhline(chance, ls="-", lw=1.8, color="#d62728", zorder=5,
                label=f"CHANCE = {chance:.4f}  ({n_cand} candidates)")
    ctrl_mean = float(np.mean(by_arm["control"])) if "control" in by_arm else None
    if ctrl_mean is not None:
        ax0.axhline(ctrl_mean, ls="--", lw=1.3, color="#444", zorder=3,
                    label=f"control mean = {ctrl_mean:.4f}")
    ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax0.set_ylim(0, max(chance * 1.20, max(means) * 1.30))
    ax0.set_xticks(xs)
    ax0.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=8.5)
    ax0.set_ylabel("plan-ranking MRR  (H=10, held-out test-20%)", fontsize=10)
    ax0.set_title("Primary metric per arm (dots = seeds, n=3)\n"
                  "EVERY arm sits at roughly HALF of chance", fontsize=11)
    ax0.grid(axis="y", alpha=0.3, zorder=0)

    # ---- right: paired differences with bootstrap 95% CI --------------------
    rows = []
    for c in comps:
        for key in ("vs_control", "vs_shuffled", "vs_time_binned"):
            if key in c:
                rows.append((c["arm"], NICE[key], c[key]["mean_diff"], c[key]["ci95"]))
    rows.reverse()

    ys = np.arange(len(rows))
    for y, (_, _, d, ci) in zip(ys, rows):
        wins, loses = ci[0] > 0, ci[1] < 0
        col = "#2ca02c" if wins else ("#d62728" if loses else "#888888")
        ax1.plot([ci[0], ci[1]], [y, y], color=col, lw=2.6, solid_capstyle="round", zorder=3)
        ax1.scatter([d], [y], color=col, s=64, zorder=4,
                    marker="o" if wins else ("x" if loses else "s"))
    ax1.axvline(0, color="k", lw=1.4, zorder=2)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([f"{a}  {lab}" for a, lab, _, _ in rows], fontsize=8.5)
    ax1.set_xlabel("paired difference in MRR (arm - baseline), 95% bootstrap CI", fontsize=10)
    ax1.set_title("Every pre-registered comparison\n"
                  "green = CI excludes 0 (win)   red = CI excludes 0 (loss)   grey = includes 0",
                  fontsize=11)
    ax1.grid(axis="x", alpha=0.3, zorder=0)

    n_win = sum(1 for _, _, _, ci in rows if ci[0] > 0)
    ng = res.get("nongameability", {})
    sound = ng.get("metric_is_sound", None)
    ci_ng = ng.get("contracted_minus_real_mrr", {}).get("ci95", [float("nan")] * 2)
    md = ng.get("contracted_minus_real_mrr", {}).get("mean_diff", float("nan"))

    fig.suptitle(
        "Phase G/H/I EXPLORATORY run - automaton FAILED Phase F "
        "(coverage 83.3% vs 90%; 3-seed reproducibility FAIL)\n"
        f"{n_win} of {len(rows)} pre-registered comparisons clear zero - but every arm is "
        "BELOW CHANCE and the metric failed its own soundness check",
        fontsize=11.5, y=0.995)
    if sound is False:
        fig.text(0.5, 0.885,
                 "SECTION-16 NON-GAMEABILITY: FAILED - contracting predictions 10x IMPROVES MRR "
                 f"(mean {md:+.4f}, 95% CI [{ci_ng[0]:+.4f}, {ci_ng[1]:+.4f}] lies strictly above 0)"
                 "   ==>   DO NOT TRUST ANY COMPARISON BELOW",
                 ha="center", fontsize=10, color="white", fontweight="bold",
                 bbox=dict(facecolor="#b22222", edgecolor="none", pad=6.0))
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    out = OUT_DIR / "phase_i_results.png"
    fig.savefig(out, dpi=145)
    print(f"wrote {out}")

    print("\n=== every comparison, primary metric (MRR) ===")
    for a, lab, d, ci in reversed(rows):
        verdict = "WIN " if ci[0] > 0 else ("LOSS" if ci[1] < 0 else "n.s.")
        print(f"  [{verdict}] {a:<26} {lab:<24} {d:+.5f}  CI [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    print(f"\nchance MRR = {chance:.5f}; control mean = {ctrl_mean:.5f}")
    print(f"non-gameability: metric_is_sound = {sound}")


if __name__ == "__main__":
    main()
