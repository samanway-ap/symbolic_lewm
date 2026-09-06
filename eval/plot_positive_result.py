"""The single comparison in this run that cleared zero, plotted in context.

`latent_subspace` vs `control` is a real, seed-consistent effect on the
primary metric (all 3 seeds positive, tight CI). This figure shows it, and
then shows the three independent reasons it cannot be reported as evidence
that the symbolic abstraction helped. All four panels use the same measured
numbers from artifacts/eval_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "artifacts"
res = json.loads((OUT / "eval_results.json").read_text())
SEEDS = [0, 1, 2]
mrr = {(x["arm"], x["seed"]): x["mrr"] for x in res["per_run"]}
comps = {c["arm"]: c for c in res["comparisons"]["mrr"]}

GREEN, GREY, RED, BLUE = "#2ca02c", "#8c8c8c", "#c0392b", "#1f77b4"

fig = plt.figure(figsize=(16, 9.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)

# ============ A: the positive result itself ==============================
ax = fig.add_subplot(gs[0, 0])
d = [mrr[("latent_subspace", s)] - mrr[("control", s)] for s in SEEDS]
ci = comps["latent_subspace"]["vs_control"]["ci95"]
md = comps["latent_subspace"]["vs_control"]["mean_diff"]
ax.axhline(0, color="k", lw=1.4, zorder=2)
ax.bar([0], [md], width=0.34, color=GREEN, alpha=0.85, zorder=3)
ax.errorbar([0], [md], yerr=[[md - ci[0]], [ci[1] - md]], fmt="none",
            ecolor="k", elinewidth=2.0, capsize=9, capthick=2.0, zorder=5)
for i, s in enumerate(SEEDS):
    ax.scatter([0.30], [d[i]], s=70, color="k", zorder=6)
    ax.annotate(f"seed {s}: {d[i]:+.5f}", (0.34, d[i]), fontsize=9, va="center")
ax.set_xlim(-0.45, 1.25)
ax.set_xticks([])
ax.set_ylabel("MRR difference (arm - control)", fontsize=10)
ax.set_title("A. THE POSITIVE RESULT\n"
             f"latent_subspace beats control: {md:+.5f}, 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]\n"
             "all 3 seeds positive - a consistent effect, not noise",
             fontsize=11, color="#1a7a1a")
ax.grid(axis="y", alpha=0.3, zorder=0)

# ============ B: disqualifier 1 - the shuffled twin ======================
ax = fig.add_subplot(gs[0, 1])
arms = ["control", "latent_subspace", "shuffled_latent_subspace"]
means = [float(np.mean([mrr[(a, s)] for s in SEEDS])) for a in arms]
cols = [GREY, GREEN, RED]
ax.bar(np.arange(3), means, color=cols, alpha=0.85, width=0.6, zorder=3)
for i, a in enumerate(arms):
    ax.scatter([i] * 3, [mrr[(a, s)] for s in SEEDS], color="k", s=34, zorder=5)
ax.axhline(means[0], ls="--", color="#444", lw=1.2, zorder=4)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["control", "latent_subspace\n(REAL signal)",
                    "shuffled_latent_subspace\n(RANDOM signal)"], fontsize=9)
ax.set_ylim(min(means) * 0.93, max(means) * 1.06)
ax.set_ylabel("plan-ranking MRR", fontsize=10)
sh = comps["shuffled_latent_subspace"]["vs_control"]["mean_diff"]
ax.set_title("B. DISQUALIFIER 1 - the random twin does just as well\n"
             f"real vs control {md:+.5f}   BUT   shuffled vs control {sh:+.5f}\n"
             "real vs its own shuffled twin: -0.00022, CI [-0.00212, +0.00261] (n.s.)",
             fontsize=11, color=RED)
ax.grid(axis="y", alpha=0.3, zorder=0)

# ============ C: disqualifier 2 - effect size vs the gap to chance =======
ax = fig.add_subplot(gs[1, 0])
n_cand = 64
chance = float(np.sum(1.0 / np.arange(1, n_cand + 1)) / n_cand)
ctrl = float(np.mean([mrr[("control", s)] for s in SEEDS]))
gap = chance - ctrl
ng = res["nongameability"]["contracted_minus_real_mrr"]["mean_diff"]
labels = ["the positive result\n(latent_subspace - control)",
          "its shuffled twin's effect",
          "METRIC ARTIFACT\n(contracting the model 10x)",
          "distance from control UP TO CHANCE"]
vals = [md, sh, ng, gap]
cols2 = [GREEN, RED, "#b22222", "#333333"]
y = np.arange(len(vals))[::-1]
ax.barh(y, vals, color=cols2, alpha=0.88, height=0.6, zorder=3)
for yy, v in zip(y, vals):
    ax.annotate(f"{v:+.5f}", (v, yy), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("magnitude in MRR units", fontsize=10)
ax.set_xlim(0, gap * 1.22)
ax.set_title("C. DISQUALIFIER 2 - the effect is dwarfed by its own context\n"
             f"the positive result is {100*md/gap:.1f}% of the gap to chance, and "
             f"{ng/md:.1f}x SMALLER than the metric's own artifact",
             fontsize=11, color=RED)
ax.grid(axis="x", alpha=0.3, zorder=0)

# ============ D: disqualifier 3 - the metric is broken ===================
ax = fig.add_subplot(gs[1, 1])
per = res["nongameability"]["per_checkpoint"]
names = list(per.keys())
real = [per[n]["real_mrr"] for n in names]
contr = [per[n]["contracted_mrr"] for n in names]
lo = min(real + contr) * 0.96
hi = max(real + contr) * 1.04
ax.plot([lo, hi], [lo, hi], ls="--", color="k", lw=1.4, zorder=2,
        label="y = x  (a sound metric would sit on or below this)")
ax.scatter(real, contr, s=95, color="#b22222", zorder=5, edgecolor="k", linewidth=0.6)
for n, xr, yc in zip(names, real, contr):
    ax.annotate(n, (xr, yc), xytext=(6, -11), textcoords="offset points", fontsize=7.6)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("real MRR", fontsize=10)
ax.set_ylabel("MRR after contracting predictions 10x", fontsize=10)
ci_ng = res["nongameability"]["contracted_minus_real_mrr"]["ci95"]
ax.legend(fontsize=8.5, loc="lower right")
ax.set_title("D. DISQUALIFIER 3 - the metric fails its own soundness check\n"
             "ALL 6/6 checkpoints score HIGHER when deliberately crippled\n"
             f"mean {ng:+.4f}, 95% CI [{ci_ng[0]:+.4f}, {ci_ng[1]:+.4f}] lies strictly above 0",
             fontsize=11, color=RED)
ax.grid(alpha=0.3, zorder=0)

fig.suptitle("The ONE positive result in this run - and the three independent reasons it does not support the hypothesis\n"
             "(exploratory run on a Phase-F-REJECTED automaton; all numbers from artifacts/eval_results.json)",
             fontsize=12.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = OUT / "phase_i_positive_result.png"
fig.savefig(out, dpi=145)
print(f"wrote {out}")
print(f"positive effect {md:+.5f} | shuffled twin {sh:+.5f} | metric artifact {ng:+.5f} | gap to chance {gap:+.5f}")
print(f"artifact / effect = {ng/md:.2f}x ; effect as % of gap to chance = {100*md/gap:.2f}%")
