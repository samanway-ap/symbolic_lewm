"""Phase G/H diagnostics figure -- everything MEASURED so far, before Phase I
scoring completes. Deliberately contains no plan-ranking numbers: none exist
yet at the time this is generated.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "artifacts"

ARM_COLORS = {
    "control": "#7f7f7f",
    "sample_prio": "#1f77b4",
    "shuffled_sample_prio": "#d62728",
    "time_binned_sample_prio": "#ff7f0e",
    "latent_subspace": "#2ca02c",
    "shuffled_latent_subspace": "#9467bd",
}

fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

# ---------------- panel 1: STEP 1 relevance subspace ----------------------
ax = fig.add_subplot(gs[0, 0])
summ = json.loads((OUT / "relevance_subspace_summary.json").read_text())
ev = np.array(summ["explained_variance_curve"])
cum = np.cumsum(ev)
r, d = summ["r"], summ["d"]
ax.plot(np.arange(1, len(cum) + 1), cum, lw=2.2, color="#1f77b4", zorder=3)
ax.axhline(0.95, ls="--", color="#444", lw=1.2, label="95% variance rule")
ax.axvline(r, ls=":", color="#d62728", lw=2.0, label=f"r = {r}")
ax.axvline(0.8 * d, ls="-.", color="#888", lw=1.6, label=f"vacuity threshold 0.8d = {0.8*d:.0f}")
ax.set_xlim(0, max(len(cum), 0.8 * d * 1.05))
ax.set_xlabel("principal component of the gradient directions", fontsize=10)
ax.set_ylabel("cumulative explained variance", fontsize=10)
ax.set_title(f"STEP 1 — relevance subspace is NOT vacuous\n"
             f"r/d = {r}/{d} = {summ['r_over_d']:.3f}  <=  0.8  ->  latent_subspace arms KEPT",
             fontsize=11)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.3, zorder=0)

# ---------------- panel 2: training loss, all 18 runs ---------------------
ax = fig.add_subplot(gs[0, 1])
runs = json.loads((OUT / "training_runs.json").read_text())
seen = set()
for run in runs:
    arm, seed = run["arm"], run["seed"]
    hist = json.loads((OUT / f"loss_history_{arm}_seed{seed}.json").read_text())
    steps = [h["step"] for h in hist]
    loss = np.array([h["pred_loss"] for h in hist])
    k = 15
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    ax.plot(steps[k - 1:], smooth, color=ARM_COLORS[arm], lw=1.4, alpha=0.85,
            label=arm if arm not in seen else None, zorder=3)
    seen.add(arm)
ax.set_xlabel("training step (5 epochs x 37 batches)", fontsize=10)
ax.set_ylabel("prediction loss (smoothed, k=15)", fontsize=10)
ax.set_title("Phase H — all 18 runs completed (6 arms x 3 seeds)\n"
             "NOT a performance comparison: latent_subspace arms optimise a\n"
             "DIFFERENT objective, so their loss is not commensurable", fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, zorder=0)

# ---------------- panel 3: STEP 2 time-binned control validation ----------
ax = fig.add_subplot(gs[1, 0])
tb = json.loads((OUT / "time_binned_signal_diagnostics.json").read_text())
bins = np.arange(tb["n_bins"])
rates = np.array(tb["per_bin_real_failure_rate"])
counts = np.array(tb["bin_counts"])
ax.bar(bins - 0.19, counts / counts.max(), width=0.36, color="#ff7f0e", alpha=0.85,
       label="samples per bin (normalised)", zorder=2)
ax.bar(bins + 0.19, rates, width=0.36, color="#8c564b", alpha=0.85,
       label="real abstraction-failure rate in bin", zorder=2)
ax.axhline(rates.mean(), ls="--", color="#444", lw=1.3,
           label=f"mean failure rate = {rates.mean():.3f}", zorder=3)
ax.set_ylim(0, 1.05)
ax.set_xticks(bins)
ax.set_xlabel("normalized-timestep bin (episode phase)", fontsize=10)
ax.set_title("STEP 2 — time-binned control is correctly matched, but the\n"
             "failure flag carries almost no phase signal (rates 0.370-0.400)\n"
             f"NMI(automaton state, time bin) = {tb['normalized_mutual_information_state_vs_timebin']:.3f}",
             fontsize=11)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(axis="y", alpha=0.3, zorder=0)

# ---------------- panel 4: the automaton's structural degeneracy ----------
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
pos = {"s0": (0.5, 0.86), "s1": (0.14, 0.44), "s2": (0.38, 0.44),
       "s3": (0.62, 0.44), "s4": (0.86, 0.44), "s5": (0.5, 0.08)}
live = {"s1": "(1,1,0)", "s2": "(0,0,0)", "s3": "(1,0,0)", "s4": "(1,1,1)"}
for name, (x, y) in pos.items():
    if name == "s0":
        c, ec = "#cfe3f7", "#1f77b4"
    elif name == "s5":
        c, ec = "#f2f2f2", "#999999"
    else:
        c, ec = "#d7efd7", "#2ca02c"
    ax.add_patch(plt.Circle((x, y), 0.072, facecolor=c, edgecolor=ec, lw=2.0, zorder=3))
    ax.text(x, y + 0.012, name, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    if name in live:
        ax.text(x, y - 0.032, live[name], ha="center", va="center", fontsize=6.5, zorder=4)
for tgt in ("s1", "s2", "s3", "s4"):
    x0, y0 = pos["s0"]; x1, y1 = pos[tgt]
    ax.annotate("", xy=(x1, y1 + 0.075), xytext=(x0, y0 - 0.075),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#1f77b4"), zorder=2)
ax.text(0.5, 0.66, "reset symbols  r0..r7  (only)", ha="center", fontsize=8.5, color="#1f77b4")
for tgt in ("s1", "s2", "s3", "s4"):
    x, y = pos[tgt]
    ax.annotate("", xy=(x + 0.026, y + 0.083), xytext=(x - 0.026, y + 0.083),
                arrowprops=dict(arrowstyle="-|>", lw=1.9, color="#d62728",
                                connectionstyle="arc3,rad=2.9"), zorder=2)
ax.text(0.5, 0.28, "ALL 12 action symbols aa0..aa11 SELF-LOOP in every live state",
        ha="center", fontsize=10, color="#d62728", fontweight="bold")
ax.text(0.5, 0.225, "-> the state never responds to any action", ha="center",
        fontsize=9.5, color="#d62728")
ax.text(0.5, 0.008, "sink, unreachable on well-formed input\n"
                    "(=> 5/6 = 83.3% is the MAXIMUM attainable coverage)",
        ha="center", fontsize=7.5, color="#666")
ax.set_title("The Phase-F-rejected machine is structurally degenerate:\n"
             "a static 4-way classification of the initial frame, not a dynamical model",
             fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1)

fig.suptitle("Phase G/H measured diagnostics — EXPLORATORY run on a Phase-F-REJECTED automaton "
             "(coverage 83.3% vs 90%; 3-seed reproducibility FAIL)\n"
             "Phase I plan-ranking scores are still computing and appear in a separate figure",
             fontsize=12.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = OUT / "phase_gh_diagnostics.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
