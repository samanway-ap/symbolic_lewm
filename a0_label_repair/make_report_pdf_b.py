"""Renders artifacts/a0_horizon_decomposition_b_report.json into a PDF:
h=0 reset test, same-sample A-E link decomposition, re-grounding sweep,
continuous h=1 diagnostics (correlation + boundary-distance stratification).
Reporting only, per instruction -- no repair chosen.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REPORT_PATH = OUT_DIR / "a0_horizon_decomposition_b_report.json"
PDF_PATH = OUT_DIR / "A0.1b_link_decomposition_report.pdf"


def load():
    return json.loads(REPORT_PATH.read_text())


def page_title(pdf, d):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.94, "A0.1b -- Link Decomposition, Re-grounding,", ha="center", fontsize=17, weight="bold")
    fig.text(0.5, 0.915, "Continuous Diagnostics", ha="center", fontsize=17, weight="bold")
    fig.text(0.5, 0.885, "Run in direct response to five corrections on A0.1 -- same frozen Lever-0 hyperplanes,\n"
              "same pickled machines, no refitting, no training", ha="center", fontsize=9.5, style="italic", color="#444")

    h0 = d["h0"]
    body = f"""
A0.1 compared quantities computed on DIFFERENT populations (arbitrary real start points vs.
episode-beginning reset-aligned walks), so the apparent role of reset-window substitution was
plausible but not demonstrated, and several other claims overreached what the data actually
showed. This experiment fixes that: for the SAME {d['n_start_points']} real (episode, t) samples
across {d['n_episodes']} episodes, roll BOTH the true window z_t and the canonical-reset
substitute r_t = choose_reset(z_t) through the SAME real 21-letter action word, and step the
SAME frozen machine on r_t with that REAL word:

  A_h  phi(ghat(z_t,u)) vs phi(z_t+h)        true-start model error
  B_h  phi(ghat(r_t,u)) vs phi(z_t+h)        reset-substituted model error
  C_h  phi(ghat(r_t,u)) vs phi(ghat(z_t,u))  reset substitution's effect on the model alone
  D_h  M(r_t,u) vs phi(ghat(r_t,u))          extraction fidelity on the REAL word distribution
  E_h  M(r_t,u) vs phi(z_t+h)                full end-to-end, same sample as A-D

Headline h=0 result (before any rollout at all):

  P[phi(r_t) = phi(z_t)], joint over 8 bits  =  {h0['P_reset_matches_start_joint8']['mean']:.3f}   (95% CI [{h0['P_reset_matches_start_joint8']['ci95'][0]:.3f}, {h0['P_reset_matches_start_joint8']['ci95'][1]:.3f}])
  mean ||r_t - z_t|| (ambient)               =  {h0['mean_d_r_z_ambient']:.2f}   (overall data-spread scale ~13.2)

Eight fixed canonical reset windows cannot approximate an arbitrary real episode's start in a
space this large and this scene-dominated (93% between-episode variance, established earlier).
Per the pre-registered rule: reset substitution fails already at h=0, so the reset interface
must be repaired before end-to-end FSM fidelity can be meaningfully interpreted.

This report stops at the decomposition. No repair is chosen here.
""".strip()
    wrapped = "\n".join(textwrap.fill(line, 100) if line.strip() else "" for line in body.split("\n"))
    fig.text(0.07, 0.83, wrapped, va="top", fontsize=8.6, family="monospace")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def page_link_decomposition(pdf, d):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    horizons = d["horizons"]
    colors = {"A": "#1b6fa8", "B": "#a02b2b", "C": "#7a4fa3", "D": "#c46a1f", "E": "#2e8b57"}
    labels = {"A": "A: true-start vs real", "B": "B: reset-sub vs real", "C": "C: reset-sub vs true-start (model only)",
               "D": "D: extraction fidelity (real words)", "E": "E: full pipeline (same sample)"}
    for ax, m in zip(axes.flat, ("2", "4", "6", "8")):
        j = d[m]["joint"]
        for k in ("A", "B", "C", "D", "E"):
            vals = [j[str(h)][k]["mean"] for h in horizons]
            ax.plot(horizons, vals, "o-", ms=3, color=colors[k], label=labels[k] if m == "2" else None)
        ax.set_title(f"m={m} (joint, {d[m]['n_states']} states)", fontsize=10)
        ax.set_xlabel("horizon h (letters)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(horizons)
    axes[0, 0].legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    fig.suptitle("Same-sample link decomposition: A-E (joint, all bits)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def page_regrounding(pdf, d):
    fig, ax = plt.subplots(figsize=(11, 7))
    horizons = d["horizons"]
    rg = d["regrounding"]
    colors = plt.cm.viridis([0.15, 0.4, 0.65, 0.9])
    for r, c in zip(("1", "2", "4", "8"), colors):
        vals = [rg[r][str(h)]["mean"] for h in horizons]
        ax.plot(horizons, vals, "o-", color=c, label=f"r={r} (re-ground every {r} letters)")
    ax.set_xlabel("horizon h (letters)")
    ax.set_ylabel("joint accuracy (all 8 bits, vs real endpoint)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(horizons)
    ax.set_title("Re-grounding sweep (m=8): accuracy resets after each grounding event,\n"
                  "decays on a schedule that depends only on steps-since-grounding", fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="#999", lw=0.5, ls=":")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_continuous(pdf, d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    c = d["continuous_h1_diagnostics"]
    bits = list(c.keys())

    ax = axes[0]
    corr = [c[b]["correlation"] for b in bits]
    ax.bar(range(len(bits)), corr, color="#1b6fa8")
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(bits, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("correlation (predicted vs. real continuous score)")
    ax.set_title("h=1 continuous prediction quality", fontsize=10)
    ax.axhline(0.9, color="#999", lw=0.5, ls=":")

    ax = axes[1]
    x = range(4)
    for bi, b in enumerate(bits):
        accs = [q["accuracy"] for q in c[b]["accuracy_by_boundary_distance_quartile"]]
        ax.plot(x, accs, "o-", ms=3, alpha=0.8, label=b)
    ax.set_xticks(x)
    ax.set_xticklabels(["closest", "Q2", "Q3", "farthest"], fontsize=8)
    ax.set_xlabel("quartile of |score| at real endpoint (distance from boundary)")
    ax.set_ylabel("binary accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Binary accuracy vs. distance from decision boundary", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)

    fig.suptitle("Continuous prediction is good; binary errors concentrate at the boundary", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def page_switch_table(pdf, d):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    c = d["continuous_h1_diagnostics"]
    rows = []
    for b, v in c.items():
        rows.append([b, f"{v['n_switch_events']}", f"{v['n_episodes_with_switch']}",
                      f"{v['recall_given_switch']:.3f}", f"{v['false_switch_rate_given_nonswitch']:.3f}",
                      f"{v['balanced_accuracy']:.3f}", "yes" if v["underpowered_switch_analysis"] else "no"])
    col_labels = ["bit", "n switch events", "n episodes", "recall|switch", "false-switch rate|non-switch",
                   "balanced acc", "underpowered?"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    for k in range(len(col_labels)):
        table[0, k].set_facecolor("#dbe7f0")
        table[0, k].set_text_props(weight="bold")
    ax.set_title("Switch-event detection at h=1 (per bit)", fontsize=12, weight="bold", pad=20)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_interpretation(pdf, d):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.94, "Reading against the pre-registered stop-point conditions", ha="center",
              fontsize=14, weight="bold")

    h0 = d["h0"]
    text = f"""
1. Reset substitution fails already at h=0 -- CONFIRMED.
   P[phi(r_t)=phi(z_t)] joint = {h0['P_reset_matches_start_joint8']['mean']:.3f}, mean ||r_t-z_t|| = {h0['mean_d_r_z_ambient']:.2f}
   (data-spread scale ~13.2). Per the pre-registered rule: the reset interface must be repaired
   before end-to-end FSM fidelity (E) can be meaningfully interpreted. E tracks B almost exactly
   at every horizon and every m -- the end-to-end number is dominated by reset substitution, not
   by extraction (D stays close to 1.0 at h=1 on the REAL-word distribution too, for every m).

2. Continuous prediction good, binary errors concentrate near the boundary -- CONFIRMED.
   Correlation 0.90-0.97 for every bit; accuracy climbs from 0.57-0.72 (closest quartile to the
   boundary) to 0.94-1.00 (farthest quartile), monotonically, for every one of the 8 bits.
   Per the pre-registered rule: the label construction is brittle -- this is NOT evidence that
   "these residual directions are not learned by the checkpoint" (A0.1's overclaim, corrected).

3. Re-grounding progressively restores accuracy -- CONFIRMED, on a clean, stationary schedule.
   Accuracy resets near-perfectly at every grounding event and decays as a function of STEPS
   SINCE GROUNDING, not absolute horizon position (r=8's h=4, ungrounded, and h=12, 4 steps past
   grounding at h=8, land within 0.001 of each other: 0.192 vs 0.193). Accumulated rollout drift
   is established as real and independent of the reset-substitution problem (re-grounding here
   is always relative to the TRUE start z_t, never r_t).

4. D_h failing on real words despite perfect random-word h=1 performance -- DOES NOT APPLY.
   D_h(h=1) = 1.000 on the REAL-word distribution too, for every m. Extraction is not implicated
   at h=1 on either word distribution.

Net picture: THREE independent, now-demonstrated defects, not one:
  - the 8-canonical-reset interface is far too coarse for a >20-dimensional between-episode
    subspace and fails before any dynamics are even involved;
  - genuine open-loop rollout drift accumulates on a repeatable schedule, independent of (1);
  - the binary tessellation threshold is brittle exactly where it matters (near boundaries,
    which is where real label switches concentrate by definition), even though the checkpoint's
    CONTINUOUS prediction of these directions is good.

No repair is chosen here. Which of the three to address first -- or the user's own suggested
reframe (jointly optimise within-episode switching and model predictability in the next label
search) -- is left for a separate decision.
""".strip()
    wrapped = "\n".join(textwrap.fill(line, 96, subsequent_indent="  ") if line.strip() else "" for line in text.split("\n"))
    fig.text(0.06, 0.90, wrapped, va="top", fontsize=8.4, family="monospace")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    d = load()
    with PdfPages(PDF_PATH) as pdf:
        page_title(pdf, d)
        page_link_decomposition(pdf, d)
        page_regrounding(pdf, d)
        page_continuous(pdf, d)
        page_switch_table(pdf, d)
        page_interpretation(pdf, d)
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
