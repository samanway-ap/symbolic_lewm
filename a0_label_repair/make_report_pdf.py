"""Renders artifacts/a0_horizon_decomposition_report.json into a PDF report:
methodology, joint-label horizon curves per m, per-bit h=1 snapshot, full
numeric tables, and the decision-logic reading -- reporting only, per the
experiment's own instruction not to choose a repair.
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
REPORT_PATH = OUT_DIR / "a0_horizon_decomposition_report.json"
PDF_PATH = OUT_DIR / "A0.1_horizon_decomposition_report.pdf"

COLORS = {
    "F_model_real": "#1b6fa8",
    "P_start_persist": "#1b6fa8",
    "F_FSM_model": "#c46a1f",
    "F_FSM_real": "#a02b2b",
    "P_closed_loop_persist": "#666666",
}


def load():
    return json.loads(REPORT_PATH.read_text())


def page_title(pdf, d):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.92, "A0.1 -- Horizon / Error Decomposition", ha="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.885, "Lever-0 residualized tessellation, frozen; no refitting, no training",
              ha="center", fontsize=10, style="italic", color="#444")

    body = f"""
Question. Lever 0 (project out the between-episode/scene subspace, redraw hyperplanes in the
166-dim orthogonal complement) moved every tessellation bit's switch rate into the [0.02, 0.25]
acceptance band -- but end-to-end fidelity (FSM vs. real trajectories) got WORSE, not better
(margins -0.41 to -0.50, vs -0.25 to -0.45 before). That comparison conflates three sources of
error: (1) symbolic-extraction error, (2) open-loop LeWM rollout error, (3) inadequacy of the
partition phi itself -- plus, discovered here, (4) a reset-window initialisation mismatch. This
experiment separates them.

Method. Hyperplanes frozen exactly as Lever 0 left them -- no refitting. One FSM extraction per
m in {{2,4,6,8}} (separation_rule=SepSeq, a validated runtime fix only). No training anywhere.
Six quantities at horizon h in {{1,2,4,8,12,16,21}} letters, all read off ONE rollout per start
point (not seven separate rollouts):

  F_TF             one-letter, teacher-forced: phi(ghat(z_t,a_t)) vs phi(z_t+1). Equals
                   F_model_real at h=1 by construction.
  F_model_real(h)  phi(ghat(z_t, u_t:t+h)) vs phi(z_t+h) -- REAL start, REAL action word, REAL
                   endpoint. No FSM. Isolates rollout drift.
  F_FSM_model(h)   M(u) vs phi(ghat(z0,u)) on held-out RANDOM words. Machine vs. the function it
                   was actually fit to. Isolates extraction error.
  F_FSM_real(h)    M(u) vs phi(z_real_endpoint) -- the ORIGINAL end-to-end quantity, now sliced
                   by horizon instead of averaged over a whole episode.
  P_start_persist(h)   phi(z_t) vs phi(z_t+h) -- copy the INITIAL label forward h steps. The
                   matched-information baseline for an open-loop h-step predictor.
  P_closed_loop_persist   the ORIGINAL 1-step persistence baseline (copies the immediately
                   preceding REAL label); reported as a constant reference line.

Data: {d['config']['n_start_points']} real start points (one per heldout episode), {d['config']['n_fsm_model_words']}
held-out random words, {d['config']['n_bootstrap']}x episode-level bootstrap for every CI.
Extracted machine sizes: {', '.join(f"m={m}: {n} states" for m, n in d['machine_n_states'].items())}.

This report stops at the decomposition. No repair is chosen here.
""".strip()

    wrapped = "\n".join(textwrap.fill(line, 100) if line.strip() else "" for line in body.split("\n"))
    fig.text(0.08, 0.83, wrapped, va="top", fontsize=9, family="monospace")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def page_joint_curves(pdf, d):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    horizons = d["horizons"]
    for ax, m in zip(axes.flat, ("2", "4", "6", "8")):
        j = d[m]["joint"]
        fmr = [j[str(h)]["F_model_real"]["mean"] for h in horizons]
        fmr_lo = [j[str(h)]["F_model_real"]["ci95"][0] for h in horizons]
        fmr_hi = [j[str(h)]["F_model_real"]["ci95"][1] for h in horizons]
        sp = [j[str(h)]["P_start_persist"]["mean"] for h in horizons]
        fm = [j[str(h)]["F_FSM_model"]["mean"] for h in horizons]
        fr = [j[str(h)]["F_FSM_real"]["mean"] for h in horizons]

        ax.fill_between(horizons, fmr_lo, fmr_hi, color=COLORS["F_model_real"], alpha=0.15)
        ax.plot(horizons, fmr, "o-", color=COLORS["F_model_real"], label="F_model_real (rollout vs real)")
        ax.plot(horizons, sp, "s--", color=COLORS["P_start_persist"], alpha=0.6,
                 label="P_start_persist (matched baseline)")
        ax.plot(horizons, fm, "^-", color=COLORS["F_FSM_model"], label="F_FSM_model (extraction fidelity)")
        ax.plot(horizons, fr, "v-", color=COLORS["F_FSM_real"], label="F_FSM_real (original end-to-end)")
        ax.axhline(0.5, color="#999", lw=0.5, ls=":")
        ax.set_title(f"m={m} (joint, all {m} bits; {d['machine_n_states'][m]} states)", fontsize=10)
        ax.set_xlabel("horizon h (letters)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(horizons)
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle("Joint-label horizon curves (all bits must match simultaneously)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def page_perbit_h1(pdf, d):
    fig, ax = plt.subplots(figsize=(11, 6))
    bits = list(d["8"]["per_bit"].keys())
    tf = [d["8"]["per_bit"][b]["F_TF"]["mean"] for b in bits]
    tf_lo = [d["8"]["per_bit"][b]["F_TF"]["ci95"][0] for b in bits]
    tf_hi = [d["8"]["per_bit"][b]["F_TF"]["ci95"][1] for b in bits]
    sp = [d["8"]["per_bit"][b]["1"]["P_start_persist"]["mean"] for b in bits]
    cl = [d["8"]["closed_loop_persistence"][b]["persistence_baseline"] for b in bits]

    x = range(len(bits))
    w = 0.27
    ax.bar([i - w for i in x], tf, width=w, color=COLORS["F_model_real"], label="F_TF (teacher-forced, h=1)")
    ax.errorbar([i - w for i in x], tf, yerr=[[a - b for a, b in zip(tf, tf_lo)], [b - a for a, b in zip(tf, tf_hi)]],
                 fmt="none", ecolor="black", capsize=3, lw=1)
    ax.bar(list(x), sp, width=w, color=COLORS["P_start_persist"], alpha=0.5, label="P_start_persist (h=1, matched)")
    ax.bar([i + w for i in x], cl, width=w, color=COLORS["P_closed_loop_persist"], alpha=0.6,
            label="P_closed_loop_persist (original baseline)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bits, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Per-bit, h=1: teacher-forced prediction vs. both persistence baselines\n"
                  "(all 8 residualized tessellation bits -- lower m levels reuse the same first-k bits)",
                  fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_table_joint(pdf, d):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    horizons = d["horizons"]
    rows = []
    for m in ("2", "4", "6", "8"):
        for h in horizons:
            j = d[m]["joint"][str(h)]
            rows.append([
                m, str(h),
                f"{j['F_model_real']['mean']:.3f}",
                f"{j['P_start_persist']['mean']:.3f}",
                f"{j['margin_model_real_vs_start_persist']:+.3f}",
                f"{j['F_FSM_model']['mean']:.3f}",
                f"{j['F_FSM_real']['mean']:.3f}",
            ])
    col_labels = ["m", "h", "F_model_real", "P_start_persist", "margin", "F_FSM_model", "F_FSM_real"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    for k in range(len(col_labels)):
        table[0, k].set_facecolor("#dbe7f0")
        table[0, k].set_text_props(weight="bold")
    ax.set_title("Joint-label results, full table (all m x h)", fontsize=12, weight="bold", pad=20)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_decision(pdf, d):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.94, "Reading against the pre-registered decision logic", ha="center",
              fontsize=15, weight="bold")

    text = """
Case 2 (extraction is the problem) -- RULED OUT at h=1.
  F_FSM_model(h=1) = 1.000 for every m: the extracted machine perfectly reproduces the model's
  OWN one-step behaviour on fresh held-out words. Extraction is not the h=1 defect.
  (It DOES degrade at long horizon -- down to 0.033 (m=8) - 0.653 (m=2) by h=21 -- a real,
  separate generalisation limit of a 10-round/cheap extraction on long unseen words, layered on
  top of everything else, not the primary story.)

Case 3 (LeWM does not locally predict these directions well) -- APPLIES, robustly, per-bit.
  Every one of the 8 individual bits loses to its OWN matched start-persistence baseline
  already at h=1 (F_TF margins -0.082 to -0.186, all bits). This is not a joint-probability
  artifact of requiring many bits to agree at once -- it holds bit by bit. The checkpoint does
  not locally predict these particular (post-scene-projection) directions well, independent of
  any rollout length.

Case 1 (compounding rollout drift) -- ALSO present, layered on top of the Case-3 deficit.
  F_model_real decays substantially further with h beyond the h=1 shortfall (joint roughly
  halves to 6x's down by h=21, depending on m). The matched P_start_persist baseline also
  decays with h (real trajectories genuinely move between cells over 21 letters) but more
  slowly than F_model_real keeps pace -- margin worsens from about -0.21 (h=1) to -0.33 (h=21)
  at m=2, similarly at every m. Rollout error is real and additive, not the sole explanation.

A fourth contributor, NOT in the pre-registered case list:
  F_FSM_real (the ORIGINAL end-to-end quantity) is far worse than F_model_real even at h=1
  (0.304 vs 0.763 at m=2; 0.014 vs 0.270 at m=8) and stays roughly FLAT across horizon rather
  than decaying with h -- inconsistent with pure rollout drift, which should worsen WITH h.
  Mechanism: F_FSM_real's starting point is never the episode's TRUE starting window -- it is
  the NEAREST of 8 fixed canonical reset windows (choose_reset), an approximation error baked
  in at step 0, independent of horizon. This plausibly explains why the ORIGINAL end-to-end
  fidelity numbers (both before and after Lever 0) were as bad as they were: a real h=1 local
  deficit, PLUS real rollout drift, PLUS a reset-window initialisation mismatch that appears to
  dominate the end-to-end number almost on its own.

Not concluded here (per instruction: report the decomposition, do not choose the repair):
  - whether the reset-window mismatch, the h=1 local-prediction deficit, or rollout drift is
    the highest-priority target;
  - whether a different checkpoint, different residual directions, or a defined grounded
    horizon h* is the right next move;
  - Case 5 (short horizon works, fails beyond h*) does not cleanly apply -- the deficit is
    already present at the shortest horizon tested (h=1), not only beyond some h*.
""".strip()
    wrapped = "\n".join(textwrap.fill(line, 96, subsequent_indent="  ") if line.strip() else "" for line in text.split("\n"))
    fig.text(0.06, 0.90, wrapped, va="top", fontsize=8.3, family="monospace")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    d = load()
    with PdfPages(PDF_PATH) as pdf:
        page_title(pdf, d)
        page_joint_curves(pdf, d)
        page_perbit_h1(pdf, d)
        page_table_joint(pdf, d)
        page_decision(pdf, d)
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
