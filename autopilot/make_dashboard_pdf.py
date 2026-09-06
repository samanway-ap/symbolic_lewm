"""Renders a data dashboard PDF from both autopilot runs (C00-C04 in
attempt1, C05-C06 in the current run) -- graphs plus a factual methodology
section. No interpretation of results; only what was hypothesised, what the
arms are, and what was measured.
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PDF_PATH = OUT_DIR / "autopilot_dashboard.pdf"

REPORT_PATHS = [OUT_DIR / "autopilot_final_report.json.attempt1", OUT_DIR / "autopilot_final_report.json"]
LOG_PATHS = [OUT_DIR / "autopilot_run.log.attempt1", OUT_DIR / "autopilot_run.log"]
LEDGER_PATHS = [OUT_DIR / "autopilot_ledger.jsonl.attempt1", OUT_DIR / "autopilot_ledger.jsonl"]

CAND_CONFIG = {
    "C00": {"m": 4, "suffix": 1, "policy": "high_W_ratio"},
    "C01": {"m": 2, "suffix": 1, "policy": "high_W_ratio"},
    "C02": {"m": 8, "suffix": 1, "policy": "high_W_ratio"},
    "C03": {"m": 4, "suffix": 1, "policy": "high_residual"},
    "C04": {"m": 4, "suffix": 1, "policy": "diverse_scene"},
    "C05": {"m": 4, "suffix": 2, "policy": "high_W_ratio"},
    "C06": {"m": 2, "suffix": 2, "policy": "high_W_ratio"},
}
ARMS = ["orthogonal", "random_dir", "random_traj", "continue"]
ARM_COLORS = {"orthogonal": "#1b6fa8", "random_dir": "#c46a1f", "random_traj": "#4c8c4a", "continue": "#8a8a8a"}


def load_all_nodes():
    nodes = {}
    total_gpu_hours = 0.0
    for p in REPORT_PATHS:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        total_gpu_hours += d["gpu_hours_used"]
        for r in d["run_log"]:
            nodes[r["id"]] = r
    return nodes, total_gpu_hours


def parse_log_metrics(log_paths):
    """Pulls per-arm E_W_mean at each stage/seed, and retrieval attrition,
    directly from the run logs (this data isn't persisted in the JSON
    reports)."""
    text = ""
    for p in log_paths:
        if p.exists():
            text += p.read_text() + "\n"

    ew = {}   # node -> stage -> arm -> [E_W_mean per seed]
    for m in re.finditer(r"\[(\w+)/(\w+)\] arm=(\w+) seed=\d+ steps=\d+ loss=[\d.]+ E_W_mean=([\d.]+)", text):
        node, stage, arm, val = m.group(1), m.group(2), m.group(3), float(m.group(4))
        ew.setdefault(node, {}).setdefault(stage, {}).setdefault(arm, []).append(val)

    attrition = {}   # node -> arm -> {stage: count}
    for m in re.finditer(r"\[(\w+)\] retrieval\[(\w+)\]: (\{.*\})", text):
        node, arm, d_str = m.group(1), m.group(2), m.group(3)
        d_str_py = d_str.replace("'", '"')
        try:
            d = json.loads(d_str_py)
        except json.JSONDecodeError:
            continue
        attrition.setdefault(node, {})[arm] = d
    return ew, attrition


def page_title(pdf, nodes, total_gpu_hours):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.94, "Orthogonal-Curriculum Autopilot -- Results Dashboard", ha="center",
              fontsize=16, weight="bold")
    fig.text(0.5, 0.915, "v6 exploratory predictor-adaptation experiment tree (E0-E4)", ha="center",
              fontsize=10, style="italic", color="#444")

    hypothesis = """
HYPOTHESIS TESTED

Among equally sized real-data training curricula, trajectories selected for local latent
displacement along W_proxy (directions the frozen symbolic abstraction cannot see) improve
short-horizon continuous prediction in that frozen target subspace more than continued
training on the original data mixture, and more than a curriculum selected via a matched
random direction.

Success tier targeted: S2_COARSE_SUCCESS -- the orthogonal arm beats BOTH random_dir and
continue, with positive lower confidence bounds, on a robust rerun.
""".strip()

    arms = """
ARMS (identical starting checkpoint, optimizer, schedule, batch order, and seed; only the
training data differs)

  orthogonal   trajectories retrieved for alignment with W_proxy (the target subspace)
  random_dir   same retrieval pipeline, aligned to a dimension-matched RANDOM subspace instead
  random_traj  uniformly sampled eligible trajectories, no directional selection at all
  continue     the original training-data mixture (the "just keep training" baseline)
""".strip()

    setup = f"""
SETUP

Checkpoint: partial (epoch 7/20, ~35% trained), encoder frozen throughout -- only the
predictor moves. 7 candidates tried (C00-C06), varying the partition width (m), the gradient
suffix length (1 or 2 real letters), and the base-point selection policy. Successive halving
per candidate: Stage 1 (1 seed, 200 updates) -> Stage 2 (2 seeds, 500 updates) -> Stage 3
(3 seeds, 1000 updates, reached by none). Total compute used: {total_gpu_hours:.3f} of a
10 GPU-hour budget.
""".strip()

    y = 0.86
    for block in (hypothesis, arms, setup):
        wrapped = "\n".join(textwrap.fill(line, 96) if line.strip() else "" for line in block.split("\n"))
        fig.text(0.07, y, wrapped, va="top", fontsize=8.6, family="monospace")
        y -= 0.10 + 0.017 * block.count("\n")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def page_status_table(pdf, nodes):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    rows = []
    for nid in CAND_CONFIG:
        cfg = CAND_CONFIG[nid]
        r = nodes.get(nid)
        if r is None:
            rows.append([nid, cfg["m"], cfg["suffix"], cfg["policy"], "not run", "-"])
            continue
        farthest = "stage3" if "stage3" in r else ("stage2" if "stage2" in r else "stage1")
        rows.append([nid, cfg["m"], cfg["suffix"], cfg["policy"], r["status"], farthest])
    col_labels = ["node", "m", "suffix", "base policy", "status", "farthest stage reached"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.8)
    for k in range(len(col_labels)):
        table[0, k].set_facecolor("#dbe7f0")
        table[0, k].set_text_props(weight="bold")
    ax.set_title("Node summary", fontsize=13, weight="bold", pad=20)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_ew_by_node(pdf, ew_data):
    node_ids = [n for n in CAND_CONFIG if n in ew_data]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    axes = axes.flat
    for ax, nid in zip(axes, node_ids):
        stages = list(ew_data[nid].keys())
        stage = stages[-1] if stages else None
        if stage is None:
            ax.axis("off")
            continue
        arm_vals = ew_data[nid][stage]
        means = [np.mean(arm_vals.get(a, [np.nan])) for a in ARMS]
        colors = [ARM_COLORS[a] for a in ARMS]
        ax.bar(ARMS, means, color=colors)
        ax.set_title(f"{nid} ({stage})", fontsize=9)
        ax.set_ylabel("E_W (mean)", fontsize=7)
        ax.tick_params(axis="x", labelsize=6, rotation=20)
        ax.tick_params(axis="y", labelsize=7)
    for ax in list(axes)[len(node_ids):]:
        ax.axis("off")
    fig.suptitle("Evaluation error by arm, per node (most advanced stage reached)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def page_gains(pdf, nodes):
    node_ids = [n for n in CAND_CONFIG if n in nodes]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(node_ids))
    width = 0.2
    for i, arm in enumerate(ARMS):
        vals = []
        for nid in node_ids:
            r = nodes[nid]
            stage_key = "stage3" if "stage3" in r else ("stage2" if "stage2" in r else "stage1")
            vals.append(r[stage_key]["gains"][arm])
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=arm, color=ARM_COLORS[arm])
        for xi, v in zip(x + (i - 1.5) * width, vals):
            if v < -0.6:
                ax.text(xi, -0.6, f"{v:.2f}", ha="center", va="top", fontsize=6, rotation=90)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.02, color="red", lw=0.6, ls="--", label="min target gain (0.02)")
    ax.set_ylim(-0.6, 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(node_ids)
    ax.set_ylabel("relative gain vs pre-training E_W\n(values beyond axis range printed as text)")
    ax.set_title("Relative gain by arm, per node (most advanced stage reached)", fontsize=12, weight="bold")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_forest(pdf, nodes):
    node_ids = [n for n in CAND_CONFIG if n in nodes]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, comp, title in zip(axes, ("adv_vs_random_dir", "adv_vs_continue"),
                                  ("orthogonal advantage vs random_dir", "orthogonal advantage vs continue")):
        ys = []
        labels = []
        for nid in node_ids:
            r = nodes[nid]
            stage_key = "stage3" if "stage3" in r else ("stage2" if "stage2" in r else "stage1")
            stage = r[stage_key]
            if comp not in stage:
                continue
            c = stage[comp]
            ys.append((c["mean"], c["ci95"][0], c["ci95"][1]))
            labels.append(f"{nid}/{stage_key}")
        y_pos = np.arange(len(labels))
        means = [v[0] for v in ys]
        los = [v[0] - v[1] for v in ys]
        his = [v[2] - v[0] for v in ys]
        colors = ["#1b6fa8" if (v[1] > 0) else ("#a02b2b" if v[2] < 0 else "#999999") for v in ys]
        for yi, m, lo, hi, c in zip(y_pos, means, los, his, colors):
            ax.errorbar([m], [yi], xerr=[[lo], [hi]], fmt="o", color="black", ecolor=c, capsize=4, elinewidth=2)
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("mean paired difference (E_W reduction), 95% CI")
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
    fig.suptitle("Advantage estimates with 95% confidence intervals\n"
                 "(blue = CI entirely positive, red = CI entirely negative, grey = crosses zero)",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    pdf.savefig(fig)
    plt.close(fig)


def load_per_node_gpu_hours():
    """Ledger events carry CUMULATIVE gpu_hours_used per node within their
    own process; deltas give the per-node cost. Attempt1 and the
    C05/C06 continuation are separate processes, each starting its own
    budget at 0 -- deltas are computed within each ledger file separately."""
    per_node = {}
    for p in LEDGER_PATHS:
        if not p.exists():
            continue
        prev = 0.0
        for line in p.read_text().splitlines():
            ev = json.loads(line)
            if ev.get("event_type") != "NODE_EVALUATED":
                continue
            cum = ev["gpu_hours_used"]
            per_node[ev["node_id"]] = cum - prev
            prev = cum
    return per_node


def page_gpu_hours(pdf, nodes, total_gpu_hours):
    node_ids = [n for n in CAND_CONFIG if n in nodes]
    per_node = load_per_node_gpu_hours()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    ax = axes[0]
    vals = [per_node.get(nid, 0.0) for nid in node_ids]
    ax.bar(node_ids, vals, color="#1b6fa8")
    ax.set_ylabel("GPU-hours")
    ax.set_title("GPU-hours per node", fontsize=11, weight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.pie([total_gpu_hours, 10.0 - total_gpu_hours], labels=[f"used\n{total_gpu_hours:.3f}h",
                                                                  f"unused\n{10 - total_gpu_hours:.3f}h"],
            colors=["#1b6fa8", "#dddddd"], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Total GPU-hours used vs 10h budget", fontsize=11, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_retrieval_attrition(pdf, attrition_data):
    node_ids = [n for n in CAND_CONFIG if n in attrition_data]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    axes = axes.flat
    stages_order = ["considered", "locality", "alignment", "non_degenerate", "selected"]
    for ax, nid in zip(axes, node_ids):
        a = attrition_data[nid].get("orthogonal")
        if not a:
            ax.axis("off")
            continue
        vals = [a.get(s, a.get("selected_pool_before_dispersion", 0) if s == "selected" else 0) for s in stages_order[:-1]]
        vals.append(a.get("selected", 0))
        ax.plot(stages_order, vals, "o-", color=ARM_COLORS["orthogonal"])
        ax.set_title(f"{nid}", fontsize=9)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
    for ax in list(axes)[len(node_ids):]:
        ax.axis("off")
    fig.suptitle("Orthogonal-arm retrieval funnel per node\n"
                 "(pool candidates -> locality -> alignment -> non-degeneracy -> selected)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def page_full_data_table(pdf, nodes):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    rows = []
    for nid in CAND_CONFIG:
        r = nodes.get(nid)
        if r is None:
            continue
        for stage_key in ("stage1", "stage2", "stage3"):
            if stage_key not in r:
                continue
            s = r[stage_key]
            adv_dir = s.get("adv_vs_random_dir", {})
            adv_cont = s.get("adv_vs_continue", {})
            rows.append([
                nid, stage_key, s.get("n_seeds", ""),
                f"{s['gains']['orthogonal']:+.3f}",
                f"{adv_dir.get('mean', float('nan')):+.5f}",
                f"[{adv_dir.get('ci95', ['', ''])[0]:+.5f},{adv_dir.get('ci95', ['', ''])[1]:+.5f}]" if adv_dir else "",
                f"{adv_cont.get('mean', float('nan')):+.5f}",
                f"[{adv_cont.get('ci95', ['', ''])[0]:+.5f},{adv_cont.get('ci95', ['', ''])[1]:+.5f}]" if adv_cont else "",
                str(s.get("forgetting", "")), str(s.get("S1_ROUTE_PASS", "")),
            ])
    col_labels = ["node", "stage", "seeds", "orth. gain", "adv vs\nrandom_dir", "CI95", "adv vs\ncontinue", "CI95",
                   "forgetting", "route\npass"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    table.scale(1, 1.5)
    for k in range(len(col_labels)):
        table[0, k].set_facecolor("#dbe7f0")
        table[0, k].set_text_props(weight="bold")
    ax.set_title("Full numeric results, every node x stage", fontsize=12, weight="bold", pad=20)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    nodes, total_gpu_hours = load_all_nodes()
    ew_data, attrition_data = parse_log_metrics(LOG_PATHS)

    with PdfPages(PDF_PATH) as pdf:
        page_title(pdf, nodes, total_gpu_hours)
        page_status_table(pdf, nodes)
        page_ew_by_node(pdf, ew_data)
        page_gains(pdf, nodes)
        page_forest(pdf, nodes)
        page_gpu_hours(pdf, nodes, total_gpu_hours)
        page_retrieval_attrition(pdf, attrition_data)
        page_full_data_table(pdf, nodes)
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
