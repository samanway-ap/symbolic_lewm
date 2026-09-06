import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
entries = [json.loads(l) for l in (OUT_DIR / "search_log.jsonl").read_text().splitlines() if l.strip()]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sizes = [e["size"] for e in entries]
n_states = [e["n_states"] for e in entries]
fidelity = [e.get("trace_fidelity", 0) for e in entries]

ax = axes[0]
ax.plot(sizes, n_states, "o-", color="tab:blue")
ax.set_xlabel("|S| (active predicates)"); ax.set_ylabel("n_states (learned automaton)")
ax.set_title("State count vs predicate-set size")
ax.axhline(50, color="gray", linestyle="--", label="target_states=50")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(sizes, fidelity, "o-", color="tab:red")
ax.axhline(0.85, color="k", linestyle="--", label="fidelity threshold=0.85")
ax.set_xlabel("|S| (active predicates)"); ax.set_ylabel("trace_fidelity (train-sample, search-time)")
ax.set_title("Fidelity vs predicate-set size (per-predicate metric, FIX 3b)")
ax.set_ylim(0, 1)
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle("Phase E Pareto front: drawer_open_close, k*=2 (FIX 1/2 rerun, 2026-08-21)")
plt.tight_layout()
plt.savefig(OUT_DIR / "phase_e_pareto_front.png", dpi=140)
print("saved", OUT_DIR / "phase_e_pareto_front.png")

import sys
sys.stdout.flush()
import os
os._exit(0)
