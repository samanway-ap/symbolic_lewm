"""Persist the PRE-REGISTERED m=8 primary Phase R/G outcome.

The m=8 run reached Phase G for 3 seeded regions and was starved at the
containment filter in all of them (the third also hit a transient HF I/O
outage). It then crashed in np.stack before writing its atlas. That outcome
is a real result -- geometric retrieval cannot supply the pre-registered
>=200 candidate trajectories per region at this granularity -- so it is
reconstructed here from the run log's own measured attrition counts rather
than being lost, and marked BLOCKED_UNDERPOWERED (explicitly NOT homogeneous).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).resolve().parents[1] / "artifacts"
LOG = OUT / "phase_rgk.log"


def main():
    text = LOG.read_text(errors="replace")

    fail_line = re.search(r"abstraction-failure rate by region: (\{.*?\})", text)
    fail_by = json.loads(fail_line.group(1).replace("'", '"')) if fail_line else {}
    seeded = re.search(r"seeded \d+ region\(s\).*?: \[(.*?)\]", text)
    seeded_cells = [s.strip().strip("'") for s in seeded.group(1).split(",")] if seeded else []

    attritions = {}
    cur = None
    for line in text.splitlines():
        g = re.search(r"=== Phase G: retrieval for region (\w+) ===", line)
        if g:
            cur = g.group(1)
        a = re.search(r"attrition: (\{.*\})", line)
        if a and cur:
            attritions[cur] = json.loads(a.group(1).replace("'", '"'))

    n_regions = re.search(r"after merging below the \d+-segment floor: (\d+) regions \((\d+) merges\)", text)

    cells = {}
    for bits, rate in fail_by.items():
        rec = {
            "cell_bits": bits, "occupancy": None, "status": "UNEXPLORED",
            "rounds_attempted": 0, "trajectory_ids_used": [], "metric_history": [],
            "block_reason": None, "eligibility_counts": {}, "coherence_fraction": None,
            "abstraction_failure_rate": rate, "merge_history": [],
        }
        if bits in attritions:
            a = attritions[bits]
            rec["status"] = "BLOCKED_UNDERPOWERED"
            rec["eligibility_counts"] = {
                "candidate_trajectories": a.get("novelty", 0),
                "min_required_candidates": 200,
                "encoded": a.get("encoded"), "containment": a.get("containment"),
                "dwell": a.get("dwell"), "non_degenerate": a.get("non_degenerate"),
            }
            rec["attrition"] = a
            rec["block_reason"] = (
                f"retrieval starved at the CONTAINMENT filter: {a.get('containment', 0)} of "
                f"{a.get('encoded', 0)} encoded unseen episodes kept >=80% of latents inside this "
                f"cell (needed >=200 candidates). NOT a homogeneity claim -- the region was never "
                f"testable.")
        elif bits in seeded_cells:
            rec["status"] = "BLOCKED_UNDERPOWERED"
            rec["block_reason"] = ("seeded for retrieval but the unseen-episode scan hit a transient "
                                     "HF video I/O outage (0 episodes decoded); never testable this run.")
        cells[bits] = rec

    atlas = {
        "granularity": "m=8 (PRE-REGISTERED PRIMARY)",
        "n_regions_after_merge": int(n_regions.group(1)) if n_regions else len(cells),
        "n_merges": int(n_regions.group(2)) if n_regions else None,
        "seeded_regions": seeded_cells,
        "cells": cells,
        "note": ("Phase K never ran at m=8: every seeded region failed Phase G's candidate-trajectory "
                   "floor. All blocks are UNDERPOWERED (retrieval starvation), never HOMOGENEOUS."),
    }
    (OUT / "atlas_m8_primary.json").write_text(json.dumps(atlas, indent=2))
    print(f"wrote atlas_m8_primary.json: {len(cells)} regions, "
          f"{sum(1 for c in cells.values() if c['status'] == 'BLOCKED_UNDERPOWERED')} blocked underpowered")


if __name__ == "__main__":
    main()
