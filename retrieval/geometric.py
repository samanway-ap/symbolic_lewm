"""Phase G -- geometric trajectory selection from UNSEEN DROID episodes.

Pool: DROID episodes NOT in the 3,184 drawer family and NOT in
episode_split.json (~92k available). Encoded lazily, latents cached, so a
cell that gets revisited across rounds never re-streams the same episode.

Four filters, ALL of which must hold for an episode to qualify for cell C:

  1. CONTAINMENT    >= 80% of its latents fall in C (exact bit match --
                    the cell boundary is a partition, not a ball)
  2. DWELL          >= 12 CONSECUTIVE frames inside C
  3. NON-DEGENERACY latent path length >= floor AND mean action magnitude
                    >= floor.
                    *** Load-bearing. Teleop data is full of pauses, and a
                    stationary arm satisfies containment and dwell PERFECTLY
                    while carrying zero dynamics information. Without this
                    the curriculum quietly fills with the robot doing
                    nothing, and every downstream "no improvement" verdict
                    becomes uninterpretable. The count rejected by this
                    filter is reported per cell for exactly that reason. ***
  4. NOVELTY        episode id unused in any prior round (checked against
                    the atlas, not against this run's local state)

No compactness filter -- the cell boundary already enforces it.

Selection among qualifiers maximises DISPERSION within the cell (greedy
farthest-point), not proximity to the centre: K trajectories hugging the
centroid would re-sample the part of the cell the model already fits best.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RETRIEVAL_DIR = OUT_DIR / "retrieval"

CONTAINMENT_MIN = 0.80
DWELL_MIN = 12
PATH_LENGTH_FLOOR_PCT = 25     # percentile of the pool's own path-length distribution
ACTION_MAG_FLOOR_PCT = 25


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def path_length(Z: np.ndarray) -> float:
    if Z.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(Z, axis=0), axis=1).sum())


def action_magnitude(raw: np.ndarray) -> float:
    """raw: (T, FRAMESKIP, 7) in droid_100 convention. Mean L2 over the 6
    non-gripper dims -- the gripper channel is [0,1] and would swamp a
    'is the arm actually moving' test."""
    if raw.size == 0:
        return 0.0
    return float(np.linalg.norm(raw.reshape(-1, raw.shape[-1])[:, :-1], axis=1).mean())


def screen_episode(Z: np.ndarray, raw: np.ndarray, tess, cell_id: int,
                     path_floor: float, action_floor: float) -> dict:
    """Returns per-filter booleans + the measurements behind them, so an
    episode's rejection reason is always attributable to one named filter."""
    ids = tess.cell_ids(Z)
    inside = (ids == cell_id)
    containment = float(inside.mean())
    dwell = longest_run(inside)
    plen = path_length(Z)
    amag = action_magnitude(raw)
    return {
        "containment": containment, "containment_ok": containment >= CONTAINMENT_MIN,
        "dwell": int(dwell), "dwell_ok": dwell >= DWELL_MIN,
        "path_length": plen, "action_magnitude": amag,
        "non_degenerate_ok": bool(plen >= path_floor and amag >= action_floor),
    }


def greedy_dispersion(Z_reps: np.ndarray, K: int, seed: int = 3072) -> list[int]:
    """Farthest-point traversal: seed with the point farthest from the mean,
    then repeatedly take the candidate maximising its minimum distance to the
    already-chosen set. Maximises coverage of the cell, not centrality."""
    n = Z_reps.shape[0]
    if n <= K:
        return list(range(n))
    centre = Z_reps.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(Z_reps - centre, axis=1)))
    chosen = [first]
    dmin = np.linalg.norm(Z_reps - Z_reps[first], axis=1)
    while len(chosen) < K:
        nxt = int(np.argmax(dmin))
        if nxt in chosen:
            remaining = [i for i in range(n) if i not in chosen]
            if not remaining:
                break
            nxt = remaining[0]
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(Z_reps - Z_reps[nxt], axis=1))
    return chosen


def write_retrieval_report(cell_bits: str, attrition: dict, selected: list[int], notes: str = "") -> Path:
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    p = RETRIEVAL_DIR / f"{cell_bits}.md"
    lines = [
        f"# Retrieval report — cell {cell_bits}", "",
        "Filter attrition (each row = candidates surviving THAT filter, applied in order).",
        "A cell starved by filter 3 (non-degeneracy) means the pool had motion-free",
        "trajectories that satisfy the geometry; a cell starved by filter 1 means the",
        "cell is simply rare in unseen data. These are different problems.", "",
        "| stage | surviving |", "|---|---|",
        f"| pool considered | {attrition.get('considered', 0)} |",
        f"| encoded successfully | {attrition.get('encoded', 0)} |",
        f"| 1. containment >= {CONTAINMENT_MIN:.0%} | {attrition.get('containment', 0)} |",
        f"| 2. dwell >= {DWELL_MIN} consecutive | {attrition.get('dwell', 0)} |",
        f"| 3. non-degeneracy (path + action floors) | {attrition.get('non_degenerate', 0)} |",
        f"| 4. novelty (unused in prior rounds) | {attrition.get('novelty', 0)} |",
        f"| **selected by dispersion** | **{len(selected)}** |", "",
        f"Rejected by non-degeneracy alone: **{attrition.get('rejected_by_non_degeneracy', 0)}** "
        f"(these satisfied containment+dwell but the arm was effectively stationary).", "",
        f"Selected episode ids: `{selected}`", "",
    ]
    if notes:
        lines += ["## Notes", "", notes, ""]
    p.write_text("\n".join(lines))
    return p
