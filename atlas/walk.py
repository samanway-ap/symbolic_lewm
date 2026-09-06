"""Correctly walk the frozen Moore machine along a REAL trajectory.

Why this module exists: the machine's alphabet includes 8 reset symbols, and
a word is well-formed ONLY if its first letter is a reset symbol (see
oracle/latent_oracle.py's `is_well_formed`). Stepping an action symbol from
the initial state instead sends the machine to the SINK, whose output is the
sink label -- which never equals a real predicate tuple. Doing that silently
produces:

  * abstraction-failure rate = 1.0 everywhere (every comparison fails), and
  * behavioural coherence = 1.0 everywhere (every episode "shares" the sink
    state, so no cell can ever be flagged incoherent)

Both look like findings and are actually the same bug. This module mirrors
gate/acceptance.py's established convention exactly -- pick the nearest reset
symbol by encoded-first-window distance, prepend it, then compare the
machine's output after k+1 letters against phi(z_{HISTORY_SIZE + k}).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.lewm_g import HISTORY_SIZE


def reset_reference(reset_windows: dict):
    names = list(reset_windows.keys())
    Z = np.stack([reset_windows[r].emb[-1].numpy() for r in names])
    return names, Z


def choose_reset(z_first: np.ndarray, reset_names: list[str], reset_Z: np.ndarray) -> str:
    """Nearest canonical reset symbol to this episode's own first window --
    the same proxy gate/acceptance.py uses for held-out episodes that are not
    themselves one of the 8 canonical reset episodes."""
    d = np.linalg.norm(reset_Z - z_first[None, :], axis=1)
    return reset_names[int(d.argmin())]


def walk(machine, Z: np.ndarray, symbol_names: list[str], reset_names: list[str],
           reset_Z: np.ndarray, predicate_fns, max_steps: int | None = None):
    """Returns (n_failures, n_compared, visited_state_ids, machine_outputs).

    Z: (T,D) real latents for one episode. symbol_names: the episode's full
    per-position symbol sequence (symbol_names[t] drives z_t -> z_{t+1})."""
    reset_sym = choose_reset(Z[HISTORY_SIZE - 1], reset_names, reset_Z)
    future = symbol_names[HISTORY_SIZE:]
    if max_steps is not None:
        future = future[:max_steps]

    machine.reset_to_initial()
    outputs = [machine.step(reset_sym)]
    visited = [machine.current_state.state_id]
    for s in future:
        outputs.append(machine.step(s))
        visited.append(machine.current_state.state_id)

    fails = total = 0
    n_cmp = min(len(outputs) - 1, max(0, Z.shape[0] - HISTORY_SIZE))
    for t in range(n_cmp):
        real = tuple(f(Z[HISTORY_SIZE + t]) for f in predicate_fns)
        total += 1
        fails += int(outputs[t + 1] != real)
    return fails, total, visited, outputs


def episode_state(machine, Z: np.ndarray, reset_names: list[str], reset_Z: np.ndarray) -> int:
    """The automaton state an episode lands in after its reset symbol -- the
    unit behavioural-coherence is measured over."""
    reset_sym = choose_reset(Z[HISTORY_SIZE - 1], reset_names, reset_Z)
    machine.reset_to_initial()
    machine.step(reset_sym)
    return machine.current_state.state_id
