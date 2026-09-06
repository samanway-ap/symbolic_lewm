"""Hand-built finite Moore machine, lifted into R^d as one-hot-plus-fixed-
noise embeddings.

This is the fixture behind "the single most valuable test in the suite"
(SYMBOLIC_ABSTRACTION_PLAN.md §15, known-automaton round trip): replace `g`
with a synthetic system whose ground truth we know exactly, and check the
Angluin loop recovers it.

Two machines:

  * `build_real_machine()`     -- the "task" automaton over actions
    {a, b, c}, 5 states, deliberately built so every state is distinguishable
    from every other (no two states have identical future behaviour) even
    though several share an output label -- this forces the learner to do
    real work, mirroring |B| << n_states in the real predicate setting.

  * `build_extended_reference()` -- `build_real_machine()` plus the reset /
    ill-formed bookkeeping states (`pre`, `sink`) over the FULL alphabet
    {r0, r1, a, b, c}. This is the ground truth the *learned* Moore machine
    should be bisimilar to, since LatentOracle's reset-symbol handling
    (Phase A.2) is itself part of `f` -- the oracle, not `g`, enforces that a
    word must start with a reset and contain no other, and that violating
    words map to a sink label forever.

The "lifting": each real-machine state gets one fixed point in R^d (a scaled
one-hot plus a small seeded perturbation, generated once -- never re-sampled
per call, since the oracle built on top of it must be pure). `step_fn`/
`label_fn` decode a window's last entry back to a state id by nearest
centroid; centroids are spaced far enough apart (`scale >> noise_std`) that
this decode is always exact, so it introduces no ambiguity of its own.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from aalpy.automata import MooreMachine

from oracle.latent_oracle import LatentOracle

ACTIONS = ("a", "b", "c")
RESETS = ("r0", "r1")
ALPHABET = list(RESETS) + list(ACTIONS)

# The ground-truth automaton's ill-formed-word output. LatentOracle's own
# sink_label defaults to "⊥" -- it must be pointed at THIS string (see
# make_oracle below), or f(epsilon) and every ill-formed word will disagree
# with build_extended_reference() on label alone, independent of anything
# the learner does.
SINK_OUTPUT = "SINK"

# Real task automaton. Built so every pair of states is distinguishable:
# e.g. s0 and s4 share output "L0" but s0 --a--> s1 (L1) while s4 --a--> s0
# (L0), so the single-letter suffix "a" already tells them apart.
_REAL_STATE_SETUP = {
    "s0": ("L0", {"a": "s1", "b": "s2", "c": "s0"}),
    "s1": ("L1", {"a": "s2", "b": "s3", "c": "s0"}),
    "s2": ("L2", {"a": "s3", "b": "s4", "c": "s1"}),
    "s3": ("L1", {"a": "s4", "b": "s0", "c": "s2"}),
    "s4": ("L0", {"a": "s0", "b": "s1", "c": "s3"}),
}
REAL_STATE_IDS = list(_REAL_STATE_SETUP.keys())

# r0 and r1 are two distinct initial conditions (A.2): they land on
# different real states, so a single learned machine must cover both from
# shared structure.
RESET_TARGETS = {"r0": "s0", "r1": "s2"}


def build_real_machine() -> MooreMachine:
    """The task automaton alone, alphabet = ACTIONS only. Used as the
    ground truth for step_fn/label_fn (never sees reset symbols)."""
    return MooreMachine.from_state_setup(dict(_REAL_STATE_SETUP))


def build_extended_reference() -> MooreMachine:
    """Ground truth for the round-trip test: full ALPHABET including the
    reset / ill-formed bookkeeping. `pre` is the initial state (its output
    is what f(epsilon) must equal)."""
    ext = {
        "pre": (
            SINK_OUTPUT,
            {**{r: RESET_TARGETS[r] for r in RESETS}, **{a: "sink" for a in ACTIONS}},
        ),
        "sink": (SINK_OUTPUT, {**{r: "sink" for r in RESETS}, **{a: "sink" for a in ACTIONS}}),
    }
    for sid, (out, trans) in _REAL_STATE_SETUP.items():
        ext[sid] = (out, {**trans, **{r: "sink" for r in RESETS}})
    return MooreMachine.from_state_setup(ext)


def lift_states_to_rd(
    d: int = 16, scale: float = 8.0, noise_std: float = 0.05, seed: int = 3072
) -> dict[str, np.ndarray]:
    """Fixed point in R^d per real-machine state: scaled one-hot + a small
    FIXED (seeded once, never re-sampled) perturbation. Deterministic across
    calls with the same seed -- a lifting, not a source of randomness."""
    rng = np.random.default_rng(seed)
    centroids: dict[str, np.ndarray] = {}
    for i, sid in enumerate(REAL_STATE_IDS):
        onehot = np.zeros(d, dtype=np.float64)
        onehot[i % d] = scale
        centroids[sid] = onehot + rng.normal(0.0, noise_std, size=d)
    # sanity: nearest-centroid decode must be exact given this margin
    for sid, z in centroids.items():
        assert nearest_state_id(z, centroids) == sid, (
            "noise_std too large relative to scale -- centroids overlap"
        )
    return centroids


def nearest_state_id(z: np.ndarray, centroids: dict[str, np.ndarray]) -> str:
    ids = list(centroids.keys())
    dists = [float(np.linalg.norm(z - centroids[i])) for i in ids]
    return ids[int(np.argmin(dists))]


def make_step_and_label_fns(
    centroids: dict[str, np.ndarray], real_machine: MooreMachine
) -> tuple[Callable, Callable]:
    """step_fn/label_fn operating on a WINDOW = tuple[np.ndarray, ...] of the
    last N embeddings, decoding only the last entry (the real machine is
    memoryless; the window exists to exercise Phase A.1's window-carrying
    oracle, mirroring `N > 1` for the real LeWM predictor)."""
    states_by_id = {s.state_id: s for s in real_machine.states}

    def step_fn(window: tuple[np.ndarray, ...], symbol: str) -> tuple[np.ndarray, ...]:
        cur_id = nearest_state_id(window[-1], centroids)
        next_state = states_by_id[cur_id].transitions[symbol]
        next_z = centroids[next_state.state_id]
        return window[1:] + (next_z,)

    def label_fn(window: tuple[np.ndarray, ...]) -> str:
        cur_id = nearest_state_id(window[-1], centroids)
        return states_by_id[cur_id].output

    return step_fn, label_fn


def make_oracle(window_N: int = 2, d: int = 16, seed: int = 3072) -> LatentOracle:
    real_machine = build_real_machine()
    centroids = lift_states_to_rd(d=d, seed=seed)
    step_fn, label_fn = make_step_and_label_fns(centroids, real_machine)

    resets = {
        r: tuple(centroids[target] for _ in range(window_N))
        for r, target in RESET_TARGETS.items()
    }
    return LatentOracle(step_fn=step_fn, label_fn=label_fn, resets=resets, sink_label=SINK_OUTPUT)
