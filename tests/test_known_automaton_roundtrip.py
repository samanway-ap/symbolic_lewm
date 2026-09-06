"""§15 'Known-automaton round trip' -- the single most valuable test in the
suite (SYMBOLIC_ABSTRACTION_PLAN.md explicitly says to write this before
Phase D). Replace `g` with a hand-built finite state machine lifted into
R^d (states as one-hot-plus-noise); the pipeline must recover it exactly.

This is also where the plan's requirement to "verify AALpy's current API
before writing against it" bites: this installed release (aalpy==1.6.2) has
no `run_TTT`. We substitute `run_Lsharp` (Vaandrager et al. -- apartness +
an observation tree, a modern successor to TTT and, like TTT, not an
observation-table algorithm like classic L*). See PRE_EXPERIMENT_NOTES.md.
"""
from __future__ import annotations

from aalpy.learning_algs import run_Lsharp
from aalpy.oracles import WMethodEqOracle
from aalpy.utils import bisimilar

from learning.sul import LeWMSUL
from tests.fixtures.synthetic_fsm import ALPHABET, SINK_OUTPUT, build_extended_reference, make_oracle


def _learn(window_N: int, seed: int):
    oracle = make_oracle(window_N=window_N, seed=seed)
    sul = LeWMSUL(oracle)
    # W-method is exhaustive up to max_number_of_states; the true machine has
    # 7 states (pre, sink, s0..s4) -- give it headroom so a wrong/oversized
    # hypothesis would still be caught rather than accidentally accepted.
    eq_oracle = WMethodEqOracle(ALPHABET, sul, max_number_of_states=12)
    learned = run_Lsharp(
        ALPHABET,
        sul,
        eq_oracle,
        automaton_type="moore",
        print_level=0,
    )
    return learned, oracle


def test_recovers_extended_reference_exactly():
    learned, _oracle = _learn(window_N=2, seed=3072)
    reference = build_extended_reference()

    assert reference.size == 7, "fixture sanity: pre, sink, s0..s4"

    cex = bisimilar(learned, reference, return_cex=True)
    assert cex is None, f"learned machine disagrees with ground truth on: {cex}"

    is_bisim = bisimilar(learned, reference)
    assert is_bisim is True

    assert learned.size == reference.size, (
        f"learned machine has {learned.size} states, expected {reference.size} "
        "(both should be minimal, so equal state count is expected whenever "
        "bisimilarity holds -- a mismatch here would itself be a red flag "
        "even if bisimilar() were somehow wrong)"
    )


def test_recovery_is_stable_across_window_size_and_seed():
    """Reproducibility check in the spirit of Phase F's acceptance gate:
    different query seeds / window sizes recovering non-isomorphic machines
    would mean the oracle construction (not the learner) is the bug."""
    reference = build_extended_reference()
    for window_N, seed in [(1, 0), (3, 7), (2, 12345)]:
        learned, _oracle = _learn(window_N=window_N, seed=seed)
        assert bisimilar(learned, reference), (
            f"window_N={window_N} seed={seed}: recovered machine not "
            "bisimilar to ground truth"
        )


def test_empty_word_hits_the_pre_state_output():
    """f(epsilon) must equal the ground truth's initial-state ('pre')
    output -- this is what SUL.step(None) exercises."""
    _learned, oracle = _learn(window_N=2, seed=3072)
    assert oracle.evaluate(()) == SINK_OUTPUT


def test_ill_formed_words_map_to_sink():
    _learned, oracle = _learn(window_N=2, seed=3072)
    # no reset as first letter
    assert oracle.evaluate(("a", "b")) == SINK_OUTPUT
    # reset appears twice
    assert oracle.evaluate(("r0", "a", "r1", "b")) == SINK_OUTPUT
    # well-formed control case should NOT be sink
    assert oracle.evaluate(("r0", "a")) != SINK_OUTPUT
