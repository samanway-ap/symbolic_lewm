"""Model-agnostic right action of A* on a windowed latent state
(SYMBOLIC_ABSTRACTION_PLAN.md, Phase A.1-A.2).

Deliberately independent of any specific model `g` -- the real LeWM predictor
is one instantiation of `step_fn`; the known-automaton round-trip test
(tests/test_known_automaton_roundtrip.py) is another. This module only
assumes:

    step_fn(window, symbol) -> window      # one abstract-action application of g
    label_fn(window) -> label              # phi

`WindowState` is left opaque -- step_fn/label_fn define its shape (a tuple of
the last N raw latents for the real model; a tuple of numpy embeddings in the
synthetic tests).

Multiple initial conditions (A.2) are handled by reset symbols: a word is
well-formed iff its first letter is a reset symbol and no reset symbol
occurs elsewhere. Ill-formed words map to `SINK_LABEL`, never raise.
"""
from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence

from oracle.cache import PrefixTrie

Symbol = Hashable
Word = tuple
WindowState = Any
Label = Hashable

SINK_LABEL = "⊥"  # bottom


class LatentOracle:
    """f(u) = phi( ghat(z0, u) ). Pure and deterministic: repeated calls, in
    any order, against any cache state (cold, warm, or built by evaluating
    the words in a different order/grouping), return the same value for the
    same word.
    """

    def __init__(
        self,
        step_fn: Callable[[WindowState, Symbol], WindowState],
        label_fn: Callable[[WindowState], Label],
        resets: dict[Symbol, WindowState],
        sink_label: Label = SINK_LABEL,
    ) -> None:
        if not resets:
            raise ValueError("at least one reset symbol/window is required")
        self.step_fn = step_fn
        self.label_fn = label_fn
        self.resets = dict(resets)
        self.reset_symbols = frozenset(resets.keys())
        self.sink_label = sink_label
        self.cache = PrefixTrie()

    def is_well_formed(self, word: Word) -> bool:
        if len(word) == 0:
            return False
        if word[0] not in self.reset_symbols:
            return False
        return not any(s in self.reset_symbols for s in word[1:])

    def evaluate(self, word: Word) -> Label:
        """Pure: no observable side effect except cache population (which is
        itself an idempotent memoisation, not part of the semantics)."""
        word = tuple(word)
        if not self.is_well_formed(word):
            return self.sink_label

        state, done_len = self.cache.longest_cached_prefix(word)
        if done_len == 0:
            state = self.resets[word[0]]
            self.cache.insert(word[:1], state)
            done_len = 1

        for i in range(done_len, len(word)):
            state = self.step_fn(state, word[i])
            self.cache.insert(word[: i + 1], state)

        return self.label_fn(state)

    def evaluate_batch(self, words: Sequence[Word]) -> list[Label]:
        """Correctness contract only: any grouping/ordering of `words` must
        yield the same per-word results as evaluating them one at a time in
        any other order, since all of them share the same prefix cache.
        Real tensor-batched vectorisation of step_fn across the model's batch
        dimension (Phase A.3) is a performance concern layered on top of this
        contract for the real LeWM predictor, not something this generic
        module needs to implement to be correct.
        """
        return [self.evaluate(w) for w in words]

    def hit_rate(self) -> float:
        return self.cache.hit_rate()
