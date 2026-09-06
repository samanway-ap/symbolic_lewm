"""§15 'Oracle purity': evaluate(u) returns identical output across calls,
cache states, and batch groupings. Property test over random words.
"""
from __future__ import annotations

import random

from hypothesis import given, settings, strategies as st

from tests.fixtures.synthetic_fsm import ALPHABET, make_oracle

WORDS = st.lists(st.sampled_from(ALPHABET), min_size=0, max_size=12).map(tuple)


@given(word=WORDS)
@settings(max_examples=300)
def test_repeated_calls_agree(word):
    """Same oracle, same word, called twice -- must agree (idempotent cache)."""
    oracle = make_oracle()
    assert oracle.evaluate(word) == oracle.evaluate(word)


@given(word=WORDS)
@settings(max_examples=300)
def test_cold_vs_warm_cache_agree(word):
    """A fresh oracle (cold cache) and one whose cache was warmed by
    evaluating unrelated words first must agree on `word`."""
    cold = make_oracle()
    warm = make_oracle()
    decoys = [tuple(random.choice(ALPHABET) for _ in range(random.randint(0, 8))) for _ in range(20)]
    for d in decoys:
        warm.evaluate(d)
    assert cold.evaluate(word) == warm.evaluate(word)


@given(words=st.lists(WORDS, min_size=1, max_size=15))
@settings(max_examples=150)
def test_evaluation_order_does_not_matter(words):
    """Evaluate the same set of words in two different orders against two
    independent oracles (sharing no cache) -- per-word results must match."""
    forward = make_oracle()
    backward = make_oracle()
    fwd_results = [forward.evaluate(w) for w in words]
    bwd_results_reversed = [backward.evaluate(w) for w in reversed(words)]
    bwd_results = list(reversed(bwd_results_reversed))
    assert fwd_results == bwd_results


@given(words=st.lists(WORDS, min_size=1, max_size=15))
@settings(max_examples=150)
def test_batch_grouping_does_not_matter(words):
    """evaluate_batch(words) must equal evaluating each word individually
    (against a fresh oracle sharing the same underlying cache), regardless of
    how the batch is internally grouped/ordered."""
    oracle = make_oracle()
    batch_results = oracle.evaluate_batch(words)
    individual_results = [oracle.evaluate(w) for w in words]
    assert batch_results == individual_results

    # also check a shuffled re-submission of the same batch is consistent
    shuffled = list(words)
    random.shuffle(shuffled)
    shuffled_results = {w: r for w, r in zip(shuffled, oracle.evaluate_batch(shuffled))}
    for w, r in zip(words, batch_results):
        assert shuffled_results[w] == r
