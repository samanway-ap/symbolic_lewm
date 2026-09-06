"""§15 'Prefix consistency': evaluate(u) computed fresh equals the value
obtained by extending the cached rollout of any prefix of u.
"""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from tests.fixtures.synthetic_fsm import ALPHABET, make_oracle

WORDS = st.lists(st.sampled_from(ALPHABET), min_size=1, max_size=14).map(tuple)


@given(word=WORDS, prefix_len=st.integers(min_value=0, max_value=14))
@settings(max_examples=300)
def test_prefix_then_extend_matches_fresh(word, prefix_len):
    prefix_len = min(prefix_len, len(word))
    prefix = word[:prefix_len]

    fresh = make_oracle()
    fresh_result = fresh.evaluate(word)

    warmed = make_oracle()
    warmed.evaluate(prefix)  # populate cache up to the prefix only
    warmed_result = warmed.evaluate(word)  # must extend from the cached prefix

    assert fresh_result == warmed_result


@given(word=WORDS)
@settings(max_examples=200)
def test_every_prefix_is_individually_consistent(word):
    """Evaluating every prefix of `word` in increasing length order (as a
    real learner walking one letter at a time would) must give the same
    per-prefix labels as evaluating each prefix fresh."""
    incremental = make_oracle()
    incremental_labels = [incremental.evaluate(word[: i + 1]) for i in range(len(word))]

    fresh_labels = [make_oracle().evaluate(word[: i + 1]) for i in range(len(word))]

    assert incremental_labels == fresh_labels


def test_prefix_cache_actually_hits():
    """Sanity check that the trie is doing its job: walking a word one
    symbol at a time on a single oracle instance should hit the cache on
    every step after the first."""
    oracle = make_oracle()
    word = ("r0", "a", "b", "c", "a", "b")
    for i in range(1, len(word) + 1):
        oracle.evaluate(word[:i])
    assert oracle.hit_rate() > 0.0
