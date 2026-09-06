"""E0c's repaired equivalence oracle.

AALpy's stock `PacOracle.find_cex` already implements the round-scaled
sample-count formula from Mohri et al.:
  num_test_cases = 1/epsilon * (log(1/delta) + round*log(2))
(verified by reading its source directly -- see preregistration.md). What it
does NOT do is scale with the CURRENT HYPOTHESIS SIZE within a round, which
is the specific mechanism SELF_IMPROVEMENT_LOOP.md v2 SS2 blames for why
instability grows with m: a 30-state hypothesis needs more evidence than a
6-state one to be tested to the same confidence, and the stock oracle tests
both the same way at a given round.

`ScaledPacOracle` adds the missing floor:
  num_test_cases = max(formula_value, min_words_per_round_coef * |Q_hyp| * |A|)
Everything else (word sampling, per-word membership comparison) is
byte-for-byte AALpy's own implementation -- duplicated here only because
`find_cex` computes and consumes `num_test_cases` in one expression, so
overriding the floor requires overriding the whole method.
"""
from __future__ import annotations

from math import ceil, log
from random import choice, randint

from aalpy.oracles import PacOracle


class ScaledPacOracle(PacOracle):
    def __init__(self, alphabet: list, sul, epsilon: float = 0.02, delta: float = 0.05,
                  min_walk_len: int = 4, max_walk_len: int = 20, min_words_per_round_coef: float = 4.0):
        super().__init__(alphabet, sul, epsilon=epsilon, delta=delta,
                           min_walk_len=min_walk_len, max_walk_len=max_walk_len)
        self.min_words_per_round_coef = min_words_per_round_coef
        self.last_num_test_cases = 0   # exposed for logging/diagnostics

    def find_cex(self, hypothesis):
        self.round += 1
        formula_cases = 1 / self.epsilon * (log(1 / self.delta) + self.round * log(2))
        floor_cases = self.min_words_per_round_coef * len(hypothesis.states) * len(self.alphabet)
        num_test_cases = max(formula_cases, floor_cases)
        self.last_num_test_cases = num_test_cases

        for _ in range(ceil(num_test_cases)):
            inputs = []
            self.reset_hyp_and_sul(hypothesis)
            num_steps = randint(self.min_walk_len, self.max_walk_len)
            for _ in range(num_steps):
                inputs.append(choice(self.alphabet))
                out_sul = self.sul.step(inputs[-1])
                out_hyp = hypothesis.step(inputs[-1])
                self.num_steps += 1
                if out_sul != out_hyp:
                    self.sul.post()
                    return inputs
            self.sul.post()
        return None
