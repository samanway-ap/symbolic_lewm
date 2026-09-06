"""AALpy System-Under-Learning adapter (SYMBOLIC_ABSTRACTION_PLAN.md, Phase D.1).

Verified against the installed aalpy==1.6.2 API before writing this (per the
plan's explicit instruction not to assume signatures):

  - aalpy.base.SUL requires pre()/step()/post(). step(letter) with
    letter=None must return the *current* output without consuming a letter
    -- SUL.query() calls step(None) for the empty-word membership query.
  - This AALpy release has no `run_TTT`. The closest available algorithms
    are `run_Lsharp` (Vaandrager, Garhewal, Rot & Wissmann -- apartness +
    observation trees, a modern successor to TTT and *not* an observation
    table like classic L*) and `run_KV` (Kearns-Vazirani, classification
    tree). We substitute `run_Lsharp` for "TTT" throughout; see
    PRE_EXPERIMENT_NOTES.md for the justification. This file does not import
    either learner -- that choice is made at the call site (tests/, and
    later learning/extract.py) so the adapter itself has no algorithm
    dependency.
"""
from __future__ import annotations

from aalpy.base import SUL

from oracle.latent_oracle import LatentOracle, Symbol


class LeWMSUL(SUL):
    """Wraps a LatentOracle as an AALpy SUL. All state lives in the growing
    word; every actual computation (well-formedness, stepping, caching,
    labelling) is delegated to the oracle, so this adapter has no logic of
    its own to get wrong.
    """

    def __init__(self, oracle: LatentOracle) -> None:
        super().__init__()
        self.oracle = oracle
        self._word: tuple[Symbol, ...] = ()

    def pre(self) -> None:
        self._word = ()

    def post(self) -> None:
        self._word = ()

    def step(self, letter):
        if letter is not None:
            self._word = self._word + (letter,)
        return self.oracle.evaluate(self._word)
