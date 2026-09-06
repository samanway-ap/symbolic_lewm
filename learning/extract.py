"""Phase D.3-D.4: run the Angluin loop (run_Lsharp, per the pre-experiment
tests' AALpy-version finding -- no run_TTT in aalpy==1.6.2) against a real
production oracle for one predicate subset. Hard caps: max_learning_rounds
(wall-clock/query proxy) and a post-hoc max_states check (AALpy has no
live mid-loop circuit breaker without deeper hooking; time-constrained
simplification -- an oversized result is still a valid DATA POINT for
Phase E's search, not a crash).

Equivalence oracle: AALpy's PacOracle (matches D.2(a)'s formula exactly),
used alone -- the custom ngram-weighted-word-distribution and
abstraction-refinement oracles from the plan's hybrid design are dropped
under time pressure. Documented simplification, not silently cut.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aalpy.learning_algs import run_Lsharp
from aalpy.oracles import PacOracle
from learning.sul import LeWMSUL
from oracle.latent_oracle import LatentOracle

MAX_STATES_HARD = 200
TARGET_STATES = 50


@dataclass
class ExtractionResult:
    status: str  # converged | aborted | error
    machine: object | None
    n_states: int
    n_queries: int
    wall_time_s: float
    predicate_subset: list


def extract(
    predicate_subset: list[str],
    oracle: LatentOracle,
    alphabet: list[str],
    max_learning_rounds: int = 12,
    pac_epsilon: float = 0.08,
    pac_delta: float = 0.1,
    max_states_hard: int | None = None,
    eq_oracle_factory=None,
    separation_rule: str = "ADS",
) -> ExtractionResult:
    """`max_states_hard` overrides MAX_STATES_HARD for this call only. Phase T2
    passes 2000 for TESSELLATION extractions, where the point is to MEASURE
    state blowup rather than to pass a gate -- see tessellation/guard.py for
    why such machines can never be used for anything else.

    `eq_oracle_factory(alphabet, sul, epsilon, delta) -> oracle`, if given,
    replaces the stock `PacOracle` -- e0c_repair/scaled_pac_oracle.py's
    `ScaledPacOracle` is the first caller (adds a hypothesis-size-scaled
    sample floor the stock oracle lacks; see preregistration.md's E0c entry).

    `separation_rule` forwards to `run_Lsharp` ("ADS" or "SepSeq" -- AALpy's
    own two options). "ADS" is AALpy's own default and was this project's
    implicit choice until profiled (preregistration.md's "E0c timing probe"/
    cProfile follow-up): ADS construction (`construct_ads_rec` and friends in
    aalpy's ADS.py) accounted for >half of total wall time in a 6-round m=2
    profile, with clearly superlinear recursive fan-out (2.3M recursive calls
    from 1283 top-level invocations). Exposed here so callers can compare
    against "SepSeq" without editing this function again.
    """
    hard_cap = MAX_STATES_HARD if max_states_hard is None else max_states_hard
    t0 = time.time()
    sul = LeWMSUL(oracle)
    if eq_oracle_factory is not None:
        eq_oracle = eq_oracle_factory(alphabet, sul, pac_epsilon, pac_delta)
    else:
        eq_oracle = PacOracle(alphabet, sul, epsilon=pac_epsilon, delta=pac_delta, min_walk_len=4, max_walk_len=20)
    try:
        result = run_Lsharp(
            alphabet, sul, eq_oracle, automaton_type="moore", separation_rule=separation_rule,
            max_learning_rounds=max_learning_rounds, print_level=0, return_data=True,
        )
        machine, info = result if isinstance(result, tuple) else (result, {})
        n_states = machine.size
        status = "converged" if n_states <= hard_cap else "aborted"
    except Exception as e:  # noqa: BLE001 -- an aborted extraction is a data point, not a crash
        return ExtractionResult(status="error", machine=None, n_states=-1,
                                 n_queries=sul.num_queries, wall_time_s=time.time() - t0,
                                 predicate_subset=predicate_subset)
    return ExtractionResult(
        status=status, machine=machine, n_states=n_states,
        n_queries=sul.num_queries, wall_time_s=time.time() - t0,
        predicate_subset=predicate_subset,
    )
