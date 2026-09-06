"""Hard guard: tessellation-derived machines are DIAGNOSTIC ONLY.

Phase T2 extracts automata from a geometrically neutral partition purely to
MEASURE state blowup against the predicate-discovery Pareto front. Such a
machine must never pass Phase F, and must never drive the Phase K
curriculum, whatever its numbers happen to be -- a uniform partition is not
a Markov partition, so signals derived from it would be structure-shaped
noise. This is enforced here rather than left to convention, per the
explicit instruction to assert it in code.
"""
from __future__ import annotations

CELL_PREDICATE_PREFIX = "cell_h"
DIAGNOSTIC_KEY = "diagnostic_only"


def is_tessellation_label_set(subset) -> bool:
    return any(str(n).startswith(CELL_PREDICATE_PREFIX) for n in (subset or []))


def assert_not_tessellation(subset, where: str) -> None:
    """Call at every point where a machine could influence an accept/reject
    decision or a training signal."""
    if is_tessellation_label_set(subset):
        raise AssertionError(
            f"{where}: refusing to use a TESSELLATION-derived label set {list(subset)!r}. "
            "Tessellation machines are diagnostic-only by pre-registration -- they may not "
            "pass Phase F nor drive the Phase K curriculum. See tessellation/guard.py."
        )


def mark_diagnostic(d: dict) -> dict:
    d[DIAGNOSTIC_KEY] = True
    d["diagnostic_note"] = (
        "Tessellation FSM: measured for blowup comparison only. Must never pass Phase F "
        "or drive the curriculum (enforced by tessellation/guard.py)."
    )
    return d
