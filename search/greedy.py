"""Phase E -- the greedy add-step's candidate-ranking proxy (plan §8.2
point 2: "estimate marginal value cheaply ...; run full extractions only
on the top 3; keep the best").

FIX 1 (2026-08-21, preregistration.md): the proxy is now trace-fidelity of
a majority-vote (state,symbol)->label table on held-out TRAIN transitions,
not block-nondeterminism reduction. Nondeterminism reduction is a
differently-shaped signal than the actual gate: a candidate predicate can
reduce nondeterminism (by partitioning states more finely) while making
fidelity worse, and vice-versa. The new proxy asks exactly the question
J(S) cares about -- "if I add this predicate, do (state,symbol)->label
transitions become more PREDICTABLE on held-out data" -- at the same
O(transitions) cost as the old proxy (no oracle calls, no Angluin
extraction: real extractions are reserved for the top-PROXY_TOPK ranked
candidates per add-step, not run blind on every candidate in the pool).

Drop/CEGAR (plan §8.3) are explicitly out of scope for this fix -- FIX 1
only concerns the add-step's ranking signal.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphabet.segments import segment_feature
from predicates.latent_pool import collect_trajectories, flatten_transitions

PROXY_TOPK = 3


def build_proxy_transitions(
    train_episode_ids: list[int],
    alphabet_feats: np.ndarray,
    alphabet_names: list[str],
    n_episodes: int = 120,
    n_positions: int = 25,
    seed: int = 3072,
) -> dict:
    """Cheap, extraction-free transition dataset for the add-step proxy:
    real latents (zt, znext) + the quantized k*-alphabet symbol for each
    single model-step transition (droid_100-convention raw actions are
    already FRAMESKIP-length per transition, i.e. exactly one k*=2 step),
    split fit/held-out. No oracle rollout, no Angluin loop."""
    rng = np.random.default_rng(seed + 17)  # offset from discover.py's own sample, not required disjoint
    sample_ids = sorted(rng.choice(
        train_episode_ids, size=min(n_episodes, len(train_episode_ids)), replace=False,
    ).tolist())
    pool = collect_trajectories(sample_ids, n_positions=n_positions, seed=seed + 17)
    zt, znext, aemb, ep_ids = flatten_transitions(pool)

    symbols = []
    for e in pool.episode_ids:
        raw = pool.raw_actions[e]  # (n_transitions, FRAMESKIP, 7), droid_100 convention
        for chunk in raw:
            f = segment_feature(chunk)
            idx = int(np.argmin(np.linalg.norm(alphabet_feats - f, axis=1)))
            symbols.append(alphabet_names[idx])
    symbols = np.array(symbols)
    assert len(symbols) == len(zt), f"{len(symbols)} symbols vs {len(zt)} transitions -- ordering mismatch"

    n = len(zt)
    perm = rng.permutation(n)
    split = int(n * 0.6)
    return {
        "zt": zt, "znext": znext, "symbols": symbols,
        "fit_idx": perm[:split], "ho_idx": perm[split:],
        "n_transitions": n, "n_episodes": len(pool.episode_ids),
    }


def _label_fn(subset: list[str], all_fns: dict[str, callable]):
    fns = [all_fns[n] for n in subset]
    return lambda z: tuple(f(z) for f in fns)


def proxy_fidelity(subset: list[str], all_fns: dict[str, callable], proxy_data: dict) -> float:
    """FIX 1's cheap add-step ranking signal. Majority-vote
    (state,symbol)->next_label table fit on `fit_idx`, scored on `ho_idx`
    (both held out from Phase C's predicate-discovery sample and from
    Phase F's canonical test-20% split). state/next_label are the
    predicate-bit tuple under `subset`, evaluated at t and t+1
    respectively. This is a majority-vote *proxy* automaton, not a real
    extracted one -- it exists purely to rank candidates before spending a
    real extraction on the top few."""
    label = _label_fn(subset, all_fns)
    zt, znext, symbols = proxy_data["zt"], proxy_data["znext"], proxy_data["symbols"]
    fit_idx, ho_idx = proxy_data["fit_idx"], proxy_data["ho_idx"]

    table: dict[tuple, Counter] = {}
    global_counter: Counter = Counter()
    for i in fit_idx:
        state = label(zt[i])
        nxt = label(znext[i])
        table.setdefault((state, symbols[i]), Counter())[nxt] += 1
        global_counter[nxt] += 1
    majority = {k: c.most_common(1)[0][0] for k, c in table.items()}
    global_majority = global_counter.most_common(1)[0][0] if global_counter else None

    if len(ho_idx) == 0:
        return 0.0
    correct = 0
    for i in ho_idx:
        state = label(zt[i])
        nxt = label(znext[i])
        pred = majority.get((state, symbols[i]), global_majority)
        correct += int(pred == nxt)
    return correct / len(ho_idx)


def block_nondeterminism_proxy(subset: list[str], all_fns: dict[str, callable], proxy_data: dict) -> float:
    """The OLD add-step proxy, retained as a reported diagnostic only
    (preregistration.md's FIX 1) -- no longer used to rank candidates."""
    label = _label_fn(subset, all_fns)
    zt, znext, symbols = proxy_data["zt"], proxy_data["znext"], proxy_data["symbols"]

    table: dict[tuple, set] = {}
    for i in proxy_data["fit_idx"]:
        state = label(zt[i])
        table.setdefault((state, symbols[i]), set()).add(label(znext[i]))
    if not table:
        return 1.0
    nondet = sum(1 for v in table.values() if len(v) > 1)
    return nondet / len(table)


def rank_by_proxy(
    remaining: list[str], current_subset: list[str], all_fns: dict[str, callable],
    proxy_data: dict, topk: int = PROXY_TOPK,
) -> list[tuple[str, float]]:
    """Ranks `remaining` candidates by proxy_fidelity(current_subset + [p]),
    descending. Returns the top-`topk` (candidate, score) pairs -- these,
    and only these, get a real extraction (plan §8.2 point 2)."""
    scored = [(p, proxy_fidelity(current_subset + [p], all_fns, proxy_data)) for p in remaining]
    scored.sort(key=lambda x: -x[1])
    return scored[:topk]
