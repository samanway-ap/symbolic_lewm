"""Phase F.2 trace fidelity + reachable-state coverage + block
nondeterminism, computed against REAL held-out rollouts (never model
rollouts -- that's the whole point, per the plan's explicit warning).

For a held-out episode not among the 8 canonical reset episodes: no
literal reset symbol matches its actual first window, so the CLOSEST
(by encoded first-window L2 distance) reset symbol is used as the
machine's starting point -- a reasonable proxy, not exact replay. Labels
from position 1 onward are what's scored (position 0 is definitionally
approximate under this proxy).

FIX 3a/3b (2026-08-21, preregistration.md's Phase F acceptance-gate
amendment, committed before this file was touched):
  (a) `trace_fidelity` is now measured only out to the candidate machine's
      own max distinguishing-suffix length (`max_distinguishing_suffix_length`
      below), not the full ~L_max rollout -- gating at L_max was circular
      (L_max was defined by a provisional predicate set's own selection
      floor, not by anything the automaton is used for downstream).
  (b) `trace_fidelity` (the gated metric) is now the mean, over active
      predicates, of each predicate's OWN per-step agreement rate --
      joint tuple-equality compounds as p^|S|, making a fixed 0.85 floor
      implicitly |S|-dependent. The old joint metric is preserved as
      `joint_trace_fidelity` and reported (never gated) alongside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphabet.segments import segment_feature
from oracle.droid_actions import load_actions_for_episodes
from oracle.droid_streaming import stream_many_windows
from oracle.lewm_g import ActionPipeline, FRAMESKIP, HISTORY_SIZE, RAW_ACTION_DIM, encode_pixel_windows_batch

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
K_STAR = 2


def quantize_to_symbols(raw_actions_converted: np.ndarray, k: int, alphabet_feats: np.ndarray, alphabet_names: list[str]) -> list[str]:
    n_chunks = raw_actions_converted.shape[0] // k
    out = []
    for c in range(n_chunks):
        chunk = raw_actions_converted[c * k:(c + 1) * k]
        f = segment_feature(chunk)
        idx = int(np.argmin(np.linalg.norm(alphabet_feats - f, axis=1)))
        out.append(alphabet_names[idx])
    return out


def load_alphabet_feats(alphabet_path: Path):
    alphabet = json.loads(alphabet_path.read_text())
    feats, names = [], []
    for s in alphabet["symbols"]:
        seg = np.array(s["medoid_segment"])
        feats.append(segment_feature(seg))
        names.append(f"a{s['symbol']}")
    return np.stack(feats), names


def max_distinguishing_suffix_length(machine, alphabet: list[str]) -> int:
    """FIX 3a: max, over all reachable-state pairs, of the shortest
    distinguishing-suffix length -- the horizon at which the machine's
    states are still meaningfully distinct from each other. Uses AALpy's
    own `MooreMachine.find_distinguishing_seq` (a validated BFS shipped
    with the library, guaranteed shortest since it explores prefix lengths
    in order), rather than a hand-rolled partition-refinement -- one less
    place for this rerun to introduce a new bug."""
    states = list(machine.states)
    n = len(states)
    if n <= 1:
        return 1
    lengths = []
    for i in range(n):
        for j in range(i + 1, n):
            seq = machine.find_distinguishing_seq(states[i], states[j], alphabet)
            if seq is not None:
                lengths.append(len(seq))
    return max(lengths) if lengths else 1


def evaluate(
    machine, label_fn, all_predicate_fns, predicate_subset: list[str],
    episode_ids: list[int], reset_windows: dict, alphabet_path: Path,
    n_positions: int = 20, seed: int = 3072,
):
    """Returns dict with trace_fidelity, reachable_state_coverage,
    block_nondeterminism, computed against REAL held-out episodes."""
    from predicates.functions import make_label_fn
    fns = [all_predicate_fns[n] for n in predicate_subset]

    alphabet_feats, alphabet_names = load_alphabet_feats(alphabet_path)
    n_raw_needed = n_positions * FRAMESKIP

    actions = load_actions_for_episodes(episode_ids, max_frames=n_raw_needed)
    usable = [e for e in episode_ids if e in actions and actions[e].shape[0] >= n_raw_needed]
    windows = stream_many_windows(usable, num_frames=n_positions, frameskip=FRAMESKIP, max_workers=16)
    usable = sorted(set(usable) & set(windows.keys()))
    if not usable:
        return {"trace_fidelity": 0.0, "joint_trace_fidelity": 0.0, "fidelity_horizon": 0,
                "per_predicate_fidelity": {}, "reachable_state_coverage": 0.0,
                "block_nondeterminism": 1.0, "n_episodes": 0}

    all_raw = np.concatenate([actions[e][:n_raw_needed] for e in usable], axis=0)
    pipeline = ActionPipeline.fit(all_raw)

    pixel_stack = np.stack([windows[e] for e in usable])
    real_trace = encode_pixel_windows_batch(pixel_stack).numpy()  # (N,n_positions,D)

    reset_names = list(reset_windows.keys())
    reset_first_z = np.stack([reset_windows[r].emb[-1].numpy() for r in reset_names])

    # FIX 3a: gate at the machine's own max distinguishing-suffix length,
    # not at the full n_positions (~L_max) rollout -- preregistration.md
    horizon = max_distinguishing_suffix_length(machine, alphabet_names)
    horizon = max(1, min(horizon, n_positions - HISTORY_SIZE))

    n_pred = len(fns)
    per_step_agree = []  # joint (all predicates match) -- diagnostic only, never gated
    per_predicate_matches = [0] * n_pred
    per_predicate_total = [0] * n_pred
    visited_states = set()
    state_symbol_to_labels = {}

    for i, eid in enumerate(usable):
        first_z = real_trace[i, HISTORY_SIZE - 1]
        d = np.linalg.norm(reset_first_z - first_z[None, :], axis=1)
        reset_sym = reset_names[int(d.argmin())]

        conv = pipeline.to_convention(actions[eid][:n_raw_needed].astype(np.float64))
        # quantize the FUTURE (post-window) actions
        future_conv = conv[HISTORY_SIZE * FRAMESKIP:]
        symbols = quantize_to_symbols(future_conv, K_STAR, alphabet_feats, alphabet_names)

        word = (reset_sym,) + tuple(symbols)
        machine.reset_to_initial()
        machine_labels = []
        cur_state_id = machine.current_state.state_id
        visited_states.add(cur_state_id)
        for sym in word:
            out = machine.step(sym)
            machine_labels.append(out)
            cur_state_id = machine.current_state.state_id
            visited_states.add(cur_state_id)
            key = (cur_state_id, sym)
            state_symbol_to_labels.setdefault(key, set()).add(out)

        real_labels = []
        for pos in range(HISTORY_SIZE, min(n_positions, HISTORY_SIZE + len(symbols) + 1)):
            z = real_trace[i, pos]
            real_labels.append(tuple(f(z) for f in fns))

        # machine_labels[0] is after the reset symbol only; cap the
        # comparison at `horizon` steps (FIX 3a), not the full trace
        n_cmp = min(len(machine_labels) - 1, len(real_labels), horizon)
        for t in range(n_cmp):
            m_lab, r_lab = machine_labels[t + 1], real_labels[t]
            per_step_agree.append(m_lab == r_lab)
            for k in range(n_pred):
                per_predicate_total[k] += 1
                if m_lab[k] == r_lab[k]:
                    per_predicate_matches[k] += 1

    joint_trace_fidelity = float(np.mean(per_step_agree)) if per_step_agree else 0.0
    per_predicate_rates = [
        (m / t if t else 0.0) for m, t in zip(per_predicate_matches, per_predicate_total)
    ]
    # FIX 3b: the GATED metric -- mean per-predicate agreement, not joint
    trace_fidelity = float(np.mean(per_predicate_rates)) if per_predicate_rates else 0.0

    total_states = machine.size
    reachable_coverage = len(visited_states) / total_states if total_states else 0.0
    nondeterministic_keys = sum(1 for v in state_symbol_to_labels.values() if len(v) > 1)
    block_nondeterminism = nondeterministic_keys / len(state_symbol_to_labels) if state_symbol_to_labels else 0.0

    return {
        "trace_fidelity": trace_fidelity,
        "joint_trace_fidelity": joint_trace_fidelity,
        "fidelity_horizon": horizon,
        "per_predicate_fidelity": dict(zip(predicate_subset, per_predicate_rates)),
        "reachable_state_coverage": reachable_coverage,
        "block_nondeterminism": block_nondeterminism,
        "n_episodes": len(usable),
        "n_states_visited": len(visited_states),
        "n_states_total": total_states,
    }
