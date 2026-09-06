# Pre-experiment tests — implementation notes

Scope: SYMBOLIC_ABSTRACTION_PLAN.md §15 ("Testing requirements"), implemented
and run *before* any repo audit (M0) or real LeWM/Franka code, exactly as the
plan's own logic allows — the highest-value test in the suite (known-automaton
round trip) is explicitly defined against a synthetic system, not the real
model, so none of §15 requires the real checkpoint or dataset.

## What's implemented

| §15 requirement | File(s) | Status |
|---|---|---|
| Oracle purity | `tests/test_oracle_purity.py` | done — property tests (hypothesis) |
| Prefix consistency | `tests/test_prefix_consistency.py` | done — property tests |
| Known-automaton round trip | `tests/test_known_automaton_roundtrip.py`, `tests/fixtures/synthetic_fsm.py` | done |
| Alphabet determinism | `tests/test_alphabet_determinism.py`, `alphabet/codebook.py` | done — checked in-process and across a subprocess boundary |
| Arm isolation | `tests/test_arm_isolation.py`, `training/masked_trainer.py`, `training/arms/{control,sample_prio,latent_subspace}.py` | done — 3 of 5 arms implemented (enough to exercise the isolation mechanism); `shuffled` and `input_patch` are not separate loss code paths per the plan (`shuffled` permutes the *signal table* upstream of an existing arm; `input_patch` needs real patch-attribution data) |

Supporting modules built to make the above possible:
- `oracle/latent_oracle.py`, `oracle/cache.py` — Phase A.1–A.3, model-agnostic.
- `learning/sul.py` — Phase D.1 AALpy adapter.

## AALpy API — verified against the installed version, not assumed

Installed: `aalpy==1.6.2`. Confirmed by inspection (not from memory) before
writing any code against it:

- **No `run_TTT` exists in this release.** The plan calls for TTT
  specifically ("discrimination trees instead of observation tables, far
  better counterexample handling"). The two candidates that share that
  property here are `run_KV` (Kearns–Vazirani, classification tree) and
  `run_Lsharp` (Vaandrager, Garhewal, Rot & Wissmann — apartness + an
  observation tree, the modern successor to TTT, from the same research
  lineage the plan cites in §16). **I substituted `run_Lsharp`** throughout
  — it is newer than TTT, not an observation-table algorithm, and supports
  Moore machines directly. This is a real deviation from the letter of the
  spec and should be confirmed against whatever the Phase D implementation
  ultimately needs (e.g. if downstream tooling specifically wants a TTT
  discrimination tree object to inspect, `run_KV`'s classification tree is
  the closer structural match and is available as a fallback).
- `SUL.step(letter=None)` is a real sentinel in the query protocol (used for
  the empty-word membership query), not an edge case I invented — confirmed
  by reading `aalpy.base.SUL.query()`.
- `aalpy.utils.bisimilar(a1, a2)` exists and is exactly the isomorphism-up-to-
  bisimulation check the round-trip test needs; used directly rather than
  hand-rolling a comparison.
- `WMethodEqOracle(alphabet, sul, max_number_of_states)` (Chow's W-method) is
  used for the round-trip test instead of the plan's hybrid PAC +
  abstraction-refinement oracle (Phase D.2) — W-method is exhaustive up to
  the given state bound, so the test is deterministic (no flake risk from a
  probabilistic oracle missing a counterexample by chance). `PacOracle` does
  exist and matches the plan's D.2(a) formula exactly if a probabilistic
  oracle is wanted later for the real system, where exhaustive testing isn't
  affordable.

## Deliberately out of scope here (needs the real repo/data — Phase A onward)

- Repo audit (M0) — window size `N`, action conditioning, latent dim,
  Fast-LeWM prefix availability, real checkpoint, real dataset loader.
- `L_max` measurement (A.4), real alphabet clustering on Franka action
  segments (Phase B.1/B.2 — the gripper special-case), predicate discovery
  from real latents (Phase C), the search loop (Phase E), acceptance gate
  (Phase F), signal extraction (Phase G), and the 5×5 training/eval runs
  (Phase H/I).

None of the above blocks §15; this is exactly the ordering the plan itself
recommends ("write it before Phase D").
