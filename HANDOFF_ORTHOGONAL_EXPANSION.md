# HANDOFF — Orthogonal-Expansion Self-Improvement Ladder

**Give this file to a new Claude Code session to resume this line of work.** It supersedes
the original `HANDOFF.md` for anything past Phase F (that file is still accurate for
environment setup, venvs, and DROID/streaming gotchas — read it too, but for outcome, read
this one). Everything below is current as of **2026-09-02**, session stopped by explicit user
request while E0c's timing probe was still running unconverged.

---

## 1. One paragraph

Starting from the Phase-F-rejected 6-state automaton (see `HANDOFF.md`), this session ran a
Phase G/H/I exploratory arm-comparison (negative — no arm beat controls, and it surfaced that
the "accepted" automaton is a degenerate constant classifier of the initial frame). The user
then supplied a formal, strictly-gated **experiment ladder** (`SELF_IMPROVEMENT_LOOP.md` /
`EXPERIMENTS_orthogonal_expansion.md`) to systematically test whether a latent world model can
be made to "self-improve" by retrieving real trajectories along directions its own symbolic
abstraction is blind to. **Both documents were revised twice more, mid-session, each revision
directly incorporating this session's own preceding results as new "established facts."** The
ladder ran E0 (FAIL — discovered predicates worse than trivial persistence), E0b (FAIL again,
for tessellation too, plus a real bug this session introduced and then caught), and stalled at
the very start of E0c (repairing the extractor before choosing a starting level) because the
prescribed repaired settings turned out to be enormously more expensive than expected — a
single extraction ran **>23 minutes without converging**, versus <1 second at the old settings.
The user stopped the run before it completed even one repaired extraction. **No level has been
selected. E1/E2/E3/E4 (geometry, capability instrument, single-round test, the loop itself)
have not run and must not run until E0c actually produces a k0.**

---

## 2. Documents: read the CURRENT version, and expect it to change again

`C:\Users\Admin\Downloads\SELF_IMPROVEMENT_LOOP.md` and
`C:\Users\Admin\Downloads\EXPERIMENTS_orthogonal_expansion.md` have been supplied to this
session **three times**, each version materially different from the last:

1. **v1** — hard gates (E0 through E5), any failure means STOP.
2. **v2** — "E0's findings are incorporated," gates become routing checkpoints, claims
   tessellation "passed its baselines" (an established fact NOT to re-derive) and sets `m=2`
   as implicitly fine via the Moore counting bound.
3. **v2, revised again** (the version in `Downloads/` as of this handoff) — adds a
   **monotonicity law** (§0.1 of the experiments doc) built directly from THIS session's own
   E0b.2 numbers, which retroactively disqualifies `m=2` and requires a new step, **E0c**,
   before E1 may run at all.

**If you are resuming this work, re-read whatever copy of these two files currently exists in
`Downloads/` in full, even if you remember an earlier version — do not assume the version
described in this handoff is still current.** The pattern established this session is that the
documents are being actively rewritten in response to what gets reported back, not executed
once as a fixed plan.

---

## 3. What actually ran, phase by phase, with exact artifact locations

| phase | verdict | key artifact | code |
|---|---|---|---|
| E0 (degeneracy gate) | **FAIL.** Rollout variance does NOT collapse (grows 0.94→6.71, ℓ=4→24) — the "attractor collapse" hypothesis is rejected. But all 3 discovered predicates, and their joint 0.9156 that passed Phase F, are beaten by trivial persistence (~0.995) by 0.075–0.088. | `artifacts/e0_report.json`, `artifacts/e0_results.md`, `artifacts/e0_rollout_collapse.png` | `e0_degeneracy/` |
| E0b (two disambiguations) | **Both routed to caution, not "proceed as written."** E0b.1: nearest-real-latent distance grows 3.8× (1.83→6.98) over 24 letters — real, moderate off-manifold drift, bounded within the overall data spread scale (~13.2). E0b.2: **all 20 tessellation bits across m=2/4/6/8, properly re-measured with bootstrap CIs, are CLEARLY WORSE than persistence** (mean margin −0.34, every CI excludes zero on the negative side). | `artifacts/e0b_report.json`, `artifacts/e0b1_manifold_check.png` | `e0_degeneracy/e0b.py`, `e0_degeneracy/run_e0b.py` |
| **Bug found and fixed during E0b** | `e0_degeneracy/baselines.py`'s `tessellation_baselines()` named bits `"h0","h1",...`; the real per-predicate keys (from `tessellation/guard.py`'s `CELL_PREDICATE_PREFIX`) are `"cell_h0","cell_h1",...`. The join in `compare_to_measured_fidelity` matched NOTHING, `tess_comparison[m]` was empty for every m, and `all(... for ... in {})` **vacuously returned True** — `e0_report.json`'s `"tessellation_bits_all_beat_baseline": true` was never actually computed. **This field in `e0_report.json` is WRONG and superseded by `e0b_report.json`'s real numbers.** Both the naming bug and the vacuous-`all()` trap are fixed in the current code (`e0_degeneracy/baselines.py`, `e0_degeneracy/run_e0.py` now raises instead of vacuously passing on an empty comparison set) — but `e0_report.json` itself was never regenerated with the fix, so treat that one field in that one file as stale/incorrect if you ever open it directly. | see `preregistration.md`'s 2026-09-02 CORRECTION entry for the full narrative | — |
| E0c (repair the extractor, choose k0) | **STOPPED, incomplete.** Config per the current documents: `epsilon=0.02` (was 0.08), `delta=0.05` (was 0.10), `max_learning_rounds=40` (was 10), plus a hypothesis-size-scaled equivalence-sample floor (`ScaledPacOracle`, since AALpy's stock `PacOracle` already implements the round-scaled formula but not this floor). **A single m=2 extraction under this config ran >23 minutes without converging** (vs <1s at the old settings) before being killed by explicit user instruction. No level has produced a result. `artifacts/e0c_report.json` and `artifacts/e0c_machines/` **do not exist**. | none produced | `e0c_repair/` (all written and syntax-checked, never successfully run to completion) |
| E1 (geometry sizing) | **Not run.** Blocked on E0c producing a k0. An earlier draft of `e1_geometry/run_e1.py` hardcoded `m=2` under the SUPERSEDED (pre-monotonicity-law) reasoning; it has been left in place but its choice of level is now known wrong and must be re-parameterised on whatever k0 E0c eventually selects. | — | `e1_geometry/` (intrinsic dimension + local rank are level-independent and usable as-is; `run_e1.py`'s machine selection needs fixing) |
| E2, E3, E4, E5, the loop itself | **Not started.** Both documents are explicit these may not run before E0–E2 pass (loop doc) / before E0c selects k0 (experiments doc). | — | — |

---

## 4a. UPDATE (2026-09-02, later same day): the diagnosis in SS4 below was wrong about WHERE the cost is

A resuming session ran `e0c_repair/profile_timing.py` (SS4 option 1, "profile before
re-attempting") instead of blindly resuming the sweep. Full writeup:
`artifacts/preregistration.md`'s "E0c timing probe" entry (2026-09-02, appended after the "GATE
E0 result" entry — read it in full before doing anything else here). One-paragraph summary:

**The equivalence oracle itself is cheap.** 6 rounds of real m=2 extraction, `find_cex` timed
directly: `num_test_cases` grew 400 → 6320 words exactly as the repaired formula/floor predicts,
and every one of those 6 calls finished in under 0.75s (2.5s total). **The actual cost is
AALpy's own `ObservationTree` bookkeeping between rounds** (`build_hypothesis` /
`process_counter_example` in `aalpy/learning_algs/deterministic/LSharp.py`, confirmed by reading
that source, not inferred) — entirely outside the oracle, and not affected by `epsilon`,
`delta`, or `ScaledPacOracle`'s floor at all. The hypothesis exploded from 11 to 74 states
between round 4 and round 5 (at **m=2**, the cheapest level in the whole sweep), and round 7's
bookkeeping alone ran past 9 minutes before being killed manually.

**Consequence: SS4's options 2 and 3 below both target the wrong mechanism.** Loosening
epsilon/delta or running the full sweep unattended does not address an AALpy-internal
bookkeeping cost that scales with hypothesis size, not with oracle sampling parameters. SS4 is
left below UNEDITED for the record (it was a reasonable hypothesis before anyone had profiled
it), but treat its "options" list as superseded by the three candidates at the end of the
preregistration.md entry: (i) cProfile `ObservationTree.process_counter_example`/
`build_hypothesis` directly to find the superlinear operation, (ii) check whether the 79-state
explosion at m=2 is real memory or hyperplane-boundary label noise (histogram
`|z @ u_i - b_i|` near zero for the states' representative words — cheap, not yet run), or
(iii) treat this as the ladder's own "monotonicity STILL VIOLATED" row and swap learners
(abstraction-refinement oracle, original plan §7.2(b), never implemented) rather than continue
tuning `run_Lsharp`.

**One of the three candidates above was also run** (it was cheap and didn't need a completed
extraction): static boundary-proximity density for the real corpus against the m=2 hyperplanes
matches a plain Gaussian almost exactly (no excess crowding at the threshold) — see
preregistration.md's addendum. This weakens but does not eliminate hypothesis (b)
(boundary-noise-driven spurious states); genuine memory (a) remains equally plausible. The
question that would actually distinguish them — trajectory-level label *stability* under small
real perturbations, not static score density — was not run.

**Follow-up, later same session: the user chose the trajectory-stability check.** Result (full
detail: preregistration.md's "E0c trajectory-stability check" entry) is decisive and points ONE
way: near-duplicate real pairs (mean d0=0.53) barely diverge over 16 real steps of IDENTICAL
actions (mean pairwise distance 1.13 -> 1.66) and their m=2 label mismatch rate actually DROPS
after the first step and stays at 0.2%-5%, nowhere near the ~0.6-0.75 ceiling that unrelated
("control", mean d0=19) pairs sit at from the start. **Hypothesis (b) (dynamical/boundary-noise
explosion) is disconfirmed; hypothesis (a) (genuine, stable memory the old under-converged
oracle never found) stands.** The round-5 11->74 state jump in the timing probe is very likely
real signal, and the remaining blocker is PURELY the AALpy bookkeeping cost identified above —
not a validity problem with the repair. Next step, not yet started: cProfile
`ObservationTree.process_counter_example`/`build_hypothesis` (candidate (i)) to find the
specific superlinear operation, now worth doing with much higher confidence a converged ~79+
state m=2 machine is a real result worth extracting efficiently, not noise to engineer around.

**SUPERSEDED (2026-09-02, later same session): new document revision (v3 loop / v4
experiments) reinterpreted the above and adopted a new Ladder A0 (label diagnostics/repair).**
Full detail: preregistration.md's "v3/v4 documents supplied" and "Ladder A0 Step 1 result"
entries — read both before doing anything with AALpy or the tessellation label function. One-
paragraph summary: the new documents argue the trajectory-stability finding doesn't establish
"genuine memory" (a locally-continuous deterministic function looks stable under tiny
perturbations regardless of whether it encodes memory) and that 74 states over 4 cells is
itself the signature of a bad partition — explicitly directing *against* the AALpy profiling
work above ("stop profiling the learner"). The 40-round SepSeq validation run in progress was
stopped without finishing. **However, checking the new documents' specific mechanism claim
against this project's own data found it does NOT hold**: tessellation's switch rate is
confirmed (independently, both splits) to sit BELOW the acceptance band's floor at all 20
bits — the same failure direction as the discovered predicates, not the opposite one the new
doc assumes — and block determinism on REAL grounded transitions is high (90–97%, clearing the
doc's own 0.70 bar), contradicting the "far-from-Markov partition" diagnosis outright. A third
explanation (not in either document) is now more likely: the PAC oracle extracts an automaton
for the model's own OPEN-LOOP rollout, which E0b.1 already showed drifts off-manifold — a real,
highly-deterministic closed-loop system can still yield an unstable-looking open-loop automaton
if rollout error compounds over the oracle's own word lengths (4–20 letters). **No repair lever
has been applied yet** — this is a three-way fork (fix the label function's switch rate / fix
the open-loop extraction horizon / dig further) handed back rather than picked unilaterally.

**No route was taken beyond that. This was handed back for a decision rather than acted on unilaterally**,
consistent with SS4's own closing instruction not to default into more unattended compute after
a user-stopped run — the choice between (i)/(ii)/(iii) changes what gets built next
(instrumentation vs. a label-function fix vs. a learner swap) and is exactly the kind of
consequential, hours-of-compute-either-way fork the loop's routing philosophy asks a human to
make deliberately.

---

## 4. The central open problem: the repaired oracle is too expensive as specified (SUPERSEDED — see 4a above)

This is the one concrete, unresolved technical question blocking everything downstream.

**What was tried:** `learning/extract.py`'s `extract()` now accepts an `eq_oracle_factory`
parameter. `e0c_repair/scaled_pac_oracle.py`'s `ScaledPacOracle` subclasses AALpy's
`PacOracle`, keeping its native round-scaled sample formula
(`1/epsilon * (log(1/delta) + round*log(2))` — verified by reading AALpy's own source, it was
already doing this) and adding the missing hypothesis-size floor
(`min_words_per_round_coef * |Q_hyp| * |A|`, coefficient 4.0 per the config).

**What happened:** with `epsilon=0.02, delta=0.05, max_learning_rounds=40`, a single m=2
(2-bit tessellation, smallest/cheapest case) extraction ran for over 23 minutes of sustained,
non-stalled CPU/GPU activity without producing a hypothesis. The equivalent extraction at the
OLD settings (`epsilon=0.08, delta=0.1, max_learning_rounds=10`, no size floor) took **0.2–0.6
seconds** throughout this whole project's history. This is not a hang (verified via
`Get-Process` CPU-time deltas — it was actively computing, tracking wall-clock almost 1:1) — it
is a genuine, enormous increase in the number of real model queries the equivalence oracle
issues per round, compounded by `max_learning_rounds` going from 10 to 40 and by larger `m`
needing proportionally more per the size floor.

**Back-of-envelope why:** `1/epsilon` alone goes from 12.5 to 50 (4×). The size floor
(`4*|Q|*|A|`, with `|A|` = 20ish symbols including resets) adds hundreds to thousands of words
per round once `|Q|` exceeds a handful of states, and each word is walked through the REAL
frozen model (2–4 real forward passes per letter). At round 20–40 the round-scaled term alone
(`round*log(2)/epsilon`) is already in the thousands of test words. This was foreseeable from
the formula but not verified against wall-clock before committing to a 12-extraction sweep —
that verification (this session's one timing probe) is the useful result to carry forward.

**Options for a resuming session, not yet decided between:**
1. **Profile before re-attempting.** Instrument `ScaledPacOracle.find_cex` to print
   `num_test_cases` per round and time per round; confirm whether cost is dominated by the
   round-scaling term, the size floor, or word length (`min_walk_len=4, max_walk_len=20` — was
   never revisited even though `E0b.1` found meaningful drift by word length ~8–12; shorter
   walks would cut cost roughly linearly and might also keep queries in the near-manifold
   region E0b.1 flagged as more trustworthy).
2. **Negotiate a less aggressive repair** (e.g. `epsilon=0.04, delta=0.08, max_learning_rounds=
   20`, or drop the size floor coefficient from 4 to 1–2) and report the resulting numbers
   honestly as a smaller repair than specified, rather than silently using the full spec and
   waiting indefinitely.
3. **Run the full prescribed sweep unattended over a long window** (likely many hours to
   plausibly over a day for all 12 extractions, back-loaded toward larger m) if wall-clock is
   not a binding constraint for the next session.
4. **Step back from the ladder.** Both documents' own pivot map (`EXPERIMENTS_orthogonal_
   expansion.md`'s §0, and the ORIGINAL v1's §4) name a smaller, already-well-defined
   alternative if this repair proves impractical: rebuild the fidelity function so a bare
   number is structurally impossible to report (baselines returned in the same call — already
   partially true for `gate/acceptance.py`'s per-predicate output, not yet true anywhere in
   `e0c_repair/` or `tessellation/`), and stop there rather than pursuing orthogonal expansion
   on top of a label function that has now failed baselines TWICE (predicates, then
   tessellation).

**Do not just re-run the exact same 12-extraction sweep unattended without picking one of
these** — option 3 alone, done blindly, could consume a very large amount of wall-clock time
for a result that might turn out the same way (monotonicity still violated, or worse) as the
un-repaired sweep, and the next session should make that call deliberately rather than by
default.

---

## 5. Code map (new this session, all syntax-checked, not all run to completion)

```
corpus/latent_corpus.py        shared latent+symbol corpus builder/cache (Corpus, build_corpus,
                                 train_ids, test_ids) -- everything below depends on this
tessellation/
  hyperplanes.py                Tessellation (uniform-sphere + median-offset cells), Voronoi
  guard.py                      CELL_PREDICATE_PREFIX="cell_h", assert_not_tessellation()
  run_phase_t.py                original tessellation sweep (m=2,4,6,8), wrote tessellation.json
                                 + tessellation_params.npz -- ALREADY RUN, reusable as-is
  run_phase_t2.py                original diagnostic-FSM extraction at old settings -- ALREADY
                                 RUN, wrote tessellation_fsm.json; superseded for anything
                                 needing the repaired oracle
spectral/, atlas/, retrieval/,  from an EARLIER (pre-ladder) exploratory pass, still on disk,
curriculum/                     not used by the ladder -- see memory for that narrative if ever
                                 needed again
e0_degeneracy/
  rollout_collapse.py            E0.1/E0.2: 500-word nested-prefix rollout from z0=r0
  baselines.py                   E0.3: persistence/majority baselines -- NAMING BUG FIXED here
  run_e0.py                      E0 driver + gate -- vacuous-all() TRAP FIXED here (raises now)
  e0b.py                         E0b.1 (manifold check) + E0b.2 (tessellation margins w/ CI,
                                 re-extracts each m fresh since aggregates aren't enough)
  run_e0b.py                     E0b driver + routing
e0c_repair/
  scaled_pac_oracle.py            ScaledPacOracle -- the repaired equivalence oracle
  suffix_depth.py                 suffix_depth_distribution(), memory_multiplier()
  reachability.py                 reached_cells_from_real_oracle() -- ALL 8 resets, real model,
                                 NEVER from the extracted machine's own graph (would be
                                 tautological -- see the module docstring)
  run_e0c.py                      the 3-seed x 4-level sweep driver -- NEVER COMPLETED A SINGLE
                                 EXTRACTION; persists machines to e0c_machines/*.pkl as they
                                 converge (none exist yet)
e1_geometry/
  intrinsic_dimension.py          TwoNN + Levina-Bickel MLE -- level-independent, usable now
  local_rank.py                   local effective rank at N base points -- level-independent
  expansion_space.py              compute_V_T_W() -- linear tessellation soft-scores +
                                 signals/relevance_subspace.py's differentiable rollout;
                                 moore_bound_check() -- NOTE: this checks vs OCCUPIED cells,
                                 which the NEWEST document says is the wrong comparison (should
                                 be vs REACHED cells, e0c_repair/reachability.py). Not yet fixed
                                 in expansion_space.py itself -- fix before reusing.
  run_e1.py                       hardcodes m=2 (SUPERSEDED reasoning) -- needs re-parameterising
                                 on E0c's k0 once one exists
learning/extract.py              gained eq_oracle_factory param (backward compatible, defaults
                                 to stock PacOracle if not given)
checkpoints.py                   load_epoch(1..20), load_random_init() -- for E2 later, unused
                                 so far, verified working
```

Also still relevant from the pre-ladder session: `gate/acceptance.py`, `gate/check_reproducibility.py`,
`predicates/functions.py`, `oracle/lewm_g.py` (gained `rollout_batch_from_convention`), all
described in the original `HANDOFF.md` and prior memory entries.

---

## 6. Pre-registration trail

`artifacts/preregistration.md` has, in order, appended and never rewritten: the original Phase
F fixes, the exploratory G/H/I pre-registration, the "new experiment ladder adopted" entry, the
GATE E0 result, the E0.3-bug CORRECTION entry (important — read this one if anything about
tessellation baselines looks inconsistent across files), the E0b result entry, and the E0c
adoption entry (config + operationalisation choices, written BEFORE the timing probe). **There
is no E0c RESULT entry** — write one when a resuming session actually gets a k0, following the
same append-only, dated convention.

---

## 7. Environment reminders carried from `HANDOFF.md` (still true)

- Two venvs: `le-wm/.venv` (torch, lerobot, av, aalpy — run anything touching the real model or
  real data from here) and `symbolic_lewm/.venv` (aalpy, numpy, no torch — sufficient for
  syntax checks and the old synthetic test suite only).
- `le-wm/eval.py` (a top-level file) shadows this repo's `eval/` PACKAGE once
  `oracle.lewm_g` puts le-wm on `sys.path[0]`. Anything needing `eval.stats` or
  `eval.plan_ranking` from a script that also touches `oracle.lewm_g` must load by explicit
  file path via `local_import.py`'s `load_local()`, not `import eval.something`.
- Every epoch (1–20) of `lewm_droid_s0`'s training was checkpointed — `checkpoints.py` already
  wraps this, no retraining needed for E2 whenever it runs.
- Background python processes on this Windows machine have, on at least two prior occasions
  this project, hung with near-zero CPU well AFTER finishing real work and writing all outputs
  to disk (cause never isolated). Most `run_*.py` entry points in this project defensively call
  `os._exit(0)` right after their last write for exactly this reason — keep doing that in any
  new driver script.
- DROID video streaming (`oracle/droid_streaming.py`) has a wall-clock timeout guard
  (`stream_many_windows`'s `per_episode_timeout_s`) added after a stalled HTTP connection once
  blocked an entire batch indefinitely — keep it in any new streaming call.
