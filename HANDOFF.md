# HANDOFF — Symbolic Abstraction of a Trained LeWM

**Give this file to a new Claude Code session to resume or extend this
project.** It is the single entry point; everything else in this repo is
detail this file points to. Read this file first, then only open the
specific artifact/report you need.

---

## 1. What this project is, in one paragraph

Extract a compact finite-state (Moore machine) symbolic abstraction from a
trained LeWM latent world model (`theta_0`), via active automata learning
(Angluin-style, AALpy), on real DROID robot manipulation data. Use the
automaton to derive masking/curriculum signals for a short fine-tune, and
measure whether that improves planning competence versus controls. Full
spec: `SYMBOLIC_ABSTRACTION_PLAN_v1.md` (original) and
`SYMBOLIC_ABSTRACTION_PLAN_v2.md` (revised after the M0 repo audit; **v2
is authoritative** — read it, not v1, if the two conflict).

## 2. Current status: STOPPED at Kill Criterion F. Negative result delivered.

The pipeline ran end-to-end through Phase F (M0 → M0.5 → M1 → M2 → M3 →
M4 → M5 → M6) and **Phase F's acceptance gate rejected every candidate
automaton found**. Per the plan's own design, this is a valid, reportable
negative result, not a failure: *"no compact symbolic model exists in
this latent space at this alphabet resolution and fidelity requirement,
on this task family."* Phases G/H/I (signal extraction, training arms,
evaluation) were **not run** — they require an accepted automaton as
their basis and none exists.

**Read `artifacts/results.md` first** for the full final writeup. This
handoff file is about *how to resume or extend*, not a restatement of the
result.

### The gate that failed (canonical test-20% split, 150 real held-out episodes)

| gate | threshold | measured | pass? |
|---|---|---|---|
| n_states | ≤ 50 | 8 | yes |
| trace_fidelity | ≥ 0.85 | **0.675** | **no** |
| block_nondeterminism | ≤ 0.10 | 0.0 | yes |
| reachable-state coverage | ≥ 90% | **87.5%** | **no** (marginal) |
| 3-seed reproducibility | isomorphic | not checked (moot) | — |

## 3. If you're resuming to try again (most likely reason to read this)

Three concrete, plausible reasons the gate failed, each independently
fixable — **fix one at a time, re-run Phase E, don't change everything at
once**:

1. **Predicate pool was thin and undiverse.** Only 32 candidates survived
   (24 `slow_feature` + 8 `vq_code`), because family (iii)
   (behavioural-metric clustering) produced 751 near-empty clusters and
   0 survived hygiene — a clustering-cap tuning bug in
   `predicates/discover.py`'s `bisim_candidates()` (the complete-linkage
   diameter cap, `diameter_cap_pct=75`, was too low, over-fragmenting the
   clusters). **Fix**: raise the cap, or fix `n_clusters` directly instead
   of a distance-based cut, and re-run `predicates/discover.py`. This is
   the single most likely lever to actually change the outcome, since it
   was never really tried.
2. **Search was a stripped-down greedy add-only sweep** (sizes 3→8, 6
   extractions), not the plan's real add/drop + CEGAR loop (`search/
   greedy.py`/`cegar.py` were never written). A proper bidirectional
   search — especially the CEGAR sub-loop, which proposes new predicates
   based on *why* an extraction failed rather than blind ranking order —
   might find a materially better subset than "the first 3-8 slow-feature
   candidates in discovery order," which is all that was actually tried.
3. **k\*=2 may be a hard floor for this model/family**, not a search
   artifact — see §5's risk register discussion. If (1) and (2) don't
   move the needle, this is the more fundamental explanation, and the
   next lever is a different task family (§4) or accepting the negative
   result as final.

To re-run the search with a fixed predicate pool:
```bash
cd /c/Users/Admin/Projects/le-wm   # NOTE: run from le-wm's venv, see §6
./.venv/Scripts/python.exe /c/Users/Admin/Projects/symbolic_lewm/predicates/discover.py
./.venv/Scripts/python.exe /c/Users/Admin/Projects/symbolic_lewm/search/run_search.py
./.venv/Scripts/python.exe /c/Users/Admin/Projects/symbolic_lewm/gate/run_acceptance_check.py
```

## 4. If you're resuming to try a different task family

M0.5's scope selection found 5 candidate families in full DROID
(`lerobot/droid_1.0.1`) besides `drawer_open_close`, all with 2×–20×
headroom over the 200-episode threshold: `marker_to_container` (3893
episodes — the pre-registered **fallback**, see `artifacts/
scope_episode_ids.json`), `towel` (4113), `tap_open_close` (759),
`microwave_open_close` (421). To switch families: redo `scope/
build_episode_split.py` on the new family's episode ids, then redo
everything from Phase A (`oracle/horizon.py`) forward — **the k-sweep is
family-specific and must be re-measured, not assumed**. Full family
episode-id lists are already computed and saved in
`artifacts/scope_episode_ids.json` (both `primary` and `fallback` keys) —
no need to re-derive them from the 95,658-episode metadata.

## 5. Critical findings that shaped every downstream decision — read before assuming anything

- **`theta_0` (the frozen checkpoint) was trained on `lerobot/droid_100`
  only** (100 episodes) — see §6 for what it is. Everything in this
  project runs `theta_0` on **full DROID** (`lerobot/droid_1.0.1`, a
  different, much larger dataset with a different action-field
  convention). This cross-dataset mismatch is the single biggest source
  of everything that went wrong; see the guardrail below.
- **Action-convention guardrail (`oracle/lewm_g.py`'s `ActionPipeline`)**:
  `droid_100`'s `action` field is clipped to ~[-1,1] (what `theta_0` was
  trained on); full DROID's `action.original` (dimension-matched, 7-dim)
  has raw unbounded values in a different scale. Every raw action in this
  project is passed through `ActionPipeline`: empirically rescaled +
  hard-clipped into `droid_100`'s own observed range (asserted), *then*
  z-scored with `droid_100`'s own fixed statistics — never the other way,
  never skipped. This is a mitigation, not proof the two datasets' action
  fields are the same physical quantity — still an **open risk**
  (`preregistration.md`).
- **k\* = 2 was forced, not chosen**, and reintroduces the exact problem
  the alphabet is supposed to solve: k=2 raw frames is `theta_0`'s own
  native step size (`FRAMESKIP=2`), so 10 of the 12 alphabet symbols
  (`stay_open`/`stay_closed`) carry little trajectory-shape information —
  two frames of continuous motion look alike regardless of direction.
  Only the 2 gripper-toggle symbols have a clear basis. This was flagged
  as a risk *before* the search ran (`preregistration.md`) and is the
  leading hypothesis for why Phase F failed. See `M1_report.md`'s full
  correction history for how k* was determined (twice — the first
  determination, k\*=4, was **later found to be based on a bug-corrupted
  measurement** and was corrected to k\*=2; both the bug and the
  correction are documented in detail there, worth reading if the k-sweep
  is ever re-run).
- **`theta_0`'s own plan-ranking MRR sits near chance** (~0.18–0.24,
  `M2_report.md`) on this data at the tested horizon. A model whose own
  multi-step predictions are only weakly informative should be expected
  to yield weakly-predictable automaton labels — this is likely a root
  cause underneath both the k* problem and the fidelity-gate failure, not
  an independent issue.

## 6. Environment — where everything lives, exactly

Two sibling project directories, two separate venvs, used together:

- **`C:\Users\Admin\Projects\le-wm`** — the LeWM model repo (forked/
  modified from `lucas-maes/le-wm`). Its venv (`le-wm/.venv`) has
  **torch, lerobot, av, scikit-learn, matplotlib, AND aalpy** (aalpy was
  added here specifically so this project's real-data/real-model scripts
  could run from one venv without duplicating the heavy torch+lerobot
  stack). **Almost every script in `symbolic_lewm/` that touches real
  data or the trained model is run from `le-wm`'s venv**, e.g.:
  ```bash
  cd /c/Users/Admin/Projects/le-wm
  ./.venv/Scripts/python.exe /c/Users/Admin/Projects/symbolic_lewm/<script>.py
  ```
  `le-wm/.stable-wm/checkpoints/lewm_droid_s0/` is `theta_0` (20 epochs,
  seed 3072, trained on `droid_100`; see `le-wm`'s own memory file
  `lewm-droid100-replication.md` for how it was trained — that was a
  separate, earlier, already-completed task in this same session).
  `le-wm/.frame_cache/actions.npy` is `droid_100`'s raw actions (32212×7)
  — this is what `ActionPipeline`'s droid_100-convention constants are
  measured from; **do not delete this file**, several modules read it at
  import/first-use time (`oracle/lewm_g.py`'s `_droid100_action_stats()`).

- **`C:\Users\Admin\Projects\symbolic_lewm`** (this repo) — the symbolic
  abstraction pipeline itself. Its own venv (`symbolic_lewm/.venv`) has
  **aalpy, numpy, scikit-learn, pytest, hypothesis** but **NOT torch or
  lerobot** — it's only sufficient for `tests/` (the pre-experiment test
  suite, all passing, uses synthetic fixtures, no real data needed). Real
  pipeline scripts (`oracle/`, `alphabet/`, `predicates/`, `search/`,
  `gate/`) import from here but must be *run* via `le-wm`'s venv (see
  above) — they `sys.path.insert` this repo's root to make that work.

- **DROID data caches** (do not delete): `lerobot/droid_1.0.1`'s episode
  metadata + actions live under
  `~/.cache/huggingface/lerobot/hub/datasets--lerobot--droid_1.0.1/` —
  **note this is a different cache root than plain `huggingface_hub`'s
  default** (`~/.cache/huggingface/hub/`); a duplicate, smaller copy of
  just the metadata also exists there from an earlier exploratory step
  and is harmless but unused. `oracle/droid_streaming.py`'s `_META_GLOB`
  and `oracle/droid_actions.py`'s `_DATA_GLOB` point at the correct
  (`lerobot/hub/`) location — if you ever see a `KeyError` on a video
  column name or an empty metadata glob, check this first.
  `droid_1.0.1`'s `data/` (actions/state, no video) was fully downloaded
  for the `drawer_open_close` family (3184 episodes, ~7.9GB) — video was
  **never bulk-downloaded** (~400GB+ for the full family across 3
  cameras); all pixel access goes through `oracle/droid_streaming.py`'s
  direct HTTP-range seek-and-decode (see §7).

## 7. Non-obvious infrastructure worth knowing about before rebuilding it

- **Streaming video without downloading it**: `oracle/droid_streaming.py`
  opens the remote `.mp4` URL directly with PyAV and seeks via HTTP range
  requests — verified empirically to decode a handful of frames in ~3s
  without pulling the full ~500MB file. This works because episodes are
  scattered across nearly 100% of chunk files regardless of family (a
  measured fact, not an assumption — see the session transcript around
  "M0.5 continued" if the reasoning is needed again), so file-level
  `episodes=[...]` selection in `LeRobotDataset` buys nothing; per-episode
  range-seeking is the only viable approach at this dataset's scale.
- **Camera name**: full DROID's actual camera keys are
  `observation.images.{exterior_1_left, exterior_2_left, wrist_left}` —
  **no "image" in the middle** (unlike `droid_100`'s
  `exterior_image_1_left`). `oracle/droid_streaming.py`'s default is
  correct; a bug where a *different* file (`predicates/latent_pool.py`)
  hardcoded the wrong (droid_100-style) name was caught and fixed during
  this session — if a new file ever hardcodes a camera string instead of
  using the shared default, check this first.
- **Resize is aspect-preserving, not square.** `utils.get_img_preprocessor`
  resizes to `img_size=224` on one dimension only — droid frames (180×320)
  become ~224×398, not 224×224. Assuming square output broke masking
  once already in the earlier LeWM-replication task and broke
  `oracle/lewm_g.py`'s batched pixel encoder once in this project too —
  **always read the actual post-resize shape back, never hardcode it.**
- **AALpy 1.6.2 has no `run_TTT`.** The plan wants TTT specifically; the
  closest available algorithms are `run_Lsharp` (used throughout this
  project — Vaandrager et al., apartness + observation trees, a modern
  successor to TTT) and `run_KV` (Kearns-Vazirani, classification tree,
  not used but available as a fallback). Verified by inspection, not
  assumed — see `PRE_EXPERIMENT_NOTES.md`.
- **`LeRobotDataset`'s own cache root** (`~/.cache/huggingface/lerobot/
  hub/`) differs from bare `huggingface_hub.snapshot_download`'s default
  (`~/.cache/huggingface/hub/`) — see §6. This caused one real,
  time-costly bug (streaming/action-loading code pointed at the wrong
  cache root) before being found and fixed.
- **The action pipeline has two stages, used at different points**:
  `ActionPipeline.to_convention(raw)` (droid_100-range, unnormalized —
  used for codebook/medoids/segments, anything human-facing) vs.
  `ActionPipeline(raw)` / the full call (convert + z-score — used only at
  the `action_encoder` input boundary, i.e. inside oracle rollouts). Mixing
  these up produces silently-wrong-scale inputs to the frozen model, not
  a crash — this exact bug (calling the full pipeline on already-converted
  medoid segments) was caught once via `alphabet/validate_split.py`'s
  `rollout_final_latent`, which deliberately calls `.normalizer(...)` only
  when its input is already-converted.

## 8. Full milestone trail — where the detail actually lives

Read these in order if you need the full history; each is self-contained
and was written contemporaneously (not reconstructed after the fact):

| Milestone | File | One-line result |
|---|---|---|
| M0 (repo audit) | `artifacts/repo_audit.json` | window_N=3, no Fast-LeWM prefix head, action-convention mismatch flagged |
| M0.5 (scope) | `artifacts/scope_report.md` | `droid_100` failed (kill criterion); `drawer_open_close` (full DROID) selected, 3184 episodes |
| M1 (horizon sweep / k\*) | `artifacts/M1_report.md` | k\*=2 (corrected after a bug was found and isolated — read this file's full correction trail, it's instructive) |
| M2 (alphabet + validation) | `artifacts/M2_report.md` | 12-symbol codebook; Kill Criterion B not triggered; canonical 80/20 split frozen (`artifacts/episode_split.json`) |
| Risk register / scope cuts | `artifacts/preregistration.md` | Every open risk and every deadline-driven simplification, logged before/as they happened |
| Running log | `artifacts/progress_log.md` | Chronological milestone log, terser than the individual reports |
| M3–M6 (predicates → search → gate) | `artifacts/acceptance_report.md` | Full Pareto front, gate table, root-cause discussion |
| Final | `artifacts/results.md` | The delivered top-line summary |

Supporting data artifacts (not narrative, but referenced by the reports
above): `alphabet_trainfit.json` (the codebook actually used, k\*=2,
train-80% only — `alphabet.json` is the earlier full-family/in-sample one,
kept for contrast per M2's methodology), `reset_episode_ids.json` (8
canonical starting episodes), `predicate_pool_summary.json` +
`predicate_functions_{slowfeature.json,networks.pt}` (the 32 surviving
candidates + everything needed to reconstruct their callable φ
functions — see `predicates/functions.py`), `search_log.jsonl` +
`best_machine.pkl` (every extraction attempted + the best one found,
rejected by Phase F but inspectable), `horizon_sweep.json/png` (final,
corrected k-sweep) and `horizon_sweep_original_unsplit.json/png` (the
bug-corrupted original, kept for the correction narrative in
`M1_report.md`).

## 9. Code map (what's implemented vs. stubbed)

Fully implemented and exercised: `oracle/` (streaming, actions, the real
`g` wrapper, the production oracle), `alphabet/` (segments, codebook,
validation — twice, in-sample and held-out), `predicates/discover.py`
(all 3 families in one consolidated script, family iii weak — see §3),
`learning/` (SUL adapter + `run_Lsharp` wrapper), `gate/acceptance.py`
(the real fidelity/coverage/nondeterminism computation).

**Not implemented** (plan sections that were never reached or were
simplified away under time pressure — see `preregistration.md`'s
"deadline-driven scope cuts" for the full disclosure): `search/greedy.py`
/`cegar.py` (the plan's real add/drop+CEGAR search — `search/run_search.py`
is a much simpler greedy-add-only stand-in), `signals/` (Phase G,
relevance subspace via autograd — directory exists, empty), `training/
arms/{shuffled*,input_patch}.py` (only `control`, `latent_subspace`,
`sample_prio` exist, from the *pre-experiment test* phase, before the
7-arm redesign in the last round of instructions was ever implemented —
Phase H never ran), `eval/` (Phase I — directory exists, empty).

## 10. Tests (all passing, don't need real data)

`tests/` — the pre-experiment test suite from before any real pipeline
code existed: oracle purity, prefix consistency, the known-automaton
round-trip (the most valuable one — synthetic FSM, no real model needed),
alphabet-determinism (subprocess-verified), arm isolation. Run from
`symbolic_lewm`'s own venv: `cd symbolic_lewm && ./.venv/Scripts/pytest.exe
tests/ -v`. These test the *mechanism*, not this specific run's result —
worth re-running if any core module (`oracle/latent_oracle.py`,
`learning/sul.py`) is changed.
