# Symbolic Abstraction of a Trained LeWM via Active Automata Learning

**Implementation specification for Claude Code.**

Target: extract a compact Moore machine (≤ 50 states) from a trained LeWM forward
model `g` on the Franka dataset, then use that machine to emit masking / curriculum
signals for a short specialised fine-tune, and measure whether planning success
improves.

---

## 0. Read this section before writing any code

This is a **research pipeline with a real chance of a negative result.** The spec is
written so that a negative result is a clean, publishable outcome rather than a
failure. Several phases have explicit **kill criteria**. Honour them. Do not weaken a
threshold to make a later phase runnable — record the failure, emit the diagnostic
artefacts, and stop.

The four things most likely to sink this, in order of probability:

1. **The action alphabet.** Franka actions are continuous and high-dimensional
   (7-DoF arm + gripper). Automata learning needs a finite alphabet. Every prior
   success in this literature (RNN language extraction, protocol learning) had a
   *natively* discrete alphabet. We do not. Phase B is the highest-risk phase and
   should be validated before anything downstream is built.
2. **The oracle horizon.** `f(u)` is computed by rolling `g` forward `|u|` times.
   Error compounds. Beyond some length `L_max`, `f` is fiction. `L_max` bounds the
   automaton complexity we can resolve at all, and it must be *measured* (Phase A.4),
   not assumed.
3. **Predicate discovery without ground truth.** The user has chosen to discover
   predicates from latents rather than hand-specify them from simulator state. This is
   the harder path: the symbols come out unnamed and there is no external referent to
   validate them against. The compensating mechanism is the state-count diagnostic
   (§D.5) — it scores a predicate set without needing to know what the predicates mean.
4. **Self-referential feedback.** The automaton is derived from `g` and then used to
   train `g`. Errors become self-confirming. The anti-feedback protocol in §H.1 is
   mandatory, not optional.

**The single most important experimental control in this entire document is the
shuffled-automaton arm (§I.3).** Without it, any improvement is uninterpretable.
Build it at the same time as the real arms, not afterwards.

---

## 1. Mathematical setting (reference)

Let `A` be a finite alphabet of abstract actions, `A*` the free monoid on `A`.
Let `Z ⊆ R^d` be the LeWM latent space, `g : Z × A_raw → Z` the trained predictor,
and `φ : Z → B` a labelling into a finite set `B`.

Extend `g` to words: `ĝ(z, ε) = z`, `ĝ(z, ua) = g(ĝ(z,u), a)` — a right action of
`A*` on `Z`. Fix `z₀` and define

```
f(u) = φ( ĝ(z₀, u) )        f : A* → B
```

The **Myhill–Nerode congruence** `u ~_f v ⟺ ∀w. f(uw) = f(vw)` is a right congruence.
`f` is realisable by a finite Moore machine iff `~_f` has finite index, and then
`A*/~_f` **is** the minimal machine. Angluin's `L*` (we will use TTT) computes this
quotient from an evaluation oracle plus an equivalence oracle.

Two facts that matter operationally:

- **`f` is deterministic even when the block-level transition relation is not.**
  `g` and `φ` are deterministic maps, so `f(u)` is single-valued for every word. A bad
  partition does not make the oracle nondeterministic — it makes the automaton
  **larger**. This is why state count is a valid quality score for a partition.
- Equivalently, in the language of symbolic dynamics: the machine is the
  **follower-set automaton** of the symbolic dynamics of `(g, P, A)`. It is finite iff
  that dynamics is sofic; it has `≈ |P|` states iff `P` is close to a Markov partition.

---

## 2. Repository integration — resolve these first

Before Phase A, inspect the actual LeWM codebase and record the answers in
`artifacts/repo_audit.json`. Do not guess any of these.

| Unknown | How to resolve | Why it matters |
|---|---|---|
| Predictor call signature | Read the predictor `forward`. LeWM conditions on a **history of N frame embeddings** with causal masking, not on `z_t` alone. | If `N > 1`, the state of the dynamical system is the window `ζ_t = (z_{t-N+1..t})`, **not** `z_t`. The oracle must carry the window. Everything downstream is defined on `ζ`, not `z`. |
| Action conditioning | LeWM injects actions via AdaLN per layer. Confirm shape, normalisation, and whether actions are absolute or deltas. | Determines how abstract actions are decoded back to `A_raw`. |
| Latent dim `d` and projection | ViT-tiny CLS → 1-layer MLP + BatchNorm. Confirm `d`. | Sizes every downstream module. |
| Is `Fast-LeWM` prefix prediction available? | Check for an action-prefix encoder. | If yes, `g(z, a_{t:t+k-1}) → z_{t+k}` in one pass — use it for macro-actions; it avoids k-step compounding and makes Phase B far cheaper. |
| Checkpoint + exact training config | | Needed to reproduce the control arm. |
| Franka dataset loader, episode boundaries, action normalisation stats | | Phase B clusters raw actions; wrong normalisation silently ruins the alphabet. |
| Native CEM planner entry point + goal sampling | | Phase I metric. |

**Write `artifacts/repo_audit.json` and stop for review before proceeding.**

---

## 3. Configuration

Single YAML, `configs/symbolic.yaml`. All thresholds live here; nothing hard-coded.

```yaml
oracle:
  window_N: null            # from repo audit
  horizon_cap: null         # L_max, measured in Phase A.4
  trace_fidelity_thresh: 0.85
  cache_dir: artifacts/oracle_cache

alphabet:
  segment_len_k: 8          # frames per macro-action
  n_codes: 12               # |A| target, excluding resets
  gripper_special_case: true
  n_reset_symbols: 4        # initial-condition selectors

predicates:
  n_candidates: 64          # candidate pool size
  max_active: 8             # max simultaneously active predicates (|B| ≤ 2^8)
  slowness_weight: 1.0
  predictability_weight: 1.0

learner:
  algorithm: TTT            # TTT preferred over classic L*
  max_states: 200           # hard abort
  target_states: 50         # acceptance gate
  eq_oracle:
    kind: hybrid            # random-word PAC + abstraction-refinement
    epsilon: 0.05
    delta: 0.05
    max_word_len: null      # = oracle.horizon_cap
    n_refinement_rounds: 20

search:
  budget_extractions: 40    # predicate-set search budget
  strategy: greedy_bidirectional

training:
  epochs: 5
  arms: [control, shuffled, latent_subspace, sample_prio, input_patch]
  seeds: [0, 1, 2, 3, 4]

eval:
  n_goals: 200
  primary_metric: planning_success_rate
```

---

## 4. Phase A — Oracle construction

### A.1 Window-aware rollout

```python
class LatentOracle:
    """Right action of A* on the LeWM latent window state."""
    def __init__(self, model, window_N: int, horizon_cap: int): ...

    def reset(self, init_id: int) -> WindowState:
        """Load a canonical initial window from the held-out set."""

    def step(self, state: WindowState, sym: Symbol) -> WindowState:
        """Apply one abstract action (which decodes to k raw actions)."""

    def evaluate(self, word: tuple[Symbol, ...]) -> Label:
        """f(u). Must be pure, deterministic, and memoised."""
```

`WindowState` holds the last `N` latents. `step` decodes a symbol to its raw action
segment (Phase B) and applies `g` for each frame, or uses the Fast-LeWM prefix head if
available.

### A.2 Multiple initial conditions

`f` depends on `z₀`. Handle it inside the alphabet, not outside: prepend reset symbols
`r_1..r_m` to `A`, with the constraint that a word is well-formed iff its first letter
is a reset and no reset appears elsewhere. Ill-formed words return a dedicated sink
label `⊥`. This is standard in automata learning of multi-init systems, and it lets a
single machine cover all initial conditions with shared structure.

Pick the `m` initial windows by k-medoids over encoded first-frames of held-out
episodes, so they span the dataset's start distribution.

### A.3 Caching — do this properly, it dominates runtime

`L*`/TTT query sets are **prefix-closed**. Store `f` in a trie keyed by word; a query
for `u` reuses the rollout state of its longest cached prefix. Expect >90% hit rate.
Persist the trie to `oracle.cache_dir` between runs; key it by
`(checkpoint_hash, alphabet_hash, predicate_hash)`.

Also batch: TTT issues queries in bursts. Expose `evaluate_batch(words)` that groups by
shared prefix and vectorises the `g` calls across the batch dimension.

### A.4 Measure `L_max` — gate, do not skip

Sample `M ≥ 200` held-out real episodes. For each, encode the true frames to get the
real latent trace, and roll `g` forward from the same start with the same actions to get
the model trace. Under a provisional predicate set (any reasonable one — this is a
dynamics measurement, not a predicate measurement), compute per-step agreement of the
predicate traces.

Define `L_max` = the largest `L` at which mean predicate-trace agreement over the first
`L` steps ≥ `oracle.trace_fidelity_thresh`.

Emit `artifacts/horizon_curve.png` (agreement vs. step) and `L_max` into the config.

> **KILL CRITERION A.** If `L_max < 10` abstract steps, stop. Distinguishing suffixes
> will be shorter than the structure we are trying to resolve, and any automaton learned
> will be an artefact of model error. Report `L_max`, the horizon curve, and halt.

---

## 5. Phase B — Action alphabet (highest risk)

Single-step Franka deltas carry essentially no symbolic content. Cluster **temporally
extended segments**.

### B.1 Segment construction

Slice every training episode into overlapping windows of `segment_len_k` raw action
vectors. Represent each segment as the concatenation of (a) the normalised action
sequence, (b) summary statistics (mean, cumulative displacement, dominant direction),
and (c) the gripper channel's binary pattern.

### B.2 Special-case the gripper

The gripper dimension is *natively* binary and is almost certainly the most
symbolically loaded channel in a Franka dataset. Do not let it be averaged into a
continuous cluster. Split the segment pool first by gripper pattern
(`stay-open`, `stay-closed`, `open→close`, `close→open`), then cluster within each
group. Guarantees the alphabet distinguishes grasp/release regardless of what the
clustering does.

### B.3 Codebook

Per gripper group, fit k-means (or a VQ codebook) to reach `n_codes` total. Each symbol
`a ∈ A` stores a **representative raw segment** (the medoid, not the centroid — the
centroid may be dynamically unrealisable).

### B.4 Validate the alphabet before building anything on it

Two checks, both mandatory:

- **Coverage.** Re-encode every training episode as a symbol sequence by nearest-medoid
  assignment. Reconstruct the latent trajectory by applying the *medoid* segments
  through `g`. Measure endpoint latent error vs. applying the true actions. Report the
  distribution.
- **Expressiveness.** Sample random goals; run CEM restricted to sequences of alphabet
  symbols and compare planning success against unrestricted CEM. This measures what the
  discretisation costs in control authority.

> **KILL CRITERION B.** If alphabet-restricted CEM achieves < 60% of unrestricted CEM
> success, the alphabet cannot express the task. Increase `n_codes` or
> `segment_len_k` and retry; if it does not recover within 3 attempts, stop and report.
> An automaton over an inexpressive alphabet is a model of nothing.

Emit `artifacts/alphabet.json` (medoids, cluster stats) and
`artifacts/alphabet_validation.md`.

---

## 6. Phase C — Predicate discovery from latents

No simulator ground truth. We generate a **pool of candidate binary predicates** and let
Phase E's state-count search select among them.

### C.1 Candidate generation — three independent families

Diversity matters more than any single method's quality; the selector will prune.

**(i) VQ bottleneck.** Train a small quantised bottleneck: `z → encoder → VQ (K codes)
→ decoder → ẑ`, with an auxiliary term requiring the code to predict the *next* code
under the abstract action. Each code, and each bit of a binarised code, is a candidate
predicate. (This is the Koul–Fern–Greydanus move, adapted.)

**(ii) Slow / piecewise-constant directions.** A good symbol changes rarely and changes
*predictably*. Score a candidate binary function `ψ` on latent trajectories by

```
score(ψ) = slowness_weight   * (1 - switch_rate(ψ))
         + predictability_weight * AUC( predict switch of ψ from (ζ_t, a_t) )
```

Generate candidates as thresholded linear projections `1[wᵀz > b]`, with `w` from a
slow-feature-analysis / temporal-coherence objective over latent trajectories, and `b`
swept. Keep the top scorers. This family is cheap and tends to find genuine
event-like structure (contact, grasp, release).

**(iii) Behavioural-metric clustering.** Fit the deterministic bisimulation metric
`d(z,z') = max_a [ c·‖ρ(z)-ρ(z')‖ + γ d(g(z,a), g(z',a)) ]` as
`d_ψ(z,z') = ‖ψ(z)-ψ(z')‖₁` via bootstrapped regression (MICo-style, target network),
bootstrapping `ρ` from family (ii). Cluster in `ψ`-space with **complete linkage** under
a diameter cap. Never single linkage — ε-aggregation is not transitive and chaining will
silently merge distant states. Cluster indicators are candidate predicates.

> **Note.** Do not cluster in raw `z`. SIGReg pushes the latent marginal toward an
> isotropic Gaussian — a deliberately unimodal, structureless density. Density-based or
> k-means clustering on `z` will find nothing meaningful. Clustering must happen in a
> *behavioural* embedding, not the latent geometry. Log a warning if anyone adds a raw-`z`
> clustering path.

### C.2 Pool hygiene

Deduplicate candidates by mutual information on held-out trajectories (drop one of any
pair with MI > 0.9). Drop candidates that are constant on > 98% of timesteps (no
information) or that switch on > 40% of timesteps (noise, not a symbol). Target a pool
of `n_candidates ≈ 64` survivors.

Emit `artifacts/predicate_pool.json` with, per candidate: family, switch rate,
predictability AUC, and a montage of decoded frames at switch points
(`artifacts/predicate_montages/`) so a human can try to name them later.

---

## 7. Phase D — The Angluin loop

### D.1 Library

Use **AALpy** (Python, supports Moore/Mealy, custom `SUL` classes, TTT and `L*`).
Verify its current API before writing against it — do not assume signatures. Implement:

```python
class LeWMSUL(SUL):
    """System-under-learning adapter: wraps LatentOracle + predicate set."""
    def pre(self):  ...   # reset to a chosen initial window
    def step(self, letter): ...   # returns the Moore output = φ(current window)
    def post(self): ...
```

Prefer **TTT** over classic `L*`: discrimination trees instead of observation tables,
far better counterexample handling, materially fewer queries.

### D.2 Equivalence oracle — hybrid, two components

A true equivalence oracle does not exist. Combine:

**(a) PAC random-word testing.** Fix a distribution `D` over words of length
`≤ L_max` — do *not* use uniform-random words; use a distribution matched to the
symbol statistics of real episodes, since that is the region where `g` is trustworthy
and where planning actually operates. At round `i` draw
`m = ⌈(1/ε)(ln(1/δ) + i·ln 2)⌉` words. Any disagreement is a counterexample.

**(b) Abstraction-refinement testing.** Maintain a partition of the latent window space.
Track, for each hypothesis automaton state, the set of latent windows reaching it. If two
windows mapped to the same automaton state diverge in label under some short suffix,
that suffix is a counterexample. This is the mechanism from Weiss–Goldberg–Yahav and it
finds structured counterexamples that random sampling misses.

Run both; feed whichever finds a counterexample first.

### D.3 Hard caps

Abort an extraction if states exceed `learner.max_states`, queries exceed a budget, or
wall-clock exceeds a limit. An aborted extraction is a **data point** (this predicate set
is bad), not an error — return `ExtractionResult(status="aborted", n_states_at_abort=...)`.

### D.4 Outputs per extraction

```python
@dataclass
class ExtractionResult:
    status: str                       # converged | aborted | timeout
    machine: MooreMachine | None
    n_states: int
    n_queries: int
    distinguishing_suffixes: dict[tuple[int,int], tuple[Symbol,...]]
    state_visitation: dict[int, int]  # on held-out real episodes
    edge_visitation: dict[tuple[int,Symbol], int]
    block_nondeterminism: float       # see D.5
    trace_fidelity: float             # see F.2
```

### D.5 The state-count diagnostic

This is the scoring function for the whole search. For a predicate set inducing
`|B|` observable labels and a reachable block count `n`:

- `n_states ≈ n` → the partition is near-Markov, the abstraction is sound.
- `n_states >> n` → the partition is destroying information the dynamics needs; the
  machine is spending states patching the hole.
- `aborted` → not sofic at this resolution, or the alphabet/horizon is inadequate.

Also record **block nondeterminism**: over held-out real transitions, the fraction of
`(block, symbol)` pairs whose observed successor block is not unique. High values mean
the partition is too coarse in a dynamically relevant direction.

---

## 8. Phase E — Refinement search over predicate sets

Search the power set of the candidate pool for a subset minimising state count subject to
fidelity. Budget: `search.budget_extractions` full extractions.

### E.1 Objective

```
J(S) = n_states(S) + λ_fid · max(0, fidelity_target - trace_fidelity(S))·1000
                   + λ_size · |S|
```

subject to `trace_fidelity(S) ≥ oracle.trace_fidelity_thresh`. The hard constraint is
fidelity; state count is what we minimise. **Never trade fidelity for compactness** — a
20-state machine that does not track the system is worse than no machine.

### E.2 Strategy — greedy bidirectional

1. Seed with the 3 highest-scoring predicates from family (ii) (slowness/predictability).
2. **Add** phase: for each candidate not in `S`, estimate marginal value cheaply
   (block nondeterminism reduction on held-out transitions — no extraction needed).
   Run a full extraction only on the top-3 estimated candidates. Keep the best.
3. **Drop** phase: every 3 add-steps, try removing each member of `S`; keep any removal
   that does not decrease fidelity below threshold.
4. Stop at `target_states`, at budget exhaustion, or when no add/drop improves `J`.

Cache extractions by predicate-set hash. Log the full search trace to
`artifacts/search_log.jsonl` — the trajectory of `(|S|, n_states, fidelity)` is itself a
reportable result about whether compact symbolic structure exists in this latent space.

### E.3 CEGAR sub-loop

When an extraction aborts on state blowup, inspect the discrimination tree for the pair
of latent windows that forced the most recent split, and the suffix that separated them.
Find the candidate predicate in the pool with the highest agreement with that split and
propose it as the next add. This makes refinement counterexample-driven rather than
blind, and gives every added symbol a *reason* recorded in the log.

---

## 9. Phase F — Acceptance gate

Proceed to masking **only** if all hold, on held-out data:

| Gate | Threshold |
|---|---|
| `n_states` | ≤ 50 |
| `trace_fidelity` at `H = L_max` | ≥ 0.85 |
| `block_nondeterminism` | ≤ 0.10 |
| Reachable-state coverage on held-out episodes | ≥ 90% of states visited |
| Reproducibility | 3 extractions with different query seeds agree up to isomorphism |

### F.2 Trace fidelity, defined precisely

Sample held-out **real** episodes. Encode to get the real latent trace. Convert to a
predicate trace via `φ`. Separately, run the automaton on the same abstract-action
sequence. Fidelity = mean per-step label agreement. Report against **real** rollouts,
never model rollouts — the whole point is to catch model error.

Also decompose the gap: (real vs. model latent traces) is *model* error, (model traces vs.
automaton) is *abstraction* error. Report both separately in
`artifacts/acceptance_report.md`. Conflating them is the most common way to fool
yourself here.

> **KILL CRITERION F.** If no predicate set within budget passes all gates, **stop and
> report the negative result.** Do not raise `max_states` above 50 to proceed. A
> 200-state machine is a lookup table, not a symbolic model, and downstream signals
> derived from it will be noise dressed as structure. The search log and the
> `(|S|, n_states, fidelity)` Pareto front are the deliverable in that case.

---

## 10. Phase G — Signal extraction

From an accepted machine, derive four signals. Each is a pure, deterministic function
computable online during training.

### G.1 Automaton state `q_t`
Run the machine along the observed abstract-action sequence. Discrete, low-dimensional
conditioning variable, available every timestep.

### G.2 Relevance subspace `V ⊆ R^d` — the principled one

For each pair of automaton states `(i,j)`, TTT gives a distinguishing suffix `e_ij`.
Define the sensitivity of the outcome of `e_ij` to a latent perturbation:

```
v_ij(z) = ∇_z  φ( ĝ(z, e_ij) )        # via autograd through the g-rollout
```

Collect `{v_ij(z)}` over sampled `z` and all pairs, and take the top-`r` principal
subspace `V`. Directions orthogonal to `V` provably do not affect any distinguishing
suffix, hence do not affect the symbolic model. Choose `r` by explained variance
(target 95%); report `r/d`.

Store `P_V` (the projector) in `artifacts/relevance_subspace.npz` along with `r`, the
explained-variance curve, and the per-pair contribution breakdown.

### G.3 Edge rarity
Visitation counts over `(state, symbol)` edges on held-out episodes → inverse-frequency
weights, clipped to `[0.2, 5.0]`.

### G.4 Abstraction-failure flag
Per timestep: 1 if the automaton's predicted next label ≠ `φ(g(ζ_t, a_t))`, else 0.
These mark where the symbols are inadequate — usually the dynamically interesting
regions.

---

## 11. Phase H — Specialised training arms

### H.1 Anti-feedback protocol (mandatory)

- Extract the automaton from a **frozen** checkpoint `θ₀`.
- Run all 5 epochs against that frozen automaton. **Do not re-extract mid-run.**
- All arms start from the identical `θ₀`, see identical data, identical batch order per
  seed, identical optimiser state, identical step count. The *only* difference is the
  loss/mask.
- If a future version re-extracts, gate each refresh on Phase F fidelity against real
  held-out rollouts. Never on model rollouts.

### H.2 The arms

| Arm | Mechanism |
|---|---|
| `control` | Unmodified LeWM objective. |
| `shuffled` | **Critical control.** Identical mask *statistics* to the best real arm, but with the automaton→signal map randomly permuted, so mask magnitude and sparsity are preserved while the symbolic content is destroyed. |
| `latent_subspace` | `ℒ_pred = ‖P_V(ẑ_{t+1} − z_{t+1})‖² + β‖P_V^⊥(·)‖²` with `β < 1`. Stops capacity going to predictable-but-irrelevant content. |
| `sample_prio` | Reweight the batch loss by `edge_rarity(q_t, a_t) × (1 + κ · abstraction_failure_flag_t)`. |
| `input_patch` | Condition the JEPA mask distribution on `q_t`: attribute each predicate back to input patches (integrated gradients), and bias the mask sampler toward implicated regions on a curriculum schedule. Most plumbing; do last. |

Keep `SIGReg(Z)` unmodified in every arm — it is what keeps the latent from collapsing,
and masking must not be allowed to interact with it.

### H.3 Implementation

Each arm is a subclass of a single `MaskedTrainer` with one overridden
`compute_loss(batch, signals)`. Signals arrive via a precomputed per-sample side table
(computed once against the frozen automaton), not recomputed in the inner loop —
the automaton is fast, but determinism and reproducibility matter more than the saving.

---

## 12. Phase I — Evaluation

### I.1 Primary metric
**Planning success rate** with LeWM's native CEM/MPC on held-out Franka goals.
`n_goals = 200`, fixed goal set across all arms and seeds. Secondary: steps-to-goal on
successes, and final goal-latent distance on failures.

### I.2 Secondary metrics (report, do not optimise)
Multi-step latent prediction error at `H ∈ {5,10,20,50}`; predicate-trace agreement vs.
real rollouts; automaton state count on re-extraction after training.

### I.3 Statistics — the part that decides whether any of this means anything

- 5 seeds per arm, **paired** by seed against `control`.
- Report mean difference with bootstrap 95% CI, not just point estimates. Five epochs
  is a small intervention; effects will be small and seed variance will be comparable.
- **Pre-register** the primary metric and the comparison before running. Write it into
  `artifacts/preregistration.md` and commit before the first training run.
- **The `shuffled` arm is the interpretive key.** If `latent_subspace` beats `control`
  but does not beat `shuffled`, then non-uniform masking helped and the symbolic
  structure contributed nothing. That is a legitimate and interesting finding — report
  it as such. Do not bury it.

### I.4 Reporting
`artifacts/results.md` with: per-arm table, paired-difference plot with CIs, the
`(|S|, n_states, fidelity)` Pareto front from Phase E, the horizon curve from A.4, and a
rendered automaton diagram (Graphviz) with states annotated by their most frequent
decoded frames.

---

## 13. File layout

```
symbolic_lewm/
  configs/symbolic.yaml
  oracle/
    latent_oracle.py        # A.1-A.2  window-aware right action of A*
    cache.py                # A.3      prefix trie, batched evaluation
    horizon.py              # A.4      L_max measurement
  alphabet/
    segments.py             # B.1-B.2
    codebook.py             # B.3
    validate.py             # B.4      coverage + CEM expressiveness
  predicates/
    vq_bottleneck.py        # C.1(i)
    slow_features.py        # C.1(ii)
    bisim_metric.py         # C.1(iii) MICo-style metric + complete-linkage
    pool.py                 # C.2      dedup, hygiene, montages
  learning/
    sul.py                  # D.1      AALpy SUL adapter
    eq_oracle.py            # D.2      hybrid PAC + abstraction-refinement
    extract.py              # D.3-D.4
    diagnostics.py          # D.5
  search/
    objective.py            # E.1
    greedy.py               # E.2
    cegar.py                # E.3
  gate/
    acceptance.py           # F
  signals/
    relevance_subspace.py   # G.2  autograd through the g-rollout
    edge_stats.py           # G.3-G.4
  training/
    masked_trainer.py       # H.3  base class
    arms/{control,shuffled,latent_subspace,sample_prio,input_patch}.py
  eval/
    planning.py             # I.1
    stats.py                # I.3
    report.py               # I.4
  artifacts/                # all outputs, git-ignored except reports
  tests/
```

---

## 14. Milestones — stop for review at each

| # | Deliverable | Gate |
|---|---|---|
| M0 | `repo_audit.json` | Human review before any code |
| M1 | `L_max` + horizon curve | Kill criterion A |
| M2 | Alphabet + validation report | Kill criterion B |
| M3 | Predicate pool + montages | Sanity check: are any predicates namable by a human? |
| M4 | First successful extraction (any size) | Pipeline works end to end |
| M5 | Search complete, Pareto front | Kill criterion F |
| M6 | Accepted machine ≤ 50 states | Acceptance gate |
| M7 | Signals extracted, `r/d` reported | If `r ≈ d`, latent-subspace masking is vacuous — say so |
| M8 | 5 arms × 5 seeds trained | Anti-feedback protocol verified |
| M9 | `results.md` | Shuffled-control comparison reported prominently |

Milestones M1, M2, M5 are the ones most likely to end the project. That is by design —
each is cheap relative to what follows it.

---

## 15. Testing requirements

- **Oracle purity.** `evaluate(u)` returns identical output across calls, cache states,
  and batch groupings. Property test over random words.
- **Prefix consistency.** `evaluate(u)` computed fresh equals the value obtained by
  extending the cached rollout of any prefix of `u`.
- **Known-automaton round trip.** Replace `g` with a hand-built finite state machine
  lifted into `R^d` (states as one-hot-plus-noise). The pipeline must recover it exactly.
  This is the single most valuable test in the suite — write it before Phase D.
- **Alphabet determinism.** Symbol→segment decoding is stable across runs given a fixed
  codebook hash.
- **Arm isolation.** With masking disabled, every arm reproduces `control` bit-for-bit
  under a fixed seed.

---

## 16. Prior work to consult

- Weiss, Goldberg & Yahav, *Extracting Automata from RNNs Using Queries and
  Counterexamples* (ICML 2018) — `L*` with a network as membership oracle, and the
  abstraction-refinement equivalence oracle used in §D.2(b). Closest precedent.
- Koul, Fern & Greydanus, *Learning Finite State Representations of Recurrent Policy
  Networks* (ICLR 2019) — quantised bottleneck → Moore machine; basis for §C.1(i).
- Danesh, Fern et al., *Re-understanding Finite-State Representations of Recurrent Policy
  Networks* (ICML 2021) — caution: minimised machines can be deceptively small.
- Ferns, Panangaden & Precup — bisimulation metrics; Castro et al. — MICo. §C.1(iii).
- Vaandrager et al. / LearnLib — mature active automata learning practice.
- Konidaris, Kaelbling & Lozano-Pérez — which symbols are *sufficient* for sound planning,
  if the project extends to operator-level models.

**No prior work applies this to a JEPA-style latent world model.** The Franka setting
(continuous 7-DoF actions, no ground-truth predicates) is harder than every precedent
above on both the alphabet and the grounding axis. Plan accordingly, and treat a clean
negative result as a successful outcome.
