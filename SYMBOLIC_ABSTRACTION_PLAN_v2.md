# Symbolic Abstraction of a Trained LeWM via Active Automata Learning

**Implementation specification for Claude Code — v2.**

Target: extract a compact Moore machine (≤ 50 states) from a trained LeWM forward
model `g` on DROID Franka data, then use that machine to emit masking / curriculum
signals for a short specialised fine-tune, and measure whether goal-directed planning
competence improves.

---

## v2 changelog — what M0 changed

The repo audit (`artifacts/repo_audit.json`) resolved every unknown in v1 and surfaced
one blocker v1 did not anticipate. Changes:

| Finding | Consequence in v2 |
|---|---|
| `history_size: 3` — predictor is **not** Markov in `z_t` | Confirmed §4.1. System state is the 3-window `ζ_t`. Already correct in v1. |
| **No Fast-LeWM prefix head** in this repo | New **Phase A.5**: sweep `segment_len_k` and choose it to maximise `L_max` measured *in letters*. `segment_len_k` is no longer a fixed config value. |
| Train/val normalisation leak (`train.py:140` vs `:149`) | New §2.3. Fix the code; **do not retrain** — the leak is common-mode across arms and cancels in paired comparison. It invalidates absolute figures only. |
| **No Franka/DROID gym environment exists** — CEM cannot execute | Phase I primary metric **replaced**. See §12. The naive "offline latent-distance proxy" is rejected as actively biased; §12.1 explains why and §12.2 gives the replacement. Phase B.4's kill criterion is rewritten in the same terms. |
| Data is **real-world DROID**, not sim | New **Phase M0.5** (§3): scope selection. A single automaton over 564 scenes is hopeless; you must narrow to one task family first. This is a new gating milestone and is as likely to end the project as any other. |

---

## 0. Read this before writing any code

This is a **research pipeline with a real chance of a negative result.** It is written so
that a negative result is a clean outcome rather than a failure. Several phases carry
explicit **kill criteria**. Honour them. Never weaken a threshold to make a later phase
runnable — record the failure, emit the diagnostics, stop.

The five things most likely to sink this, in order of probability:

1. **Scope.** DROID is in-the-wild real-robot data across 564 scenes and 86 buildings,
   with no repeated resets and no two episodes sharing an initial condition. An automaton
   over the whole dataset does not exist in any useful sense. Phase M0.5 narrows to a
   single task family; if no family has enough episodes, the project ends there.
2. **The action alphabet.** 7-DoF continuous control. Every precedent for this
   construction had a natively discrete alphabet. We do not.
3. **The oracle horizon.** `f(u)` compounds `g` `|u| × k` times, with no prefix head to
   shortcut it. `L_max` bounds the automaton complexity we can resolve at all, and must
   be *measured* (Phase A.4–A.5), not assumed.
4. **Predicate discovery without ground truth.** Symbols come out unnamed with no
   external referent. The compensating mechanism is the state-count diagnostic (§8.5),
   which scores a predicate set without knowing what the predicates mean.
5. **Self-referential feedback.** The automaton is derived from `g` and then used to
   train `g`. The anti-feedback protocol in §11.1 is mandatory.

**The two most important experimental controls in this document are the shuffled-automaton
arm (§12.4) and the choice of offline metric (§12.2).** Get either wrong and no result
here is interpretable. Build both before any masking arm.

### A note on what the automaton will actually describe

DROID is teleoperated demonstration data: near-optimal, narrow action distribution. `g`
has never seen most action sequences. Random words in `L*` will be wildly
out-of-distribution and `g`'s answers there are fabrication. Consequently the PAC
guarantee holds only with respect to the demonstration distribution, and **the automaton
will describe the demonstration manifold, not the dynamics.** State this limitation in
any writeup. It is not a bug in the method; it is what the data supports.

---

## 1. Mathematical setting (reference)

Let `A` be a finite alphabet of abstract actions, `A*` the free monoid on `A`.
Let `Z ⊆ R^d` be the latent space, `ζ = (z_{t-2}, z_{t-1}, z_t)` the 3-window state,
`g` the trained predictor, and `φ` a labelling into a finite set `B`.

Extend `g` to words: `ĝ(ζ, ε) = ζ`, `ĝ(ζ, ua) = g(ĝ(ζ,u), a)` — a right action of `A*`
on window space. Fix `ζ₀` and define

```
f(u) = φ( ĝ(ζ₀, u) )        f : A* → B
```

The **Myhill–Nerode congruence** `u ~_f v ⟺ ∀w. f(uw) = f(vw)` is a right congruence.
`f` is realisable by a finite Moore machine iff `~_f` has finite index, and then
`A*/~_f` **is** the minimal machine.

Two facts that matter operationally:

- **`f` is deterministic even when the block-level transition relation is not.** `g` and
  `φ` are deterministic, so `f(u)` is single-valued for every word. A bad partition does
  not make the oracle nondeterministic — it makes the automaton **larger**. This is why
  state count is a valid quality score for a partition.
- In the language of symbolic dynamics, the machine is the **follower-set automaton** of
  the symbolic dynamics of `(g, P, A)`. It is finite iff that dynamics is sofic, and has
  `≈ |P|` states iff `P` is near-Markov.

---

## 2. Established facts from M0

### 2.1 Predictor

`history_size = 3`, causal attention. The state object throughout is `ζ`, a 3-window of
latents. Initial conditions require 3 real frames.

### 2.2 No prefix head

`g` applies one raw step at a time. An abstract letter of `k` raw steps costs `k`
applications and `k`-fold error compounding. `segment_len_k` is therefore a *swept*
parameter (§5.5), not a chosen one.

### 2.3 Normalisation leak — fix, do not retrain

`train.py:140` fits action/proprio z-score statistics on the full dataset before the
split at `train.py:149`.

- **Fix the code now** so all downstream fitting uses train-split statistics only.
- **Do not retrain `θ₀`.** All arms branch from the same contaminated checkpoint, see the
  same data, and are compared pairwise. The leak is common-mode and cancels exactly in
  the differences, which is what the experiment measures.
- It *does* invalidate any absolute number reported as clean held-out performance.
  Record this in `artifacts/preregistration.md`. Retrain only if absolute figures are
  needed for publication.

### 2.4 No environment

`stable_worldmodel.solver.CEMSolver` requires a live Gymnasium env to report task
success. No Franka/DROID env is registered, and DROID is real-hardware in-the-wild data,
so none can be built to match. Planning success as originally specified is not
computable. §12 replaces it.

---

## 3. Phase M0.5 — Scope selection (new gating milestone)

Run this before Phase A. It is cheap and it is the most likely place for the project to
end.

### 3.1 Find a task family

DROID episodes carry natural-language task annotations and scene/building identifiers.
Group episodes by (normalised task string, scene id). For each group record: episode
count, mean length, action-distribution spread, and visual variance of first frames.

Select a **single task family** satisfying:

- ≥ 200 episodes
- consistent scene (a single camera viewpoint and workspace layout)
- a task with visible discrete structure — reach / grasp / lift / place / release. A
  continuous repositioning task has no symbols to find and should not be chosen.

Prefer a family involving the gripper, since the gripper channel is the one natively
discrete signal in the whole action space.

Emit `artifacts/scope_report.md`: the ranked candidate families, the chosen one, and the
episode count. **All later phases operate on this subset only.** Record the subset's
episode ids in `artifacts/scope_episode_ids.json` and use it everywhere.

> **KILL CRITERION M0.5.** If no family reaches 200 episodes in a consistent scene, stop.
> With fewer, the held-out sets for `L_max`, trace fidelity, and the 5-seed evaluation are
> all too small to support any conclusion. Report the family-size distribution and halt.

### 3.2 Consequence for initial conditions

Real data means no two episodes share an initial condition. Reset symbols (§4.2) select
among `m` real episode-start windows drawn from the chosen family, picked by k-medoids on
encoded first frames. They span the family's start distribution; they do not reproduce it.

---

## 4. Phase A — Oracle construction

### 4.1 Window-aware rollout

```python
class LatentOracle:
    """Right action of A* on the LeWM 3-window state."""
    def __init__(self, model, window_N: int = 3, horizon_cap: int = ...): ...
    def reset(self, init_id: int) -> WindowState: ...
    def step(self, state: WindowState, sym: Symbol) -> WindowState:
        """Decode symbol to its raw action segment; apply g once per raw step."""
    def evaluate(self, word: tuple[Symbol, ...]) -> Label:
        """f(u). Pure, deterministic, memoised."""
```

### 4.2 Multiple initial conditions

`f` depends on `ζ₀`. Handle inside the alphabet: prepend reset symbols `r_1..r_m` to `A`;
a word is well-formed iff its first letter is a reset and no reset appears elsewhere.
Ill-formed words return a sink label `⊥`. Standard for multi-init systems, and it lets one
machine cover all starts with shared structure.

### 4.3 Caching — dominates runtime, do it properly

`L*`/TTT query sets are **prefix-closed**. Store `f` in a trie keyed by word; a query for
`u` reuses the rollout state of its longest cached prefix. Expect > 90% hit rate. Persist
between runs keyed by `(checkpoint_hash, alphabet_hash, predicate_hash)`. Expose
`evaluate_batch(words)` grouping by shared prefix and vectorising `g` across the batch
dimension — TTT issues queries in bursts.

### 4.4 Measure `L_max` at fixed `k`

Sample `M ≥ 200` held-out episodes from the scoped family. Encode true frames for the real
latent trace; roll `g` from the same start with the same actions for the model trace.
Under a provisional predicate set (any reasonable one — this is a dynamics measurement),
compute per-step predicate-trace agreement.

`L_max(k)` = the largest number of **letters** `L` at which mean agreement over the first
`L` letters ≥ `oracle.trace_fidelity_thresh`.

### 4.5 Sweep `k` — new, replaces the fixed `segment_len_k`

There is a real optimum. Larger `k` gives more symbolic content per letter but compounds
error faster; smaller `k` compounds less but needs longer words to express anything, and
word length is capped by `L_max` itself.

For `k ∈ {2, 4, 8, 16}`: build a provisional alphabet (§5), measure `L_max(k)`, and record
`(k, L_max(k), raw_horizon = k · L_max(k))`.

**Choose `k* = argmax_k L_max(k)`** — maximise the trustworthy horizon *in letters*, since
letters are what the automaton is built from, not raw steps.

Emit `artifacts/horizon_sweep.png` (agreement-vs-letter curves, one per `k`) and write
`k*` and `L_max(k*)` into the config.

> **KILL CRITERION A.** If `max_k L_max(k) < 10` letters, stop. Distinguishing suffixes
> would be shorter than the structure we are trying to resolve, and any automaton learned
> would be an artefact of model error rather than of the system.
>
> **Escape hatch, only if the sweep fails:** train a Fast-LeWM-style action-prefix encoder
> on the frozen LeWM encoder (a few hours) and re-run the sweep. This changes the artefact
> under study, so record it prominently and re-run every prior measurement against the new
> predictor. Do not do this pre-emptively.

---

## 5. Phase B — Action alphabet

Single-step Franka deltas carry essentially no symbolic content. Cluster **temporally
extended segments** of length `k`.

### 5.1 Segments
Slice scoped-family episodes into overlapping windows of `k` raw action vectors.
Represent each as: normalised action sequence ⊕ summary statistics (mean, cumulative
displacement, dominant direction) ⊕ the gripper channel's binary pattern.

### 5.2 Special-case the gripper
Natively binary and almost certainly the most symbolically loaded channel in DROID. Do not
let it be averaged into a continuous cluster. Split the segment pool first by gripper
pattern (`stay-open`, `stay-closed`, `open→close`, `close→open`), then cluster within each
group. Guarantees the alphabet distinguishes grasp from release whatever the clustering does.

### 5.3 Codebook
Per gripper group, fit k-means / VQ to reach `n_codes` total. Each symbol stores a
**medoid** raw segment, not a centroid — a centroid may be dynamically unrealisable.

### 5.4 Coverage validation
Re-encode every scoped episode as a symbol sequence by nearest-medoid assignment.
Reconstruct the latent trajectory by applying medoid segments through `g`; measure endpoint
latent error against applying the true actions. Report the distribution.

### 5.5 Expressiveness validation — rewritten for the offline setting

v1 compared alphabet-restricted CEM against unrestricted CEM. With no environment, run the
comparison **on the plan-ranking metric of §12.2** instead:

- Unrestricted: candidate negatives drawn as arbitrary action sequences.
- Restricted: candidates constrained to alphabet-expressible sequences (medoid
  concatenations), with the positive being the alphabet-quantised version of the true
  sequence.
- Compare MRR.

> **KILL CRITERION B.** If restricted MRR < 60% of unrestricted MRR, the alphabet cannot
> express the task. Increase `n_codes`, or revisit `k`, and retry. If it does not recover
> within 3 attempts, stop. An automaton over an inexpressive alphabet is a model of nothing.

Emit `artifacts/alphabet.json` and `artifacts/alphabet_validation.md`.

---

## 6. Phase C — Predicate discovery from latents

No ground truth. Generate a **pool of candidates** and let Phase E's state-count search
select among them.

### 6.1 Three independent families
Diversity matters more than any single method's quality; the selector prunes.

**(i) VQ bottleneck.** Train a small quantised bottleneck `ζ → enc → VQ(K) → dec → ζ̂`,
with an auxiliary term requiring the code to predict the *next* code under the abstract
action. Codes and binarised code bits are candidates. (Koul–Fern–Greydanus, adapted.)

**(ii) Slow / piecewise-constant directions.** A good symbol changes rarely and
*predictably*:

```
score(ψ) = w_slow * (1 - switch_rate(ψ)) + w_pred * AUC(predict switch of ψ from (ζ_t, a_t))
```

Candidates are thresholded linear projections `1[wᵀz > b]`, with `w` from a
slow-feature / temporal-coherence objective and `b` swept. Cheap, and tends to find
genuine event structure (contact, grasp, release).

**(iii) Behavioural-metric clustering.** Fit
`d(ζ,ζ') = max_a [ c·‖ρ(ζ)-ρ(ζ')‖ + γ d(g(ζ,a), g(ζ',a)) ]` as `‖ψ(ζ)-ψ(ζ')‖₁` via
bootstrapped regression (MICo-style, target network), bootstrapping `ρ` from family (ii).
Cluster in `ψ`-space with **complete linkage** under a diameter cap — never single
linkage; ε-aggregation is not transitive and chaining will silently merge distant states.

> **Do not cluster in raw `z`.** SIGReg pushes the latent marginal toward an isotropic
> Gaussian — a deliberately unimodal, structureless density. k-means or density clustering
> on `z` will find nothing meaningful. Clustering must happen in a *behavioural* embedding.
> Log a warning if any raw-`z` clustering path is added.

### 6.2 Pool hygiene
Deduplicate by mutual information on held-out trajectories (drop one of any pair with
MI > 0.9). Drop candidates constant on > 98% of timesteps (no information) or switching on
> 40% (noise, not a symbol). Target ≈ 64 survivors.

Emit `artifacts/predicate_pool.json` with per-candidate family, switch rate,
predictability AUC, and a montage of decoded frames at switch points
(`artifacts/predicate_montages/`) so a human can attempt to name them.

---

## 7. Phase D — The Angluin loop

### 7.1 Library
Use **AALpy** (Python; Moore/Mealy, custom `SUL`, TTT and `L*`). Verify its current API
before writing against it.

```python
class LeWMSUL(SUL):
    def pre(self): ...    # reset to a chosen initial 3-window
    def step(self, letter): ...   # Moore output = φ(current window)
    def post(self): ...
```

Prefer **TTT** over classic `L*`: discrimination trees rather than observation tables,
better counterexample handling, materially fewer queries.

### 7.2 Equivalence oracle — hybrid

**(a) PAC random-word testing.** Fix a distribution `D` over words of length `≤ L_max`.
**Do not use uniform-random words.** DROID is demonstration data; uniform words are far
off-distribution and `g`'s answers there are fabrication. Use a distribution matched to the
symbol statistics of real scoped episodes (an n-gram model over the symbol sequences is
sufficient). At round `i` draw `m = ⌈(1/ε)(ln(1/δ) + i·ln 2)⌉` words.

**(b) Abstraction-refinement testing.** Maintain a partition of window space; track which
windows reach each hypothesis state. If two windows at the same state diverge in label
under some short suffix, that suffix is a counterexample. (Weiss–Goldberg–Yahav.) Finds
structured counterexamples random sampling misses.

Run both; use whichever finds a counterexample first.

### 7.3 Hard caps
Abort if states exceed `max_states`, queries exceed budget, or wall-clock exceeds a limit.
An aborted extraction is a **data point** (this predicate set is bad), not an error.

### 7.4 Outputs

```python
@dataclass
class ExtractionResult:
    status: str                       # converged | aborted | timeout
    machine: MooreMachine | None
    n_states: int
    n_queries: int
    distinguishing_suffixes: dict[tuple[int,int], tuple[Symbol,...]]
    state_visitation: dict[int, int]
    edge_visitation: dict[tuple[int,Symbol], int]
    block_nondeterminism: float
    trace_fidelity: float
```

### 7.5 The state-count diagnostic
For a predicate set inducing reachable block count `n`:

- `n_states ≈ n` → near-Markov partition, sound abstraction.
- `n_states >> n` → the partition destroys information the dynamics needs; the machine is
  spending states patching the hole.
- `aborted` → not sofic at this resolution, or alphabet/horizon inadequate.

Also record **block nondeterminism**: over held-out real transitions, the fraction of
`(block, symbol)` pairs whose observed successor block is not unique.

---

## 8. Phase E — Refinement search over predicate sets

Budget: `search.budget_extractions` full extractions.

### 8.1 Objective

```
J(S) = n_states(S) + λ_fid · max(0, fid_target - trace_fidelity(S))·1000 + λ_size · |S|
```

subject to `trace_fidelity(S) ≥ thresh`. Fidelity is the hard constraint; state count is
what we minimise. **Never trade fidelity for compactness** — a 20-state machine that does
not track the system is worse than no machine.

### 8.2 Greedy bidirectional
1. Seed with the 3 highest-scoring family-(ii) predicates.
2. **Add:** estimate marginal value cheaply (block-nondeterminism reduction on held-out
   transitions, no extraction needed); run full extractions only on the top 3; keep the best.
3. **Drop:** every 3 add-steps, try removing each member; keep removals that hold fidelity.
4. Stop at `target_states`, budget exhaustion, or no improving move.

Cache by predicate-set hash. Log to `artifacts/search_log.jsonl` — the
`(|S|, n_states, fidelity)` trajectory is itself a reportable result about whether compact
symbolic structure exists in a SIGReg'd latent space.

### 8.3 CEGAR sub-loop
On abort, inspect the discrimination tree for the pair that forced the most recent split
and the suffix separating them; propose the pool candidate with highest agreement with that
split. Every added symbol then has a recorded reason.

---

## 9. Phase F — Acceptance gate

Proceed to masking **only** if all hold on held-out scoped data:

| Gate | Threshold |
|---|---|
| `n_states` | ≤ 50 |
| `trace_fidelity` at `H = L_max` | ≥ 0.85 |
| `block_nondeterminism` | ≤ 0.10 |
| Reachable-state coverage on held-out episodes | ≥ 90% of states visited |
| Reproducibility | 3 extractions, different query seeds, agree up to isomorphism |

**Trace fidelity, precisely.** Sample held-out **real** episodes; encode to get real latent
traces; convert to predicate traces via `φ`; run the automaton on the same abstract-action
sequence; report mean per-step label agreement. Against **real** rollouts, never model
rollouts.

Decompose the gap and report both separately in `artifacts/acceptance_report.md`:
(real vs. model latent traces) is *model* error; (model traces vs. automaton) is
*abstraction* error. Conflating them is the most common way to fool yourself here.

> **KILL CRITERION F.** If no predicate set within budget passes all gates, **stop and
> report the negative result.** Do not raise `max_states` above 50. A 200-state machine is
> a lookup table, not a symbolic model, and signals derived from it are noise wearing
> structure's shape. The search log and the `(|S|, n_states, fidelity)` Pareto front are
> the deliverable in that case.

---

## 10. Phase G — Signal extraction

Four signals, each a pure deterministic function computable online.

**G.1 Automaton state `q_t`.** Run the machine along the observed abstract-action sequence.

**G.2 Relevance subspace `V` — the principled one.** For each state pair `(i,j)`, TTT gives
a distinguishing suffix `e_ij`. Compute
`v_ij(ζ) = ∇_ζ φ( ĝ(ζ, e_ij) )` by autograd through the `g`-rollout. Collect over sampled
`ζ` and all pairs; take the top-`r` principal subspace. Directions orthogonal to `V`
provably do not affect any distinguishing suffix, hence do not affect the symbolic model.
Choose `r` by 95% explained variance; **report `r/d`**.

> If `r ≈ d`, the latent-subspace arm is vacuous — every direction matters. Say so
> explicitly rather than running a masking arm that masks nothing.

Store `P_V`, `r`, the explained-variance curve, and per-pair contributions in
`artifacts/relevance_subspace.npz`.

**G.3 Edge rarity.** Inverse-frequency weights over `(state, symbol)` visitation on
held-out episodes, clipped to `[0.2, 5.0]`.

**G.4 Abstraction-failure flag.** 1 where the automaton's predicted next label ≠
`φ(g(ζ_t, a_t))`. Marks where the symbols are inadequate — usually the dynamically
interesting regions.

---

## 11. Phase H — Specialised training arms

### 11.1 Anti-feedback protocol (mandatory)
- Extract from a **frozen** checkpoint `θ₀`.
- Run all 5 epochs against that frozen automaton. **Never re-extract mid-run.**
- All arms start from identical `θ₀`, identical data, identical batch order per seed,
  identical optimiser state, identical step count. The *only* difference is the loss/mask.
- Any future refresh must be gated on Phase F fidelity against **real** held-out rollouts.

### 11.2 The arms

| Arm | Mechanism |
|---|---|
| `control` | Unmodified LeWM objective. |
| `shuffled` | **Critical control.** Identical mask *statistics* to the best real arm, with the automaton→signal map randomly permuted: mask magnitude and sparsity preserved, symbolic content destroyed. |
| `latent_subspace` | `ℒ_pred = ‖P_V(ẑ_{t+1} − z_{t+1})‖² + β‖P_V^⊥(·)‖²`, `β < 1`. |
| `sample_prio` | Batch loss reweighted by `edge_rarity(q_t, a_t) × (1 + κ · failure_flag_t)`. |
| `input_patch` | JEPA mask distribution conditioned on `q_t`; attribute predicates to input patches (integrated gradients) and bias the mask sampler toward implicated regions on a curriculum. Most plumbing; do last. |

Keep `SIGReg(Z)` unmodified in every arm — it is what prevents latent collapse, and masking
must not be allowed to interact with it. **This is load-bearing given §12.1.**

### 11.3 Implementation
Each arm subclasses one `MaskedTrainer` with a single overridden
`compute_loss(batch, signals)`. Signals arrive from a precomputed per-sample side table
built once against the frozen automaton, never recomputed in the inner loop —
reproducibility matters more than the saving.

---

## 12. Phase I — Evaluation (rewritten)

### 12.1 Why the obvious offline proxy is rejected

The tempting substitute for planning success is: run CEM in latent space, report
`‖ẑ_H − z_g‖`. **Do not do this.** It measures CEM's optimisation success against the
model's own objective, not the model's accuracy. A model that contracts its latent scale in
dynamically hard directions scores *better*.

This is not a hypothetical weakness — it is a bias pointing directly at the intervention
under test. The `latent_subspace` arm explicitly downweights `P_V^⊥`, i.e. trains the model
to care less about certain directions. It would score an improvement on a self-scored
latent-distance metric essentially by construction, with no capability gain. And the
`shuffled` control would not catch it, because shuffled also downweights *some* subspace.
The proxy defeats the experiment's only safeguard.

Every metric below is adjudicated against **real held-out data**, and none is self-scored.

### 12.2 Primary metric — offline plan ranking

Reframe from "did the plan succeed" to "can the model's objective identify a correct plan
among distractors."

For each held-out segment of the scoped family:
1. Set `z_g = encode(o_{t+H})` from the real future frame.
2. Build a candidate set: the **true** action sequence `a_{t:t+H}`, plus `K = 63` negatives —
   sequences from other episodes, time-shifted sequences from the same episode, and
   perturbed variants of the true one (include hard negatives deliberately; easy negatives
   inflate the score).
3. Score each candidate by the model's predicted terminal distance to `z_g`.
4. Record the rank of the true sequence.

Report **MRR** and **top-1 / top-5 accuracy**. `n_eval_segments = 500`, `H` swept over
`{5, 10, 20}` letters (capped at `L_max`).

A contracted or collapsed model ranks at chance here. There is no way to win by shrinking
the latent — which is exactly the property §12.1 requires.

### 12.3 Secondary metrics (report, do not optimise)
- **Reachability AUC.** Given `(ζ_t, z_g)`, discriminate reachable-within-`H` (real
  positives at horizon `H`) from not (other episodes, far horizons). Tests the implicit
  reachability structure a planner depends on.
- **Real-rollout predicate-trace fidelity** at `H ∈ {5,10,20}`.
- **Multi-step latent prediction error** against real encoded futures.
- **Automaton state count on re-extraction** after training. Report only; it is
  self-referential and must not be optimised.

### 12.4 Statistics — the part that decides whether any of this means anything
- 5 seeds per arm, **paired** by seed against `control`.
- Report mean paired difference with bootstrap 95% CI, not point estimates. Five epochs is
  a small intervention; effects will be comparable to seed variance.
- **Pre-register** the primary metric and comparison in `artifacts/preregistration.md`,
  committed before the first training run. Include the §2.3 normalisation caveat there.
- **The `shuffled` arm is the interpretive key.** If `latent_subspace` beats `control` but
  not `shuffled`, then non-uniform masking helped and the symbolic structure contributed
  nothing. That is a legitimate and interesting finding. Report it prominently; do not bury it.

### 12.5 Reporting
`artifacts/results.md`: per-arm table; paired-difference plot with CIs; the
`(|S|, n_states, fidelity)` Pareto front from Phase E; the `k`-sweep horizon curves from
A.5; and a Graphviz automaton diagram with states annotated by their most frequent decoded
frames.

---

## 13. Configuration

```yaml
scope:
  episode_ids_file: artifacts/scope_episode_ids.json   # from M0.5
  min_episodes: 200

oracle:
  window_N: 3                # confirmed by audit
  segment_len_k: null        # from A.5 sweep — do NOT set by hand
  horizon_cap: null          # L_max(k*), from A.5
  trace_fidelity_thresh: 0.85
  cache_dir: artifacts/oracle_cache

alphabet:
  k_sweep: [2, 4, 8, 16]
  n_codes: 12
  gripper_special_case: true
  n_reset_symbols: 4

predicates:
  n_candidates: 64
  max_active: 8
  slowness_weight: 1.0
  predictability_weight: 1.0

learner:
  algorithm: TTT
  max_states: 200
  target_states: 50
  eq_oracle:
    kind: hybrid
    word_distribution: ngram_from_real   # NOT uniform
    epsilon: 0.05
    delta: 0.05
    n_refinement_rounds: 20

search:
  budget_extractions: 40

training:
  epochs: 5
  arms: [control, shuffled, latent_subspace, sample_prio, input_patch]
  seeds: [0, 1, 2, 3, 4]

eval:
  primary_metric: plan_ranking_mrr
  n_eval_segments: 500
  n_negatives: 63
  horizons: [5, 10, 20]
```

---

## 14. File layout

```
symbolic_lewm/
  configs/symbolic.yaml
  scope/
    task_families.py        # M0.5  grouping, ranking, selection
  oracle/
    latent_oracle.py        # A.1-A.2
    cache.py                # A.3
    horizon.py              # A.4-A.5  L_max and the k-sweep
  alphabet/
    segments.py             # B.1-B.2
    codebook.py             # B.3
    validate.py             # B.4-B.5  coverage + plan-ranking expressiveness
  predicates/
    vq_bottleneck.py        # C.1(i)
    slow_features.py        # C.1(ii)
    bisim_metric.py         # C.1(iii)
    pool.py                 # C.2
  learning/
    sul.py                  # D.1
    eq_oracle.py            # D.2  ngram word distribution + refinement
    extract.py              # D.3-D.4
    diagnostics.py          # D.5
  search/{objective,greedy,cegar}.py
  gate/acceptance.py
  signals/
    relevance_subspace.py   # G.2
    edge_stats.py           # G.3-G.4
  training/
    masked_trainer.py
    arms/{control,shuffled,latent_subspace,sample_prio,input_patch}.py
  eval/
    plan_ranking.py         # I.2  PRIMARY
    reachability.py         # I.3
    stats.py                # I.4
    report.py               # I.5
  artifacts/
  tests/
```

---

## 15. Milestones — stop for review at each

| # | Deliverable | Gate |
|---|---|---|
| M0 | `repo_audit.json` | ✅ complete |
| **M0.5** | `scope_report.md`, chosen task family | **Kill criterion M0.5** |
| M1 | `k`-sweep + `L_max(k*)` | Kill criterion A |
| M2 | Alphabet + plan-ranking expressiveness | Kill criterion B |
| M3 | Predicate pool + montages | Sanity: can a human name any of them? |
| M4 | First successful extraction, any size | Pipeline works end to end |
| M5 | Search complete, Pareto front | Kill criterion F |
| M6 | Accepted machine ≤ 50 states | Acceptance gate |
| M7 | Signals extracted, `r/d` reported | If `r ≈ d`, say the subspace arm is vacuous |
| M8 | 5 arms × 5 seeds trained | Anti-feedback protocol verified |
| M9 | `results.md` | Shuffled-control comparison reported prominently |

M0.5, M1, M2 and M5 are the likely termination points. Each is cheap relative to what
follows it. That ordering is deliberate.

---

## 16. Testing requirements

- **Oracle purity.** `evaluate(u)` identical across calls, cache states, batch groupings.
  Property test over random words.
- **Prefix consistency.** Fresh `evaluate(u)` equals the value from extending any cached
  prefix rollout.
- **Known-automaton round trip.** Replace `g` with a hand-built finite state machine lifted
  into `R^d` (one-hot plus noise, 3-window wrapped). The pipeline must recover it exactly.
  **The most valuable test in the suite — write it before Phase D.**
- **Alphabet determinism.** Symbol→segment decoding stable across runs given a codebook hash.
- **Arm isolation.** With masking disabled, every arm reproduces `control` bit-for-bit under
  a fixed seed.
- **Metric non-gameability.** Feed the plan-ranking metric a deliberately contracted model
  (scale all latents by 0.1). MRR must not improve. If it does, the metric is broken —
  fix it before trusting any result.

---

## 17. Prior work

- Weiss, Goldberg & Yahav, *Extracting Automata from RNNs Using Queries and
  Counterexamples* (ICML 2018) — `L*` with a network as membership oracle; the
  abstraction-refinement equivalence oracle of §7.2(b). Closest precedent.
- Koul, Fern & Greydanus, *Learning Finite State Representations of Recurrent Policy
  Networks* (ICLR 2019) — quantised bottleneck → Moore machine; basis for §6.1(i).
- Danesh, Fern et al., *Re-understanding Finite-State Representations of Recurrent Policy
  Networks* (ICML 2021) — minimised machines can be deceptively small.
- Ferns, Panangaden & Precup (bisimulation metrics); Castro et al. (MICo) — §6.1(iii).
- Vaandrager et al. / LearnLib — mature active automata learning practice.
- Khazatsky et al., *DROID* (RSS 2024) — the dataset; note 564 scenes, 86 buildings,
  teleoperated, no simulator.
- Konidaris, Kaelbling & Lozano-Pérez — which symbols suffice for sound planning, if this
  extends to operator-level models.

**No prior work applies this to a JEPA-style latent world model, and none does it on
real in-the-wild data without a simulator.** DROID is harder than every precedent above on
three axes simultaneously: continuous 7-DoF alphabet, no ground-truth predicates, and no
executable environment. Plan accordingly, and treat a clean negative result as a successful
outcome.
