# Gradient-only selection in the live pruning loop

The offline ranker study ([ranker.md](ranker.md)) found that ranking actions by their units' gradient scores was as
good as anything it learned. This experiment runs that selector in the live loop: prune, gate, keep or roll back.
It measures how much of Qwen3-1.7B the selector can prune within the 3-point worst-category limit, checked on
questions that no earlier experiment used.

* Code: [`capprune/gradient_live.py`](../capprune/gradient_live.py) (the loop, method kind `ranked_search`) and
  [`capprune/gradient_audit.py`](../capprune/gradient_audit.py) (freeze, stall diagnostic, audit, report)
* Configs: [`configs/gradient_live.yaml`](../configs/gradient_live.yaml) and
  [`configs/gradient_live_smoke.yaml`](../configs/gradient_live_smoke.yaml)
* Outputs: `~/cap-prune-runs/gradient-live/` (run) and `~/cap-prune-runs/gradient-live/_analysis/` (analysis,
  `report.md`). The Gemini, Ollama, fixed-state and ranker outputs are only read.

## The live run

Seed 0, starting from dense Qwen3-1.7B. At each parent checkpoint:

1. **Profile** with the ranker's protocol: fixed calibration subsets, with activations and gradients at every
   parent. The profile is saved before it is used.
2. **Candidates** come from the ranker's generator:
   * the greedy-risk set, the lowest z_sens set and the lowest z_grad set
   * seven depth windows
   * two seeded random sets

   Every action removes 4 units × 5 blocks × 64 = **1,280 MLP channels**. Illegal actions are dropped and logged:
   a unit without room under the 0.875 cap, an increment outside [0.01, 0.10], or a different step size.
3. **Rank** with the offline gradient-only rule: lowest mean z_grad first, ties broken by generator index. The
   whole ordered shortlist is logged.
4. **Build** the top-ranked trial with the structured MLP-channel operator (`raise_unit`).
5. **Gate** it with the locked threshold-aware capability gate:
   * 3 points per MMLU category
   * cumulative paired loss against the original dense model
   * α = 0.05 per side, minimum sample and batch size 100
   * questions from the ranker's label set (5,214); budget = the whole set, so the bound is exact when it runs out
   * a fresh question order for every attempt
6. **Decide.**
   * ACCEPT: the trial becomes the next parent.
   * REJECT or INCONCLUSIVE: the parent is restored bit for bit (digest-checked) and the run **ends**. This is a
     *candidate failure*.

**Caps** (pilot values; set in the config):
* 10 proposals.
* 30,000 gate questions in total. The cap is enforced across attempts: if the gate's next batch would go over it,
  that batch is not scored, the parent is restored and the run ends with a **budget stop** (`question_cap`). A
  budget stop is logged and counted separately from candidate failures. If what is left of the cap cannot cover
  the next attempt's first sample (400 questions), the run stops before that attempt.

**Paired scoring.** The gap diagnostic showed that bf16 batch composition alone flips about 1 % of answers.
Every batch the gate reveals is therefore scored on a second, never-modified dense copy of the model and on the
trial, with the same rows and the same batch layout. The dense copy is checked to be bit-identical to the
original weights. This dense rescoring is overhead of the reference: it is reported separately and does **not**
count toward the question cap.

**Logged at every attempt** (`journal.jsonl`):
* the ordered shortlist, and any illegal actions that were dropped
* the chosen action
* proposed and actual channels removed
* the gate decision, with per-category estimate, bounds, status and broken/fixed counts
* questions used by this attempt and in total
* the order seed
* GPU seconds: profiling, trial build, trial scoring, dense-reference scoring, gate wall time and rollback
* parent and rollback verification

**Checkpoint recovery.** This is the existing `RunContext` machinery: sealed checkpoints after every attempt,
content-addressed masks, the supervisor and the science-hash guard. Profiles and paired batches are cached
under deterministic keys, so a resumed run replays to the identical trajectory. The smoke test kills the worker
four times to check this.

## Stall diagnostic

This runs after the run has finished and its final checkpoint has been frozen (`gradient-live freeze` records
the state id and mask SHA-256 and copies the mask). The stalled parent is the frozen checkpoint, because the
failed trial was rolled back.
* The run's own saved profile rebuilds the identical shortlist. This is checked against the journal.
* **Every** candidate is gated with the same gate and the stalled attempt's question order.
* Dense answers are shared across candidates batch by batch.
* The live top-1's batches are read from the run's cache (read-only), so its decision must reproduce the live
  one.
* **Reported:** the rank of the first passing candidate, how many pass, and which categories blocked the rest:
  *confirmed* violations (lower bound above 3 points, i.e. REJECT) separately from *unresolved* ones
  (INCONCLUSIVE).
* **Costs** are reported separately from the live run.

The rule was fixed in the config before the run (`near_top_rank: 3`):
* the first pass is at rank ≤ 3 → test a budgeted top-k fallback
* the first pass is at a lower rank → investigate ranking signals
* every candidate is rejected with a confirmed violation → test a smaller step on the same parent

## Audit data

`gradient-live audit-data` checks every prepared MMLU split against the artifacts of the earlier experiments:
* the ranker's split
* the fixed-state pool scores
* the experiment-1 pilot items
* the gate-tuning items
* the earlier final audits (5-shot, so their MMLU-dev exemplars count as used)
* the gap diagnostic

**No MMLU question is unused**:

| split | questions | used before | unused |
|---|---|---|---|
| probe (pruning-time pool) | 5,614 | 5,614 | 0 |
| final (held-out) | 8,428 | 8,428 | 0 |
| validation (gate tuning) | 1,531 | 1,531 | 0 |
| dev (5-shot exemplars) | 285 | 285 | 0 |

**Fresh audit set: MMLU-CF validation** (`microsoft/MMLU-CF`, revision `c25b89a9…`, per-field files pinned by
SHA-256).
* MMLU-CF is a decontaminated MMLU-style benchmark: 4-choice questions in 14 fields.
* The fields map onto MMLU's official subcategory → category table:
  * STEM: Biology, Chemistry, Computer Science, Engineering, Math, Physics
  * Humanities: History, Law, Philosophy
  * Social Sciences: Economics, Psychology
  * Other: Business, Health, Other
* No earlier experiment touched it.
* Nine items whose question text equals an MMLU question are excluded. All nine are generic stems such as "Which
  statement is false?".
* That leaves **9,991 questions**: STEM 4,657, Humanities 1,650, Social Sciences 861, Other 2,823.
* It is a different question source from the gate's pool, so it tests transfer to new items *and* a modest shift
  in distribution.

**Exploratory, reused set: the MMLU held-out final split** (8,428). This split is in-distribution, but the earlier
audits and the gap diagnostic have already used it, so its results are labelled exploratory.

## Audit

The checkpoint is the frozen one, verified by SHA-256 throughout. The protocol is the live gate's:
* 0-shot, `max_seq_len` 2048, a 32,768-token batch budget, masked bf16 weights
* each category in a seeded order, in batches of 100 (the gate's batch size)
* dense and pruned scored on **identical batches**

Reported:
* the fraction of all model parameters physically removed, counted on the materialised model
* MLP sparsity
* overall paired loss
* each category's paired loss, with 95 % stratified paired-bootstrap intervals and simultaneous (Bonferroni over 4
  categories) one-sided 95 % upper bounds
* whether the point estimates and the intervals support the 3-point limit

## Results: RTX 5090 pilot, seed 0 (2026-11-23)

Full report: `~/cap-prune-runs/gradient-live/_analysis/report.md`, which includes every attempt, each attempt's
ordered shortlist, and all bounds.

**Bottom line.** All 10 attempts passed the gate.
* **Size:** 7.44 % MLP sparsity (12,800 of 172,032 channels). That is **4.57 % of all parameters physically
  removed** (78.6 M of 1.72 B, counted on the materialised model).
* **Fresh audit (MMLU-CF):** worst category Other +0.21 points, overall +0.08. Every simultaneous 95 % upper bound
  is ≤ 1.63 points, so the audit supports the 3-point limit.
* **Why it stopped:** the 10-proposal cap, a budget stop. The selector never stalled, and the offline probe of the
  next attempt passes as well.

### Live run

| attempt | parent MLP sparsity | action (rank 1 = lowest z_grad) | gate | questions (cumulative) | worst category on the pool |
|---|---|---|---|---|---|
| 1 | 0.00 % | layers 22–25 | accept | 1,300 (1,300) | STEM +0.75 |
| 2 | 0.74 % | 21–24 | accept | 1,300 (2,600) | Social Sciences −0.33 |
| 3 | 1.49 % | 22–25 | accept | 1,600 (4,200) | Social Sciences +0.25 |
| 4 | 2.23 % | 21, 23–25 | accept | 2,000 (6,200) | Humanities +0.83 |
| 5 | 2.98 % | 22–25 | accept | 1,500 (7,700) | STEM +0.40 |
| 6 | 3.72 % | 20–23 | accept | 2,800 (10,500) | STEM +1.25 |
| 7 | 4.46 % | 20, 24–26 | accept | 1,900 (12,400) | STEM +1.29 |
| 8 | 5.21 % | 21–23, 25 | accept | 2,700 (15,100) | STEM +0.88 |
| 9 | 5.95 % | 21–24 | accept | 2,600 (17,700) | STEM +1.44 |
| 10 | 6.70 % | 20, 23–25 | accept | 2,500 (20,200) | STEM +0.88 |

* **Checks.** Every attempt removed exactly 1,280 channels, and no candidate was illegal. Parent digests were
  verified at every attempt. The run never restarted.
* **Where the pruning went.** Everything was taken from layers 20–26: L23 lost 2,880 of 6,144 channels, L24 2,560,
  L22 and L25 2,240 each.
* **Agreement with the offline study.** At the first two parents (the dense model and ranker trajectory T2's
  first parent), the live z_grad scores match the ranker's within 5e-7 and the top-1 pick is the same. The ranker's
  cached exact label-set losses put those picks at +0.09 (2nd best of 13) and −0.06 (best of 11).
* **Cost.** 375 s of search wall time:
  * 52 s profiling
  * 160 s trial scoring
  * 160 s dense-reference rescoring (not counted toward the cap)

  That works out to 2,020 gate questions per decision.

### Stall diagnostic

**There was no live stall**, so none of the three stall rules applies.

As a substitute, the frozen checkpoint was profiled offline with the live protocol, and its full 11-action
shortlist was gated exactly as attempt 11 would have been: same gate, that attempt's question order.
* **10 of 11 pass**, including the gradient top-1 (layers 20–22 and 26, accepted after 1,600 questions).
* The only failure is random set 2 (layers 0, 5, 6, 20): a confirmed Other violation of +3.42 points (exact at
  exhaustion).
* Cost, kept separate from the live run: 33,367 gate questions plus 4,703 dense-reference questions, 314 GPU s.

**Next step:** raise the caps in a new run directory to reach the first stall, then rerun `diagnose`. The live
pilot stayed within its 10-proposal cap.

### Audit of the frozen checkpoint (`03810f5b84f494f6`, 0-shot gate protocol, identical dense/pruned batches)

Losses are paired, in accuracy points. Intervals are 95 % paired stratified bootstrap (10,000 draws). The upper
bound is simultaneous over the four categories (Bonferroni, one-sided 98.75 % each).

| category | fresh: MMLU-CF val, loss [95 % CI] | upper bound | exploratory: MMLU held-out (reused), loss [95 % CI] | upper bound |
|---|---|---|---|---|
| STEM | +0.00 [−0.56, +0.56] (n 4,657) | +0.62 | +0.48 [−0.48, +1.43] (n 1,892) | +1.59 |
| Humanities | +0.12 [−0.85, +1.03] (n 1,650) | +1.15 | −0.21 [−0.85, +0.42] (n 2,824) | +0.53 |
| Social Sciences | +0.00 [−1.39, +1.39] (n 861) | +1.63 | −0.16 [−0.87, +0.54] (n 1,847) | +0.65 |
| Other | +0.21 [−0.43, +0.85] (n 2,823) | +0.99 | +0.32 [−0.54, +1.18] (n 1,865) | +1.29 |
| overall | +0.08 [−0.30, +0.45] | +0.40 | +0.07 [−0.32, +0.46] | +0.39 |

* **Dense accuracy:** 49.6 % on MMLU-CF and 55.3 % on MMLU held-out. The latter matches the gap diagnostic's
  55.5 %, which used another batch layout.
* **Masked vs. materialised weights:** they agree on 255 of 256 fresh items.
* **Both audits:** the point estimates meet the 3-point limit, and so do all simultaneous upper bounds.
* **The MMLU column is exploratory:** the earlier audits and the gap diagnostic had already used those questions.

### Limitations

* **One seed and one trajectory.** The gate's per-candidate guarantee does not cover the search as a whole. The
  fresh audit is the check on the final result.
* **The fresh set is a different source.** MMLU-CF's fields map onto MMLU's categories, but its questions are
  harder (dense 49.6 % vs. 55.3 %) and are not MMLU's own. It shows that the losses carry over to new items under
  a modest change in distribution. It is not a resample of MMLU.
* **The audit uses the gate's 0-shot protocol**, as specified. The earlier 5-shot final audits are not directly
  comparable.
* **Speed was not measured.** Removing 4.6 % of parameters gave no measurable speed-up in earlier pilots.

## Follow-up (2026-11-23): materialised verification and the extended search

Reports:
* `~/cap-prune-runs/gradient-live/_analysis/audit_materialized/audit_materialized.md` (pilot checkpoint, materialised)
* `~/cap-prune-runs/gradient-live-ext/_analysis/report.md` (extended run)

The pilot's own outputs are unchanged: all 50 of its files were hash-checked before and after.

### 1. The pilot checkpoint as it would be deployed

`gradient-live audit-materialized` checks two things before scoring: the frozen id `03810f5b84f494f6`, and the
mask's SHA-256 against both the config and `frozen.json`. It then materialises the checkpoint with
`pruning/materialize.py`.

**Size checks:**
* 1,720,574,976 → 1,641,931,776 parameters: **78,643,200 removed (4.571 %)**. This is exactly the masks' count of
  12,800 channels × 6,144 parameters per channel.
* Every MLP's intermediate size matches its mask.
* Every kept gate/up row and down column is bit-identical to the masked model's.

**Scoring setup.** Dense, masked and materialised models were scored on the original audit's batches: the same
ids, the same seeded order, batches of 100, the 0-shot prompt, bf16 and a 32,768-token budget.

**Dense reference.** The rescored dense and masked answers reproduce the original audit's cache on 9,991 of 9,991
MMLU-CF items, in 102 of 102 batches with the same ids and budget. The provenance therefore matches exactly.

| MMLU-CF val (9,991) | masked loss | materialised loss [95 % CI] | materialised simultaneous one-sided 95 % UB | prediction agreement |
|---|---|---|---|---|
| STEM | +0.00 | +0.00 [−0.56, +0.56] | +0.62 | 4,642 / 4,657 |
| Humanities | +0.12 | +0.12 [−0.85, +1.03] | +1.15 | 1,646 / 1,650 |
| Social Sciences | +0.00 | +0.23 [−1.16, +1.63] | +1.86 | 855 / 861 |
| Other | +0.21 | +0.11 [−0.53, +0.74] | +0.85 | 2,814 / 2,823 |
| overall | +0.08 | +0.07 [−0.30, +0.44] | +0.38 | 9,957 / 9,991 (99.66 %) |

**The materialised model meets the 3-point limit**, and the simultaneous bounds support it. The worst category is
Social Sciences at +0.23. On the exploratory MMLU held-out set, the worst category is Other at +0.48 with an
upper bound of +1.45.

### 2–3. Extended search and its first stall

**Registration.** Before launch, `gradient-live register` recorded:
* the caps: 30 proposals and 100,000 paired gate questions, including the prefix
* the replay source
* the audit-set decision

**Replay instead of branching.** Branching a finished run directory is not supported: the science-hash guard
refuses to resume it with new caps, and the pilot had to stay untouched. The extension therefore replays from dense
with seed 0. Attempts 1–10 reproduced the pilot exactly:
* parents, ordered shortlists and chosen actions
* trial masks and order seeds
* decisions, question counts and per-category counts
* the final mask content

If any of these had differed, the loop would have stopped with `prefix_mismatch` before extending. Mask files are
compared by content: safetensors writes metadata keys in an unspecified order, so equal masks can have different
file hashes.

**Where it stopped.** Attempts 11–15 were accepted; attempts 14 and 15 only just passed, with STEM certified at
+2.24 and +2.50.

| attempt | parent MLP sparsity | rank-1 action | gate | paired questions (cumulative) |
|---|---|---|---|---|
| 11 | 7.44 % | layers 20–22, 26 | accept | 1,600 (21,800) |
| 12 | 8.18 % | 23–26 | accept | 2,200 (24,000) |
| 13 | 8.93 % | 18, 19, 21, 22 | accept | 2,200 (26,200) |
| 14 | 9.67 % | 20, 23–25 | accept (STEM +2.24, exact) | 4,861 (31,061) |
| 15 | 10.42 % | 17, 21, 22, 26 | accept (STEM +2.50, exact) | 4,461 (35,522) |
| 16 | 11.16 % | 18–20, 26 | **reject: Social Sciences +3.36, exact** | 1,430 (36,952) |

The stop reason is `candidate_rejected`, the first top-ranked failure. The last accepted checkpoint
`e65e9461315ef207` (11.16 % MLP sparsity) was frozen before any diagnostic or audit ran.

**Cost** (synchronised GPU seconds):

| scope | paired questions | item–model evaluations run | profiling | pruning | dense scoring | candidate scoring | search |
|---|---|---|---|---|---|---|---|
| pilot | 20,200 | 40,400 | 51.9 | 0.18 | 159.8 | 160.2 | 375.0 |
| replayed prefix | 20,200 | 40,400 | 51.0 | 0.12 | 158.0 | 158.6 | 370.8 |
| incremental (11–16) | 16,752 | 33,504 | 30.6 | 0.05 | 136.9 | 137.2 | 306.8 |
| cumulative | 36,952 | 73,904 | 81.6 | 0.17 | 294.9 | 295.7 | 677.5 |

No pair came from a cache: this was a fresh run directory with a single worker start.

**First stall.** The stalled parent's 11-action shortlist was gated offline with the same gate and attempt 16's
question order. The top-1's decision was reproduced from the run's cache.
* **3 of 11 pass:** rank 2 (layers 20–23, STEM +2.58 exact), rank 3 (24–27) and rank 5 (greedy).
* **8 are rejected** with confirmed violations: STEM 5, Social Sciences 3. None are inconclusive.
* **Recommendation, from the rule fixed before the pilot:** test a **budgeted top-2 fallback** in a separate run.
  Reaching the first pass costs 6,091 questions at this parent.
* **Diagnostic cost:** 29,734 paired questions and 267 GPU s, reported separately.

### 4. Audit of the extended checkpoint (materialised; exploratory)

No suitable never-used set exists. The criteria and candidates were fixed in `configs/gradient_live_ext.yaml`
before launch:
* every MMLU question has been used, and MMLU derivatives reuse those questions
* MMLU-CF's test split is not public
* MMLU-Pro has no non-MMLU Humanities items, and 83 % of its items have 10 options

Both audits below are therefore exploratory.

**Size:** 117,964,800 parameters removed, **6.856 %** (11.16 % MLP sparsity), matching the masks exactly.

| | MMLU-CF val: loss [95 % CI] | UB | MMLU held-out: loss [95 % CI] | UB |
|---|---|---|---|---|
| STEM | +1.57 [+0.58, +2.56] | +2.71 | **+3.65** [+2.11, +5.18] | +5.39 |
| Humanities | +1.39 [−0.18, +3.03] | +3.21 | +1.38 [+0.28, +2.51] | +2.62 |
| Social Sciences | +0.58 [−1.63, +2.79] | +3.14 | +1.79 [+0.49, +3.14] | +3.30 |
| Other | +2.16 [+0.99, +3.33] | +3.51 | +1.23 [−0.21, +2.63] | +2.84 |
| overall | +1.62 [+0.97, +2.28] | +2.17 | +1.95 [+1.29, +2.60] | +2.49 |

UB = simultaneous one-sided 95 % upper bound.

**The extended checkpoint is not supported.**
* On MMLU-CF the point estimates are below 3 points, but three simultaneous upper bounds exceed it.
* On MMLU held-out, STEM's point estimate is above the limit.

The last five accepts pushed STEM close to the limit on the gate pool, and that margin did not carry over to other
questions. This repeats the gap diagnostic's finding: a certificate on the reused pool is not a certificate on new
items. Choosing an intermediate checkpoint by these audits would select on audit data, so no checkpoint was picked
that way.

## Follow-up 2 (2026-11-23): compression boundary, gate-margin calibration, top-2 fallback

Everything here is **exploratory**. No untouched audit set exists (see the registration records), so the boundary
map, the margin calibration and the policy audits all use the two already-inspected sets: MMLU-CF validation and
MMLU held-out. The completed pilots were only read; their 56 and 57 files were hash-checked before and after.

Code and outputs:
* Code: [`capprune/gradient_boundary.py`](../capprune/gradient_boundary.py), plus top-k fallback and `gate_margin` in
  `gradient_live.py`.
* `~/cap-prune-runs/gradient-boundary/report.md`: boundary map and calibration.
* `~/cap-prune-runs/gradient-compare/report.md`: the A/B/C table.

### 1. Compression boundary (A's trajectory, materialised, matched 0-shot)

**How the states were recovered:**
* Attempts 11 and 12 were replayed from attempt 10's frozen mask with the run's persisted profiles. Their mask hashes
  match the journal.
* Attempts 13–15 were replayed as well, and each replay equals its saved mask.
* The rank-2 candidate of attempt 16 (layers 20–23 from `e65e9461315ef207`) was rebuilt. Its mask hash is present
  in the stall diagnostic's cache.

**Reuse:**
* Dense answers came from the two earlier materialised audits, whose dense scorings agree item for item.
* The 4.57 % and 6.86 % anchors reused all 18,419 of their materialised answers (mask hashes matched).
* Every other state cost 18,419 materialised evaluations, about 110 s.

| state | params removed | MMLU-CF: worst category | MMLU-CF: largest UB | MMLU held-out: worst category | MMLU held-out: largest UB | all UB ≤ 3 |
|---|---|---|---|---|---|---|
| attempt 10 | 4.57 % | Soc +0.23 | +1.86 | Other +0.48 | +1.45 | yes |
| attempt 11 | 5.03 % | — (all ≤ 0) | +1.51 | Other +0.16 | +1.18 | yes |
| attempt 12 | 5.49 % | Soc +0.70 | +2.67 | STEM +0.85 | +2.06 | **yes** |
| attempt 13 | 5.94 % | Soc +1.51 | **+3.48** | STEM +0.63 | +1.90 | no (MMLU-CF Social Sciences interval) |
| attempt 14 | 6.40 % | Other +2.52 | +3.95 | **STEM +3.75** | +5.55 | no |
| attempt 15 | 6.86 % | Other +2.16 | +3.51 | **STEM +3.65** | +5.39 | no |
| attempt 16, rank 2 | 7.31 % | Other +2.30 | +3.95 | **STEM +4.23** | +6.03 | no |

UB = largest simultaneous one-sided 95 % upper bound.

**Reading the map:**
* The last state the audit supports is **5.49 %** (attempt 12).
* Attempt 13 fails only because one interval is wide: its point estimates are ≤ +1.51 everywhere, but MMLU-CF
  Social Sciences has just 861 items.
* The point estimates first go over the limit at **6.40 %** (attempt 14).
* The rank-2 candidate adds **+0.50 [+0.23, +0.77]** pts of overall loss over its parent on MMLU held-out, and
  +0.15 [−0.12, +0.40] on MMLU-CF.

### 2. Margin calibration: inconclusive

The rule was recorded before the new states were scored: take the smallest margin in {0, 0.5, 1.0, 1.5, 2.0} for
which every development state passing the gate at tolerance 3 − m meets the audit criterion (all simultaneous upper
bounds ≤ 3 on both sets), with at least one pruned state passing.

* At margin 0 the gate replays reproduce all 7 live decisions.
* Attempts 14, 15 and the rank-2 candidate are blocked from margin 1.0 upward.
* Attempt 13 passes the gate even at tolerance 1.0, yet fails the audit criterion. Its failure is the audit
  interval described above, which no pruning-time margin can see.
* So no margin qualifies: the calibration is **inconclusive**. Three states (10–12) pass and meet the criterion at
  every margin, and none are blocked that meet it.
* As pre-declared, arm C used the largest grid margin, 2.0 pts (pruning-time tolerance 1.0), and is **exploratory**.
* Cost: 19,550 new evaluations; 106,215 pairs were read from the runs' caches.

### 3–4. Live policies (same dense start, seed 0, generator, step and caps)

Before launch, `gradient-live register` recorded each arm's caps, fallback setting and margin; it checks the margin
against the calibration file. A is the extended top-1 run, reused because its settings match.

* B replayed A exactly through attempt 16, final mask included.
* In top-2, rank 2 is tried only after rank 1 fails and the parent is restored (digest-verified). It reuses the
  parent's profile and shortlist, gets its own attempt number and question order, and counts against both caps.

| | A: top-1, gate 3.0 | B: top-2, gate 3.0 | C: top-2, gate 1.0 (exploratory margin) | B − A (fallback) | C − B (stricter gate) |
|---|---|---|---|---|---|
| stop | rank 1 rejected (#16) | top-2 exhausted (#21) | top-2 exhausted (#10) | | |
| accepted steps (rank-2 rescues) | 15 (0) | 17 (2 of 3 triggers) | 7 (1 of 2 triggers) | | |
| verified params removed | 6.86 % | 7.77 % | **3.20 %** | +0.91 pp | −4.57 pp |
| MMLU-CF: worst category [95 % CI] / largest UB | Other +2.16 [+0.99, +3.33] / +3.51 ✗ | Other +2.23 [+1.03, +3.44] / +3.65 ✗ | Other +0.14 [−0.35, +0.64] / +1.05 ✓ | overall −0.01 [−0.28, +0.27] | overall −1.55 [−2.21, −0.88] |
| MMLU held-out: worst category [95 % CI] / largest UB | STEM +3.65 [+2.11, +5.18] / +5.39 ✗ | STEM +4.12 [+2.54, +5.71] / +5.92 ✗ | Other +0.05 [−0.64, +0.75] / +0.86 ✓ | overall **+0.39 [+0.12, +0.65]** | overall −2.41 [−3.11, −1.72] |
| paired gate questions | 36,952 | 51,274 | 33,206 | +14,322 | −18,068 |
| item–model evaluations run | 73,904 | 102,548 | 66,412 | +28,644 | −36,136 |
| end-to-end search (s) | 678 | 962 | 597 | +284 | −365 |
| terminal stall: passing ranks | 2, 3, 5 of 11 | 5, 6 of 11 | 3 of 11 | | |

The overall-loss differences are paired on identical audit items.

* The fallback's own candidates cost B 11,222 questions, 22,444 evaluations and 196 s, and C 5,561 questions,
  11,122 evaluations and 90 s.
* The stall diagnostics cost 27,347 (B) and 22,716 (C) paired questions, reported separately from the runs.

**Conclusions (exploratory).**
1. **Top-2 did not extend the compression the audit supports.** It pushed the pruning-time-certified frontier from
   6.86 % to 7.77 %, at +39 % questions and +42 % search time. But B's checkpoint fails the audit more clearly than
   A's (held-out overall +0.39 pts worse). Neither top-3.0 arm's final checkpoint is supported. The
   audit-supported point on this trajectory (5.49 %) was found only by inspecting the audit sets.
2. **The 2-point margin makes the result verifiable but gives up compression.** C's 3.20 % passes the audit with
   room to spare: worst upper bound +1.05. That is less than the 10-proposal pilot's verified 4.57 %, and the
   calibration could not justify a smaller margin.
3. **Ranking misses recur**: rank 1 failed at 3 of B's last 6 proposals and at 2 of C's 10, and the passing actions
   sit at ranks 2–6. But rescuing them added only compression the gate certifies and the audit does not. A custom
   controller trained to find gate-passing actions would optimise the wrong target. The binding problem is that the
   gate's certificate on the reused pool does not carry over to new items. The next steps are a fresh confirmation
   sample or population-level bounds for each accept, and an untouched audit set. A controller makes sense only if
   it is trained against an audit-aligned label.

## Commands

```powershell
python -m pytest tests/test_gradient_live.py -q                                # CPU smoke, ~2 min
python -m capprune gradient-live audit-data --config configs/gradient_live.yaml  # inventory + fresh set (CPU)
python -m capprune supervise                --config configs/gradient_live.yaml  # live run, RTX 5090
python -m capprune gradient-live freeze     --config configs/gradient_live.yaml
python -m capprune gradient-live diagnose   --config configs/gradient_live.yaml  # GPU
python -m capprune gradient-live audit      --config configs/gradient_live.yaml  # GPU
python -m capprune gradient-live report     --config configs/gradient_live.yaml  # -> _analysis/report.md

# follow-up: materialised verification of the pilot checkpoint, then the extended search
python -m pytest tests/test_gradient_live_ext.py -q                                          # CPU smoke, ~2 min
python -m capprune gradient-live audit-materialized --config configs/gradient_live.yaml      # -> _analysis/audit_materialized/
python -m capprune gradient-live audit-data         --config configs/gradient_live_ext.yaml  # inventory (no fresh set)
python -m capprune gradient-live register           --config configs/gradient_live_ext.yaml  # caps, BEFORE the launch
python -m capprune supervise                        --config configs/gradient_live_ext.yaml  # replay 1-10, then extend
python -m capprune gradient-live freeze             --config configs/gradient_live_ext.yaml
python -m capprune gradient-live diagnose           --config configs/gradient_live_ext.yaml
python -m capprune gradient-live audit-materialized --config configs/gradient_live_ext.yaml
python -m capprune gradient-live report             --config configs/gradient_live_ext.yaml  # -> gradient-live-ext/_analysis/report.md
```
