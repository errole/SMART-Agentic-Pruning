# Study 2 plan: DRAFT, NOT REGISTERED (2026-11-27)

> **DRAFT. This plan is not registered, and nothing in it is in force.**
> * No model has been scored for it, and no new model has been sealed.
> * The numbers in §8–§9 are planning estimates from EXPLORATORY data.
> * Registration needs, in order:
>   1. decisions on the open questions (§12);
>   2. the new models sealed;
>   3. this file renamed without "DRAFT";
>   4. the file committed with every registration hash, before any sealed item is read.

**Question.** How much training-free unstructured pruning survives per-domain certification? How does that depend on
model size, and on how sparsity is allocated across layers?

**Built on:**
* [`study2_groundwork.md`](study2_groundwork.md): P1–P7;
* the unstructured feasibility study ([`unstructured_feasibility_report.md`](unstructured_feasibility_report.md));
* paper A's confirmatory study ([`confirmatory_study_plan.md`](confirmatory_study_plan.md)).

**Zeroed weights give no memory or speed gain on standard hardware.** They are still stored and multiplied. Every
"share" below is a count of zeroed weights, not a saving.

## 1. Models

| role | model | revision | parameters | status |
|---|---|---|---|---|
| size ladder | Qwen/Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` | 4,022,468,096 | to be sealed |
| size ladder | Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | 8,190,735,360 | **sealed**, registration `f96e8ef2…`, admissible |
| size ladder | Qwen/Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` | 14,768,307,200 | to be sealed |
| second family | one of P6's candidates (§12, question 2) | P6 | 7.3–8.2 B | to be sealed |
| development only | Qwen/Qwen3-1.7B | `70d244cc…` | 1,720,574,976 | **exploratory only**: its items are seen |

**Variant.** All are post-trained, as Qwen3-8B is. The revisions are the Hub's current commits, read on 2026-11-27.
Weight hashes are pinned when each model is downloaded.

**Before any other scoring, each new model:**
* is sealed with the confirmatory procedure (`confirmatory build-data`), with a new split seed:
  * proposed: 2026112801 for Qwen3-4B, 2026112802 for Qwen3-14B, 2026112803 for the second family;
* passes the admissibility test on its gate pool: dense 0-shot accuracy ≥ 40 % overall and ≥ 30 % in every category.
  * **A model that fails is replaced by amendment** before any of its runs.

**Memory.** One resident copy, with the dense weights restored from disk between levels (P2). The expected peak of
Qwen3-14B in bf16 is about 44 GB of the 96 GB card.

## 2. Data

Carried over from paper A unchanged:
* **Sources:** MMLU test and validation, and MMLU-CF validation, pinned by revision and SHA-256.
* **Deduplication:** 25,432 items.
* **Per-model confirmation sets:** STEM 3,600, Humanities 3,000, Social Sciences 3,000, Other 3,500 (13,100),
  stratified and seeded.
* **Sealing rules:** the confirmation rows live only in `sealed/`, and only `confirm` reads them.
  * It refuses unless the runs are frozen and this plan, recording the registration hash, is committed.

**Disclosure on Qwen3-8B.** Its sealed set has never been read. But its screening pool was scored in the feasibility
study, dense and pruned at 11 levels on 9,927 items. That informed this design: the level grid and the power estimates
of §8. The confirmation data are untouched. The design is not blind to the model.

## 3. Arm U: uniform one-shot Wanda (every model)

**Operator:** Wanda, unstructured, as published.
* |W| × ‖X‖₂, per output row.
* int(columns × s) weights pruned per row (`unstructured_count: floor`).
* Every linear of every block (q, k, v, o, gate, up, down); embeddings and the output head are not pruned.
* Sequential, no weight update.
* The cached-block-input path (P1), whose masks are identical to the original path's (hash-checked).

**Calibration:** seeds 0, 1 and 2, each choosing 128 of the model's 256-window C4 train pool (2,048 tokens per window)
with `calibration_indices(seed, 128)`. The seeds' windows overlap, so their masks are correlated.

**Levels:** 5, 10, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35 and 40 % layer sparsity.
* Each level is pruned from the dense weights, restored from disk, with the checksum verified.
* **The order is fixed by the grid, so there is no screening.**

**Freeze.** Before any sealed item is read, every level of every seed is pruned. Its mask hash and achieved accounting
are written to a freeze list, which is hashed (paper A's `freeze` step). At confirmation, a regenerated mask must match
its frozen hash.

**Stored per scored item:** dense and pruned correctness, **and the four letter log-probabilities** (P5).

## 4. Arm A: agent-allocated sparsity (Qwen3-8B only; optional; costed in §9)

**The base paper's loop,** as implemented by `paper_agent` (P7):
* per-linear increments δ ∈ [0.01, 0.15] chosen by an LLM controller from z-scored Wanda and gradient profiles;
* the perplexity rollback against the parent at τ = 0.15 on C4 validation (32 × 2,048);
* a 50 % target, the agent's stop flag, and at most 40 iterations (patience 10);
* the controller is `gemini-3-flash-preview` at temperature 0.5 (§12, question 3).

**Seeds:** three (0, 1, 2). Each selects the calibration and profile windows, and the controller is sampled.

**Freeze:** each run's accepted checkpoints are frozen in ascending order of zeroed weights, with their mask hashes,
before any sealed item is read.

**Registration fixes every point the paper leaves open, as resolved in `paper_agent`:**
* the unit is a linear;
* δ is absolute;
* per-row Wanda, cumulative masks;
* gradients every third iteration;
* history window 1;
* the count rule (floor).

## 5. Confirmation: paper A's rule, unchanged

**Per run, a fixed sequence over its frozen order** (arm U: ascending levels; arm A: ascending accepted checkpoints).
* **Each test:** the pruned model against the dense model on the sealed 13,100 items (0-shot, identical batches).
* **Bound:** per category, the Clopper–Pearson union bound on the lost and gained counts, with no finite-pool
  correction.
* **Decision:** the IUT across the four categories at α = 0.05. It passes only if every category's one-sided 95 % upper
  bound is ≤ 3 points.
* **Stop:** at the first failure. The claim is the last passing level or checkpoint.

**Every level and checkpoint is also scored, for description only** (paper A, amendment 3).

**Multiplicity.** Each run's claim has a family-wise error ≤ 0.05 within its own sequence. Runs, seeds, models and arms
are not corrected against each other. The hypotheses (§7) are support criteria over those claims, not further tests.

## 6. Endpoints

**Primary:** the **certified zeroed share of all parameters** per run. That is the zeroed weights of the claimed level
or checkpoint divided by all model parameters (embeddings and head included); 0 if nothing is claimed.

**Registered secondaries:**
* **S1, the fixed exact test's claim.**
  * Candidate (ii), grid v2 (P4; [`bound_development_plan.md`](bound_development_plan.md) §8, note of 2026-11-27),
    replaces the CP union bound in the same IUT and fixed sequence.
  * Labelled "secondary bound; method development".
* **S2, the claim at a 2-point limit** (CP union).
* **S3, 5-shot MMLU accuracy at the claimed level** (dense and pruned, on the sealed set).
* **S4, C4 validation perplexity relative to dense,** at the claimed level and at the first failing level. It is
  measured at every level anyway.
* **S5, paired disagreement per level:** lost + gained, per category.
* **S6, certified share against confirmation sample size:** the claim recomputed on seeded subsamples of each sealed
  set at 0.5× and 0.25× the category sizes, as in paper A's S4.

## 7. Hypotheses, support criteria and best attainable outcomes

**Accounting.** At a given layer-sparsity level, the zeroed share of all parameters differs by model, because
embeddings differ (⟨`accounting.json`⟩):

| level | Qwen3-4B | Qwen3-8B | Qwen3-14B | Llama-3.1-8B-I | Olmo-3-7B-I | Granite-3.3-8B-I |
|---|---|---|---|---|---|---|
| (prunable share) | 0.9033 | 0.8480 | 0.8946 | 0.8691 | 0.8874 | 0.9753 |
| 10 % | 9.03 % | 8.47 % | 8.95 % | 8.68 % | 8.86 % | 9.74 % |
| 15 % | 13.55 % | 12.71 % | 13.42 % | 13.03 % | 13.30 % | 14.62 % |
| 20 % | 18.06 % | 16.96 % | 17.89 % | 17.38 % | 17.74 % | 19.50 % |
| 25 % | 22.58 % | 21.20 % | 22.37 % | 21.73 % | 22.18 % | 24.38 % |
| 30 % | 27.09 % | 25.43 % | 26.84 % | 26.06 % | 26.61 % | 29.25 % |
| 40 % (top) | 36.13 % | 33.91 % | 35.78 % | 34.76 % | 35.49 % | 39.01 % |

| | hypothesis (default design) | support criterion | best attainable outcome | expected (§8) |
|---|---|---|---|---|
| **H1** size | certified share non-decreasing 4B → 8B → 14B, and strictly larger at 14B than at 4B, in all three seeds | per seed s: share₄ ≤ share₈ ≤ share₁₄ and share₁₄ > share₄ | attainable, but **only if 8B certifies a strictly higher level than 4B in every seed, and 14B a strictly higher level than 4B** (see the flag below); impossible if 4B certifies the top level | unknown: Qwen3-1.7B certifies through 10 % (exploratory); Qwen3-8B through 20–25 % |
| **H2** level | Qwen3-8B certifies at least the 20 % level in all three seeds | each seed's claim ≥ 20 % | attainable; the best is 40 % (33.91 %) | P(through 20 %) ≈ 1.00 per seed (plug-in) |
| **H3** transfer | the second-family model certifies ≥ 10 % of all parameters in all three seeds | each seed's share ≥ 10 % | attainable; **≥ 10 % needs the 15 % level** for every candidate (the 10 % level gives 8.7–9.7 %); the best is 34.8–39.0 % | unknown (no data on any candidate) |
| **H4** allocation | arm A certifies more than arm U on Qwen3-8B in ≥ 2 of 3 seeds | share(A, seed i) > share(U, seed i) for ≥ 2 of i = 0, 1, 2 | attainable unless arm U's claim is its top level and arm A has no certified checkpoint above it | unknown |

**Flags on the default design** (decisions in §12):
* **H1 as written can fail through accounting alone.**
  * Qwen3-8B's prunable share (0.848) is lower than Qwen3-4B's (0.903) and Qwen3-14B's (0.895). Its untied 151,936 ×
    4,096 embedding and head are 15 % of the model.
  * If 4B and 8B certify the same level, 8B's share is smaller, and "non-decreasing" fails. If 4B and 14B certify the
    same level, 14B's share is smaller than 4B's too.
  * So H1 in shares requires a strictly higher level at 8B than at 4B in every seed, and at 14B than at 4B. One grid
    step up suffices at every level below the top (40 %).
  * Proposed: state H1 on the **certified layer-sparsity level** (non-decreasing, strictly higher at 14B than at 4B),
    and keep shares as the reported endpoint.
* **H2 is close to a foregone conclusion.** On the feasibility data it is certified with probability ≈ 1. It checks
  that the feasibility result replicates on sealed data; it does not discriminate between outcomes.
* **H3's threshold is a grid level.** With the 12-level grid, "≥ 10 % of all parameters" is the same as "certifies the
  15 % level" for every candidate.
* **H4 needs a seed pairing.** Arm A's seed i is paired with arm U's seed i: the same calibration pool draw, with arm A
  also sampling the controller.

## 8. Expected power, from the feasibility estimates

**Method** (⟨`power.json`⟩, `capprune groundwork power`).
* The feasibility per-item paired outcomes are resampled with replacement to the confirmation sizes. Each item keeps its
  outcomes at every level, so levels stay correlated.
* Then paper A's rule is applied: CP union, IUT, 3 points, α = 0.05, fixed sequence. 4,000 replicates.
* **The feasibility estimates are plugged in as true rates.** Their own uncertainty is ignored, so these numbers are
  optimistic near the boundary.
* Only the feasibility levels (5, 10, …, 40 %) can be resampled. The 17.5, 22.5, 27.5 and 32.5 % levels have no data,
  and the probability of passing them lies between the neighbouring levels' values.

**Qwen3-8B** (feasibility n 3,000 / 3,000 / 1,254 / 2,673, resampled to 3,600 / 3,000 / 3,000 / 3,500).
P(certified through the level) for one seed:

| bound, confirmation size | 5 % | 10 % | 15 % | 20 % | 25 % | 30 % | 35 % | 40 % |
|---|---|---|---|---|---|---|---|---|
| CP union, 13,100 (primary) | 1.000 | 1.000 | 1.000 | **1.000** | 0.786 | 0.183 | 0.011 | 0.000 |
| exact score v2, 13,100 (S1) | 1.000 | 1.000 | 1.000 | 1.000 | 0.970 | 0.671 | 0.238 | 0.002 |
| CP union, 0.5 × (S6) | 1.000 | 1.000 | 0.995 | 0.936 | 0.251 | 0.013 | 0.000 | 0 |
| CP union, 2 × | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 | 0.669 | 0.124 | 0 |

* **The likeliest claim on Qwen3-8B is 25 % (21.2 % of all parameters), or 30 % with the exact test.**
* In all three seeds, P(H2) ≈ 1.00 under the plug-in. Across seeds it lies between P³ and P for any level: the seeds'
  masks differ, but they share the sealed items.

**Qwen3-1.7B** (exploratory, 3,000 per category): P(through 10 %) = 1.000, P(through 15 %) ≈ 0.000 with either bound.
* Its 15 % losses (+1.8 to +2.8 points) put the upper bounds above 3.
* P5 traces about 0.9 points of that to a shift in letter preference.
* **It brackets what a small model may do.** Qwen3-4B has no data.

**Qwen3-14B and the second family: no data.** Their power is unknown until they are scored.

## 9. GPU time

**Measured rates** (P1, P2; the RTX PRO 6000; one resident copy, cached Wanda):
* **Qwen3-8B:** prune 109 s; restore 5 s; 0-shot scoring 24.3 ms per item (feasibility); C4 perplexity 14 s.
* **Qwen3-1.7B:** prune 36 s; scoring 7.8 ms per item; perplexity 5 s.
* **Scaled from 8B by prunable parameters:** 4B × 0.52–0.6 and 14B × 1.9.
* **5-shot prompts** are taken to be about 6 × the 0-shot tokens. This is an assumption to check with a token count,
  which needs no scoring.

| per model | sealing + admissibility | dense 0-shot, 13,100 items | 3 seeds × 12 levels (prune + score + ppl) | S3: 5-shot, dense + 3 claims | total |
|---|---|---|---|---|---|
| Qwen3-4B | ~3 min | ~3 min | 36 × ~270 s ≈ 2.7 h | ≈ 1.3 h | **≈ 4.1 h** |
| Qwen3-8B | done | ~5 min | 36 × ~450 s ≈ 4.5 h | ≈ 2.1 h | **≈ 6.7 h** |
| Qwen3-14B | ~9 min | ~10 min | 36 × ~860 s ≈ 8.6 h | ≈ 4.0 h | **≈ 12.9 h** |
| second family (8B class) | ~5 min | ~5 min | ≈ 4.5 h | ≈ 2.1 h | **≈ 6.8 h** |
| **core study** | | | | | **≈ 30 h of GPU** (≈ 21 h without S3) |
| **arm A** (Qwen3-8B, 3 seeds; P7) | | | search ≈ 0.5–1 h + confirmation of 19–36 checkpoints per seed ≈ 5.4–10 h | ≈ 1.6 h | **≈ 7.5–12 h** + 1–3 h of controller waiting + API cost |

* Scoring the sealed sets is about three quarters of the core cost. Pruning was the largest cost before P1 (955 s per
  8B level).
* Downloads before sealing: Qwen3-4B 8 GB, Qwen3-14B 30 GB, the second family about 15–16 GB.

## 10. Carried over from paper A unchanged

* **Data:** the sources, revisions, file hashes, deduplication, the confirmation-set sizes and stratification, and the
  sealing code and rules.
* **Admissibility:** the test and its thresholds.
* **Plan order:** the plan is committed with each registration hash before `confirm` reads a sealed row.
* **Scoring:** 0-shot letter log-probabilities, identical dense and candidate batches (100 per batch, question order
  seed 2026, a 32,768-token budget), and the score cache.
* **The confirmation rule:** the CP union bound, the per-category IUT, 3 points, α = 0.05, the fixed sequence, and
  freeze-before-read.
* **Description:** every level or checkpoint scored; simultaneous Bonferroni bounds.
* **The perplexity gate of arm A:** C4 validation, 32 × 2,048, τ = 0.15 against the parent (paper A's arm (a)).

## 11. Deviations that would require an amendment

* **Any change to:**
  * a model, a revision or a weight hash;
  * a split seed or a sealed set;
  * the confirmation sizes;
  * the admissibility thresholds.
* **Replacing a model** that fails admissibility, or that cannot be downloaded or loaded.
* **The operator:** the count rule, the eligible linears, the per-row Wanda definition, sequential order, or
  calibration size, source or seeds. The cached path stays only while its masks equal the original path's: a hash
  mismatch is a deviation.
* **The level grid,** its order, or regenerating a level whose mask hash does not match the freeze list.
* **The confirmation rule:** bound, limit, α, IUT, fixed sequence, or reading sealed items before the freeze.
* **The primary endpoint,** its denominator, or a hypothesis's criterion.
* **Arm A:** the controller model or its settings, the prompt, the target, the iteration cap, the rollback rule, or
  dropping it after any of its runs started.
* **Scoring:** shots, prompt format, batch layout, budget, or dropping the stored log-probabilities.
* **Running a secondary on a subset** (for example S3 on fewer items).
* **Moving to other hardware or library versions.** The mask hashes must reproduce first.

## 12. Open questions for the user

1. **H1 wording.** Shares (as written; it can fail through accounting alone) or certified levels (proposed)?
2. **Second family.** Choose one:
   * Llama-3.1-8B-Instruct: needs Meta's access approval for this account;
   * Olmo-3-7B-Instruct: open data; needs the cached path extended to mixed sliding-window layers;
   * Granite-3.3-8B-Instruct: no code change; paper A's family, which an earlier rule excluded.
3. **Arm A.** Run it (≈ 7.5–12 h of GPU, plus a paid API key for `gemini-3-flash-preview` and the risk that a preview
   model changes or is retired) or drop it, and H4 with it?
4. **S3 (5-shot) on all 13,100 items** (≈ 9.5 h) or on a registered subset?
5. **Qwen3-8B's disclosure (§2).** Is a design informed by its screening-pool results acceptable for a confirmatory
   claim on its untouched sealed set?
6. **Level names.** `unstructured_feasibility.level_name` rounds 17.5 % to `s18` and 22.5 % to `s22` (Python rounds
   halves to even). Study 2's code must name levels to one decimal. This is an implementation item, listed so it is not
   forgotten.

## 13. What this is not

* It is not registered, and nothing here may be scored until it is.
* It is not a claim about any model.
* The exploratory feasibility results it uses are not confirmatory.
