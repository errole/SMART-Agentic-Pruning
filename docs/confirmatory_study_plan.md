# Confirmatory study plan (v1, 2026-11-24)

Committed and pushed **before any confirmatory model is scored on any item**. Any change before the full matrix needs
a dated amendment in this file, committed before any further confirmation scoring.

**Amendments to this plan:** amendment 3 (2026-11-24, after the pilot): every frozen checkpoint is scored, and
secondary endpoints S1–S4 are added (§10). Amendments 1 and 2 are the protocol's.

**Development model.** Qwen3-1.7B is the development model. None of its results count as confirmatory.

**Protocol in force.** [`screen_confirm_protocol.md`](screen_confirm_protocol.md) with amendments 1 and 2. Claims:
[`paper/claims.md`](paper/claims.md).

**Code.**
* `capprune/confirmatory.py`: build-data, admissibility, freeze, confirm and report.
* `capprune/gradient_live.py`: the live loop, with the perplexity screen, the KL probe and the counterfactual log.
* `capprune/screen_confirm.py`: `iut_test` and `fixed_sequence`.
* `capprune/signals.py`: `pick_action`.

**Configs.**
* `configs/confirmatory.yaml`: shared by both models.
* `configs/confirmatory_smollm3.yaml` and `configs/confirmatory_granite.yaml`: one per model.

## 1. Models

| | model 1 | model 2 |
|---|---|---|
| model | `HuggingFaceTB/SmolLM3-3B-Base` | `ibm-granite/granite-3.3-2b-base` |
| family | Hugging Face SmolLM | IBM Granite |
| revision | `d78a42f79198603e614095753484a04c10c2b940` | `c15dc681f4b2aeb4e576cdaae5434edfc17ed14d` |
| weights (SHA-256) | `7e270ac5…7d49`, `c6a6e769…87ae` | `e9f76a73…7844`, `5545dacc…68c2` |
| licence / access | Apache-2.0, not gated | Apache-2.0, not gated |
| parameters | 3.08 B | 2.53 B |
| layers × MLP channels | 36 × 11,008 | 40 × 8,192 |
| MLP | `gate_proj`, `up_proj`, `down_proj` (SiLU) | the same |
| published MMLU (base) | MMLU (cloze) 44.1 and MMLU-Pro (multiple choice) 32.7, both 0-shot (model card) | MMLU 54.33, OLMES (model card) |

**Selection criteria:** a family other than Qwen and different from each other; 1–4 B parameters; a gated MLP that the
structured MLP-channel operator supports; open weights pinned by revision; dense MMLU well above chance; fits the
RTX 5090 for gradient profiling.

**Candidates rejected:**
* **Llama-3.2-3B:** gated (manual approval), and this project's Hugging Face account has no access.
* **Phi-4-mini:** `Phi3MLP` fuses `gate_up_proj`, which the operator does not support.
* **OLMo-2-0425-1B:** its card gives no base-model MMLU. 0-shot letter scoring is doubtful at 1.5 B.

**GPU fit (RTX 5090, 31.8 GiB, measured on C4 calibration text only; no benchmark item scored):**
* **Without checkpointing:** the 2,048-token gradient pass of SmolLM3 peaked at 37.6 GiB with the dense copy on the
  GPU. It spilled to system memory and took 329 s per profile.
* **With checkpointing:** `profiling.gradient_checkpointing: true` gives the same computation; the gradient profile is
  bitwise identical, measured. SmolLM3 then profiles in 10.2 s at a 22.4 GiB peak.
* **Granite:** profiles in 12.6 s at a 19.2 GiB peak with checkpointing. The comparison pass without checkpointing
  spills on this GPU and was stopped after 20 minutes. The identical-profile check was completed on SmolLM3 only.
* **Results:** `~/cap-prune-runs/confirmatory/gpu_fit_check_{smollm3,granite}.json` and
  `profile_memory_experiment.json`.
* **Scripts:** `scripts/confirmatory_gpu_fit_check.py`, `scripts/profile_memory_experiment.py`.

**Model admissibility (pre-registered; each model's first scoring).**
* The test: dense 0-shot accuracy on that model's gate pool must be ≥ 40 % overall and ≥ 30 % in every category
  (`capprune confirmatory admissibility`).
* If it fails, the model is replaced by amendment before any of its runs.

## 2. Data, sealed per model before the model is scored on anything

**Sources** (`confirmatory.sources`):
* MMLU test and validation from `cais/mmlu` at `c30699e8356da336a370243923dbaf21066bb9fe`. The prepared rows were
  checked identical to the hub revision, and the files are pinned by SHA-256.
* MMLU-CF validation from `microsoft/MMLU-CF` at `c25b89a9…`, with 14 files pinned by SHA-256 and mapped to MMLU's
  four categories.
* Few-shot exemplars come from MMLU dev only (unused at 0-shot).

**Deduplication.** Pooled items: 25,573.
* 135 duplicate question texts are dropped (normalized question + choices; the first occurrence is kept, in the order
  MMLU test, MMLU validation, MMLU-CF).
* 6 items equal to few-shot exemplars are dropped.
* That leaves **25,432**.

**The confirmation set.** Per model: seeded (split seed 2026112401 for model 1, 2026112402 for model 2), stratified
by category, and within a category allocated to the three sources in proportion to their size.
* **Sizes**, from the amendment-1 sample-size table (n80 at a 1.0-point true loss, with at least 3,000):
  * STEM **3,600**
  * Humanities **3,000**
  * Social Sciences **3,000**
  * Other **3,500**
  * total 13,100: 7,317 MMLU test, 788 MMLU validation, 4,995 MMLU-CF.
* **No shortfall in any category**, with 900 items per category reserved for screening (100 probe + 200 hold-out +
  600 gate minimum).
* **The categories are fixed and will not change.**

**The other three sets.** These come from the remaining items, derived exactly as the live loop derives them:
* **probe:** 100 per category, from the ranker split, seed 4242;
* **hold-out slice:** 200 per category, seed 9091, carved from the gate items in every arm;
* **gate pool:** the rest.

| set | STEM | Humanities | Social Sciences | Other | total |
|---|---|---|---|---|---|
| confirmation | 3,600 | 3,000 | 3,000 | 3,500 | 13,100 |
| gate | 4,231 | 3,574 | 954 | 2,373 | 11,132 |
| probe | 100 | 100 | 100 | 100 | 400 |
| hold-out | 200 | 200 | 200 | 200 | 800 |

The sizes are identical for both models, but the items differ.

**Registrations** (`~/cap-prune-runs/confirmatory/<model>/registration.json`; first 16 hex digits of the file's
SHA-256):

| model | registration file | confirmation set | gate | probe | hold-out |
|---|---|---|---|---|---|
| model 1 | `762e792bc9adf1ed` | `9c086901e745aaef` | `109ce6633bc92cdf` | `05be7ffaa5cc6a74` | `aacea46795307c4e` |
| model 2 | `a10c2bbe7208f4f7` | `46e748d0eff1892b` | `fd6b9b9c10ef19fd` | `a6f85b337fe3a863` | `7eab6a999164db13` |

The set hashes are of the sorted qids. Both registrations were made at commit `f3092cf` (clean tree).

**Sealing rules.**
* **Where the confirmation rows live:** in `sealed/confirmation.jsonl`. Screening sees only their ids
  (`sealed/confirmation_qids.json`), for the overlap check.
* **No confirmation text in screening data:** the model's data folder has an empty `mmlu_final.jsonl` and an empty
  `mmlu_dev.jsonl`. Its `mmlu_probe.jsonl` holds the screening pool only, with no confirmation question text.
* **A run refuses to start** if its gate, probe, hold-out or sealed set differs from the registration
  (`ranked_search.split_registration`), or if any screening item is a sealed id.
* **When the confirmation rows can be read:** only `confirm` reads them. It refuses unless every run of its batch is
  frozen and this plan is committed and records the registration hash.

**Power** (per-category one-sided test at α = 0.05 against 3 points, at the sizes above).
* **Planning rates:** the development model's paired disagreement rates (Qwen3-1.7B run A: final, median and maximum
  over attempts).
* **Joint power:** every category certifies, with categories independent.

| rates | true loss | STEM | Humanities | Social Sciences | Other | joint |
|---|---|---|---|---|---|---|
| final (8.0–11.3 %) | 0 | 0.99 | 1.00 | 1.00 | 1.00 | **0.99** |
| final | 1.0 pt | 0.81 | 0.84 | 0.87 | 0.80 | **0.48** |
| final | 1.5 pts | 0.48 | 0.52 | 0.56 | 0.48 | 0.07 |
| median (2.8–3.4 %) | 1.0 pt | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| maximum (10.6–22.0 %) | 1.0 pt | 0.78 | 0.53 | 0.73 | 0.46 | 0.14 |

A checkpoint whose true losses are near 1 point in all four categories at once has about a coin-flip chance of
certifying. The study can certify checkpoints whose losses are near 0, not ones that sit near the tolerance.

## 3. Screening

Each model has 3 arms × seeds 0, 1, 2, all with the gradient ranker.

**Common to all arms:**
* **Selection:** gradient top-1 (ascending mean z_grad, ties by generator index), using the ranker's candidate
  generator.
* **Step:** 4 units × k blocks of 64 MLP channels, matched to the development step (0.744 % of MLP channels):
  * model 1: k = 12, 3,072 channels, 0.775 %;
  * model 2: k = 10, 2,560 channels, 0.781 %.
* **Depth windows:** every 4 layers.
* **Profiling:**
  * the ranker protocol: 16 activation and 8 gradient samples of 2,048 C4 tokens;
  * gradient checkpointing (section 1);
  * the seed selects the calibration windows and the gate question orders.
* **Caps:** 30 gated proposals, and 100,000 paired screening questions **including the KL-probe items**.
* **Stop:**
  * arms (a) and (b): at the first rank-1 rejection (or inconclusive result);
  * arm (c): when KL's pick is rejected (or inconclusive);
  * every arm: when a cap is reached.

**The arms:**
* **(a) `ppl_top1`: perplexity-rollback screen.**
  * Accept if the trial's perplexity on the C4 validation set (32 × 2,048 tokens) is at most 15 % above its parent's
    (τ = 0.15, as in the base paper).
  * No capability gate. Top-1.
* **(b) `cap_top1`: the capability gate at 3 points.**
  * The exact finite-pool, threshold-aware gate: α = 0.05 per side, minimum sample = batch = 100, uniform spending,
    budget = the gate pool, a fresh question order per attempt.
  * Top-1.
* **(c) `cap_klprobe`: as (b), with `ranked_search.fallback_policy: kl_probe`** (amendment 2).
  * After a rank-1 rejection, ranks 2–11 are scored on the probe items, KL's pick is chosen with
    `capprune.signals.pick_action`, and the pick is gated.
  * The counterfactual log is on: at each rejection, the KL, S1c and TV picks and gradient rank 2 are scored on the
    hold-out slice. It is recorded with its costs, never used for a decision, and not charged to the cap.

## 4. Confirmation (amendment 1, exactly)

* **Freeze.** After a batch of runs ends, each run's accepted checkpoints are listed in ascending order of parameters
  removed, with their mask hashes. The list is written and hashed (`confirmatory freeze`) **before any confirmation
  item is read**.
* **Test, per run:** fixed-sequence testing over that order (`confirmatory confirm`). For each checkpoint:
  * **Model:** the checkpoint, materialized, against the dense model, in identical batches (0-shot), on the sealed set.
  * **Bound:** per category, the Clopper–Pearson union bound on the lost and gained counts, with no finite-pool
    correction.
  * **Decision:** the per-category IUT at α = 0.05. Pass only if every category's one-sided 95 % upper bound is
    ≤ 3 points.
  * **Stop and claim:** stop at the first failure; the run's claim is the last passing checkpoint (none if the first
    fails).
  * **Scoring order:** the sequence is evaluated lazily, so checkpoints after a failure are never scored as tests.
    (Amendment 3, §10: every checkpoint is now scored, but only for secondary endpoints; the decision is unchanged.)
* **Description only.**
  * Simultaneous Bonferroni bounds for every tested checkpoint.
  * The run's final screened checkpoint, scored once if the sequence did not reach it, and labelled as not part of
    any claim.
* **Batches.**
  * `pilot` = model 1, seed 0, all three arms.
  * `main` = model 1's seeds 1 and 2 (6 runs), and all nine runs of model 2.
  * One model's confirmation set is read once per frozen batch.
* **The pilot reads model 1's confirmation set before model 1's seeds 1 and 2 are screened.** Screening never reads
  confirmation items (enforced), so later screening cannot depend on them except through a plan change, which needs
  an amendment.
* **Multiplicity.** Each run's claim has family-wise error ≤ 0.05 within its own sequence. Runs and arms are not
  corrected against each other; they are compared on the endpoints below.

## 5. Endpoints

**Primary: confirmed parameters removed per run.** This is the materialized parameter count of the claimed
checkpoint, as a fraction of the dense model; 0 if nothing is certified.

**Secondary:**
1. **Evaluation cost**, each reported separately:
   * screening: gate questions and GPU seconds;
   * KL-probe items and seconds;
   * counterfactual items and seconds;
   * confirmation items and seconds;
   * wall time per run.
2. **C1, the pool-to-confirmation gap.** For each tested checkpoint of arms (b) and (c), per category:
   (confirmation estimate) − (the gate's pool estimate at acceptance).
3. **C4, perplexity-arm violations.** In arm (a):
   * the position of the first checkpoint that fails confirmation;
   * whether the arm's final screened checkpoint (descriptive) has an estimated loss above 3 points in any category.

**C6 test.** Data: the counterfactual logs of arm (c), pooled over models and seeds. **The unit is the rejection.**
* **Primary:** the hold-out worst-category net flip loss of KL's pick vs gradient rank 2. Two-sided exact sign test,
  ties excluded.
* **Secondary:** TV vs KL, and S1c vs KL, with the same test.
* **Post hoc caveat:** TV was favoured **post hoc** on Qwen3-1.7B (`tv_posthoc.md`), so a TV result here is the first
  test of that observation, not a confirmation.
* **Expected power:** every arm-(c) run has at least one rank-1 rejection, and more only if a KL pick is accepted. That
  makes about 6–12 rejections in all, so C6 is expected to be underpowered.

## 6. Analysis

* **Within each model, paired across seeds (the seed is the unit):**
  * per-seed differences in confirmed parameters removed, (b) − (a) and (c) − (b);
  * reported per seed, with the mean and the range.
* **With 3 seeds there is no significance test.** "Consistent" means the same sign in all 3 seeds.
* **Models are not pooled for the primary endpoint.**

**What counts as support:**

| claim | supported if |
|---|---|
| C1 | In both models, the mean worst-category gap (confirmation − pool) over the tested checkpoints of arms (b) and (c) is > 0, and its 95 % bootstrap interval over checkpoints excludes 0. |
| C2 | Population re-gating (rule v1, `docs/population_gate_rule.md`, applied retrospectively to the arm-(b) runs' reveal sequences) stops within the first 3 steps in at least 2 of 3 seeds in both models. |
| C3 | The procedure runs as specified: frozen orders, the sealed set read only by `confirm`, no checkpoint tested outside its order. Validity itself rests on the simulation in `tests/test_screen_confirm.py`; it is a property of the procedure, not an empirical finding. |
| C4 | In both models, arm (a) confirms less than arm (b) in at least 2 of 3 seeds, **and** arm (a)'s final screened checkpoint exceeds 3 points (estimate) in some category in at least 2 of 3 seeds. |
| C5 | Not tested by the pilot. A later replay on these runs' gate caches (threshold-aware vs round-robin vs an e-process test). |
| C6 | KL's pick beats gradient rank 2 in a majority of rejections and the sign test gives p < 0.05. Otherwise "not established". |

## 7. Pilot and stopping

1. **Pilot:** model 1, seed 0, arms (a), (b) and (c), end to end through confirmation, using the existing
   checkpointing, replay and supervisor.
2. **Then stop and report:**
   * per-arm confirmed compression;
   * per-category bounds;
   * power achieved;
   * costs (screening, probe, counterfactual and confirmation, separately);
   * wall time per run;
   * whether the plan must change.
3. **Changes before the full matrix** need a dated amendment here, committed before any further confirmation scoring.
4. **Status of the pilot runs:**
   * If no amendment changes how they were screened or confirmed, they are part of the confirmatory data.
   * Otherwise they are reported as pilot only, and model 1 seed 0 is rerun.

## 8. Practices

* Commit before running. Each run records the commit and whether code or configs had uncommitted changes.
* **Always check test exit codes; never pipe a test command in a way that hides a failure.**
* Every score is persisted before use (crash replay), and every output is written once.
* Backups follow `docs/research_records.md`.

## 9. Limitations

* Two models, three seeds, MMLU-format items only.
* MMLU is public and may be in pretraining data; MMLU-CF validation was built to be contamination-free.
* The planning rates come from Qwen3-1.7B. The actual power depends on each checkpoint's disagreement rate.
* The confirmation set is reused across a model's runs and arms. Each claim is per run.
* Social Sciences has the smallest gate pool (954 items), so its gate decisions use the whole pool more often.

## 10. Amendment 3 (2026-11-24): every frozen checkpoint scored; secondary endpoints S1–S4 added after the pilot

Committed and pushed **before any further scoring.** Since the pilot, no item has been scored: no screening of model 1
seeds 1 and 2 or of model 2, and no confirmation scoring.

**Reason.** The pilot (model 1, seed 0; [`confirmatory_pilot_report.md`](confirmatory_pilot_report.md)) showed that
the primary endpoint cannot separate the arms when confirmation fails before they diverge.
* The arms' trajectories are identical until a screen first rejects (step 7 in the pilot).
* Confirmation failed at checkpoint 4, so all three arms confirmed the same 56.6 M parameters.

The secondary endpoints below describe how the arms differ in that case.

**Not changed:** the primary endpoint and its analysis (§5, §6); the models (§1); both models' sealed data splits and
registrations (§2); the arms, step, caps and stop rules (§3); the confirmation rule (§4: Clopper–Pearson union bound,
per-category IUT at α = 0.05 against 3 points, fixed sequence over the frozen order, stop at the first failure); the
batches; the C1–C6 criteria (§6).

### A3.1 Every frozen checkpoint is scored

* `confirm` scores **every** checkpoint of each run's frozen order, in that order, on the full confirmation set, with
  the same materialization, identical batches and score cache as §4.
* **The primary decision does not change.** It is `fixed_sequence` at 3 points over the frozen order, stopping at the
  first failure. The scores of checkpoints after the first failure never enter it.
* Those later scores serve only the secondary endpoints S1–S4 and description. They replace §4's single descriptive
  score of the final checkpoint.
* Each checkpoint's per-item outcomes (dense and checkpoint correctness on every confirmation item) are stored with the
  run's confirmation record. S3 and S4 are computed from exactly these scores.

### A3.2 Secondary endpoints S1–S4 (added after the pilot)

Wherever they are reported, they are labelled **"added after the pilot"**.
* **Reporting:** per run, and per arm within each model: the mean and range over seeds, with the seed as the unit, and
  the per-seed paired differences (b) − (a) and (c) − (b).
* **Two versions of each:** (i) all seeds, and (ii) excluding model 1 seed 0 (the pilot).
* **Status:** descriptive. No test and no support criterion.

**S1, screening overshoot.**
* The parameters removed at the screen's final checkpoint (the last position of the frozen order), minus the confirmed
  parameters removed (the primary endpoint).
* Both are materialized parameter counts. Reported in parameters and as a fraction of the dense model.
* 0 for a run with no accepted checkpoint.

**S2, evaluation cost per confirmed parameter.** The numerator is screening + KL probe + confirmation. The
counterfactual log is reported separately and is not included; profiling appears under wall time and is not included.
Two units, both pre-specified:
* **Items:** the questions charged to the cap (gate questions + KL-probe items), plus the confirmation items that the
  primary procedure needs.
  * Confirmation items = the set size (13,100) × (1 dense reference + the checkpoints the fixed sequence tested, up to
    and including the first failure, or all of them if none fails).
  * Arm (a)'s screen scores no benchmark items, so its screening cost in items is 0. Its perplexity evaluations appear
    in GPU seconds.
* **GPU seconds:** gate seconds (the journal's `gate_s`: the trial and dense-reference scoring, or arm (a)'s
  perplexity evaluation), plus KL-probe seconds, plus confirmation seconds.
  * Confirmation scores are shared between runs and cached. A run's confirmation seconds are therefore its confirmation
    items × the model's measured seconds per confirmation item.
  * That rate is measured over the model's confirmation passes, excluding items replayed from the cache and the pilot's
    first (spilled) pass.
* **The denominator:** confirmed parameters removed, in millions.
* **When nothing is confirmed,** S2 is undefined. Those runs are reported as "none confirmed" and counted, not
  averaged.

**S3, confirmed compression at a 5-point limit.**
* The same procedure (Clopper–Pearson union bound, per-category IUT at α = 0.05, fixed sequence over the same frozen
  order, stop at the first failure) with a 0.05 tolerance instead of 0.03, on the same scores.
* The result is the materialized parameters removed of its claim; 0 if the first checkpoint fails.
* Not a claim of the study.

**S4, certifiable compression vs confirmation sample size.**
* **Sizes:** 500, 1,000 and 2,000 items per category, and the full set (3,600 / 3,000 / 3,000 / 3,500).
* **Draws:** 50 per size.
  * Draw d of size n takes, in each category, a uniformly random subset of n items without replacement, from the
    category's items sorted by qid.
  * The generator is `numpy.random.default_rng([20261124, n, d, crc32(category)])`.
  * Each model has one set of draws, shared by all its runs and checkpoints (paired).
  * All 50 draws of the full set are the full set, so that size has one value, equal to the primary endpoint.
* **Per draw:** the 3-point procedure of the primary endpoint, applied to the subsample's counts over the frozen order;
  the result is the claim's parameters removed.
* **Reported:** per run, the median and interquartile range (25th–75th percentile, linear interpolation) over the 50
  draws; per arm and model, the median and IQR over all draws of its seeds.
* **Status:** descriptive.

### A3.3 Data splits and C1

* Both models' sealed splits are kept as registered.
* Under C1, the report gives each model's Social Sciences gate-pool size (954 items in both) and its
  pool-vs-confirmation gap at every tested checkpoint, next to the other categories' gaps.
* **C1's bootstrap interval** (§6 left the details open; fixed here, before the data): percentile interval,
  10,000 resamples of the checkpoints, `numpy.random.default_rng(0)`. The checkpoints are the tested checkpoints of
  arms (b) and (c), counted per run as §6 states, so a checkpoint that both arms share is counted once in each run.

### A3.4 The pilot

* S1–S4 are also computed for the pilot runs (model 1, seed 0) from their frozen orders. **They are labelled post
  hoc.**
* The pilot's already-scored checkpoints are replayed from the score cache. `confirm` refuses if a rerun's primary
  claim differs from the recorded one.
* The pilot's earlier records are kept, moved to `batches/pilot/superseded/`.
* Under §7.4, the pilot remains confirmatory data. Amendment 3 changes neither how the pilot was screened nor its
  primary confirmation.

### A3.5 Order of work

* **Screening order:** model 1's seeds 1 and 2 (all three arms; batch `main` of model 1) are screened first, then
  model 2's seeds 0–2 (all three arms; batch `main` of model 2).
* **Per batch:** each batch is frozen after its runs end, and confirmed only after freezing. The pilot batch is
  re-confirmed for S1–S4.
* **Hardware:** every run uses the same RTX 5090, with no other GPU job running. Screening and confirmation never run
  at the same time.

### A3.6 The report

The report leads with the primary endpoint per arm (paired across seeds within each model, the seed as the unit). It
then gives:
* C1, per category;
* C4: arm (a)'s per-category loss at its screened endpoint, descriptive, together with the §6 criterion;
* C6: KL vs gradient rank 2 over all rejections, sign test; TV and S1c vs KL as secondary;
* S1–S4;
* wall time and cost per run;
* every deviation from this plan.

The study summary is produced by `capprune confirmatory study`.
