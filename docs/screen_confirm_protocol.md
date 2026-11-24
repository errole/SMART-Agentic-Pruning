# Screen-then-confirm protocol (v1, amendments 1 and 2 of 2026-11-23)

Committed before any new scoring. It separates the two jobs that the live gate currently does at once:
* **screening:** choosing which steps to take, cheaply, on the reused pruning-time pool;
* **confirmation:** a certificate on fresh items.

The original protocol is commit `15171d3` and stays the record of v1. This file is the protocol in force.

## Amendment 1 (2026-11-23), made before any confirmation data was touched

**What changes:** the confirmation rule only (section 2) and therefore its sample sizes (section 3). Screening is
unchanged.

1. **The pass decision is an intersection-union test (IUT).** A checkpoint passes only if, in **every** category, its
   one-sided 95 % upper bound on the paired loss is ≤ 3 points. There is no Bonferroni correction in the decision.
   The simultaneous Bonferroni bounds (δ = α/4 per category) are still reported, as description only.
2. **Multiple checkpoints are handled by fixed-sequence testing.**
   * All accepted checkpoints of a run are ordered by parameters removed (ascending), and the order is frozen before
     any confirmation scoring.
   * Each is tested at α = 0.05, and testing stops at the first failure.
   * The claimed result is the last passing checkpoint.

**Why.**
* **Bonferroni is not needed for the decision.**
  * The null hypothesis for a checkpoint is "some category loses ≥ 3 points": a union of per-category nulls.
  * A test that rejects it only when every per-category test rejects at α has level α (the IUT principle).
  * Bonferroni over the categories made v1's decision conservative. It inflated the confirmation sample size by about
    1.45×: 18,250 → 12,600 items at the planning value.
* **Fixed-sequence testing keeps a usable claim when the final checkpoint fails.**
  * v1 tested only the final checkpoint, so a failure left nothing certified.
  * Fixed-sequence testing certifies the most compressed checkpoint that passes, before the first failure, using the
    same confirmation items.
  * Its family-wise error rate is at most α under any dependence between the checkpoints' tests, because a false
    claim requires the first false null in the fixed order to be rejected.
* **Checked by simulation** (`tests/test_screen_confirm.py`; 3,000 seeded runs per scenario). The family-wise error
  stays at or below 0.05:

  | scenario | family-wise error |
  |---|---|
  | one category exactly at 3 points | 0.002 |
  | losses only, exactly at 3 points | 0.012 |
  | all four categories at 3 points | 0.000 |
  | a bad checkpoint first in the sequence | 0.001 |
  | losses that grow past 3 points | 0.000 |

  The Clopper-Pearson union bound is conservative, so the realised error is well under α.

## Amendment 2 (2026-11-23): the screening proposer, made before any confirmatory run

**What changes:** the proposer and the stopping rule of screening (section 1). The gate and confirmation are
unchanged.

**Policy:**
* Gradient top-1 at every parent.
* After a rank-1 rejection, run a **KL probe** over ranks 2–11:
  * 100 items per category, on probe items disjoint from the gate's items;
  * charged to the arm's budget.
* Gate KL's choice, and stop if that is rejected too.

**Why** ([`matched_signals_ordinary.md`](matched_signals_ordinary.md), merged in `18d8a43`):
* **Where the gate accepted rank 1** (12 ordinary parents), KL's choice was worse than rank 1: 5 better, 7 worse,
  sign test p = 0.77. A probe at every step would cost budget and lose accuracy.
* **Where the gate rejected rank 1** (4 parents):
  * KL beat rank 1 at 3 of 4, under the pre-registered rule;
  * KL beat the gradient's rank 2 (the top-2 fallback) at 3 of 4, post hoc, with mean regret 0.49 vs 1.29 points.
* **The evidence is thin:** 4 parents, not significant. The benefit of this policy is therefore claim **C6** of the
  confirmatory study, a **secondary** claim.
* **S1c and total variation (TV) are recorded as secondary signals, not adopted.** S1c had the lowest regret of the
  measured signals, but it was not a policy option in the plan.

## 1. Screening (choose the steps), as amended

* **Gate:** the existing exact finite-pool capability gate. It is threshold-aware, uses the ranker's 5,214-item label
  set, budget = pool, α = 0.05 per side, minimum sample = batch = 100, uniform spending, a fresh question order per
  attempt, and a **3-point** tolerance per category.
* **Proposer:** gradient top-1 at every parent, with the ranker's candidate generator, the matched 1,280-channel step
  and the 11-action shortlist (amendment 2).
* **After a rank-1 rejection (or inconclusive decision): the KL probe.**
  * **Probe items:** the ranker's probe split.
    * 100 items per category (400), drawn by seed 4242 from the pruning-time pool.
    * Disjoint from the gate's 5,214 items, and the same items at every probe.
    * They are not confirmation items.
  * **Scoring:** the dense model and the parent's ranks 2–11 on the probe items, with the gate's 0-shot prompt, in
    identical batches. Dense answers are scored once and cached.
  * **Signal and choice:** per category, the mean KL(dense ‖ candidate), with both distributions renormalised over
    A–D. Choose the action with the lowest worst-category value; ties go to the lower gradient rank.
  * **Gate:** KL's choice goes through the same gate, with its own attempt number and question order.
    * Accept: it becomes the new parent, and gradient top-1 resumes.
    * Reject or inconclusive: stop.
  * **Budget:** every probe is charged to the arm.
    * 10 actions × 400 = **4,000 paired items** count toward the paired-question cap, plus 400 dense items the first
      time.
    * KL's choice counts as a proposal toward the 30-proposal cap.
  * **Recorded, not used for any decision:** per probe and candidate, on the probe items:
    * the net flip loss (S0);
    * S1c, the Δ log p(correct) over A–D on dense-correct items;
    * TV, the total variation ½ Σ |p − q| over A–D renormalised.
* **Stop:** when KL's choice is rejected (or inconclusive), or when a cap is reached (30 proposals, 100,000 paired
  questions, probe items included).
* **What a screening accept means:** loss on the pool's items only. It is not a claim about new items.

## 2. Confirmation (certify the result), as amended

* **Units:** every accepted checkpoint of the screening run, materialised, each against the dense model.
* **Order:** by parameters removed, ascending. It is written with the checkpoints' mask hashes to a registration file
  and hashed **before** any confirmation item is scored.
* **Items:** fresh items, never used in screening, tuning, calibration or any earlier audit.
  * Each category's items are drawn at random from that category's target distribution.
  * Each checkpoint and the dense model are scored once on the same items, with dense and checkpoint in identical
    batches (the gate's 0-shot prompt).
* **Bound:** per category, the exact one-sided population upper bound of rule v1
  (`population_regate.population_bounds`). This is a Clopper-Pearson union bound on the lost and gained counts, with no
  finite-pool correction, at δ = α = 0.05.
* **Pass, per checkpoint (IUT):** every category's 95 % upper bound ≤ **3.0 points** (`screen_confirm.iut_test`).
* **Sequence:** test the checkpoints in the frozen order, each at α = 0.05, and stop at the first failure
  (`screen_confirm.fixed_sequence`).
  * The certified result is the last passing checkpoint.
  * If the first one fails, nothing is certified.
  * Checkpoints after the first failure are not tested, and no checkpoint outside the frozen list may be tested on
    the same items.
* **Reported as description only:** point estimates and the simultaneous Bonferroni upper bounds (δ = α/4 per
  category) for every tested checkpoint.

## 3. Confirmation sample size per category, as amended

Computed by `capprune screen-confirm sizes --config configs/screen_confirm_v2.yaml` (output
`~/cap-prune-runs/screen-confirm/confirmation_sizes_v2.{json,md}`; the v1 sizes stay in `confirmation_sizes.*`).

**Inputs.** The paired disagreement rates q = (lost + gained)/n of run A, from
`~/cap-prune-runs/population-regate/regate.json`:
* its final checkpoint (attempt 15);
* the median and maximum over its attempts.

The expected lost and gained rates are p± = (q ± true loss)/2, and every bound is per category at δ = 0.05.
* **n50:** the bound at the expected counts certifies (about 50 % power).
* **n80:** 80 % of 2,000 seeded simulated samples certify.

| category | q: final / median / max | true loss | n50 (final q) | n80 (final q) | n80 (median q) | n80 (max q) |
|---|---|---|---|---|---|---|
| STEM | 11.3 / 3.2 / 12.0 % | 0.0 pts | 1,000 | 1,650 | 550 | 1,750 |
| STEM | | 1.0 pts | 2,150 | 3,600 | 1,150 | 3,900 |
| STEM | | 1.5 pts | 3,800 | 6,500 | 1,900 | 6,650 |
| Humanities | 8.6 / 3.4 / 15.0 % | 0.0 pts | 800 | 1,300 | 600 | 2,150 |
| Humanities | | 1.0 pts | 1,650 | 2,800 | 1,250 | 4,700 |
| Humanities | | 1.5 pts | 3,050 | 4,900 | 2,050 | 8,350 |
| Social Sciences | 8.0 / 2.8 / 10.6 % | 0.0 pts | 750 | 1,200 | 500 | 1,600 |
| Social Sciences | | 1.0 pts | 1,600 | 2,700 | 1,000 | 3,450 |
| Social Sciences | | 1.5 pts | 2,800 | 4,650 | 1,650 | 6,050 |
| Other | 11.2 / 3.1 / 22.0 % | 0.0 pts | 1,000 | 1,650 | 550 | 3,000 |
| Other | | 1.0 pts | 2,200 | 3,500 | 1,150 | 6,650 |
| Other | | 1.5 pts | 3,650 | 6,450 | 1,850 | 11,750 |

**Planning value.** Use the final-checkpoint q at a true loss of 1.0 point, with 80 % power per category: about
**2,700–3,600 fresh items per category**, roughly 12,600 in total (v1: 3,750–5,300 and 18,250).
* **Power is per category.** A checkpoint passes only if all four categories certify. If every category sits at its
  n80 with a true loss of 1.0 point, the joint power is roughly 0.8⁴ ≈ 0.4. Larger samples, or smaller true losses in
  most categories, are needed for high joint power.
* **The same items serve every checkpoint** in the fixed sequence, so the sample size is set by the most compressed
  checkpoint one hopes to certify.

## Status

This is a protocol for future runs. No confirmation has been run, and no confirmation item has been scored.

The amendment-2 proposer is not yet implemented in the live loop. The existing `ranked_search` supports gradient
top-k fallback (`fallback_k`), not the KL probe.
