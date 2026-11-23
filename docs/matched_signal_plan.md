# Analysis plan: matched-compression signal test (exploratory; no audit sets)

Recorded before any scoring. `capprune matched-signals register` writes the split and records the SHA-256 of this
file (line endings normalised) and of the split; scoring and analysis refuse to run if either changes.

## Question

At matched compression and an equal number of questions, does KL between the dense and pruned answer distributions pick
actions that keep held-out flip loss lower than the gate's flip statistic does? If so, from which sample size?

## Candidates

* **Parents:** the stall parents of the three live runs.
  * A: `gradient-live-ext`, attempt 16; parent `e65e9461315ef207`.
  * B: `gradient-top2`, attempt 21; parent `4ac58c5aa8348773`.
  * C: `gradient-top2-strict`, attempt 10; parent `05e68e072909705a`.
* **Actions:** each parent's full shortlist of 11 equal-size actions (1,280 MLP channels each), as recorded by the
  stalled attempt and gated by the stall diagnostic. 33 candidates in total.
* **Verification:**
  * the parent mask must equal the journal's `parent_mask_hash`;
  * every rebuilt candidate mask (parent + action + the stalled attempt's persisted profile) must appear in the stall
    diagnostic's cache keys;
  * for the live-gated action, it must also equal the journal's `trial_mask_hash`.

## Split and scoring

* **Split:** the 5,214-item gate pool (the ranker's label set), split into two halves within each category. Sort the
  qids, permute them with `numpy.random.default_rng([split_seed, crc32(category)])`, and put the first ceil(n/2) in
  half 1.
* **Scoring:** the dense model and every candidate, on the whole pool, in the `pool-signals` layout: per category in
  seeded order (seed 2026), 100 items per batch, a 32,768-token budget, 0-shot, masked bf16 weights. Both halves are
  therefore scored in identical batches for every model.
* **Saved:** the full-vocabulary log-probabilities of the four letters.
* **Reuse:** the `pool-signals` cache is reused only if all of these hold:
  * the pool hash, protocol, model identity (mask hash) and scoring code all match, and the cached run came from a clean
    tree;
  * re-scoring the first dense batch of every category reproduces the cached log-probabilities exactly.

  Otherwise everything is scored afresh.

## Signals (per category, on a subsample of half 1)

* **S0 net flip loss:** (lost − gained)/n. This is the gate's statistic.
* **S2 KL:** mean KL(dense ‖ pruned), with both distributions renormalised over A–D.
* **S1c:** mean [log p_dense(answer) − log p_pruned(answer)] over A–D-renormalised distributions, **restricted to the
  subsample items the dense model answers correctly**. This tests the post-hoc observation of Phase 2.

**Subsamples:** per category, n ∈ {100, 200, 300, 400} items of half 1, 20 draws each, drawn without replacement with
`default_rng([subsample_seed, n, draw, crc32(category)])`. Every candidate and signal uses the same items.

## Target

The net flip loss per category on all of half 2, against the dense model. The worst-category target is its maximum
over the categories.

## Analyses (per parent and pooled; descriptive, 3 parents x 11 actions)

1. **Ranking:** Spearman correlation across the 11 actions between the signal and the target, per category. Average
   it over the categories that vary, then over the draws.
2. **Choice:** each signal picks the action with the lowest worst-category value (maximum over categories); ties go
   to the lower gradient rank. **Regret** = the chosen action's worst-category half-2 net flip loss minus the best
   action's.
   * References: gradient rank 1 (no measurement); a **random choice** among the 11 equal-size actions (the
     no-information baseline: expected regret = the mean over actions); the oracle (0).
3. **Pass/fail call on half 2 at 3 points.** The truth is that every category's half-2 net flip loss is ≤ 3 points.
   * S0 calls pass if every category's subsample value is ≤ 3 points.
   * S2 and S1c have no 3-point scale. Each is mapped to net flip loss by a straight-line fit on the OTHER two
     parents' actions (full half 1, pairs = action x category), and calls pass if every mapped category value is
     ≤ 3 points.
   * Accuracy is reported against sample size, along with the base rate and the majority-class accuracy.
4. **Lead criterion:** the smallest n in the grid at which KL's mean regret, pooled over parents and draws, is lower
   than S0's, and stays lower at every larger n. If no such n exists, that is the answer.
5. **Supplementary, outside the grid:** the same analyses on all of half 1 (one draw), to show where each signal
   levels off. This row does not enter the lead criterion.

```json
{"plan_version": 1,
 "parents": [{"name": "A", "run": "gradient-live-ext/gradient_top1__t00__s0", "attempt": 16, "diagnostic": "gradient-live-ext/_analysis/diagnostic"},
             {"name": "B", "run": "gradient-top2/gradient_top2__t00__s0", "attempt": 21, "diagnostic": "gradient-top2/_analysis/diagnostic"},
             {"name": "C", "run": "gradient-top2-strict/gradient_top2__t00__s0", "attempt": 10, "diagnostic": "gradient-top2-strict/_analysis/diagnostic"}],
 "split_seed": 20261123, "subsample_seed": 7, "sizes": [100, 200, 300, 400], "draws": 20,
 "order_seed": 2026, "batch_size": 100, "tolerance": 0.03, "signals": ["S0", "S2", "S1c"],
 "reuse_from": "pool-signals"}
```
