# Screen-then-confirm protocol (v1)

Committed before any new scoring. It separates the two jobs that the live gate currently does at once:
* **screening:** choosing which steps to take, cheaply, on the reused pruning-time pool;
* **confirmation:** one certificate for the final model, on fresh items.

## 1. Screening (choose the steps)

* **Gate:** the existing exact finite-pool capability gate. It is threshold-aware, uses the ranker's 5,214-item label
  set, budget = pool, α = 0.05 per side, minimum sample = batch = 100, uniform spending, a fresh question order per
  attempt, and a **3-point** tolerance per category.
* **Proposer:** gradient top-1, with the ranker's candidate generator and the matched 1,280-channel step.
* **Stop:** at the first top-1 rejection (or inconclusive decision), or when a cap is reached (30 proposals, 100,000
  paired gate questions).
* **What a screening accept means:** loss on the pool's items only. It is not a claim about new items.

## 2. Confirmation (certify the result)

* **Unit:** the frozen final checkpoint of the screening run, materialised, against the dense model.
* **Items:** fresh items, never used in screening, tuning, calibration or any earlier audit. Each category's items are
  drawn at random from that category's target distribution. All items are scored once, with dense and checkpoint in
  identical batches (the gate's 0-shot prompt).
* **Test:** **one** test with the full α = 0.05: Bonferroni over the 4 categories (δ = 0.0125 each), with no spending
  over candidates or looks, because it is a single look.
* **Bound:** per category, the exact one-sided population upper bound of rule v1
  (`population_regate.population_bounds`), i.e. a Clopper-Pearson union bound on the lost and gained counts with no
  finite-pool correction.
* **Pass:** every category's upper bound ≤ **3.0 points**.
* **If it fails:** the screening run's checkpoint is not certified. No other checkpoint may be tested on the same
  confirmation items.

## 3. Confirmation sample size per category

Computed by `capprune screen-confirm sizes --config configs/screen_confirm.yaml` (output
`~/cap-prune-runs/screen-confirm/confirmation_sizes.json`). The inputs are the paired disagreement rates
q = (lost + gained)/n of run A in `~/cap-prune-runs/population-regate/regate.json`:
* its final checkpoint (attempt 15, the state a confirmation would test);
* the median and maximum over its attempts.

The expected lost and gained rates are p± = (q ± true loss)/2.
* **n50:** the bound at the expected counts certifies (about 50 % power).
* **n80:** 80 % of 2,000 seeded simulated samples certify.

| category | q: final / median / max | true loss | n50 (final q) | n80 (final q) | n80 (median q) | n80 (max q) |
|---|---|---|---|---|---|---|
| STEM | 11.3 / 3.2 / 12.0 % | 0.0 pts | 1,550 | 2,400 | 800 | 2,500 |
| STEM | | 1.0 pts | 3,500 | 5,300 | 1,650 | 5,650 |
| STEM | | 1.5 pts | 6,050 | 9,300 | 2,800 | 10,000 |
| Humanities | 8.6 / 3.4 / 15.0 % | 0.0 pts | 1,250 | 1,900 | 800 | 3,050 |
| Humanities | | 1.0 pts | 2,650 | 4,000 | 1,750 | 6,850 |
| Humanities | | 1.5 pts | 4,700 | 7,200 | 2,950 | 12,200 |
| Social Sciences | 8.0 / 2.8 / 10.6 % | 0.0 pts | 1,150 | 1,750 | 700 | 2,200 |
| Social Sciences | | 1.0 pts | 2,450 | 3,750 | 1,450 | 4,900 |
| Social Sciences | | 1.5 pts | 4,400 | 6,850 | 2,300 | 8,750 |
| Other | 11.2 / 3.1 / 22.0 % | 0.0 pts | 1,550 | 2,400 | 750 | 4,350 |
| Other | | 1.0 pts | 3,350 | 5,200 | 1,600 | 9,700 |
| Other | | 1.5 pts | 5,900 | 9,300 | 2,650 | 17,250 |

**Planning value.** Use the final-checkpoint q at a true loss of 1.0 point, with 80 % power: about **3,750–5,300 fresh
items per category**, roughly 18,000 in total. That is more than any untouched MMLU-format set available to this
project (see `configs/gradient_live_ext.yaml`, `fresh_search`). The smaller numbers at the median q apply only to
checkpoints that disagree with the dense model as little as early steps do.

## Status

This is a protocol for future runs. No confirmation has been run.
