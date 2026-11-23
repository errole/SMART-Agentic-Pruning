# Analysis plan: continuous paired signals on the gate pool (Phase 2, exploratory)

Recorded and committed before any log-probability was scored. Its parameters are in the JSON block below and are read
by `capprune pool-signals`.

## Question

Does a continuous paired signal, measured on the gate's pruning-time pool, track the exploratory audit losses of run
A's checkpoints better than the pool flip statistic the gate uses?

## States

The 7 states of the compression-boundary map (`~/cap-prune-runs/gradient-boundary/states.json`):
* the accepted checkpoints after A's attempts 10–15;
* the attempt-16 rank-2 candidate. Attempt 16's rank-1 trial was rejected, and the rank-2 candidate is the only
  attempt-16 state with audit results.

Every mask file is re-verified against its recorded SHA-256 and mask hash before scoring.

## Scoring

* Items: the gate pool (the ranker's label set, 5,214 items).
* Protocol: the live gate's 0-shot prompt and scorer, with a 32,768-token batch budget.
* Weights: masked bf16 weights, which is the gate's representation. The dense model is a separate, verified,
  never-modified copy.
* Batches: identical for every model. Each category in a seeded order (seed 2026), 100 items per batch, with the same
  rows and token budget.
* Saved per item and model: the full-vocabulary log-softmax at the last position for the four answer letters, the
  prediction and correctness.

## Signals (per state and category, on the pool)

| id | signal |
|---|---|
| S0 | **net flip loss** (lost − gained)/n: the gate's statistic, i.e. the pool flip rate |
| S0b | gross flip rate (lost + gained)/n |
| S1 | mean change in the correct answer's log-probability: log p_dense(answer) − log p_pruned(answer), full-vocabulary log-probabilities |
| S2 | mean KL(dense ‖ pruned) between the answer distributions renormalised over A–D |

## Targets and comparison (descriptive, exploratory)

The targets are the exploratory audit losses in `~/cap-prune-runs/gradient-boundary/boundary.json` (materialised
model vs. dense), on MMLU-CF validation and on MMLU held-out separately. For each signal, target set and level, I
report Spearman ρ and Pearson r:

1. **per category:** 28 pairs (7 states × 4 categories), the signal in category c against the audit loss in
   category c;
2. **worst category:** 7 pairs, the maximum of the signal over categories against the worst-category audit loss;
3. **overall:** 7 pairs, the item-weighted mean over categories against the overall audit loss.

"Tracks better" means a higher correlation than S0 at the same level. There are no significance tests: there are only
7 states, the 28 per-category pairs are not independent, and the audit targets themselves carry ±0.6–1.4-point
sampling error.

```json
{"plan_version": 1, "states_file": "gradient-boundary/states.json", "audit_file": "gradient-boundary/boundary.json",
 "order_seed": 2026, "batch_size": 100, "signals": ["S0", "S0b", "S1", "S2"], "levels": ["per_category", "worst", "overall"],
 "targets": ["comparison", "exploratory"]}
```
