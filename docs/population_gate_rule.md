# Pre-registered rule: retrospective population-level re-gating (rule v1)

Recorded before any re-gating result was computed. `capprune population-regate` reads its parameters from the JSON
block below and refuses to run if this file's SHA-256 differs from the hash recorded at registration
(`<out_dir>/rule_registration.json`).

## Question

The live gate's bounds are exact for the **finite pool** of 5,214 pruning-time items and may reveal the whole pool, so
an ACCEPT certifies the loss on those items only. Would a gate that certifies the **population** loss (the loss on new
items from the same distribution) have accepted the same steps?

## Data

* Runs: A = `gradient-live-ext`, B = `gradient-top2`, C = `gradient-top2-strict` (all completed; only read).
* Units: every gated candidate (attempt) of a run, in journal order, including the top-2 fallback candidates.
* Before an attempt's data are used, its trial mask is rebuilt on CPU from the verified parent chain, the recorded
  action and the run's persisted profile, and its mask hash must equal the journal's `trial_mask_hash`. Its cached
  paired batches (keyed by that hash and the attempt's order seed) must reproduce the journal's final per-category
  counts (n, lost, gained).
* For every revealed item, x = dense_correct - trial_correct, in {+1 (lost), 0, -1 (gained)}.

## Bound

In category c the revealed items are treated as an i.i.d. sample from an infinite population, with
p+ = P(x = +1), p- = P(x = -1) and population loss mu_c = p+ - p-. **No finite-pool correction.** At a look with n
items, k+ lost and k- gained:

* UB_c = CP_upper(k+, n; d/2) - CP_lower(k-, n; d/2)
* LB_c = CP_lower(k+, n; d'/2) - CP_upper(k-, n; d'/2)

CP_* are exact one-sided Clopper-Pearson binomial bounds. P(UB_c < mu_c) <= d (union bound over the two counts), and
likewise for LB_c with d'. This is the existing gate's construction with the binomial in place of the hypergeometric.

## Error spending (per run)

* False-accept budget `alpha_false_accept` for the whole run, split uniformly over `candidates_per_run` gated
  candidates (the arms' proposal cap; a top-2 fallback candidate counts as one candidate).
* Within a candidate: Bonferroni over the 4 categories. Within a category: uniform over that category's planned looks
  in the gate's own plan (minimum sample 100, then batches of 100, ending at the category's pool size; budget = pool).
* So a look gets d = alpha_false_accept / (candidates_per_run * 4 * J_c), where J_c is the category's number of
  planned looks. The sum over every candidate, category and look is <= alpha_false_accept. The same applies to
  false rejects with `alpha_false_reject`.

## Decision for one candidate (the gate's existing reveal schedule)

Replay the live gate's recorded reveal sequence: first every category's minimum sample, then the batches in the order
of the gate's trace. After each reveal, compute every category at its current look:

* **ACCEPT** if UB_c <= tolerance for all four categories;
* **REJECT** if LB_c > tolerance for some category;
* otherwise take the next recorded reveal.

When the recorded reveals run out, the decision is **INCONCLUSIVE (revealed items)**: the population rule could not
decide on the items the live gate scored. No new items are scored (Phase 1 is CPU-only).

## Run-level re-decision

Walk the candidates in journal order along the live trajectory. The first candidate whose population decision is
not ACCEPT is where the run would first have stopped (rejected or inconclusive). The run would stop at that
candidate's parent checkpoint. Its total parameters removed (materialised count: removed MLP channels x 6,144) are
compared with:

* the exploratory audit boundary (attempt 12 of A: 5.485 %, the last state whose simultaneous upper bounds were
  <= 3 on both inspected sets);
* arm C's endpoint (3.20 %, pruning-time tolerance 1.0 point).

Later accepts are reported too, for every candidate, but they are counterfactual: the live trajectory continued past
that point.

## Required pool sizes (descriptive)

Per category, from the observed paired disagreement rate q = (k+ + k-)/n of each attempt in run A:

1. The one-sided width UB - (p+ - p-) that the full pool (n = N_c) would give at the observed rates, with the look
   budget d of the gate's plan.
2. The items needed to certify the tolerance when the true population loss mu is 0, 1.0 or 1.5 points, at the median
   and at the maximum observed q:
   * p+ = (q + mu)/2, p- = (q - mu)/2;
   * looks every 100 items, d = alpha_false_accept / (candidates_per_run * 4 * J), with J = n/100 looks up to n;
   * **n50** = smallest look n with UB(expected counts) <= tolerance (about 50 % power);
   * **n80** = smallest look n at which at least 80 % of 2,000 simulated samples (seed 0) certify.

```json
{
  "rule_version": 1,
  "runs": {"A": "gradient-live-ext/gradient_top1__t00__s0", "B": "gradient-top2/gradient_top2__t00__s0",
           "C": "gradient-top2-strict/gradient_top2__t00__s0"},
  "tolerance": 0.03,
  "alpha_false_accept": 0.05,
  "alpha_false_reject": 0.05,
  "candidates_per_run": 30,
  "categories": ["STEM", "Humanities", "Social Sciences", "Other"],
  "min_sample": 100,
  "batch_size": 100,
  "audit_boundary_params_fraction": 0.05484901344979226,
  "arm_c_params_fraction": 0.03199525784571215,
  "sensitivity_true_loss": [0.0, 0.01, 0.015],
  "n80_draws": 2000,
  "n80_seed": 0
}
```
