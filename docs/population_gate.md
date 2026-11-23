# Population-level re-gating and continuous pool signals (2026-11-23)

The live gate uses exact finite-pool bounds and may reveal its whole 5,214-item pool, so an ACCEPT certifies the loss
on those items only. Checkpoints accepted at 5.94–7.31 % removal then failed the 3-point audit criterion. This study
asks two questions, without running any new pruning search:
1. Would a population-level gate have stopped earlier?
2. Does a continuous paired signal on the pool predict the audit better than the gate's flip statistic?

All audit comparisons are exploratory, because the audit sets had been inspected before.

* Code: [`capprune/population_regate.py`](../capprune/population_regate.py) and
  [`capprune/pool_signals.py`](../capprune/pool_signals.py)
* Pre-registered: [`population_gate_rule.md`](population_gate_rule.md) (rule v1, commit `c22d7e1`) and
  [`pool_signals_plan.md`](pool_signals_plan.md) (commit `609d0d2`), each committed and hashed before computing
* Outputs: `~/cap-prune-runs/population-regate/` (`report.md`, `phase1_summary.md`, `regate.json`) and
  `~/cap-prune-runs/pool-signals/` (`report.md`, `phase2_summary.md`, `logprobs.npz`, `signals.json`)

## Phase 1: retrospective re-gating (CPU only)

**The rule.** For each category, an exact Clopper-Pearson upper bound on the population paired loss: no finite-pool
correction, Bonferroni over the 4 categories, uniform error spending over the gate's planned looks and over the 30
candidates of a run (top-2 candidates included), α = 0.05, tolerance 3 points. It is applied to the live gate's
recorded reveal sequence.

**Verification.**
* All 47 trial masks were rebuilt on CPU and match their journals.
* The cached pairs reproduce every journal count, and every live decision.
* The error spent is exactly 0.05 per run.

| run | first population non-accept | would stop at |
|---|---|---|
| A (top-1) | attempt 1, inconclusive | **0.00 %** of parameters |
| B (top-2) | attempt 1, inconclusive | **0.00 %** |
| C (top-2, live tolerance 1.0) | attempt 3, inconclusive | **0.91 %** |

Projection, not part of the rule: with the whole pool revealed at the observed rates, all three runs would stop after
3 steps, at **1.37 %**. That is below arm C's 3.20 % and the 5.485 % audit boundary.

**Pool size needed.** With this error budget, certifying 3 points needs about 1,200–1,500 items per category at zero
true loss and median disagreement (2.8–3.4 %); three of the four categories' pools are smaller than that. Near the
boundary, at 8–12 % disagreement, certifying a real 1.0–1.5-point loss takes about 10–50 times each pool.

## Phase 2: continuous signals on the gate pool (GPU, exploratory)

The dense model and A's 7 boundary states were scored on all 5,214 gate-pool items with identical batches, saving the
four letters' log-probabilities.

| signal | MMLU-CF: per category / worst / overall | MMLU held-out: per category / worst / overall |
|---|---|---|
| S0 net flip loss (the gate's statistic) | +0.60 / +0.82 / +0.86 | +0.90 / +0.89 / +0.93 |
| S0b gross flip rate | +0.82 / +0.93 / +0.93 | +0.83 / +0.93 / +0.82 |
| S1 Δ log p(correct) | −0.36 / −0.54 / −0.50 | −0.15 / −0.54 / −0.29 |
| S2 KL(dense‖pruned) over A–D | +0.80 / +0.96 / +0.93 | +0.80 / +0.96 / +0.82 |

Values are Spearman ρ, with 28, 7 and 7 pairs. They are descriptive: 7 checkpoints from one trajectory.

* **KL is a smoother signal than the net flip loss.** It rises almost monotonically along the trajectory and jumps
  threefold where the audit losses jump. It tracks the audit better on MMLU-CF and for the worst category.
* **The net flip loss remains competitive on MMLU held-out**, which shares the pool's distribution.
* **The planned S1 is confounded by calibration shifts.** On items the dense model gets wrong, it swings with the
  pruned model's confidence. A post-hoc split shows that on dense-correct items it grows steadily with pruning. That
  needs a pre-registered test on new checkpoints.

## Commands

```powershell
python -m pytest tests/test_population_regate.py tests/test_pool_signals.py -q
python -m capprune population-regate register --config configs/population_regate.yaml   # hash the rule first
$env:CUDA_VISIBLE_DEVICES=""; python -m capprune population-regate run --config configs/population_regate.yaml
python -m capprune pool-signals score   --config configs/pool_signals.yaml   # GPU, ~6 min
python -m capprune pool-signals analyze --config configs/pool_signals.yaml
```
