# Claims (C1–C6 and the negative result N)

**The paper in one sentence.** What it takes to certify worst-domain accuracy during training-free LLM structured
pruning:
* pool-certified per-step gates overfit under adaptive search;
* per-step population certificates are infeasible at benchmark scale;
* screen-then-confirm (a per-category intersection-union test plus fixed-sequence testing over checkpoints) gives a
  valid guarantee;
* we measure the confirmed compression and its evaluation cost.

**How to read this file.**
* Every claim has a status, its evidence so far, and the experiment that would confirm it.
* Evidence paths are relative to the repo or to `~/cap-prune-runs/`. They were checked when this file was written
  (2026-11-23).
* Anything that could not be verified from the repo or the runs folder is marked **unverified - check**.
* All evidence so far comes from **one model (Qwen3-1.7B), seed 0**. Every held-out set used so far had been
  inspected before it was used, so all of it is exploratory. Confirmation means the confirmatory study on other
  models, with sealed confirmation sets.

| claim | status |
|---|---|
| C1 pool-certified gates overfit under search | exploratory support (one model, one seed) |
| C2 per-step population certification infeasible | supported retrospectively (one model) |
| C3 screen-then-confirm is valid | protocol and simulation only; not yet run on a model |
| C4 perplexity gating hides domain loss | exploratory support; never compared at matched confirmed compression |
| C5 adaptive allocation reduces screening cost | partial: question savings in replay; negligible end to end |
| C6 (secondary) KL probe picks a better fallback | thin exploratory support (4 parents); implemented, not yet tested |
| N an LLM controller adds no value | negative result; report as an ablation |

---

## C1. Gates certified on a reused question pool overfit it: more search gives more gate passes and worse transfer

**Status:** exploratory support from one model and one seed. The held-out sets had been inspected before.

**Evidence so far:**

1. **Three live arms at 3 points, same start and seed** (`~/cap-prune-runs/gradient-compare/report.md`,
   `comparison.json`; doc [`docs/gradient_live.md`](../gradient_live.md)). Every accepted step was certified at
   ≤ 3 points per category on the 5,214-item gate pool. Yet for the final checkpoints of A and B, the held-out
   simultaneous upper bound exceeds 3 points on both audit sets.

   | arm | accepted steps | held-out worst category | MMLU-CF worst category |
   |---|---|---|---|
   | A, gradient top-1 | 15 | STEM +3.65 [+2.11, +5.18]; simultaneous UB +5.39, not supported | Other +2.16; UB +3.51, not supported |
   | B, top-2 (more search per parent) | 17 (2 rank-2 rescues) | STEM +4.12; UB +5.92 | (see paired difference) |
   | C, gate tolerance 1.0 point | 7 | Other +0.05; UB +0.86, supported | (UB supported) |

   * **B vs A, paired on identical items:** more passes and worse transfer on the held-out final split, overall
     +0.39 [+0.12, +0.65] points. On MMLU-CF validation the difference is −0.01 [−0.28, +0.27].
   * **Set labels:** the report names these sets `exploratory` and `comparison`; see `configs/gradient_live_ext.yaml`.
2. **Gap diagnostic** (greedy arm of the first live-gate run; `~/cap-prune-runs/live-gate-diagnostics/greedy_cap_gap/`,
   [`docs/live_gate.md`](../live_gate.md)). STEM loss:
   * pruning-time pool: +2.78 (0-shot), +2.30 (5-shot);
   * held-out split: +4.23 (0-shot), +5.18 (5-shot);
   * held-out minus pool: +2.88 [+0.05, +5.68] at 5-shot, +1.45 [−1.32, +4.20] at 0-shot.
   * The design cannot separate selection (overfitting the reused pool) from the pool's own sampling error, which
     is about ±2.1 points for STEM.

**Gaps:**
* One seed and one model.
* The "more search" contrast is A vs B only.
* The B−A transfer difference shows on one held-out set, not on MMLU-CF.
* The selection effect is not separated from sampling error.

**Confirming experiment:**
* Replicate on the confirmatory models.
* For every accepted checkpoint, measure the gap between its pool loss (gate) and its confirmation loss on the
  sealed set, as a function of the amount of search: steps taken and candidates gated.
* Compare it with a no-search control: the same compression reached by a fixed schedule.

---

## C2. Per-step population-level certification is infeasible at MMLU scale

**Status:** supported retrospectively on one model, by exact bounds on the recorded runs.

**Evidence so far** (`~/cap-prune-runs/population-regate/`: `regate.json`, `report.md`, `phase1_summary.md`; doc
[`docs/population_gate.md`](../population_gate.md); rule [`docs/population_gate_rule.md`](../population_gate_rule.md),
committed in `c22d7e1` before computing):

* **The rule:** a Clopper–Pearson population bound per category, Bonferroni over 4 categories, uniform spending over
  looks and over 30 candidates, α = 0.05, 3 points.
* **Verification:** all 47 trial masks were rebuilt, and the live decisions were reproduced.
* **Where it would stop:**

  | run | first population non-accept | would stop at (parameters removed) |
  |---|---|---|
  | A | attempt 1 | 0.00 % |
  | B | attempt 1 | 0.00 % |
  | C | attempt 3 | 0.91 % |

  * Full-pool projection: all runs stop after 3 steps, at 1.37 %.
  * The live runs and the 5.485 % audit boundary are far beyond that.
* **Sample size:** certifying 3 points needs about 1,200–1,500 items per category even at zero true loss and median
  disagreement. Three of the four category pools are smaller.
* **Confirmation sizes** (`~/cap-prune-runs/screen-confirm/confirmation_sizes_v2.md`, amendment 1): one test of
  one checkpoint needs 2,700–3,600 fresh items per category at a true loss of 1.0 point and 80 % power.

**Gaps:**
* Retrospective, on one model.
* The 1,200–1,500 figure assumes the observed disagreement rates.

**Confirming experiment:**
* The same re-gating and sample-size analysis on the confirmatory models' screening runs: disagreement rates
  measured there, and the items needed per step against the screening pool sizes.

---

## C3. Screen-then-confirm gives a valid per-category guarantee

**Status:** protocol and simulation only. **Not yet run on a model.**

**Evidence so far:**
* **Protocol:** [`docs/screen_confirm_protocol.md`](../screen_confirm_protocol.md).
  * v1: commit `15171d3`.
  * Amendment 1: commit `23f7cb2`, the per-category IUT (every category's one-sided 95 % bound ≤ 3 points) and
    fixed-sequence testing over checkpoints in ascending order of parameters removed.
  * Code: `capprune/screen_confirm.py`, `iut_test` and `fixed_sequence`.
* **Simulation** (`tests/test_screen_confirm.py`, 3,000 seeded runs per scenario): the family-wise error is
  0.000–0.012, at or below 0.05. This includes true losses exactly at 3 points: one category, losses only, all
  categories, a bad checkpoint first, and losses growing past 3.
* **Validity argument:**
  * The per-category bound is an exact Clopper–Pearson union bound.
  * The IUT has level α for the union null "some category ≥ 3 points".
  * Fixed-sequence testing controls FWER at α without assumptions on the dependence between checkpoints.
  * These are standard results (Learn then Test, Pareto Testing; see [`related_work.md`](related_work.md)).

**Gaps:**
* No model run.
* The guarantee is for the population that the confirmation items are drawn from, so it rests on the sealed set
  being a random sample from each category's target distribution. That is a property of the confirmatory plan, not
  of the code.

**Confirming experiment:** the confirmatory study. For each model:
* screen;
* freeze the ordered checkpoint list;
* score the sealed confirmation set once;
* report the certified checkpoint.

Validity is checked by simulation (above) and, where possible, by agreement with a second sealed sample.

---

## C4. Perplexity gating hides domain loss

**Status:** exploratory support. **Never compared at matched confirmed compression.**

**Evidence so far:**
* **Gemini-controller live-gate run** (`~/cap-prune-runs/live-gate/_live/report.md`, [`docs/live_gate.md`](../live_gate.md)):
  * Arm A (LLM + perplexity gate) accepted two steps that raised perplexity by only 3.1 % and 3.7 %. It ended at
    10.4 % MLP sparsity (6.4 % of parameters) with held-out STEM +18.2 points (+14.3 overall).
  * Arm C (greedy + capability gate) ended at 3.72 % MLP sparsity with STEM +5.2.
* **Ollama 7B local pilot** (`~/cap-prune-runs/live-gate-ollama/`, [`docs/live_gate.md`](../live_gate.md)):
  * Arm A (perplexity gate) accepted 10/10 steps, with perplexity never rising more than 1.2 % per step. It ended
    at 3.94 % MLP sparsity with held-out STEM +7.61.
  * Arm B (capability gate, same controller) ended at 1.15 % with worst category Other +1.61.

**Gaps:**
* The arms ended at different compression levels.
* The controllers took different step sizes under different feedback.
* One seed.
* Audits on inspected sets.

**Confirming experiment:** confirmatory arm (a), perplexity gate, vs arm (b), capability gate, on the same models.
Compare the worst-category confirmation loss at matched confirmed compression, or the confirmed compression at a
matched loss.

---

## C5. Adaptive allocation reduces screening cost

**Status:** partly supported. The question savings come from faster rejections; end to end the saving is about 1 %
or less.

**Evidence so far:**
* **Adaptive replay**, threshold-aware vs round-robin, Qwen3-1.7B pilot, budget 3,000 per candidate
  (`~/cap-prune-runs/adaptive/pilot/replay-tol03-nolat/summary.md` and `replay-tol05-nolat/summary.md`):

  | tolerance | threshold-aware | round-robin | saving |
  |---|---|---|---|
  | 3 points | 7,700 questions | 10,700 | **−28.0 %** |
  | 5 points | 8,500 questions | 10,300 | −17.5 % |

  * **Prompt figure differs:** Prompt 1 says 27 % at 3 points; the summary file says 28.0 %. Unverified - check
    which figure is intended.
  * No false accepts or false rejects under any policy.
* **Live identical rejection** (Ollama pilot, arms B vs D, [`docs/live_gate.md`](../live_gate.md)): the same
  STEM +3.3 trial was rejected with 3,061 questions (threshold-aware) against 4,861 (round-robin), −37 %.
* **Where the savings come from:** ACCEPT costs exactly the same under every policy that samples only open domains
  ([`docs/adaptive_evaluation.md`](../adaptive_evaluation.md) §2, enforced by tests).
* **End to end:** the replays' reconstructed search wall time.

  | tolerance | threshold-aware | round-robin | saving |
  |---|---|---|---|
  | 3 points | 2,608.8 s | 2,636.1 s | −1.0 % |
  | 5 points, first qualifying checkpoint | 2,200.6 s | 2,216.5 s | −0.7 % |

  * Prompt 1's "< 1 %" holds at 5 points; at 3 points it is 1.0 %.
  * The wall time is reconstructed from measured parts, not timed end to end.

**Gaps:**
* One model.
* The replay pool is fixed.
* The live comparison has one identical rejection; after it the trajectories diverge.

**Confirming experiment:** replay on the confirmatory models' screening caches. Compare threshold-aware with
round-robin and with an adaptive Learn-then-Test e-process test (Zecchin et al. 2024), at the same error level.
Report questions and end-to-end GPU time.

---

## C6 (secondary). After a rank-1 rejection, a KL probe picks a better fallback than gradient rank 2

**Status:** secondary claim with thin exploratory evidence. The policy is protocol amendment 2 (`3f16846`; edits in
`8e98392`). It is implemented in the live loop (`ranked_search.fallback_policy: kl_probe`, `d0ff66a`) but not yet
tested on a real search.

**Evidence so far:**
* **Stall parents** (`~/cap-prune-runs/matched-signals/`, [`docs/matched_signals.md`](../matched_signals.md); plan
  `5077db4`). At the 3 stall parents, KL's top-1 choice has a regret of 0.27–0.36 points, against 0.56–0.88 for net
  flip, at 100–400 items per category.
* **Ordinary parents** (`~/cap-prune-runs/matched-signals-ordinary/`, `comparison.md`,
  [`docs/matched_signals_ordinary.md`](../matched_signals_ordinary.md); plan `212dd5d`):
  * **KL everywhere is rejected:** where the gate accepted rank 1 (12 parents), KL is better at 5 and worse at 7,
    sign test p = 0.77.
  * **After a rank-1 rejection:** KL beats rank 1 at 3 of 4 parents under the pre-registered rule.
  * **Post hoc** (`after_rejection.md`): KL chosen among ranks 2–11 beats gradient rank 2 at 3 of 4 parents, with
    mean regret 0.49 vs 1.29 points at n = 100.
* **Secondary signals, post hoc** (`~/cap-prune-runs/matched-signals-ordinary/tv_posthoc.md`, commit `c5c3ecf`):
  * On ordinary parents, TV's regret is 0.42–0.58 points, against KL 0.63–0.68 and rank 1 0.55.
  * Over all 16 parents, TV beats net flip at 12–13. This is not adopted (amendment 2).
  * S1c is recorded likewise.

**Gaps:**
* 4 rejected parents, 3 of them from runs that share their early trajectory.
* No sign test is significant.
* The target noise is about 1.3 points per category.

**Confirming experiment:**
* Arm (c) runs the amended proposer with the counterfactual log (`ranked_search.counterfactual`). At every rank-1
  rejection it scores the KL, S1c and TV picks and gradient rank 2 on a hold-out slice that neither the gate nor
  the probe uses.
* The unit of evidence is the rejection, pooled over models and seeds.

---

## N. Negative: an LLM controller adds no value

**Status:** negative result, to be reported as an ablation.

**Evidence so far:**
* **Ollama 7B pilot** ([`docs/live_gate.md`](../live_gate.md), `~/cap-prune-runs/live-gate-ollama/`):
  * Qwen2.5-7B produced 30/30 valid proposals but was a timid pruner.
  * Under capability feedback, arms B and D put almost every step on `L1.mlp`, one unit at a time, and no arm
    approached the 10 % target.
* **Fixed-state comparison** ([`docs/fixed_state.md`](../fixed_state.md), `~/cap-prune-runs/fixed-state/compare/`).
  Mean worst-category loss on 4 frozen states, in points:

  | condition | loss |
  |---|---|
  | best shortlist action (oracle) | +1.15 |
  | random shortlist pick | +3.17 |
  | Qwen2.5-14B / 7B, shortlist | +3.93 / +3.81 |
  | Qwen2.5-14B / 7B, free-form | +3.64 / +3.70 |

  Every LLM condition is worse than a random pick from the shortlist.
* **Learned ranker** ([`docs/ranker.md`](../ranker.md), `~/cap-prune-runs/ranker/data/`):
  * On unseen trajectories, ridge equals gradient-only: worst-category loss +1.10 vs +1.09, difference +0.01
    [−0.13, +0.14].
  * Gradient-only is the strongest simple baseline, and it needs no LLM.

**Gaps:**
* The completed controller comparisons use small local models (Qwen2.5-7B/14B).
* The base paper's controller (`gemini-3-flash-preview`) was run in the live-gate experiment
  (`~/cap-prune-runs/live-gate/`), but its capability-gate arm paused at 7 of 10 attempts, unaudited, when the
  free-tier quota ran out, and its round-robin arm never started (`docs/live_gate.md`). So N is not established for
  that controller. **Unverified - check** whether it changes the result.

**Confirming experiment:** an ablation in the confirmatory study. Run the same screen-then-confirm pipeline with
the LLM controller in place of gradient top-1, at matched evaluation budget, on at least one confirmatory model.
