# Analysis plan: matched-compression signal test on ordinary parents (exploratory; no audit sets)

Recorded before any scoring. `capprune matched-signals register --config configs/matched_signals_ordinary.yaml` writes
the split and records the SHA-256 of this file (line endings normalised), which includes the parent list below, and of
the split. Scoring, analysis and comparison refuse to run if either changes.

## Question

1. **Lead question.** On parents where gradient rank 1 succeeds, does KL selection match or beat it?
2. **Recommendation.** Which policy should the confirmatory experiment use: KL everywhere, KL only after a rank-1
   rejection, or gradient only?

## Parents and actions

* **Parents.** The ranker study's trajectory parents (`~/cap-prune-runs/ranker/data/decisions.json`), 13 in total:
  * the shared dense parent D0, once, with the shortlist recorded in T1's step-0 decision;
  * the 12 non-dense parents T1–T4 × steps 1–3, at 0.74, 1.49 and 2.23 % MLP sparsity.
* **Excluded:**
  * the 3 stall parents already analysed (`matched-signals`). None of their mask hashes is among these parents.
  * D0's three other step-0 decisions, which are the same parent.
  * The fixed-state development checkpoints (group G). They are not trajectory parents, and their shortlists have 9
    actions plus LLM proposals.
* **Actions.** Each parent's existing equal-size shortlist: 11 actions of 1,280 MLP channels each (143 candidates).
* **Gradient rank.** Ascending mean z_grad, ties by generator index (the gradient-only rule).
* **Verification.** The ranker kept no masks or profiles, so each trajectory is rebuilt from dense:
  * profile each parent with the ranker's protocol (activation subset of `profile_seed`);
  * rebuild every shortlisted action;
  * follow the trajectory's chosen action to the next parent.

  Every rebuilt parent mask must equal the decision's `parent_mask`, and every rebuilt candidate mask must equal the
  candidate's recorded `mask`. Otherwise nothing is scored.

## Unchanged from `matched_signal_plan.md` (v1, commit `5077db4`)

* **Split:** the same split of the 5,214-item gate pool. Its SHA-256 must equal v1's registered split
  (`26afbba9c55be1c8…`).
* **Scoring:** the pool-signals layout. Dense is reused from pool-signals under the same provenance checks and the
  exact spot re-score.
* **Signals:** S0 net flip, S2 KL over A–D, S1c Δ log p(correct) on dense-correct items.
* **Subsamples:** n ∈ {100, 200, 300, 400} per category of half 1, 20 seeded draws each, the same seeds.
* **Target:** half-2 net flip loss per category, worst category.
* **Analyses:** choice, ties and regret; gradient-rank-1, random and oracle references; Spearman correlation;
  pass/fail calls with the leave-one-parent-out map; the lead criterion.

## Added analyses (the `compare` phase)

**Unit of evidence: the parent.** Each parent's regret for a signal at n is its mean over the 20 draws.

1. **Regret tables.** The v1 regret table for these 13 parents, for the 3 stall parents (from
   `matched-signals/analysis.json`), and for all 16 together.
2. **Per-parent table.** For every parent:
   * whether gradient rank 1 was accepted;
   * rank 1's and the best action's half-2 worst-category loss;
   * the regret of rank 1, S0, KL and S1c at n = 100 and 400;
   * the share of draws in which KL's choice passes (≤ 3 points on half 2).
3. **Sign tests across parents.**
   * Comparisons: KL vs S0, and KL vs gradient rank 1.
   * Groups: these parents, the stall parents, and all 16.
   * Per parent: better, tie (identical mean regret) or worse. Two-sided exact binomial p on the non-tied parents.
4. **Lead question, at every n.**
   * **"Rank 1 succeeds"** means the screening gate accepted gradient rank 1 at that parent (the ranker's recorded
     gate replay; at the stall parents rank 1 was rejected).
   * Secondary definition: rank 1 passes the half-2 truth (worst category ≤ 3 points).
   * On those parents, count where KL matches or beats rank 1 (KL regret ≤ rank-1 regret), give the mean difference
     and the sign test, and compare the share of draws whose KL choice passes with rank 1's pass status.
5. **Probe cost per step.** 11 actions × 4 categories × n paired items, i.e. 4,400–17,600 for n = 100–400, plus 4n
   dense items unless cached. It is set beside the screening gate's typical cost per candidate: the median and
   interquartile range of paired gate questions per attempt in runs A, B and C, and per candidate in the ranker's gate
   replays.

## Recommendation rule (fixed now)

Evaluated at **n = 100 per category**, the cheapest probe (4,400 paired items per step), with parents as units.
* **R** = parents where gradient rank 1 was rejected: the 3 stall parents plus any ordinary parent with a rank-1
  rejection.
* **S** = parents where it was accepted.

1. If, on R, KL's regret is lower than rank 1's in fewer than a strict majority of parents → **gradient only**.
2. Otherwise, if on S KL's regret is lower than rank 1's in more parents than it is higher, AND the mean difference
   (KL − rank 1) over S is below 0 → **KL everywhere**.
3. Otherwise → **KL only after a rank-1 rejection**.

The other sample sizes are reported beside it. After a rejection, KL may still choose rank 1 in this analysis; how
often it does is reported.

```json
{"plan_version": 2,
 "source": "ranker",
 "ranker_decisions": "ranker/data/decisions.json",
 "parents": [{"name": "D0",   "trajectory": "T1", "step": 0, "decision": 4,  "parent_mask": "f28bf44b0ae6421067fb"},
             {"name": "T1s1", "trajectory": "T1", "step": 1, "decision": 5,  "parent_mask": "ce146e0f4a77779e96b8"},
             {"name": "T1s2", "trajectory": "T1", "step": 2, "decision": 6,  "parent_mask": "9b08ed9c6094abfff347"},
             {"name": "T1s3", "trajectory": "T1", "step": 3, "decision": 7,  "parent_mask": "eef256d5e51e04940c43"},
             {"name": "T2s1", "trajectory": "T2", "step": 1, "decision": 9,  "parent_mask": "1da77586f3f1f2dd871f"},
             {"name": "T2s2", "trajectory": "T2", "step": 2, "decision": 10, "parent_mask": "b19d62d97ab244ab099c"},
             {"name": "T2s3", "trajectory": "T2", "step": 3, "decision": 11, "parent_mask": "f7d374334f500db66aa5"},
             {"name": "T3s1", "trajectory": "T3", "step": 1, "decision": 13, "parent_mask": "645f10eb10b14e923497"},
             {"name": "T3s2", "trajectory": "T3", "step": 2, "decision": 14, "parent_mask": "c9c59bb3d2d9d464936f"},
             {"name": "T3s3", "trajectory": "T3", "step": 3, "decision": 15, "parent_mask": "5243eec56eb358c2ed68"},
             {"name": "T4s1", "trajectory": "T4", "step": 1, "decision": 17, "parent_mask": "0d95b8ede3d53ad64eb0"},
             {"name": "T4s2", "trajectory": "T4", "step": 2, "decision": 18, "parent_mask": "2494d2cfd61c39101f6a"},
             {"name": "T4s3", "trajectory": "T4", "step": 3, "decision": 19, "parent_mask": "79c50c4b9eff79f1698e"}],
 "exclude_parent_masks": ["7965b9b5716226cec724", "b9528d1ba65e8ecf7140", "06dd3215864e2bbc4894"],
 "split_seed": 20261123, "split_sha256_must_equal": "26afbba9c55be1c83fa57e82716466addfdf9358df5055ac50531489f49d5c4f",
 "subsample_seed": 7, "sizes": [100, 200, 300, 400], "draws": 20,
 "order_seed": 2026, "batch_size": 100, "tolerance": 0.03, "signals": ["S0", "S2", "S1c"],
 "reuse_from": "pool-signals",
 "stall_analysis": "matched-signals/analysis.json",
 "recommendation_n": 100,
 "gate_cost_runs": {"A": "gradient-live-ext/gradient_top1__t00__s0", "B": "gradient-top2/gradient_top2__t00__s0",
                    "C": "gradient-top2-strict/gradient_top2__t00__s0"}}
```
