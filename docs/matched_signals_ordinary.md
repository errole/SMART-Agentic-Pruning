# Matched-signal test on ordinary parents (2026-11-23, exploratory)

**Lead question.** On parents where gradient rank 1 succeeds, does KL selection match or beat it?

**No.** The gate accepted gradient rank 1 at 12 parents. There, KL's choice is better than rank 1 at 5 parents and
worse at 7, with no ties, at every sample size.
* KL's mean regret is 0.07–0.13 points higher (sign test p = 0.77).
* KL's choice passes the 3-point half-2 criterion in 87 % of draws. Rank 1 passes at 11 of the 12 parents.
* On ordinary parents, the gradient's own rank 1 is about as good as any measured signal: mean regret 0.55 points,
  against KL 0.63–0.68, net flip 0.63–0.78 and S1c 0.53–0.58.

**Recommendation for the confirmatory experiment: KL only after a rank-1 rejection.**
* **Pre-registered rule, n = 100.**
  * At the 4 parents where the gate rejected rank 1, KL beats it at 3.
  * Where rank 1 was accepted, KL does not beat it (5 better, 7 worse).
  * The rule gives the same answer at every n.
* **Supplementary, post hoc.** After a rejection, the real alternative is the gradient's rank 2. KL choosing among
  ranks 2–11 beats rank 2 at 3 of the 4 rejected parents: mean regret 0.49 vs 1.29 points at n = 100.
* **Cost.** Probing only after rejections spends 4,400 paired items (about 36 GPU-seconds) per rejection. That is about
  1.6–2.2 typical gate runs, and rejections were rare: 1 of 13 ordinary parents, and run A's first came at attempt 16.
* **The evidence is thin.** Only 4 rejected parents, no sign test significant, and the target noise is about 1.3
  points per category. The confirmatory experiment should test this policy; this study does not establish it.

* Code: [`capprune/matched_signals.py`](../capprune/matched_signals.py), with `source: ranker` and the `compare` and
  `after-rejection` phases · tests [`tests/test_matched_signals.py`](../tests/test_matched_signals.py)
* Plan: [`matched_signal_plan_ordinary.md`](matched_signal_plan_ordinary.md), committed in `212dd5d`. The parent list
  and the plan were registered before scoring (plan SHA-256 `ab5141e4…`, LF; split `26afbba9…`, identical to the stall
  test's).
* Protocol in force: [`screen_confirm_protocol.md`](screen_confirm_protocol.md), amendment 1 (`23f7cb2`).
* Outputs: `~/cap-prune-runs/matched-signals-ordinary/`:
  * `comparison.md` / `.json`: the parent-level results;
  * `report.md` / `analysis.json`: the v1 analysis on these parents;
  * `after_rejection.md` / `.json`: supplementary;
  * provenance and scores: `registration.json`, `split.json`, `manifest.json`, `logprobs.npz`, `scores.sqlite`, logs.

## Design

* **Parents:** the ranker study's 13 trajectory parents.
  * The dense parent D0, once, with T1's step-0 shortlist.
  * T1–T4 at steps 1–3 (0.74, 1.49 and 2.23 % MLP sparsity).
  * The 3 stall parents (5–13 %) are analysed separately and are not among them.
* **Actions:** each parent's 11-action equal-size shortlist (1,280 channels each), 143 candidates in all.
* **Verification.** The ranker kept no masks or profiles, so each trajectory was rebuilt from dense:
  * profile each parent with the ranker's activation subset;
  * rebuild every shortlisted action on the parent's weights;
  * follow the chosen action to the next parent.

  All 13 parent masks and all 143 candidate masks equal the ranker's recorded hashes, checked before any scoring.
* **Unchanged from the stall test:**
  * the split, subsample sizes and draws;
  * the signals (net flip S0, KL S2, S1c), the target (half-2 net flip, worst category) and the baselines (gradient
    rank 1, random);
  * the scoring layout. Dense was reused from pool-signals after an exact spot re-score.
* **Cost:** 745,602 item-model evaluations, 6,072 GPU-seconds. Scoring ran on a clean tree (`ae50920`).
* **Unit of evidence: the parent**, with each parent's regret averaged over its 20 draws. Groups: ordinary (13), stall
  (3), all (16).

## Results

### Regret of the top-1 choice

Regret is in points of half-2 worst-category net flip loss above the parent's best action; each value is a mean over
parents.

| n per category | ordinary: S0 / KL / S1c / rank 1 / random | stall: S0 / KL / S1c / rank 1 / random | all 16: S0 / KL / S1c / rank 1 / random |
|---|---|---|---|
| 100 | 0.78 / 0.68 / 0.58 / 0.55 / 1.44 | 0.88 / 0.36 / 0.37 / 2.13 / 1.56 | 0.80 / 0.62 / 0.54 / 0.85 / 1.46 |
| 200 | 0.70 / 0.66 / 0.56 / 0.55 / 1.44 | 0.65 / 0.30 / 0.39 / 2.13 / 1.56 | 0.69 / 0.60 / 0.53 / 0.85 / 1.46 |
| 300 | 0.69 / 0.63 / 0.54 / 0.55 / 1.44 | 0.56 / 0.27 / 0.37 / 2.13 / 1.56 | 0.67 / 0.56 / 0.50 / 0.85 / 1.46 |
| 400 | 0.63 / 0.65 / 0.53 / 0.55 / 1.44 | 0.64 / 0.27 / 0.38 / 2.13 / 1.56 | 0.63 / 0.58 / 0.51 / 0.85 / 1.46 |

### Sign tests across parents

Counts are KL better / tie / worse, followed by the two-sided p.

| comparison | n = 100 | n = 200 | n = 300 | n = 400 |
|---|---|---|---|---|
| KL vs net flip, ordinary (13) | 8/0/5, p 0.58 | 7/0/6, 1.00 | 8/0/5, 0.58 | 7/0/6, 1.00 |
| KL vs net flip, stall (3) | 3/0/0, 0.25 | 3/0/0, 0.25 | 3/0/0, 0.25 | 2/1/0, 0.50 |
| KL vs net flip, all (16) | 11/0/5, 0.21 | 10/0/6, 0.45 | 11/0/5, 0.21 | 9/1/6, 0.61 |
| KL vs gradient rank 1, ordinary (13) | 5/0/8, 0.58 | 5/0/8, 0.58 | 5/0/8, 0.58 | 5/0/8, 0.58 |
| KL vs gradient rank 1, stall (3) | 3/0/0, 0.25 | 3/0/0, 0.25 | 3/0/0, 0.25 | 3/0/0, 0.25 |
| KL vs gradient rank 1, all (16) | 8/0/8, 1.00 | 8/0/8, 1.00 | 8/0/8, 1.00 | 8/0/8, 1.00 |

* **On ordinary parents, KL and net flip are indistinguishable** (mean difference −0.10 to +0.02 points).
* **KL's advantage over net flip in the first study came from the stall parents.** Pooled over all 16 parents it
  points the same way (9–11 of 16) but is not significant.

### Per parent

`comparison.md` has the full table. Regrets below are at n = 100, KL vs rank 1.

KL lost at 7 of the 12 accepted parents:
* T4s3: 2.03 vs 0.70;
* T4s1: 0.73 vs 0.30;
* D0, T1s1 and T2s1: rank 1 is the best action (regret 0), KL 0.35–0.70;
* T3s1 and T3s2: narrowly, by less than 0.1.

KL won at the other 5:
* T2s2: 1.15 vs 1.85;
* T3s3: 0.55 vs 1.03;
* T2s3: 1.35 vs 1.61;
* T1s2 and T1s3: by about 0.2.

At the rejected parents:
* A and B: KL won by 2.5–2.8;
* C: KL won by 0.04;
* T4s2: KL lost, 0.87 vs 0.70.

### Cost per step

| item | paired items | GPU s (est.) |
|---|---|---|
| KL probe, n = 100 per category (11 actions × 4 × 100) | 4,400 (+400 dense if not cached) | 36 |
| KL probe, n = 400 | 17,600 (+1,600) | 143 |
| screening gate per candidate, runs A / B / C (median [IQR]) | 2,100 [1,575–2,625] / 2,000 [1,500–2,700] / 3,450 [2,825–4,060] | |
| screening gate per candidate, ranker gate replays (176) | 2,765 [1,900–3,538] | |

## Interpretation

* **Gradient rank 1 is a good choice at ordinary parents.**
  * When the gate accepts it, it is usually near the best equal-size action (median regret below 0.5 points).
  * A noisy measurement over 11 actions does not reliably improve on it: with about 1.3 points of target noise,
    KL's choice drifts to actions that look slightly better on a subsample.
* **The gradient ranking is weakest where the gate rejects rank 1.**
  * Rank 1's regret at the rejected parents is 0.7–3.0 points (A 2.48, B 2.99, C 0.91, T4s2 0.70).
  * At A and B, a measured signal recovers almost all of it. At C and T4s2 it does not.
* **This favours a hybrid policy:** gradient top-1 while it is accepted, and a KL probe over the shortlist after a
  rejection.
  * The probe costs about two gate runs, and only at rejections.
  * It replaces the top-2 fallback, which in run B went to an action 3.3 points worse than the best.
* **S1c** (Δ log p on dense-correct items) had the lowest regret of the measured signals on ordinary parents and over
  all 16. It was not a policy option in the plan, so it is a candidate to pre-register, not a basis for switching now.

## Limits

* Exploratory:
  * 16 parents, of which 4 are rank-1 rejections. Three of those come from the A/B/C runs, which share their early
    trajectory.
  * Signals and target come from the same 5,214-item pool used during pruning, with no audit sets.
  * No sign test reaches p < 0.05.
* The target noise (about 1.3 points per category) is comparable to most of the regret differences.
* The after-rejection comparison with rank 2 is post hoc.

## Commands

```powershell
python -m capprune matched-signals register        --config configs/matched_signals_ordinary.yaml
python -m capprune matched-signals score           --config configs/matched_signals_ordinary.yaml   # GPU, ~102 min
python -m capprune matched-signals analyze         --config configs/matched_signals_ordinary.yaml
python -m capprune matched-signals compare         --config configs/matched_signals_ordinary.yaml
python -m capprune matched-signals after-rejection --config configs/matched_signals_ordinary.yaml   # supplementary
```
