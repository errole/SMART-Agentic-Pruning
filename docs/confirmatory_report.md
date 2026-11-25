# Confirmatory study: report (both models, all seeds; 2026-11-25)

The study follows [`confirmatory_study_plan.md`](confirmatory_study_plan.md) v1 (commit `f0152c8`) with amendments 1–3.
Amendment 3 (§10, commit `125db01`) was committed and pushed before any of the scoring reported here. It adds S1–S4
**after the pilot**, and the pilot's values for them are **post hoc**.

* **Models:** SmolLM3-3B-Base (model 1, 3,075,098,624 parameters) and Granite-3.3-2B-Base (model 2, 2,533,525,504).
* **Arms:** (a) perplexity rollback, (b) capability gate, (c) capability gate + KL probe.
* **Seeds:** 0–2 per model. Model 1 seed 0 is the pilot ([`confirmatory_pilot_report.md`](confirmatory_pilot_report.md)).
* **Confirmation:** each model has its own sealed set of 13,100 items (STEM 3,600, Humanities 3,000, Social Sciences
  3,000, Other 3,500).
  * The claim comes from a fixed sequence over each run's frozen order. Each step is a per-category IUT with a one-sided
    95 % Clopper–Pearson union bound against 3 points, stopping at the first failure.
  * Under amendment 3, every frozen checkpoint is also scored. Checkpoints after the first failure are descriptive only.
* **Generated outputs:**
  * the study summary, `~/cap-prune-runs/confirmatory/study/study.md` and `study.json`, written by
    `capprune confirmatory study`;
  * per-batch reports with every checkpoint's per-category bounds,
    `~/cap-prune-runs/confirmatory/<model>/batches/<batch>/report.md`.

All tables below are taken from `study.json`.

## Primary endpoint: confirmed parameters removed (seed = unit, paired within model)

**Model 1, SmolLM3-3B-Base**

| seed | (a) perplexity | (b) gate | (c) gate + KL | (b) − (a) | (c) − (b) |
|---|---|---|---|---|---|
| 0 (pilot) | 56.6 M (1.84 %) | 56.6 M (1.84 %) | 56.6 M (1.84 %) | 0 | 0 |
| 1 | 37.7 M (1.23 %) | 37.7 M (1.23 %) | 37.7 M (1.23 %) | 0 | 0 |
| 2 | 75.5 M (2.46 %) | 75.5 M (2.46 %) | 75.5 M (2.46 %) | 0 | 0 |
| **mean** [range] | **56.6 M** (1.84 %) [37.7, 75.5] | **56.6 M** (1.84 %) [37.7, 75.5] | **56.6 M** (1.84 %) [37.7, 75.5] | 0 [0, 0] | 0 [0, 0] |

**Model 2, Granite-3.3-2B-Base**

| seed | (a) perplexity | (b) gate | (c) gate + KL | (b) − (a) | (c) − (b) |
|---|---|---|---|---|---|
| 0 | 47.2 M (1.86 %) | 47.2 M (1.86 %) | 62.9 M (2.48 %) | 0 | +15.7 M |
| 1 | 62.9 M (2.48 %) | 62.9 M (2.48 %) | 78.6 M (3.10 %) | 0 | +15.7 M |
| 2 | 47.2 M (1.86 %) | 47.2 M (1.86 %) | 47.2 M (1.86 %) | 0 | 0 |
| **mean** [range] | **52.4 M** (2.07 %) [47.2, 62.9] | **52.4 M** (2.07 %) [47.2, 62.9] | **62.9 M** (2.48 %) [47.2, 78.6] | 0 [0, 0] | +10.5 M [0, +15.7] |

**No difference between arms is consistent** (the same sign in all 3 seeds, §6) in either model.
* **(b) − (a)** is 0 in all six model × seed pairs.
* **(c) − (b)** is one Granite step (+15.7 M, 0.62 %) in model 2 seeds 0 and 1, and 0 everywhere else.
* With 3 seeds there is no significance test (§6), and the models are not pooled.

**Why the arms coincide.** All three arms make the same gradient top-1 steps from the same dense model and seed until a
screen first rejects. Arm (a) never rejects (30 of 30 steps accepted in every run). Arm (b) stops at its first
rejection. Arm (c) continues from that point with KL-probe picks. So the arms can differ only after (b)'s screen ends.

| model, seed | (b)'s screen ends after | the 3-point sequence first fails at | the binding constraint | outcome |
|---|---|---|---|---|
| 1, 0 | step 6 | step 4 | confirmation, before the arms diverge | all equal |
| 1, 1 | step 5 | step 3 | confirmation, before the arms diverge | all equal |
| 1, 2 | step 5 | step 5 (inside (b)'s screen) | confirmation, before the arms diverge | all equal |
| 2, 0 | step 3 (all 3 confirmed) | step 4 in (a), the step (b) rejected | **the gate** | (c)'s KL pick at step 4 confirmed: **+1 step** |
| 2, 1 | step 4 (all 4 confirmed) | step 5 in (a), the step (b) rejected | **the gate** | (c)'s KL pick at step 5 confirmed: **+1 step** |
| 2, 2 | step 4 | step 4 (inside (b)'s screen) | confirmation, before the arms diverge | all equal |

* Checked against the frozen orders and journals:
  * in every pair the three arms' checkpoints are identical (same mask hashes) up to the end of (b)'s screen;
  * arm (a)'s next step is exactly the candidate (b) rejected;
  * arm (c)'s next step is its KL pick.
* In model 1 and in model 2 seed 2, confirmation at 3 points fails inside the shared prefix. The amendment-3 reason
  (seen in the pilot) recurs in 4 of 6 model × seed pairs.
* Where the gate stopped first (model 2 seeds 0 and 1):
  * the step it rejected also failed confirmation in arm (a);
  * arm (c)'s KL-probe replacement passed and was confirmed.

**All 18 runs:**

| model | run | stop | accepts / attempts | screened final | tested / frozen | confirmed (checkpoint) |
|---|---|---|---|---|---|---|
| 1 | ppl_top1 s0 (pilot) | proposal cap | 30 / 30 | 566.2 M | 4 / 30 | 56.6 M (3) |
| 1 | cap_top1 s0 (pilot) | candidate rejected | 6 / 7 | 113.2 M | 4 / 6 | 56.6 M (3) |
| 1 | cap_klprobe s0 (pilot) | fallback exhausted | 9 / 14 | 169.9 M | 4 / 9 | 56.6 M (3) |
| 1 | ppl_top1 s1 | proposal cap | 30 / 30 | 566.2 M | 3 / 30 | 37.7 M (2) |
| 1 | cap_top1 s1 | candidate rejected | 5 / 6 | 94.4 M | 3 / 5 | 37.7 M (2) |
| 1 | cap_klprobe s1 | fallback exhausted | 6 / 9 | 113.2 M | 3 / 6 | 37.7 M (2) |
| 1 | ppl_top1 s2 | proposal cap | 30 / 30 | 566.2 M | 5 / 30 | 75.5 M (4) |
| 1 | cap_top1 s2 | candidate rejected | 5 / 6 | 94.4 M | 5 / 5 | 75.5 M (4) |
| 1 | cap_klprobe s2 | fallback exhausted | 8 / 12 | 151.0 M | 5 / 8 | 75.5 M (4) |
| 2 | ppl_top1 s0 | proposal cap | 30 / 30 | 471.9 M | 4 / 30 | 47.2 M (3) |
| 2 | cap_top1 s0 | candidate rejected | 3 / 4 | 47.2 M | 3 / 3 | 47.2 M (3) |
| 2 | cap_klprobe s0 | fallback exhausted | 6 / 10 | 94.4 M | 5 / 6 | 62.9 M (4) |
| 2 | ppl_top1 s1 | proposal cap | 30 / 30 | 471.9 M | 5 / 30 | 62.9 M (4) |
| 2 | cap_top1 s1 | candidate rejected | 4 / 5 | 62.9 M | 4 / 4 | 62.9 M (4) |
| 2 | cap_klprobe s1 | fallback exhausted | 8 / 14 | 125.8 M | 6 / 8 | 78.6 M (5) |
| 2 | ppl_top1 s2 | proposal cap | 30 / 30 | 471.9 M | 4 / 30 | 47.2 M (3) |
| 2 | cap_top1 s2 | candidate rejected | 4 / 5 | 62.9 M | 4 / 4 | 47.2 M (3) |
| 2 | cap_klprobe s2 | fallback exhausted | 7 / 12 | 110.1 M | 4 / 7 | 47.2 M (3) |

**Power achieved** (joint power of the IUT at the observed disagreement of the claimed checkpoint):

| true loss in every category | model 1 | model 2 |
|---|---|---|
| 0 | 0.99–1.00 | 0.97–1.00 |
| 1.0 point | 0.45–0.68 | 0.35–0.59 |
| 1.5 points | 0.06–0.16 | 0.04–0.12 |

## C1: pool-to-confirmation gap (confirmation estimate − gate pool estimate at acceptance; arms (b) and (c))

**§6 criterion:** in both models, the mean worst-category gap over the tested checkpoints is > 0 and its 95 % bootstrap
interval excludes 0. The bootstrap was fixed in A3.3: percentile, 10,000 resamples, `default_rng(0)`, with checkpoints
counted per run.

| model | tested checkpoints | STEM: mean / max gap | Humanities | **Social Sciences** | Other | worst-category gap: mean [95 % CI] | met |
|---|---|---|---|---|---|---|---|
| 1 | 24 | −0.39 / +0.63 | +0.14 / +2.44 | **+0.76 / +2.70** | −0.28 / +1.31 | **+0.39 [+0.18, +0.59]** | yes |
| 2 | 26 | +0.14 / +1.81 | −0.71 / +1.29 | **−0.10 / +1.17** | +0.80 / +1.96 | **−0.26 [−0.38, −0.12]** | no |

**C1 is not supported:** the criterion holds in model 1 and fails in model 2.
* **Model 1:** the pool estimate was optimistic. The gap is concentrated in Social Sciences, up to +2.70 points at
  seed 0 checkpoint 3, where the pool said −1.20 and confirmation said +1.50.
* **Model 2:** the pool was on average pessimistic in the worst category.
  * Humanities is −0.71 on average.
  * Other is +0.80 on average.
  * Social Sciences averages −0.10.

**Social Sciences gate pool:** 954 items in both models. The other categories' pools are STEM 4,231, Humanities 3,574
and Other 2,373. Its gap at every tested checkpoint is in the table below.

**Descriptive, amendment 3** (every scored checkpoint of (b) and (c), including those after the first failure):
* model 1: +0.59 [+0.43, +0.75] over 39 checkpoints;
* model 2: −0.15 [−0.31, +0.01] over 32 checkpoints.

**Post hoc sensitivity: each distinct checkpoint counted once.** Arms (b) and (c) share their early checkpoints, and the
pre-specified count includes a shared checkpoint once in each run.
* model 1: +0.39 [+0.10, +0.67] over 12 checkpoints;
* model 2: −0.19 [−0.38, +0.01] over 15 checkpoints.

The verdict is the same.

Gap per distinct tested checkpoint (points; worst-category gap = confirmation worst category − pool worst category):

| model | seed | checkpoint | arms | STEM | Humanities | **Social Sciences** | Other | worst-category gap |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | (b)+(c) | −0.50 | −0.28 | **+0.43** | −0.58 | −0.49 |
| 1 | 0 | 2 | (b)+(c) | −1.05 | −0.97 | **−0.32** | +0.21 | −0.39 |
| 1 | 0 | 3 | (b)+(c) | +0.25 | −0.53 | **+2.70** | −0.33 | +1.11 |
| 1 | 0 | 4 | (b)+(c) | +0.13 | −0.75 | **+0.83** | −0.31 | +0.83 |
| 1 | 1 | 1 | (b)+(c) | −0.98 | +1.17 | **+0.73** | −0.52 | +0.43 |
| 1 | 1 | 2 | (b)+(c) | +0.13 | +1.08 | **+0.32** | −0.40 | +0.32 |
| 1 | 1 | 3 | (b)+(c) | −0.66 | −1.10 | **+1.44** | −1.08 | +0.67 |
| 1 | 2 | 1 | (b)+(c) | +0.18 | +0.34 | **+0.60** | +1.31 | +0.30 |
| 1 | 2 | 2 | (b)+(c) | −0.66 | +0.97 | **+0.53** | −0.17 | +0.14 |
| 1 | 2 | 3 | (b)+(c) | −0.66 | −0.45 | **−0.11** | −0.61 | −0.12 |
| 1 | 2 | 4 | (b)+(c) | +0.63 | −0.23 | **+0.65** | −0.28 | +0.65 |
| 1 | 2 | 5 | (b)+(c) | −1.45 | +2.44 | **+1.31** | −0.60 | +1.20 |
| 2 | 0 | 1 | (b)+(c) | +1.81 | −1.27 | **+0.80** | +1.23 | −0.77 |
| 2 | 0 | 2 | (b)+(c) | −0.56 | −1.95 | **+1.17** | +1.91 | −0.24 |
| 2 | 0 | 3 | (b)+(c) | −0.86 | −1.70 | **−1.01** | +0.13 | −0.39 |
| 2 | 0 | 4 | (c) | −1.02 | −0.90 | **−0.17** | +0.96 | +0.38 |
| 2 | 0 | 5 | (c) | −1.33 | +1.02 | **−0.31** | +0.74 | +0.61 |
| 2 | 1 | 1 | (b)+(c) | +0.08 | −1.07 | **−0.70** | +1.53 | −0.47 |
| 2 | 1 | 2 | (b)+(c) | −0.88 | +0.09 | **+0.10** | −0.28 | −0.28 |
| 2 | 1 | 3 | (b)+(c) | +0.42 | +0.32 | **−0.69** | +0.49 | −0.50 |
| 2 | 1 | 4 | (b)+(c) | −0.20 | +0.37 | **−0.79** | +1.76 | −0.46 |
| 2 | 1 | 5 | (c) | −1.26 | −0.04 | **−0.18** | +0.40 | +0.23 |
| 2 | 1 | 6 | (c) | −0.17 | +1.29 | **−0.47** | −0.24 | −0.24 |
| 2 | 2 | 1 | (b)+(c) | +0.39 | −1.50 | **+0.63** | −0.17 | −0.43 |
| 2 | 2 | 2 | (b)+(c) | +1.73 | −1.43 | **−0.66** | +1.96 | −0.18 |
| 2 | 2 | 3 | (b)+(c) | +0.32 | −1.64 | **−0.42** | +1.31 | +0.32 |
| 2 | 2 | 4 | (b)+(c) | +1.46 | −0.11 | **+0.82** | −0.47 | −0.47 |

## C4: arm (a), the perplexity screen (descriptive)

Arm (a) accepted all 30 proposals in every run. Its screened endpoint removes 566.2 M parameters (18.4 %) in model 1 and
471.9 M (18.6 %) in model 2.

Per-category estimated loss at that endpoint, on the confirmation set (points):

| model | seed | first failing checkpoint | STEM | Humanities | Social Sciences | Other | > 3 points | (a) confirms less than (b) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 4 | +9.92 | +10.50 | +17.97 | +14.43 | yes | no |
| 1 | 1 | 3 | +11.19 | +10.67 | +20.73 | +14.91 | yes | no |
| 1 | 2 | 5 | +12.39 | +10.63 | +20.67 | +13.31 | yes | no |
| 2 | 0 | 4 | +16.83 | +20.13 | +26.23 | +25.69 | yes | no |
| 2 | 1 | 5 | +17.92 | +18.53 | +32.13 | +24.94 | yes | no |
| 2 | 2 | 4 | +14.31 | +14.63 | +24.70 | +19.60 | yes | no |

**§6 criterion:** (a) confirms less than (b) in ≥ 2 of 3 seeds, **and** its endpoint exceeds 3 points in ≥ 2 of 3
seeds, in both models. **Not met.**
* The second part holds in 6 of 6 seeds: losses of +10 to +32 points, worst in Social Sciences in every seed.
* The first part holds in 0 of 6. Confirmation stops arm (a) at 2–4 steps, which is never less than (b)'s claim,
  because the arms share those steps (see the primary endpoint).

The fixed sequence protects arm (a)'s claim. The perplexity screen itself goes 26–28 steps past the last confirmable
checkpoint.

## C6: counterfactual log of arm (c), pooled over models and seeds (the rejection is the unit)

The measure is the hold-out worst-category net flip loss of each proposer's pick at each rank-1 rejection (lower is
better). The test is a two-sided exact sign test, ties excluded.
* The plan expected 6–12 rejections; there were 21, because more KL picks were accepted than expected.
* TV was favoured **post hoc** on Qwen3-1.7B, so TV vs KL here is the first test of that observation.

| comparison | rejections | better | worse | ties | p (two-sided) |
|---|---|---|---|---|---|
| **primary: KL vs gradient rank 2, all** | 21 | 11 | 8 | 2 | **0.648** |
| secondary: TV vs KL, all | 21 | 5 | 4 | 12 | 1.000 |
| secondary: S1c vs KL, all | 21 | 5 | 8 | 8 | 0.581 |
| primary, excluding the pilot | 17 | 7 | 8 | 2 | 1.000 |
| TV vs KL, excluding the pilot | 17 | 5 | 2 | 10 | 0.453 |
| S1c vs KL, excluding the pilot | 17 | 5 | 7 | 5 | 0.774 |

**§6 criterion** (KL beats rank 2 in a majority of rejections and p < 0.05): **not met**. KL vs gradient rank 2 is
"not established".
* The pilot's 4 of 4 did not hold up. In model 1 seeds 1–2, KL was better in 1 of 5 rejections.
* In model 2, KL was better in 6, worse in 4, with 2 ties. One tie is a rejection where KL's pick was rank 2 itself.
* TV and KL picked the same candidate in 12 of 21 rejections; those are exactly the 12 ties.

| model | seed | rejection | KL (rank) | gradient rank 2 | TV (rank) | S1c (rank) | KL vs rank 2 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | +3.00 (10) | +6.00 | +3.00 (10) | +3.00 (8) | better |
| 1 | 0 | 2 | +3.50 (8) | +6.50 | +4.00 (9) | +5.50 (5) | better |
| 1 | 0 | 3 | +4.00 (7) | +7.00 | +4.00 (7) | +4.00 (6) | better |
| 1 | 0 | 4 | +2.50 (8) | +5.50 | +4.50 (7) | +2.50 (8) | better |
| 1 | 1 | 1 | +5.00 (11) | +4.00 | +3.00 (5) | +3.00 (4) | worse |
| 1 | 1 | 2 | +5.00 (5) | +6.00 | +6.50 (11) | +3.50 (6) | better |
| 1 | 2 | 1 | +4.00 (10) | +3.50 | +2.00 (6) | +2.00 (7) | worse |
| 1 | 2 | 2 | +4.50 (6) | +4.00 | +4.50 (6) | +6.50 (11) | worse |
| 1 | 2 | 3 | +6.50 (5) | +5.00 | +4.50 (7) | +4.00 (6) | worse |
| 2 | 0 | 1 | +1.50 (2) | +1.50 | +1.50 (2) | +2.00 (10) | tie |
| 2 | 0 | 2 | +2.00 (11) | +3.50 | +2.00 (11) | +2.00 (11) | better |
| 2 | 0 | 3 | +6.50 (10) | +7.00 | +6.50 (10) | +7.00 (2) | better |
| 2 | 1 | 1 | +4.50 (4) | +5.50 | +4.50 (4) | +5.50 (2) | better |
| 2 | 1 | 2 | +2.00 (8) | +3.50 | +2.00 (8) | +1.50 (3) | better |
| 2 | 1 | 3 | +3.00 (7) | +3.00 | +4.50 (8) | +4.50 (8) | tie |
| 2 | 1 | 4 | +2.00 (6) | +3.00 | +2.00 (6) | +4.00 (7) | better |
| 2 | 1 | 5 | +4.00 (6) | +3.50 | +4.00 (6) | +4.00 (6) | worse |
| 2 | 2 | 1 | +6.00 (8) | +3.00 | +3.50 (5) | +6.00 (8) | worse |
| 2 | 2 | 2 | +6.50 (4) | +3.50 | +5.50 (11) | +6.50 (4) | worse |
| 2 | 2 | 3 | +7.00 (4) | +4.00 | +7.00 (4) | +7.50 (5) | worse |
| 2 | 2 | 4 | +5.00 (11) | +5.50 | +5.00 (11) | +5.00 (11) | better |

## S1–S4 (added after the pilot; the pilot's values are post hoc; descriptive)

Per arm: mean [range] over seeds. Then the per-seed paired differences, in seed order.

### Model 1, all seeds (seed 0 = pilot, post hoc)

| endpoint | (a) | (b) | (c) | (b) − (a) per seed | (c) − (b) per seed |
|---|---|---|---|---|---|
| S1 overshoot | 509.6 M [490.7, 528.5] | 44.0 M [18.9, 56.6] | 88.1 M [75.5, 113.2] | −453.0, −471.9, −471.9 M | +56.6, +18.9, +56.6 M |
| S2 items per confirmed M | 1,195 [1,041, 1,388] | 2,007 [1,531, 2,572] | 2,828 [2,232, 3,331] | +762, +1,183, +490 | +1,003, +759, +701 |
| S2 GPU s per confirmed M | 19.4 [16.1, 23.4] | 33.6 [24.2, 44.6] | 48.5 [37.7, 58.2] | +13.6, +21.2, +8.0 | +17.4, +13.6, +13.5 |
| S3 confirmed at 5 points | 81.8 M [56.6, 94.4] | 81.8 M [56.6, 94.4] | 88.1 M [56.6, 113.2] | 0, 0, 0 | 0, 0, +18.9 M |

### Model 1, excluding seed 0 (the pilot)

| endpoint | (a) | (b) | (c) | (b) − (a) per seed | (c) − (b) per seed |
|---|---|---|---|---|---|
| S1 overshoot | 509.6 M [490.7, 528.5] | 37.7 M [18.9, 56.6] | 75.5 M [75.5, 75.5] | −471.9, −471.9 M | +18.9, +56.6 M |
| S2 items per confirmed M | 1,215 [1,041, 1,388] | 2,051 [1,531, 2,572] | 2,781 [2,232, 3,331] | +1,183, +490 | +759, +701 |
| S2 GPU s per confirmed M | 19.8 [16.1, 23.4] | 34.4 [24.2, 44.6] | 47.9 [37.7, 58.2] | +21.2, +8.0 | +13.6, +13.5 |
| S3 confirmed at 5 points | 75.5 M [56.6, 94.4] | 75.5 M [56.6, 94.4] | 84.9 M [56.6, 113.2] | 0, 0 | 0, +18.9 M |

### Model 2, all seeds

Excluding model 1 seed 0 changes nothing here.

| endpoint | (a) | (b) | (c) | (b) − (a) per seed | (c) − (b) per seed |
|---|---|---|---|---|---|
| S1 overshoot | 419.4 M [408.9, 424.7] | 5.2 M [0.0, 15.7] | 47.2 M [31.5, 62.9] | −424.7, −408.9, −408.9 M | +31.5, +47.2, +47.2 M |
| S2 items per confirmed M | 1,342 [1,249, 1,388] | 1,596 [1,404, 1,921] | 2,456 [2,152, 2,896] | +74, +155, +533 | +690, +917, +975 |
| S2 GPU s per confirmed M | 18.6 [16.8, 19.5] | 21.7 [18.9, 26.5] | 37.2 [31.2, 45.3] | +0.2, +2.1, +7.1 | +11.5, +16.2, +18.7 |
| S3 confirmed at 5 points | 73.4 M [47.2, 110.1] | 57.7 M [47.2, 62.9] | 104.9 M [78.6, 125.8] | 0, 0, −47.2 M | +31.5, +62.9, +47.2 M |

### S4: certifiable compression vs confirmation sample size

Each cell is the 3-point procedure applied to 50 seeded, category-stratified subsamples per size. It shows the median
[IQR] over all draws of the arm's seeds. The full set is a single value per seed.

| per-category n | model 1, every arm | model 2, (a) and (b) | model 2, (c) |
|---|---|---|---|
| 500 | 0.0 M [0.0, 0.0] | 0.0 M [0.0, 0.0] | 0.0 M [0.0, 0.0] |
| 1,000 | 18.9 M [0.0, 18.9] | 31.5 M [15.7, 31.5] | 31.5 M [15.7, 31.5] |
| 2,000 | 37.7 M [18.9, 56.6] | 47.2 M [31.5, 47.2] | 47.2 M [31.5, 47.2] |
| full (3,000–3,600) | 56.6 M [47.2, 66.1] | 47.2 M [47.2, 55.1] | 62.9 M [55.1, 70.8] |

Excluding model 1 seed 0 leaves model 1's S4 medians and IQRs unchanged. Per-run medians are in `study.md`.

### What S1–S4 show (descriptive)

* **S1, overshoot.** The ordering (a) ≫ (c) > (b) holds in every seed of both models:
  * (a): about 410–530 M parameters screened out beyond what could be confirmed, roughly 16–17 % of the model;
  * (b): 0–57 M. In model 2 seeds 0 and 1, (b) confirmed its whole screen;
  * (c): 31–113 M.
* **S2, cost.** Per confirmed parameter, (a) < (b) < (c) in every seed of both models, in items and in GPU seconds.
  The perplexity screen scores no benchmark items. Arm (c) spends more of the cap than it gains in confirmed
  compression.
* **S3, a 5-point limit.** (c) ≥ (b) in every seed.
  * In model 2, (c) − (b) is positive in all 3 seeds (+31.5 to +62.9 M).
  * In model 1, it is positive in seed 2 only.
  * (b) − (a) is 0 except in model 2 seed 2. There, (a) reaches 110.1 M and (b)'s screen ends at 62.9 M.
* **S4, sample size.** At 500 items per category, nothing is certifiable in any run. Certifiable compression grows
  with n, and the full sets (3,000–3,600 per category) give the primary endpoint. A smaller confirmation set would
  have certified less in every run.

## Wall time and cost per run

* **Wall time:** from the supervisor log, first "starting attempt" to "done".
* **GPU seconds** (in parentheses): taken from each run's journal.
* **Confirmation seconds:** estimated as the run's confirmation items × the model's measured rate (A3.2). Model 1 is
  13.1 ms per item and model 2 is 10.3 ms per item; see deviation 1 for model 1.
* The counterfactual log is not charged to the cap and is not in S2.

| model | run | wall (min) | worker starts | gate questions (s) | KL-probe items (s) | charged to cap | counterfactual items (s) | profiling s | confirmation items (est. s) | S2 items / confirmed M | S2 GPU s / confirmed M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ppl_top1 s0 (pilot) | 8.1 | 1 | 0 (190, perplexity) | 0 | 0 | 0 | 265 | 65,500 (859) | 1,157 | 18.5 |
| 1 | cap_top1 s0 (pilot) | 17.3 | 1 | 43,162 (961) | 0 | 43,162 | 0 | 62 | 65,500 (859) | 1,919 | 32.1 |
| 1 | cap_klprobe s0 (pilot) | 38.9 | 2 | 83,959 (1,758) | 16,000 (190) | 99,959 | 10,400 (121) | 80 | 65,500 (859) | 2,922 | 49.6 |
| 1 | ppl_top1 s1 | 8.9 | 1 | 0 (197, perplexity) | 0 | 0 | 0 | 276 | 52,400 (687) | 1,388 | 23.4 |
| 1 | cap_top1 s1 | 18.1 | 1 | 44,673 (996) | 0 | 44,673 | 0 | 64 | 52,400 (687) | 2,572 | 44.6 |
| 1 | cap_klprobe s1 | 28.0 | 1 | 65,327 (1,414) | 8,000 (96) | 73,327 | 6,400 (77) | 77 | 52,400 (687) | 3,331 | 58.2 |
| 1 | ppl_top1 s2 | 8.2 | 1 | 0 (187, perplexity) | 0 | 0 | 0 | 273 | 78,600 (1,031) | 1,041 | 16.1 |
| 1 | cap_top1 s2 | 14.4 | 1 | 36,985 (794) | 0 | 36,985 | 0 | 53 | 78,600 (1,031) | 1,531 | 24.2 |
| 1 | cap_klprobe s2 | 33.6 | 1 | 77,928 (1,672) | 12,000 (142) | 89,928 | 8,800 (104) | 79 | 78,600 (1,031) | 2,232 | 37.7 |
| 2 | ppl_top1 s0 | 9.9 | 1 | 0 (243, perplexity) | 0 | 0 | 0 | 323 | 65,500 (677) | 1,388 | 19.5 |
| 2 | cap_top1 s0 | 7.4 | 1 | 16,600 (388) | 0 | 16,600 | 0 | 44 | 52,400 (541) | 1,462 | 19.7 |
| 2 | cap_klprobe s0 | 21.8 | 1 | 44,781 (996) | 12,000 (154) | 56,781 | 4,800 (70) | 76 | 78,600 (812) | 2,152 | 31.2 |
| 2 | ppl_top1 s1 | 9.9 | 1 | 0 (242, perplexity) | 0 | 0 | 0 | 323 | 78,600 (812) | 1,249 | 16.8 |
| 2 | cap_top1 s1 | 9.6 | 1 | 22,854 (509) | 0 | 22,854 | 0 | 55 | 65,500 (677) | 1,404 | 18.9 |
| 2 | cap_klprobe s1 | 34.3 | 1 | 70,843 (1,554) | 20,000 (253) | 90,843 | 10,400 (139) | 98 | 91,700 (948) | 2,321 | 35.0 |
| 2 | ppl_top1 s2 | 9.9 | 1 | 0 (243, perplexity) | 0 | 0 | 0 | 323 | 65,500 (677) | 1,388 | 19.5 |
| 2 | cap_top1 s2 | 10.7 | 1 | 25,127 (576) | 0 | 25,127 | 0 | 55 | 65,500 (677) | 1,921 | 26.5 |
| 2 | cap_klprobe s2 | 28.0 | 1 | 55,151 (1,256) | 16,000 (204) | 71,151 | 8,800 (119) | 87 | 65,500 (677) | 2,896 | 45.3 |

* Arm (c) stayed within the 100,000-question cap in every run; the most it used was 99,959, in the pilot.
* Cached scores are shared between runs: an identical checkpoint is scored once.

**Confirmation passes:**

| batch | new checkpoints scored | replayed from cache | wall |
|---|---|---|---|
| model 1 `pilot`, re-confirmed | 26 (340,600 items, 3,832 s) | 19 | 64.5 min |
| model 1 `main` | 64 (838,400 items, 11,702 s) | 20 | 196.6 min |
| model 2 `main` | 100, plus the dense reference (1,323,100 items, 13,671 s) | 22 | 229.9 min |

The re-confirmed pilot's primary claims matched the recorded ones: checkpoint 3 and 56.6 M in all three arms, which
`confirm` checks. The pre-amendment records are in `batches/pilot/superseded/`.

**Matrix timeline (2026-11-24 to 2026-11-25), one GPU, one step at a time:**

| step | start | end | duration |
|---|---|---|---|
| model 1 screening, seeds 1–2 | 20:15 | 22:07 | 1 h 51 min |
| full test suite (154 passed, exit 0) | 22:07 | 22:23 | 15.7 min |
| model 1 freeze, then confirm `pilot` and `main` | 22:23 | 02:44 | 4 h 21 min |
| model 2 screening, seeds 0–2 | 02:44 | 05:06 | 2 h 22 min |
| model 2 freeze, then confirm `main` | 05:06 | 08:56 | 3 h 50 min |
| reports and study summary | 08:56 | 08:56 | under 1 min |

## C3, the procedure record; C2 and C5

* **C3.** Each batch was frozen before it was confirmed:
  * model 1 `main` was frozen at 22:23, and model 2 `main` at 05:06;
  * the pilot was frozen on 2026-11-24 at `c71f869`.

  Every confirmation pass ran at commit `21ab08d` with a clean tree. The sealed confirmation sets are read only by
  `confirm`. The primary claims come from the fixed sequence over each frozen order. Checkpoints after a first failure
  were scored only as amendment-3 description.
* **C2** (retrospective population re-gating) and **C5** (a gate-cache replay) are separate analyses. They are not part
  of this report.

## Deviations from the plan

1. **GPU memory contention during model 1.**
   * **The rule:** A3.5 requires no other GPU job.
   * **What happened:** a desktop lighting application (Corsair iCUE, its `QmlRenderer` process) held about 2.7 GB of the
     GPU's 32.6 GB until about 01:25 on 2026-11-25, when it was closed. It ran no compute. But with it open, the
     confirmation process spilled about 1.3 GB into shared system memory.
   * **Effect on model 1's main confirmation:** checkpoints took 200–330 s each, against 134–146 s after the
     application was closed.
   * **What this affects:**
     * model 1's confirmation wall time;
     * model 1's measured confirmation rate: 13.1 ms per item per A3.2, against about 10.6 ms per item from the
       per-checkpoint wall timings after closing;
     * therefore the level of model 1's S2 GPU seconds. At the unaffected rate they would be 6–16 % lower, most for
       arm (a), whose cost is mostly confirmation.
   * **What it does not affect:**
     * the per-seed S2 differences between arms in model 1, because there all three arms of a seed test the same
       checkpoints;
     * any score or decision: the scores do not depend on speed, and no out-of-memory fallback occurred.
   * **Not known:** the application was probably also open during model 1's screening (pilot and seeds 1–2). Those
     screening wall times may be slightly inflated; this cannot be checked after the fact.
   * **Model 2** ran entirely with it closed, so wall times are not strictly comparable between the models.
2. **CPU-only tests overlapped model 1 screening.** Test runs during code work overlapped the end of `ppl_top1` seed 1
   and the start of `cap_top1` seed 1, about 20:21–20:29 on 2026-11-24.
   * The GPU was hidden from them (`CUDA_VISIBLE_DEVICES=-1`), so no GPU job overlapped.
   * Those two runs' wall times may be slightly inflated by CPU contention.
3. **Commits differ by batch.** This is recorded, not a change of procedure.

   | what | commit |
   |---|---|
   | pilot screening | `f0152c8` (freeze at `c71f869`; see the pilot report) |
   | model 1 seeds 1–2 screening | `125db01`, the amendment-3 commit |
   | model 2 screening, and every freeze and confirmation since | `21ab08d` |

   * Between `125db01` and `21ab08d`, only the confirmation, report and study code, the CLI, the configs' `secondary` and
     `study` keys, the tests and the driver script changed. The screening code is identical.
   * All 18 runs record a clean tree and the same RTX 5090.
4. **The pilot's arm (c) restarted once.** `cap_klprobe` seed 0 has 2 worker starts. This is the supervisor heartbeat
   bug fixed before the matrix, already in the pilot report, and its wall time includes the restart.
5. **The C1 count includes shared checkpoints.** A checkpoint shared by (b) and (c) enters once per run, as A3.3
   pre-specified, which narrows the interval. The post hoc distinct-checkpoint sensitivity above leaves the verdict
   unchanged.

Nothing else departed from the plan. The data splits, arms, caps, confirmation rule and primary endpoint are as
registered.

## Summary of the claims

| claim | result |
|---|---|
| Primary endpoint | No consistent difference between arms in either model. Model 1: identical in every seed. Model 2: (a) = (b); (c) − (b) = +15.7, +15.7, 0 M. |
| C1 | Not supported: met in model 1 (+0.39 [+0.18, +0.59]), not in model 2 (−0.26 [−0.38, −0.12]). |
| C3 | The procedure ran as specified (see the C3 section above). |
| C4 | Not supported: (a) never confirms less than (b), although its endpoints exceed 3 points in every seed (+10 to +32). |
| C6 | Not established: KL beats rank 2 in 11 of 19 untied rejections, p = 0.648. |
| C2, C5 | Not evaluated here. |
| S1–S4 (added after the pilot) | Overshoot (a) ≫ (c) > (b); cost per confirmed parameter (a) < (b) < (c) in every seed; at 5 points (c) ≥ (b) in every seed; nothing certifiable at 500 items per category. |

## Reproduce

From a clean checkout of `21ab08d`, on a GPU with nothing else running:

```bash
bash scripts/confirmatory_matrix.sh .venv/Scripts/python.exe ~/cap-prune-runs/confirmatory/matrix-logs
```

The script runs these steps in order and stops at the first non-zero exit code:
1. the recorded test suite;
2. `confirmatory freeze`, `confirm` and `report` for model 1 (`pilot` and `main`);
3. `supervise` for model 2's seeds 0–2, then `freeze`, `confirm` and `report`;
4. `confirmatory study`.

Confirmation is cached, so a rerun replays the stored scores.
