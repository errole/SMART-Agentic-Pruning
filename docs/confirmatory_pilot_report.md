# Confirmatory study: pilot report (model 1, seed 0; 2026-11-24)

The pilot follows [`confirmatory_study_plan.md`](confirmatory_study_plan.md) v1 (commit `f0152c8`), unamended. Under
the plan, these three runs are confirmatory data unless a later amendment changes how they were screened or
confirmed.

* **What was run:** SmolLM3-3B-Base, seed 0, arms (a) perplexity rollback, (b) capability gate and (c) capability
  gate + KL probe, end to end.
* **Order of events:** admissibility, then screening, freeze, confirmation and report.
* **Full tables:** `~/cap-prune-runs/confirmatory/smollm3-3b/batches/pilot/report.md` (and `report.json`,
  `confirm.json`, `*.order.json`, `*.confirm.json`).

## Results

### Admissibility (the model's first scoring)

Dense 0-shot accuracy on the 11,132-item gate pool:

| overall | STEM | Humanities | Social Sciences | Other |
|---|---|---|---|---|
| 54.5 % | 50.8 % | 52.0 % | 66.0 % | 60.1 % |

The model is admissible (thresholds: 40 % overall, 30 % per category). Granite-3.3-2B-Base was also checked and is
admissible: 46.6 % overall, 41.6–54.2 % per category.

### Screening and confirmation

| arm | stop | accepts / attempts | screened final: MLP sparsity, parameters removed | **confirmed** checkpoint, parameters removed |
|---|---|---|---|---|
| (a) perplexity rollback, τ = 15 % | proposal cap | 30 / 30 | 23.26 %, 566.2 M (18.4 %) | **3 of 30: 56.6 M (1.84 %)** |
| (b) capability gate, 3 points | rank-1 rejection (Social Sciences +3.46) | 6 / 7 | 4.65 %, 113.2 M (3.7 %) | **3 of 6: 56.6 M (1.84 %)** |
| (c) gate + KL probe | KL pick rejected (Social Sciences +3.35) | 9 / 14 | 6.98 %, 169.9 M (5.5 %) | **3 of 9: 56.6 M (1.84 %)** |

**All three arms certified the same checkpoint.**
* The arms use the same gradient top-1 selection from the same dense model and seed, so their trajectories are
  identical until a screen first rejects. Arms (a) and (b) share steps 1–6, and (b) and (c) share steps 1–6.
* The fixed sequence failed at checkpoint 4 (75.5 M parameters) in every arm. Social Sciences' upper bound there is
  +3.67 points, against a limit of 3.
* That is before any divergence, so all three runs claim checkpoint 3.

**Per-category bounds** (confirmation estimate / one-sided 95 % upper bound, points; the IUT decides):

| checkpoint (parameters removed) | STEM | Humanities | Social Sciences | Other | pass |
|---|---|---|---|---|---|
| 1 (18.9 M) | +0.22 / +1.27 | −0.17 / +0.86 | +0.23 / +1.29 | −0.74 / +0.35 | yes |
| 2 (37.7 M) | +0.14 / +1.47 | −0.30 / +0.88 | +0.80 / +2.14 | +0.11 / +1.45 | yes |
| 3 (56.6 M) | +0.50 / +1.90 | −0.27 / +1.00 | +1.50 / +2.86 | +0.06 / +1.43 | **yes (claim)** |
| 4 (75.5 M) | +1.03 / +2.55 | +0.50 / +1.97 | +2.17 / **+3.67** | +0.74 / +2.20 | no (stop) |

**Each run's final screened checkpoint, scored as description only (not part of any claim)** (estimates, points):

| arm | STEM | Humanities | Social Sciences | Other |
|---|---|---|---|---|
| (a) | +9.92 | +10.50 | +17.97 | +14.43 |
| (b) | +1.47 | +1.87 | +3.47 | +1.69 |
| (c) | +1.89 | +1.93 | +3.63 | +1.91 |

### Power achieved

At checkpoint 3, the observed paired disagreement was STEM 9.3 %, Humanities 6.1 %, Social Sciences 7.2 % and Other
8.6 %. With those rates and the confirmation sizes, the IUT's joint power is:

| true loss in every category | joint power |
|---|---|
| 0 | 1.00 |
| 1.0 point | 0.66 |
| 1.5 points | 0.15 |

The joint power at 1.0 point is above the plan's 0.48, because disagreement was below the planning rate.

### Secondary endpoints (one seed: descriptive)

**C1, pool-to-confirmation gap.** This is confirmation worst-category estimate minus the gate's pool worst-category
estimate at acceptance, in arm (b).

| checkpoint | 1 | 2 | 3 | 4 | final (6) |
|---|---|---|---|---|---|
| gap (points) | −0.49 | −0.39 | +1.11 | +0.83 | +1.16 |

* Mean +0.45 points; arm (c) is the same, with +1.25 at its final.
* The gap is concentrated in Social Sciences: at checkpoint 3 the pool estimate was −1.20 and the confirmation
  estimate +1.50, a gap of +2.70. The gate's Social Sciences pool has only 954 items.
* The plan's C1 test (bootstrap over checkpoints, both models) cannot be judged from one run.

**C4, perplexity-arm violations.** All 30 perplexity steps were accepted (each step raised perplexity by at most
1.4 %).
* The arm's final checkpoint has estimated losses of +9.9 to +18.0 points in every category.
* Its first confirmation failure is checkpoint 4, the same as the other arms.

**C6, counterfactual log** (arm (c), 4 rank-1 rejections, hold-out worst-category net flip):

| rejection | KL pick | gradient rank 2 |
|---|---|---|
| 1 | +3.00 | +6.00 |
| 2 | +3.50 | +6.50 |
| 3 | +4.00 | +7.00 |
| 4 | +2.50 | +5.50 |

* KL is better at 4 of 4 (sign-test p = 0.125; descriptive, one run).
* TV vs KL: 2 ties, 2 KL better. S1c vs KL: 3 ties, 1 KL better.
* Three of the four KL picks were accepted by the gate; the fourth (rank 8) was rejected and ended the run.

### Cost, each reported separately

| item | arm (a) | arm (b) | arm (c) |
|---|---|---|---|
| screening: gate questions | 0 | 43,162 | 83,959 |
| screening: GPU s (profiling + gate / perplexity) | 461 | 1,025 | 1,838 |
| KL probe: paired items, s | – | – | 16,000, 190 |
| counterfactual: paired items, s (not charged to the cap) | – | – | 10,400, 121 |
| questions charged to the cap | 0 | 43,162 | 99,959 (cap 100,000) |
| **wall time per run** | 8.1 min | 17.4 min | 38.9 min (including one restart; see below) |

**Confirmation (shared by the three runs; identical checkpoints are scored once):**
* Scored: dense once, plus 7 distinct checkpoints (1–4 and the three final checkpoints) on 13,100 items each.
* Final pass: 81,500 checkpoint items in 1,003 s, and 2,800 dense items in 33 s (the rest of dense was cached).
* A first, stopped pass scored about 10,200 checkpoint items and 10,200 dense items in about 31 minutes; see below.

## What went wrong and was fixed

Neither issue changed a procedure. Both are fixed in code.

1. **Supervisor heartbeat** (commit `c71f869`).
   * **The bug:** on Windows the venv's `python.exe` is a launcher. The supervisor ignored the worker's heartbeats
     (different process ID) and killed every worker 30 minutes after it started.
   * **In the pilot:** arm (c) was killed once at 30 minutes while making progress. It resumed from its checkpoint and
     replayed, so its results are unaffected.
   * **Why not seen before:** earlier runs all finished within 30 minutes.
2. **Confirmation memory** (commit `31a9982`).
   * **The bug:** the first confirm pass held four model-sized objects and spilled past the 32 GB GPU.
   * **What happened:** it was stopped after 102 of 131 batches of the first checkpoint, and the batches were kept.
   * **The fix:** confirmation now materializes each checkpoint from the single dense model. The weights are identical,
     so the cached batches were reused.
3. **Profiling memory, before the pilot** (commit `bdcdb22`, recorded in the plan): gradient checkpointing, with a
   bitwise-identical profile.

## Does anything in the plan have to change?

**Nothing had to change to run the pilot.** Two design issues deserve a decision before the full matrix. If either
change is adopted, it needs a dated amendment committed before any further confirmation scoring.

1. **The primary endpoint cannot separate the arms when confirmation fails before the arms diverge.**
   * **Why:** the arms differ only after a screen first rejects. Here the screens diverged at step 7, but
     confirmation failed at step 4, so all three arms confirmed exactly the same compression.
   * **Likely in other seeds too:** the gate accepted steps whose confirmed Social Sciences loss was 2–3.5 points.
   * **Option A:** keep the primary endpoint as it is. Equal confirmed compression despite very different screened
     compression is itself a result for C1 and C4.
   * **Option B:** also pre-specify two secondary endpoints, which do separate the arms here:
     * **screening overshoot:** screened parameters removed minus confirmed parameters removed. Pilot: (a) 509.6 M,
       (b) 56.6 M, (c) 113.2 M;
     * **cost per confirmed parameter.**
   * **Recommendation:** Option B. It adds endpoints without changing any procedure.
2. **Social Sciences limits the gate.**
   * **The problem:** its gate pool is only 954 items. Pool estimates there were up to 2.7 points below confirmation,
     and Social Sciences ended every run.
   * **Option:** reducing its confirmation set from 3,000 to 2,700 (the amendment-1 n80 value) would move 300 items to
     the gate, about +31 %. Social Sciences' per-category power at a 1.0-point true loss would fall from about 0.87 to
     about 0.80.
   * **Constraint:** the data are sealed, so this means re-sealing, and all of model 1's runs would have to be rerun.
   * **Recommendation:** **leave the split as sealed.** Report the Social Sciences gap as part of C1.
3. **Model 2** has passed admissibility and has not been screened. Its batch (`main`, nine runs) is ready.

**Estimated wall time of the remaining matrix:** about 65 minutes of screening per seed and model, plus about 20
minutes of confirmation per batch. Model 1's seeds 1 and 2 and model 2's three seeds come to roughly 6–7 hours of GPU
time.
