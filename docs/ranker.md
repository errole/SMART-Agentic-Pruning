# Learned ranking of structured pruning actions (offline comparison, 2026-11-22)

A small supervised model, one ridge regression per MMLU category, takes over proposal selection from the LLM
controllers. The existing three-point capability gate still makes the final decision. The model being pruned is
Qwen3-1.7B.
* Code: [`capprune/ranker.py`](../capprune/ranker.py)
* Configs: [`configs/ranker.yaml`](../configs/ranker.yaml) and
  [`configs/ranker_smoke.yaml`](../configs/ranker_smoke.yaml)
* Outputs: `~/cap-prune-runs/ranker/data/`

The Gemini, Ollama and fixed-state outputs are only read.

## Data

**Questions.** The pruning-time pool (MMLU test, 40 %) is split once by seed:
* **Probe set:** 100 questions per category (400). Only the probe-based selectors use it.
* **Label set:** the other 5,214 questions. They provide the labels (exact per-category loss vs. the dense model)
  and the capability gate's questions (threshold-aware, 3 points, budget = the whole label set, one question
  order per parent).

The held-out audit split is not used.

**Trajectories.** Four new trajectories (T1–T4) start from the dense model, and each takes 3 accepted steps.
* **Candidate set at each parent:** the greedy-risk set, lowest z_sens, lowest z_grad, seven depth windows and
  two seeded random unit sets, all with matched removal (4 units × 5 blocks = 1,280 MLP channels).
* **Next parent:** a seeded random accepted candidate that is not a parent in any other trajectory.
* **Dense parent:** shared by every trajectory, so it forms its own group (D0) and is used for training only.

**Development.** The fixed-state pilot's four checkpoints (G: dense, 1.12 %, 2.42 %, 3.35 %), their 9-action
shortlists and the LLMs' choices, all rescored under this protocol.

**Per state–action record:**
* trajectory and parent IDs, parent-mask hash, parent sparsity and measured parent loss
* affected layers, channels and parameters removed
* Wanda-sensitivity and gradient z-scores and depth features of the chosen units
* probe-set loss and label-set loss per category, the gate decision and questions
* measured probe and label scoring times

Scores are cached by (parent-mask hash, action, evaluation-config hash).

## Selectors (same candidate sets)

| selector | rule | uses scoring before choosing? |
|---|---|---|
| random | expected value of a uniform pick | no |
| greedy | lowest mean 0.5·(z_sens + z_grad) (the existing heuristic) | no |
| gradient | lowest mean z_grad | no |
| probe | lowest measured worst-category loss on the 400 probe questions | yes: probes every candidate |
| ridge | lowest predicted worst-category cumulative loss (parent loss + predicted increment) from pre-scoring features | no |
| ridge + probe | as ridge, with the four probe losses added as features | yes: probes every candidate |
| gradient / ridge sequential @ probe budget | gate the ranked candidates in order until one passes, within the probe selector's question budget | no |

**Ridge details.**
* Features: parent sparsity and per-category loss; mean/min/max z_sens and z_grad; risk; mean/min/max depth;
  early/mid/late fractions; current and post-action unit sparsity; parameter share.
* Target: the per-category increment over the parent.
* Scaling and alpha are fitted inside the training trajectories only, with leave-one-trajectory-out CV.
* Test: leave one new trajectory out. The development checkpoints are predicted from the new trajectories only;
  D0 is excluded there because it duplicates their dense parent.

**Cost per decision.**
* profiling of the parent (shared)
* for probe selectors, 400 questions on every candidate, chosen or not
* the gate questions of the chosen candidate

GPU seconds come from measured timings. Offline data collection and training are reported separately.

## Commands

```powershell
python -m pytest tests/test_live_gate.py -q -k ranker          # CPU smoke: trajectory split, pre-scoring features, probe cost
python -m capprune ranker collect  --config configs/ranker.yaml  # GPU, 92.7 min measured on the RTX 5090
python -m capprune ranker evaluate --config configs/ranker.yaml  # CPU, seconds -> report.md, evaluation.json
```

## Results (run of 2026-11-22)

**Bottom line.** On unseen trajectories, the learned ridge selector ties gradient-only selection: worst-category loss
+1.10 vs. +1.09 points, difference +0.01 [-0.13, +0.14]. Its per-decision cost is also the same (18.0 vs. 18.1 GPU s),
and it needed 92.7 minutes of offline collection. Measured probes add cost but not accuracy. The precondition for
the live comparison (beating the strongest simple baseline) was not met, so the live comparison was not run.

### Smoke test

`tests/test_live_gate.py::test_ranker_splits_by_trajectory_uses_only_prescoring_features_and_charges_probes`
runs collect and evaluate on the tiny CPU model and checks three things:
* **Trajectory split.** Every non-dense parent mask belongs to one trajectory, and the dense parent is the separate
  D0 group. Every fold tests exactly one whole trajectory. Evaluation refuses a decision file whose parent mask
  appears on both sides of a split.
* **Pre-scoring features.** The ridge and gradient rankings don't change when the test candidates' labels and probe
  answers are scrambled.
* **Probe cost.** Every probe-based decision is charged |candidates| × 400 probe questions plus the probe GPU
  seconds.

The real run passed the same split check in every fold.

### Data collected

* **Trajectories.** Four new trajectories, each with parents at 0.74 %, 1.49 % and 2.23 % sparsity. Every action
  removes 1,280 MLP channels (7.86 M parameters).
* **Candidates.** 11 per parent, scored on all 400 probe and 5,214 label questions.
* **Totals.** 192 distinct state–action scorings plus the 4 rescored development checkpoints (13 candidates each:
  9 on the shortlist plus 4 LLM free-form actions).

Candidates accepted by the gate per parent:

| trajectory | dense | step 1 | step 2 | step 3 |
|---|---|---|---|---|
| T1 | 11/11 | 10/11 | 10/11 | 11/11 |
| T2 | 11/11 | 11/11 | 11/11 | 11/11 |
| T3 | 10/11 | 10/11 | 10/11 | 6/11 |
| T4 | 10/11 | 6/11 | 2/11 | 3/11 |

### Unseen trajectories (leave one trajectory out, 12 test decisions)

**Mean worst-category loss vs. dense, with regret vs. the best equal-removal candidate at the same parent**
(accuracy points; lower is better):

| selector | T1 | T2 | T3 | T4 | all 12 | gate accept |
|---|---|---|---|---|---|---|
| random (expected) | +1.09 (+1.07) | +0.80 (+0.91) | +2.45 (+1.33) | +4.34 (+2.12) | +2.17 (+1.35) | 77 % |
| greedy risk | +1.03 (+1.01) | +0.63 (+0.74) | +1.69 (+0.57) | +3.24 (+1.02) | +1.65 (+0.84) | 92 % |
| gradient only | +0.19 (+0.17) | **+0.07** (+0.18) | +1.58 (+0.46) | **+2.51** (+0.29) | **+1.09** (+0.27) | 92 % |
| probe only | **+0.07** (+0.06) | +0.58 (+0.68) | **+1.21** (+0.09) | +2.84 (+0.62) | +1.17 (+0.36) | 92 % |
| ridge | +0.11 (+0.09) | +0.12 (+0.22) | +1.49 (+0.37) | +2.68 (+0.46) | +1.10 (+0.29) | 92 % |
| ridge + probe | +0.11 (+0.09) | +0.24 (+0.34) | +1.61 (+0.49) | +2.68 (+0.46) | +1.16 (+0.35) | 92 % |
| best candidate (oracle) | +0.02 | -0.11 | +1.12 | +2.22 | +0.82 | — |

Every non-random selector was accepted at all 9 parents in T1–T3 and at 2 of 3 in T4. Gradient, probe and ridge
were all rejected at T4/2, where only 2 of the 11 candidates pass. Accept rates therefore don't separate the
selectors.

**Cost per decision** (questions include the probe questions; GPU seconds = profiling + probes + gate, measured):

| selector | T1 | T2 | T3 | T4 | all: questions (probe + gate) | all: GPU s (profiling + probe + gate) |
|---|---|---|---|---|---|---|
| random | 3,137 / 20.3 s | 2,686 / 18.3 s | 3,067 / 19.7 s | 2,615 / 17.6 s | 2,876 (0 + 2,876) | 19.0 (4.9 + 0 + 14.0) |
| greedy | 3,200 / 20.8 s | 2,400 / 16.8 s | 3,320 / 20.9 s | 3,741 / 23.0 s | 3,165 (0 + 3,165) | 20.4 (4.9 + 0 + 15.4) |
| gradient | 2,500 / 17.2 s | 1,667 / 13.3 s | 3,233 / 20.5 s | 3,407 / 21.5 s | 2,702 (0 + 2,702) | 18.1 (4.9 + 0 + 13.2) |
| probe | 6,867 / 38.8 s | 6,733 / 38.7 s | 7,733 / 42.6 s | 8,094 / 44.5 s | 7,357 (4,400 + 2,957) | 41.1 (4.9 + 21.8 + 14.4) |
| ridge | 2,400 / 16.7 s | 1,733 / 13.6 s | 3,200 / 20.3 s | 3,428 / 21.6 s | 2,690 (0 + 2,690) | 18.0 (4.9 + 0 + 13.1) |
| ridge + probe | 7,000 / 39.4 s | 6,233 / 36.4 s | 7,633 / 42.1 s | 7,807 / 43.2 s | 7,168 (4,400 + 2,768) | 40.3 (4.9 + 21.8 + 13.5) |

**Spending the probe budget on the gate instead.** "Gradient / ridge sequential at the probe budget" gates the
ranked candidates in turn within the probe selector's question budget. It changes the outcome only at T4/2. There,
three candidates are gated and all are rejected, so worst-category loss and accept rate are unchanged and the
decision costs more questions (2,979 / 2,967 per decision on average).

**Ranking quality** (mean Spearman correlation within each parent between the selector's ranking and the true
worst-category loss):

| greedy | gradient | probe | ridge | ridge + probe |
|---|---|---|---|---|
| +0.22 | +0.73 | +0.53 | +0.75 | +0.74 |

**Measurement uncertainty.** Paired bootstrap over label questions: 2,000 draws, resampled within each category,
with one draw shared by all candidates. Choices stay fixed, because they don't depend on label answers. This
covers question sampling only, not variation between trajectories.

The median standard error of one category loss is 0.77 points on the label set (about 1,300 questions per
category) and 2.96 points on the probe set (100 per category). The probe error alone is about the size of the
3-point tolerance.

| A − B (mean worst-category loss) | difference | 95 % interval | share of draws where A is lower |
|---|---|---|---|
| ridge − gradient | +0.01 | [-0.13, +0.14] | 46 % |
| ridge − probe | -0.07 | [-0.43, +0.28] | 64 % |
| ridge + probe − ridge | +0.06 | [-0.09, +0.21] | 21 % |
| probe − gradient | +0.09 | [-0.30, +0.46] | 35 % |
| ridge + probe − gradient | +0.07 | [-0.10, +0.23] | 22 % |
| gradient − greedy | -0.56 | [-1.07, -0.07] | 99 % |
| probe − greedy | -0.48 | [-0.90, -0.07] | 99 % |
| greedy − random | -0.52 | [-0.94, -0.11] | 99 % |

**What ridge learned.** Ridge alpha was 100 or 1,000 in every fold, i.e. heavy shrinkage. Ridge picked "layers
24–27" in 11 of the 12 unseen decisions (layers 20–23 in the other) and in all 4 development decisions. In effect it
learned a fixed depth rule next to gradient's picks (layers 21–25).

### Development: the fixed-state pilot's four checkpoints (G)

The models were trained on T1–T4 only. These checkpoints and the "lowest z_grad" finding were inspected in the
fixed-state study, so this is development evidence, not a test.
* **LLM rows:** the pilot's recorded Qwen2.5 choices (3 repeats per state), reused because the parent masks and
  shortlist are identical. Their candidates are rescored on this label set.
* **Probe cost:** 9 × 400 = 3,600 questions per decision.

| selector | dense | s1 (1.12 %) | s2 (2.42 %) | s3 (3.35 %) | mean | gate accept |
|---|---|---|---|---|---|---|
| gradient | +0.09 | +0.00 | +2.58 | +1.64 | +1.08 | 100 % |
| probe | +0.69 | +0.00 | +3.10 (reject) | +1.12 | +1.23 | 75 % |
| ridge = ridge + probe | +0.06 | -0.11 | +2.33 | +1.12 | **+0.85** | 100 % |
| best shortlist candidate | +0.06 | -0.11 | +2.33 | +1.12 | +0.85 | — |
| greedy / random | | | | | +3.34 / +2.72 | 25 % / 58 % |
| Qwen2.5-14B shortlist | +1.89 | +2.33 | +5.68 | +3.53 | +3.36 | 50 % |
| Qwen2.5-7B shortlist | | | | | +3.19 | 33 % |
| Qwen2.5-14B free-form | | | | | +3.15 | 58 % |
| Qwen2.5-7B free-form | | | | | +3.20 | 50 % |

### Offline cost (reported separately)

| item | value |
|---|---|
| distinct state–action scorings | 192 (about 1.08 M questions) |
| scoring GPU time | 87.8 min |
| profiling GPU time | 1.7 min |
| collection wall time | 92.7 min (including model loading and gate replays) |
| ridge training and evaluation | 0.03 s CPU |

Against probe-only selection, ridge saves about 23 GPU s per decision, so the collection pays for itself after
roughly 240 decisions. Against gradient-only selection it saves nothing.

### Answers

1. **Do measured capability probes help?** Not beyond the gradient score.
   * A 400-question probe (100 per category) has a 3-point standard error per category, so it cannot resolve
     differences on the scale of the tolerance. Its within-parent rank correlation with the true loss is 0.53, vs.
     0.73 for z_grad.
   * Probe-only is +0.09 [-0.30, +0.46] vs. gradient-only. Adding probe features to ridge changes the loss by
     +0.06 [-0.09, +0.21].
   * Probes do carry signal: probe-only beats greedy risk by 0.48 [0.07, 0.90] points. But they cost 4,400
     questions per decision, more than the gate itself.
2. **Does learning improve on the probe-only and gradient-only baselines?**
   * **Probe-only:** same quality (-0.07 [-0.43, +0.28]) at 37 % of the questions and 44 % of the GPU time.
   * **Gradient-only:** no (+0.01 [-0.13, +0.14]). Ridge mostly picks layers 24–27, which is about as good as
     gradient's layers 21–25.
   * On the development checkpoints ridge matched the oracle (+0.85 vs. +1.08 for gradient). Those states were
     already inspected, and the unseen trajectories don't reproduce the gap.
3. **Does any improvement survive the full evaluation cost?**
   * **Vs. gradient-only:** there is no improvement to survive. Ridge's per-decision cost equals gradient-only's
     (18.0 vs. 18.1 GPU s), and it also needs 92.7 minutes of offline collection.
   * **Vs. probe-only:** the only advantage is cost (about 23 GPU s and 4,700 questions per decision), not quality.
   * The live comparison was therefore not run.

### Limitations

* 12 unseen decisions from 4 trajectories, all at 0.74–2.23 % parent sparsity with a fixed 0.74 % step.
* Next parents are random accepted candidates, so the trajectories are mild. T1 and T2 rarely leave a selector much
  to gain.
* The bootstrap covers question sampling only. With 4 trajectories, the variation between trajectories is not
  estimated.
* The candidate set is fixed and shared by all selectors. A learned model might help more with a larger or more
  varied action space, where a depth rule is not near-optimal.
