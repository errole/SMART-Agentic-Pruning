# Compensation feasibility plan (2026-11-26)

**Status: EXPLORATORY.** It runs on the development model, Qwen3-1.7B, whose items have all been seen before. It makes
no claim. It decides whether a larger study, across operators and model sizes, is worth doing.

This plan is committed and pushed **before any model is scored**. That includes generating the reference trajectory,
which runs the model. The decision rule (§5) is not changed after scoring starts.

**Standing rules:**
* No sealed confirmation set (`~/cap-prune-runs/confirmatory/<model>/sealed/`) is opened.
* Neither confirmatory model (SmolLM3-3B-Base, Granite-3.3-2B-Base) is loaded.
* No recorded result is changed.
* Nothing else uses the GPU while models are scored.
* Every test exit code is checked.

**Hardware.** One NVIDIA RTX PRO 6000 Blackwell (96 GB). Timings are not comparable with the earlier RTX 5090 runs.

## 1. Question

In the confirmatory study, certified compression was limited by how much accuracy each step costs, not by the screen
or the bound. Our operator removes MLP channels and leaves every remaining weight unchanged. This experiment tests one
change: after each removal, refit the layer's output weights so that the pruned layer reproduces the dense layer's
output. **Does that let a trajectory go further before it loses 2 points in its worst category?**

## 2. The operator (`capprune/compensate.py`)

For each MLP whose channels are removed in a step:
* **Notation.** K is the set of kept channels and N the set of all channels (6,144 per layer).
* **Hidden activations.** h are the dense MLP's hidden activations over all N channels, computed with the dense
  `gate_proj` and `up_proj` weights, act(gate(x)) · up(x), in float32. They are taken on the layer's **current**
  inputs x, i.e. the MLP inputs produced by the current pruned and compensated model, over the calibration tokens.
* **Gram matrix.** C = Σ h hᵀ over all calibration tokens, accumulated in float64.
* **Refit of the output weights on K:**
  * W′ = W_dense[:, N] · C[N, K] · (C[K, K] + λI)⁻¹;
  * λ = 0.01 × mean(diag(C[K, K]));
  * solved with a Cholesky factorization of C[K, K] + λI, never an explicit inverse;
  * computed in float64, then stored in the model's dtype (bfloat16).
* **What stays unchanged:** the `gate_proj` and `up_proj` rows of kept channels, and every other weight. No gradient
  step is taken anywhere.
* **Order within a step.** The step's new mask is applied first. Then the step's layers are refit one at a time, in
  ascending depth. Each layer's inputs come from the model with all earlier refits of the step applied.
* **Which layers.** Only the layers whose channels are removed in this step are refit. A layer refit in an earlier
  step keeps its refit weights until it is pruned again. Its next refit starts from the dense weights, with the
  smaller kept set, and still targets the dense layer's output.

This is ridge-regularized least-squares reconstruction of the dense layer output. It is the closed form behind
OBS-style structured pruning, as in SparseGPT (Frantar & Alistarh, 2023). **It is not a reproduction of any published
method, and it is reported as our own simple variant.**

**Calibration.** 128 windows of 2,048 C4 training tokens. They come from the prepared C4-train calibration pool
(256 windows), chosen by `PreparedData.calibration_indices(seed=1, n=128)`. The trajectory's profiling uses seed 0, so
the two selections can share windows; the report gives the overlap. No benchmark item is used for calibration.

**Persistence and identity.** A compensated checkpoint is no longer defined by its mask.
* **Before the next step starts,** each step's refit weights (the `down_proj` columns of the kept channels, for each
  refit layer) are written with their SHA-256.
* **Checkpoint identity:** a checkpoint is the hash of (mask hash, the SHA-256 of every refit layer's current
  weights).
* **Resume:** the run resumes from the last persisted step by loading the persisted weights. Weights are never
  recomputed to reproduce an earlier step, and a step that was not fully persisted is recomputed from the start.

## 3. Design: the same masks, with and without compensation

**1. Reference trajectory (mask-only).** Gradient top-1 under the Qwen development protocol, seed 0, through the
existing live loop (`configs/compensation_feasibility.yaml`).
* **Inherited from `gradient_live.yaml`:** structured MLP channels; 4 layers × 5 blocks of 64 channels per step, i.e.
  1,280 channels and 7,864,320 parameters; the 0.875 per-unit cap; the profiling protocol.
* **No gate.** The loop's perplexity screen runs with a threshold of 10⁹, so it accepts every proposal. It only
  records C4 validation perplexity at each step.
* **Length.** The trajectory has **40 steps**. The loop is run to **60 proposals**, the most the extension rule (§5)
  can ask for, because the runner cannot extend a finished run (raising the cap changes the run's science hash).
  * The generator does not depend on any scoring, so steps 41–60 are exactly what an extension would produce.
  * Steps 41–60 are neither scored nor analysed unless the extension rule applies. If it does, they are scored
    mask-only before any compensated model is scored.
* **Masks:** every accepted mask is kept, and its hash is verified against the journal.
* **Check against the earlier run.** The first steps should reproduce the trial mask hashes of the existing gradient
  top-1 run on Qwen (`~/cap-prune-runs/gradient-live-ext/gradient_top1__t00__s0/journal.jsonl`: 15 accepted steps,
  then a 16th trial that its gate rejected). That run's gate accepted the first 15 steps, and gating does not change
  the ranking, so all 16 trial masks should match. Profiling ran on an RTX 5090 then and an RTX PRO 6000 now, so
  numerical differences could change a ranking. **The report gives the first step at which the hashes differ, if
  any.**

**2. Compensated trajectory.** The same masks, replayed in the same order, with §2 applied after each step.
* **This is conservative for compensation:** the channels were chosen for the mask-only model, not for the
  compensated one. The report says so.

**3. Evaluation set** (written to `~/cap-prune-runs/compensation-feasibility/eval/`):
* **Pool.** The Qwen items, built by `confirmatory.source_rows` and `confirmatory.dedupe`:
  * MMLU test and validation from the prepared Qwen data folder (`~/cap-prune-runs/data`, pinned SHA-256);
  * MMLU-CF validation (pinned files);
  * deduplicated by normalized question and choices;
  * few-shot exemplars excluded.

  The pool comes from the same sources as the two confirmatory models' sealed sets, so it overlaps them. The sealed
  files are not read, and Qwen's scores cannot affect the completed confirmatory study.
* **Sample.** 3,000 items per MMLU category. Within each category, items are sorted by qid and chosen without
  replacement with `numpy.random.default_rng([20261126, crc32(category)])`. If a category has fewer than 3,000 items,
  all are used and the report says so.
* **Recorded:** the SHA-256 of the sorted qids, per category and overall.
* **Labelling.** These items have all been seen on Qwen before, which is why this is exploratory.

**4. Scoring.**
* **Models:** the dense model, and every step of both trajectories.
* **Method:** each checkpoint is materialized, with channels physically removed. It is scored 0-shot against the
  dense model in identical batches (`pool_signals.LogprobScorer`), with every batch persisted to a score cache before
  use.
* **Perplexity:** C4 validation perplexity (the prepared 32 × 2,048-token set) for the dense model and at every step
  of both trajectories.

## 4. Quantities

**Paired loss.** For step t of a trajectory and category c: L_c(t) = (lost − gained) / n on that category's
evaluation items, against the dense model. Lost means dense right and checkpoint wrong.

**Worst-category loss.** W(t) = max over c of L_c(t). W(0) = 0 by definition, for the dense model.

**Reach.** For a trajectory, its reach R is the largest step t such that W(s) ≤ 0.02 (2.0 points) for every
s = 1, …, t. R = 0 if W(1) > 0.02.
* R_mask is the reach of the mask-only trajectory, and R_comp that of the compensated one.
* Both are computed over the same trajectory length L.
* The reach uses point estimates on 3,000 items per category, so it is noisy. One category's estimate has a standard
  error of about 0.5–0.7 points at 7–15 % disagreement, and the worst of four is biased upward. The rule is a go/no-go
  screen for the larger study, not a test.

## 5. Decision rule

**Order of work:**
1. Score the dense model and the mask-only trajectory (steps 1–40), and compute R_mask.
2. **If 2 × R_mask + 2 > L** (L = 40), extend the trajectory to L = min(60, ⌈2.5 × R_mask⌉) steps. Score the added
   mask-only steps and recompute R_mask over the new L, all before any compensated model is scored. Record the
   extension.
3. **If 2 × R_mask + 2 > 60** after that, stop and ask. The compensated trajectory is not run.
4. Otherwise, compute the compensated trajectory over the L steps, score it, and compute R_comp.

**Verdict:**
* **GO:** R_comp ≥ 2 × R_mask **and** R_comp − R_mask ≥ 5 steps.
* **NO-GO:** R_comp < 1.5 × R_mask.
* **INCONCLUSIVE:** anything between. It is reported, and the decision is the user's.

**Attainability check (written before any scoring).**
* **Range.** R_comp can be anything from 0 to L.
* **When GO can be met.** GO needs an R_comp with max(2 × R_mask, R_mask + 5) ≤ R_comp ≤ L. This exists if and only if
  L ≥ max(2 × R_mask, R_mask + 5).
* **The order of work guarantees it** whenever the compensated trajectory is run:
  * that only happens when L ≥ 2 × R_mask + 2;
  * if R_mask ≥ 5, then 2 × R_mask ≤ L;
  * if R_mask ≤ 4, then R_mask + 5 ≤ 9 ≤ 40 ≤ L.
* **So GO is attainable in every case the design allows.** Its best attainable outcome is R_comp = L, every
  compensated step within 2 points, which is GO.
* **NO-GO** is attainable whenever R_mask ≥ 1, with R_comp < 1.5 × R_mask.
* **If R_mask = 0,** NO-GO cannot occur (R_comp ≥ 0), and the verdict is GO (R_comp ≥ 5) or INCONCLUSIVE.
* **Extension cases:**
  * extension happens only if R_mask ≥ 20, and then L = ⌈2.5 × R_mask⌉ ≥ 2 × R_mask + 2;
  * for 24 < R_mask ≤ 29 the cap of 60 still leaves 2 × R_mask + 2 ≤ 60;
  * R_mask ≥ 30 stops the experiment (step 3).
* **R_mask = 40, censored at L = 40,** extends to L = 60, and R_mask is recomputed.

## 6. Also reported (descriptive)

* **Per step and category:**
  * the paired loss of each operator against dense, with one-sided 95 % Clopper–Pearson union bounds;
  * the direct paired difference, mask-only minus compensated, on the same items: P(compensated right, mask-only
    wrong) − P(mask-only right, compensated wrong), with a two-sided 95 % Clopper–Pearson union interval.
* **The pre-registered confirmation rule** applied to each trajectory on this evaluation set: per-category IUT,
  Clopper–Pearson union bound, 3 points, α = 0.05, fixed sequence. Labelled **"on seen items, not a certificate"**.
* **Perplexity** relative to dense at every step, for both operators.
* **For each refit:**
  * the conditioning of C[K, K] (largest / smallest eigenvalue, without and with the ridge);
  * the relative reconstruction error per layer on the calibration tokens, before (the dense columns of K) and after
    the refit;
  * the time and peak GPU memory per compensation step.
* **One figure:** worst-category loss against parameters removed, for both operators, with the 2- and 3-point lines.
* **Calibration overlap:** the profiling and compensation windows they share.

## 7. Tests (CPU, toy MLP; `tests/test_compensate.py`)

* The refit is a no-op when no channel is removed.
* With a tiny λ it matches `numpy.linalg.lstsq`.
* It recovers the output exactly when each removed channel is a linear combination of kept ones.
* A rank-deficient C does not crash, and the ridge term is applied.
* Resuming after a simulated crash mid-step gives the same checkpoint hashes as an uninterrupted run, from the
  persisted weights.

## 8. Outputs

* **Run files:** `~/cap-prune-runs/compensation-feasibility/`, with `reference/` (the trajectory run), `eval/` (the
  evaluation set and scores), `compensated/` (refit weights, manifest, per-step records) and `analysis/` (verdict,
  tables, figure). Each records the commit, the tree state and the GPU model.
* **Report:** `docs/compensation_feasibility_report.md`. It gives the verdict against §5, the §6 tables and the figure,
  with every number citing its run file.
* **Tests:** the full test suite, passing with exit code 0, before the pull request.

## References

* Frantar, E. and Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.
  arXiv 2301.00774.
