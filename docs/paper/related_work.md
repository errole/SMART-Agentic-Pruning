# Related work

**How these notes were made.** Each paper's arXiv abstract page and HTML full text were read on 2026-11-23 before it
was described here. Where only part of the full text was read verbatim, details that rest on an automatic summary of
the rest are marked *(summary only - check)*. Short quotes are verbatim from the arXiv HTML.

## What is NOT novel in CAP-Prune

* **Flips and KL as compression metrics.** Dutta et al. (2024) propose both and show they are well correlated.
  Felicio (2026) shows TV converts to decision flips more stably than KL.
* **Learn then Test with fixed-sequence testing for choosing a compression level.**
  * Angelopoulos et al. (LTT) give the framework and fixed-sequence testing.
  * Alvarez (2024) applies it to the pruning ratio, in vision.
  * Pareto Testing (Laufer-Goldshtein et al. 2022) uses it with one null per configuration, "some risk exceeds its
    limit". Its combined p-value is the maximum over risks, which is the same construction as our per-category
    intersection-union test. Its configurations include the number of attention heads pruned.
  * Our confirmation rule (amendment 1: a per-category IUT plus fixed-sequence testing over checkpoints) is therefore
    an application of known multiple-testing machinery, not a new test.
* **Per-step gates do not certify multistep paths.** Zhang (2026) states it for group-robust LLM pruning: "a
  terminal guarantee needs fresh control at each reached comparison or a trajectory-wide continuation bound".
* **An LLM agent choosing pruning steps under a rollback gate** is the base paper (Kodathala & Vunnam 2026).

## Candidate novelty, to be established by the confirmatory study

* Per-domain (per-category) accuracy guarantees during iterative, training-free LLM structured pruning, under a
  matched evaluation budget.
* Quantified pool overfitting: the gap between the pool-certified loss and the confirmation loss as a function of
  search (claim C1).
* The sample-size infeasibility of per-step population certification at benchmark scale (C2).
* Confirmed compression against evaluation cost, end to end (C3, C5).

---

## Compression metrics: flips, KL, TV

### Dutta, Krishnan, Kwatra, Ramjee. "Accuracy is Not All You Need." arXiv 2407.09141 (2024; NeurIPS 2024)

* **What it does.** It evaluates compressed LLMs (quantization: BnB, GPTQ, AWQ, SmoothQuant; also layer dropping,
  Wanda and SliceGPT) with two "distance" metrics beside accuracy:
  * **flips:** answers that change between correct and wrong relative to the baseline;
  * **KL divergence** over the multiple-choice answer tokens.
* **Models and tasks:** Llama2-7b/13b/70b chat and Yi-6b/34b chat, on MMLU (5-shot), PIQA, HellaSwag, ARC,
  LAMBADA, Winogrande, GSM8K, TriviaQA and MT-Bench.
* **Main finding:** accuracy changes are often 0–2 %, while flips reach "up to 13.6 %". On MMLU, flips and KL have
  a Spearman correlation of 0.981 across model-scheme pairs.
* **Quote:** "We propose two such metrics, KL-Divergence and % flips, and show that they are well correlated."
* **No per-category guarantee.** Results are per benchmark, with no MMLU per-category breakdown and no confidence
  intervals or certification. Its limitations note that "Flips is only a warning".
* **Relation to CAP-Prune:**
  * Our net flip loss and KL signals are these metrics, applied per MMLU category during iterative pruning.
  * The flips-vs-KL relation is theirs.
  * Our gate turns flip counts into per-category statistical tests, which this paper does not do.

### Felicio. "How Divergence Becomes Decision Flips in Compressed Language Models." arXiv 2610.00694 (2026)

* **What it does.** It asks which divergence between dense and compressed next-token distributions converts into
  changed greedy decisions, and in what form.
  * **Scale:** 802 perturbed or compressed configurations of 19 checkpoints (Gemma-3, Qwen3, Qwen2.5, Mistral,
    Phi, OLMo-2, Llama-3-8B) and nine families (RTN, GPTQ, AWQ, SparseGPT, Wanda, magnitude pruning, KV-cache
    quantization, layer removal, noise), over five corpora.
  * **Flip definition:** an argmax disagreement at each teacher-forced position of the dense model's greedy
    output. This is not a correct-to-wrong transition on a multiple-choice item.
* **Main findings:**
  * The flip/TV ratio has a median of 1.049 (95 % within 0.89–1.29, coefficient of variation 9.4 %), against a
    coefficient of variation of 34 % for flip/√KL.
  * Across reports, KL misorders compressors whose flip rates differ by 10 % or more in 10.8 % of pairs; TV does so
    in 1.3 %.
  * Within one report, KL and TV rank alike.
* **Quotes:**
  * "We show that total variation, not KL, answers this directly"
  * "Within a report KL ranks compressors correctly; across reports it does not."
* **Not universal.** The TV ratio "stayed in its band for 37 of 38 checkpoints but fell below one on code for two
  models."
* **No guarantee.** Its bound (flip ≤ F(h) + 2TV/h) is deterministic and loose, and "the near-unit ratio is
  explained, not guaranteed."
* **Relation to CAP-Prune:**
  * "TV tracks flips better than KL across settings" is accurate for comparisons across reports, not within one
    report.
  * Our post hoc TV result (`tv_posthoc.md`) is in a different setting: per-category, multiple-choice choice among
    equal-size actions. It is consistent with TV being at least as good a choice signal as KL there.
  * We record TV as a secondary signal (amendment 2) and do not adopt it.

## Group-robust pruning and multistep guarantees

### Zhang. "When Compression Scores Cannot Decide: Information Boundaries for Group-Robust LLM Pruning." arXiv 2608.02940 (2026)

*Read verbatim: the abstract, sections 1–4 and the start of 5. Later sections: summary only - check.*
* **What it does.** It treats each compression score as an "information interface". It shows, by matched-observation
  reversals (Proposition 1), that pooled statistics, downstream continuation and reference-path curvature can each
  fail to determine the order of endpoints for the worst group.
  * Quote: "every deterministic rule that reads only that observation fails in one world."
* **Endpoints:**
  * per-group forward KL from teacher to student (J_g);
  * per-group token-loss increase (E_g);
  * worst-group perplexity inflation.

  The paper notes that a guarantee on one endpoint does not transfer to another.
* **Multistep paths.** Section 3.4: "A finite-menu guarantee ends when one candidate is accepted", and "a terminal
  guarantee needs fresh control at each reached comparison or a trajectory-wide continuation bound." A margin
  condition gives a pathwise result only when declared bounds cover every reachable path.
* **Results:**
  * Dense: an early-preserving allocation reduced worst-group perplexity inflation by 12.6–20.9 % across three
    dense LLMs.
  * Mixture-of-experts (OLMoE): a pooled endpoint refresh lowered held-out worst-group teacher KL by 15.8 %.
  * Task checks (MMLU, PopQA, PKU-SafeRLHF) *(summary only - check)*.
* **No statistical certification.** Proposition 2 bounds selection under a uniform endpoint error ε; no tail bound
  for trained networks is established *(summary only - check)*.
* **Relation to CAP-Prune.** This is the closest work.
  * Our observation that pool-certified per-step gates do not certify the final checkpoint (C1) is an instance of
    its multistep point, and is not new.
  * We differ in three ways:
    * the endpoint is accuracy per MMLU category (paired flips), not perplexity or KL;
    * the guarantee is a finite-sample per-category test of the final checkpoints on sealed items (C3), rather than
      a selection bound;
    * we measure the sample sizes and the evaluation cost that such certification requires (C2, C5).

## Risk control by multiple testing

### Angelopoulos, Bates, Candès, Jordan, Lei. "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control." arXiv 2110.01052 (2021, rev. 2022)

*Read verbatim: the introduction, section 2 and the start of section 3. The rest: summary only - check.*
* **The guarantee.** For a grid of parameters λ_j, test the nulls H_j: R(λ_j) > α with valid p-values (for
  example Hoeffding–Bentkus) and any FWER-controlling procedure at δ. Theorem 1 then gives
  P(sup over λ ∈ Λ̂ of R(λ) ≤ α) ≥ 1 − δ, with probability over an i.i.d. calibration set.
* **Fixed-sequence testing.** Test in a fixed order and stop at the first non-rejection. The order must be chosen
  "without looking at the calibration data"; "Any ordering leads to valid error control."
* **Scope.** Multiple risks are handled with a null that holds "for some l" (section 2.4). Non-monotone risks are
  allowed.
* **Relation to CAP-Prune.** Amendment 1 uses exactly this:
  * one null per checkpoint;
  * the checkpoints ordered by parameters removed and frozen before scoring;
  * fixed-sequence testing at α;
  * the claim is the last passing checkpoint.

  The LTT guarantee is about the population that the confirmation items are drawn from, which is why the sealed
  set must be an i.i.d. sample per category.

### Zecchin, Park, Simeone. "Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection." arXiv 2409.15844 (2024, rev. 2025)

*Read verbatim: sections 1–4.2. Section 5 onward: summary only - check. No formal aLTT theorem was found in the
verbatim part.*
* **What it does.** aLTT replaces LTT's p-values with e-processes, giving anytime-valid sequential, data-dependent
  multiple testing with early termination. It stops when enough hyperparameters are certified or at t_max.
* **The guarantee:** FWER control, Pr[|Λ^unrel ∩ Λ̂| ≥ 1] ≤ δ, with an FDR variant via e-BH.
* **Quote:** "aLTT can reduce the number of testing rounds, making it particularly well-suited for scenarios in
  which testing is costly".
* **Experiments:** offline RL policy selection, and prompt selection with Llama3-8B *(summary only - check)*.
* **Relation to CAP-Prune.** This is the comparator for C5: an adaptive, anytime-valid test that saves evaluations,
  against our threshold-aware allocation within an exact finite-pool test. Neither the paper nor our work so far
  compares the two.

### Laufer-Goldshtein, Fisch, Barzilay, Jaakkola. "Efficiently Controlling Multiple Risks with Pareto Testing." arXiv 2210.07913 (2022)

*Read verbatim: sections 1–5. Experiments: summary only - check.*
* **The setting.** It controls several risks at once (Q_i(τ) ≤ α_i for all i, with probability ≥ 1 − δ) for a
  configurable model while optimizing other objectives.
  * **Per-configuration null:** "∃ i with Q_i(τ) > α_i", tested with the maximum of the per-risk p-values.
  * **Order:** the configurations are ordered along a Pareto front, with the order computed on a separate split,
    then tested by fixed-sequence testing.
* **Pruning.** The configurations include "the number of attention heads pruned", along with token dropping and
  early exit. This is inference-time, configurable computation: "does not require model re-training".
* **Worst-class experiments.** One risk per class, for worst-class accuracy *(summary only - check)*.
* **Relation to CAP-Prune.** Our per-category IUT is its "some risk exceeds its limit" null with max-p combination,
  with the four MMLU categories as the risks. Our ordering by compression plays the role of its Pareto ordering.
  The machinery is not new.

### Alvarez. "Confident magnitude-based neural network pruning." arXiv 2408.04759 (2024)

*Read verbatim to section 4.1.3. The rest: summary only - check.*
* **What it does.** It calibrates the global magnitude-pruning ratio λ with LTT and fixed-sequence testing so that
  P(E[ℓ] ≤ α) ≥ 1 − δ on the calibration distribution.
* **Experiments, vision only:**
  * MNIST with a 784-128-128-10 MLP: λ = 0.78 (accuracy loss) or 0.80 (disagreement with the full model), at
    α = δ = 0.05;
  * PolypGen segmentation with a U-Net *(summary only - check)*.
* **The guarantee is marginal:** expected risk over the calibration distribution, with no per-group or per-class
  guarantee.
* **Relation to CAP-Prune.** LTT for choosing a pruning level is not new. We differ in four ways:
  * LLMs;
  * iterative structured pruning;
  * per-category (worst-domain) guarantees through the IUT;
  * screening and confirmation separated under a measured evaluation budget.

## Structured LLM pruning with retraining

### Muralidharan et al. "Compact Language Models via Pruning and Knowledge Distillation." arXiv 2407.14679 (2024) - Minitron

*Read verbatim to section 4.1. The rest: summary only - check.*
* **What it does.** It prunes Nemotron-4 15B in depth and width (MLP neurons, attention heads, embedding channels)
  with activation-based importance from 1,024 calibration samples, then retrains with distillation (logit KL).
* **Retraining:**
  * "we use 1.8 billion tokens (400 steps) for lightweight retraining" during the architecture search;
  * 94B tokens for the final models, "a fraction (<3%) of the original training data".
* **Results (MMLU 5-shot):**
  * Minitron 8B: 63.8 (Nemotron-4 15B: 66.6);
  * Minitron 4B: 58.6.
  * At 8B, the pruning baselines without comparable retraining score lower: LLM-Pruner 25.2, SliceGPT 37.1.
* **No accuracy without retraining.** The paper reports no accuracy for the final-size pruned model without
  retraining; width pruning beats depth "only after retraining".

### Sreenivas, Muralidharan et al. "LLM Pruning and Distillation in Practice: The Minitron Approach." arXiv 2408.11796 (2024, rev. 2024)

* **What it does.** It compresses Llama 3.1 8B to 4B (width and depth variants) and Mistral NeMo 12B to 8B, by
  structured pruning, then distillation.
* **Teacher correction:** about 100B tokens of fine-tuning of the teacher, because the original pretraining data is
  not available.
* **Distillation tokens:** 94B (Llama) and 380B (NeMo).
* **Results (MMLU 5-shot):**
  * Llama-3.1-Minitron-4B-Width: 60.5;
  * Llama-3.1-Minitron-4B-Depth: 58.7 (Llama 3.1 8B: 65.3);
  * MN-Minitron-8B: 69.5.
* **No accuracy without retraining.** All accuracies are after distillation; the pruned-only starting point is
  described only qualitatively.

**Relation of both Minitron papers to CAP-Prune.** Large compression of LLMs is done with retraining at the scale of
tens to hundreds of billions of tokens. CAP-Prune is training-free and certifies small compression steps, so the
compression levels it can confirm (a few percent of parameters on Qwen3-1.7B so far) are not comparable to
Minitron's. The papers set the context; they are not competitors.

## The base paper

### Kodathala, Vunnam. "LLMs can Compress LLMs: Adaptive Pruning by Agents." arXiv 2601.09694 (2026)

* **What it does.** A foundation-model agent (gemini-3-flash-preview, temperature 0.5) chooses per-layer sparsity
  increments (δ ∈ [0.01, 0.15]) at each iteration. Its inputs are z-scored layer profiles: a Wanda-style score and
  gradient importance. Within a layer, pruning follows Wanda's score.
* **Gate:** a checkpoint rollback when the perplexity ratio exceeds 1.15 (τ = 0.15).
* **Models and evaluations:** Qwen3-4B and Qwen3-8B, evaluated on 5-shot MMLU, FreebaseQA and WikiText-2
  perplexity.
* **Main numbers:** Qwen3-8B at 43.02 % sparsity, MMLU 56.67 vs 77.38 dense and 36.29 for the 4:8 baseline.
* **Not reported:**
  * statistical tests or confidence intervals;
  * an ablation that isolates the LLM agent from a rule-based selector;
  * a cost beyond "a single NVIDIA A100 80GB GPU".
* **Internal inconsistencies noted by the reader:** about 45 % vs 43.02 % sparsity, and two perplexity tables that
  do not reconcile.
* **Relation to CAP-Prune.** We start from this loop (iterative pruning, a z-scored profile, an LLM controller, a
  rollback gate) and change three things:
  * the gate is a per-category capability test instead of perplexity (C4);
  * the controller is ablated: gradient-only selection is as good as or better than LLM controllers here (N);
  * the final model is certified per category on sealed items (C3).

  Our Gemini live-gate run used the paper's controller (`gemini-3-flash-preview`), but it is incomplete
  (`~/cap-prune-runs/live-gate/`; [`docs/live_gate.md`](../live_gate.md)):
  * the perplexity-gate arm finished;
  * the capability-gate arm paused at 7 of 10 attempts, unaudited, when the free-tier quota ran out;
  * the round-robin arm never started.

  Our completed controller comparisons use Qwen2.5-7B/14B.
