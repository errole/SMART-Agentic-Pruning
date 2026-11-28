# Study 2 groundwork (2026-11-27)

**Status: EXPLORATORY engineering and planning work. Nothing in it is a claim.**
* No model was scored on any benchmark item.
* No model was sealed, and nothing was registered.
* The GPU work was Wanda pruning, C4 perplexity, and forward passes on C4 text. Those passes have the shape of a
  scoring batch but use no benchmark text.
* The models were the two feasibility models at their pinned revisions. No other process used the GPU meanwhile.

**Code:**
* `capprune/study2_groundwork.py`
* `capprune/pruning/wanda.py` (cached path)
* `capprune/pruning/reload.py`
* `capprune/pruning/masks.py` (`LiveDenseStore`, CPU masks)
* `capprune/supervisor.py`
* `capprune/utils/runtime.py` (heartbeat)
* `capprune/bounds.py` (exact test v2)

**Config:** `configs/study2_groundwork.yaml`.

**Run files:** `~/cap-prune-runs/study2-groundwork/`.

**Tests:** `tests/test_study2_groundwork.py`.

The draft plan built on this groundwork is [`study2_plan_DRAFT.md`](study2_plan_DRAFT.md).

## P1. Wanda with cached block inputs

**Change.**
* `sequential_wanda(..., cache_inputs=True)` captures, once per calibration batch, the hidden states that enter the
  first decoder layer, together with the keyword arguments the model passes to that layer.
* It then runs each layer on its cached inputs: once to gather the activation norms, and once more, after pruning, to
  produce the next layer's inputs. This is the published Wanda code's `prepare_calibration_input` loop.
* The original path ran every calibration batch from the embeddings up to layer *i*, for every *i*, so its cost was
  quadratic in depth. It is still the default (`cache_inputs=False`), so no recorded experiment's code path changes.
* The batches, the layer calls and their arguments are identical in both paths, so the masks are identical.
* The cached path refuses models whose layers receive different attention masks (sliding-window layers mixed with
  full-attention ones). See P6.

**Check against the feasibility run's recorded mask hashes.**
* The calibration windows are the same (seed 0, 128 × 2,048 C4 train tokens; the indices match `dense.json`).
* Every level was pruned one-shot from the dense weights.

| model | level | path | prune time | recorded (feasibility) | mask hash | matches |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 5 % | original, re-timed now | 252.7 s | 277.4 s | `b08d4975dbdea428…` | yes |
| Qwen3-1.7B | 5 % | cached | **35.6 s** | 277.4 s | `b08d4975dbdea428…` | yes |
| Qwen3-1.7B | 30 % | cached | **35.5 s** | 274.5 s | `de628fb0ffa04daf…` | yes |
| Qwen3-8B | 5 % | cached | **108.4 s** | 1,073.1 s (first level) | `8bd43cadb4223969…` | yes |
| Qwen3-8B | 30 % | cached | **109.2 s** | 955.9 s | `302ec13b5fa85a07…` | yes |

⟨`qwen3-1.7b/wanda_speed.json`, `qwen3-8b/wanda_speed.json`⟩

**Speed-up and time per level.**
* Pruning is 7.1–7.8 times faster on Qwen3-1.7B and 8.8–9.9 times faster on Qwen3-8B.
* A whole level (restore, prune, score the evaluation set, C4 perplexity), from the feasibility records and these
  measurements:
  * Qwen3-1.7B: **374 s before, about 136 s after** (12,000 items).
  * Qwen3-8B: **1,209 s before, about 369 s after** (9,927 items).
* At the 30 % level of both models, C4 validation perplexity after the cached prune is bit-identical to the recorded
  value: 24.421236848892516 for Qwen3-1.7B and 16.377032669534998 for Qwen3-8B.

## P2. One resident copy of the model

**Change.**
* `pruning/reload.py::restore_dense` copies every prunable linear's dense weight from the pinned safetensors files
  into the live model. Before a level, it replaces the second, dense copy of the model that the feasibility run kept
  on the GPU.
* An exact integer checksum of the prunable weights' bits (`fingerprint`) is taken at load and compared after every
  restore.
* `LiveDenseStore` serves the live weights as the dense ones and refuses a second read of a unit before the next
  restore.
* The keep-masks are created on the CPU and moved to the GPU only while their layer is pruned.
  (`PruneState(mask_device="cpu")`, `offload_masks=True`.)
* Each level's state is released before the next level.

**Check.**
* On both models, the 30 % level was pruned after restoring from disk, and its hash equals the hash recorded from a
  fresh load (P1).
* The restore takes 0.84 s (2.8 GB) on Qwen3-1.7B and 5.1 s (13.9 GB) on Qwen3-8B, with the files in the OS cache.
* The checksum matched both times.

**Peak GPU memory.** These are `torch.cuda.max_memory_allocated` figures (tensors only), in GB.
* The scoring proxy is a 32,768-token batch (16 × 2,048, 32 × 1,024, 64 × 512 or 128 × 256 tokens) cut from C4
  validation text. It is left-padded like the scorer's batches and run through `LogprobScorer`'s forward call
  (`logits_to_keep=1`).

| | Qwen3-1.7B | Qwen3-8B | Qwen3-14B (expected) | Qwen3-4B (expected) |
|---|---|---|---|---|
| resident weights | 3.44 | 16.38 | 29.54 | 8.04 |
| + pruning (cached Wanda) | +4.04 | +8.00 | +14.17 | +8.00 |
| + C4 perplexity (32 × 2,048) | +4.99 | +5.76 | +10.21 | +5.76 |
| + scoring batch (32,768 tokens) | +1.84 | +3.58 | +6.34 | +3.58 |
| **level peak, one resident copy** | **8.43** | **24.39** | **43.71** | **16.05** |
| level peak recorded in the feasibility run (two copies) | 14.69 | 59.37 | | |

⟨`memory.json`⟩

**How the expected peaks are computed.**
* Formula: resident weights (from the pinned `config.json`, on the meta device) + the largest Qwen3-8B extra × a growth
  factor.
* Growth factor: the largest ratio of hidden size, intermediate size or largest linear, never below 1. That is 1.77 for
  Qwen3-14B, set by the largest linear (89.1 M vs 50.3 M weights).
* This rule is conservative. Applied from Qwen3-1.7B to Qwen3-8B (factor 4.0), it predicts 16.1, 20.0 and 7.4 GB of
  extra memory where 8.0, 5.8 and 3.6 GB were measured.
* **The expected peak for Qwen3-14B in bf16 is about 44 GB, 43 % of the card.**
  * The card has 97,887 MiB (102.6 GB), of which the desktop uses about 2.7 GB.
  * The allocator's reserve comes on top of the allocated peak: 0.7 GB at 8B (25.1 GB reserved against 24.4 GB
    allocated). So does the CUDA context, a few hundred MB.

**Why the old layout would not fit Qwen3-14B.**
* Two copies of the weights: 59.1 GB.
* One bool mask per weight, held on the GPU: 13.2 GB per mask set.
* The feasibility runner kept up to three mask sets alive: the previous level's state and the regeneration check's
  state are still referenced while the next level is pruned. On Qwen3-8B this is the jump of the peak from 45.5 GB
  (5 %) to 59.4 GB (10 % onward), which is 2 × 6.95 GB.
* With pruning activations, that is about 113 GB.

## P3. The hang test that timed out at d77b9f3

**Diagnosis: a real defect in the supervisor, triggered by a step of the system clock, not a timing margin of the
test.**

**Evidence.**
* The failed run's files are kept in
  `~/cap-prune-runs/_tests/2026-11-27_d77b9f34e08d/basetemp/test_hang_is_detected_and_work0/`.
* The supervisor started the worker at 11:30:11 by the then-current clock. The worker's last heartbeat is at
  1795797018.46 (11:30:18), when its injected hang began.
* The supervisor's log ends with "starting attempt 1". It never logged a hang, and the ledger stayed "running" until
  the test's 600 s limit killed it.
* The Windows System log (Kernel-General event 1, process `svchost.exe`, the Windows Time service) shows the clock
  being stepped back by 17,999,422 ms (5 h) at 16:30:28.34 UTC by the old clock, which was 11:30:28 local.
  * The clock had been running 5 h fast. The time service corrected it from time.windows.com (Time-Service events 35
    and 37 at the same moment).
  * That is 10 s after the last heartbeat and 2.5 s before the supervisor's next 5-second check could have fired.

**Mechanism.**
* The supervisor computed `time.time() − heartbeat_ts > hang_timeout_s`, a wall-clock time minus a wall-clock time.
* After the −5 h step, the difference was about −17,992 s. The hang could not be seen for about 5 hours, so the test's
  600 s limit expired first.
* The same arithmetic means that a forward step longer than `hang_timeout_s` (1,800 s in production) would kill a
  healthy worker.
* The worker's heartbeat throttle also compared wall-clock times: after a backward step, a worker would stop writing
  heartbeats for the size of the step.

**Fix.**
* The supervisor treats any change of the worker's heartbeat (a new `seq`, time stamp, phase or detail) as progress,
  and times the absence of change on its own monotonic clock.
* The worker is identified by a launch token passed in `CAPPRUNE_LAUNCH_ID` and written into the heartbeat. No clock
  is involved.
  * The earlier pid/start-time rule remains for heartbeats without a token.
* The heartbeat's throttle and the supervisor's backoff sleeps use the monotonic clock.
* The heartbeat's `ts` stays a wall-clock time, for people to read.

**Tests** (`tests/test_study2_groundwork.py`):
* a hang is detected within a second after a −5 h wall-clock step;
* a progressing worker survives a +5 h step;
* a heartbeat from another launch is not progress;
* the heartbeat keeps writing after a backward step;
* the backoff sleep ignores the wall clock.

The original end-to-end test (`tests/test_recovery.py::test_hang_is_detected_and_worker_restarted`) is unchanged.

**Side effect on records (nothing changed).**
* Files and commits written before 06:30:28 EST on 2026-11-27 carry times 5 h fast. That covers at least commits
  `4712ef0` and `d77b9f3`, and the first test record's `started` (11:17).
* So `d77b9f3` (11:16:47) shows a later date than its child `ff9a8e0` (09:23:17). The record's `finished` (06:42) also
  reads as earlier than its `started`.
* Git's parent order and every hash are unaffected, and the time measurements of the runs use monotonic counters.

## P4. The exact test: the registered fix

**Definition (grid v2, in `capprune/bounds.py::exact_score_critical_v2`; the v1 code is unchanged because bound-dev's
recorded results used it).**
* **Grid:** the v1 grid, plus a uniform grid of spacing 1/(8n) in p⁻ over [0, (1 − tol)/2). That is 5,273–14,507
  points.
* **Critical value:** z\* is the largest z (bisection to 10⁻⁹) at which the exact size is ≤ α − 10⁻⁴ at every grid
  point.
* **Certificate over the continuum, which is the "check off the grid".** On the boundary p⁺ = p⁻ + tol, the size
  S(g) = P_g(Z ≤ z\*) has S′ = Cov(U, 1_R), where U is the score of g. So:
  * |S′| ≤ √(n I(g) S(1 − S)), with I(g) = 1/(g + tol) + 1/g + 4/(1 − tol − 2g);
  * hence both S and 2 arcsin √S are Lipschitz on every interval.
* **Branch and bound:** every interval between grid points is bounded from its end values, and split until the bound
  is ≤ α. A point above α would be added to the grid and z\* recomputed. None was found.
* **Independent check:** 3,000 seeded random nuisance values (1,000 in the top 10 % of the range) and the data range
  q = 4–15 % in steps of 0.05 points, with both the v2 and the v1 critical values.

**Results** (α = 0.05; ⟨`exact_test.json`⟩, 529 s on 8 CPU cores):

| n | limit | z\* v1 | z\* v2 | largest size found, v2 | certified bound on the size, v2 | v1 at the independent points: largest size (disagreement q there), points above α |
|---|---|---|---|---|---|---|
| 1,254 | 2 pts | −1.668701 | −1.668701 | 0.048673 | ≤ 0.05 | 0.048673, none |
| 3,000 | 2 pts | −1.650109 | −1.651774 | 0.049841 | ≤ 0.05 | **0.050039 (98.9 %), 27** |
| 3,500 | 2 pts | −1.656834 | −1.656834 | 0.049513 | ≤ 0.05 | 0.049513, none |
| 3,600 | 2 pts | −1.666667 | −1.666667 | 0.049451 | ≤ 0.05 | 0.049308, none |
| 1,254 | 3 pts | −1.651740 | −1.656783 | 0.049851 | ≤ 0.05 | **0.050179 (99.9 %), 45** |
| 3,000 | 3 pts | −1.650523 | −1.651633 | 0.049874 | ≤ 0.05 | **0.050007 (98.9 %), 13** |
| 3,500 | 3 pts | −1.649793 | −1.650747 | 0.049892 | ≤ 0.05 | **0.050003 (98.5 %), 10** |
| 3,600 | 3 pts | −1.660929 | −1.660929 | 0.049418 | ≤ 0.05 | 0.048875, none |

**The exact size of the fixed test is ≤ 0.05 on and off the grid at every n and limit asked for.**
* The proof is the certificate. Floating point and the < 10⁻¹⁵ of trinomial mass outside the ±9 SD windows are
  covered by a 10⁻¹² slack.
* **v1 was invalid at more places than the 2026-11-25 note says:**
  * n = 1,254 at 3 points (+1.8 × 10⁻⁴);
  * n = 3,000 at 2 points;
  * n = 3,500 at 3 points (+2.6 × 10⁻⁶). The review's check there found none.
  * Every excess is at 98–100 % disagreement.
  * Inside the data range (q = 4–15 %), v2's size is at most 0.04925.
* The fix lowers z\* only where v1 had an excess. The test is not made noticeably more conservative.

**Recorded:** a dated note in [`bound_development_plan.md`](bound_development_plan.md) §8. bound-dev was not rerun.

## P5. The Qwen3-1.7B oddity: answer letters

**What is stored.**
* The score cache's dense records hold, per item, the four letter log-probabilities, the predicted letter and
  correctness.
* The pruned levels' records hold **correctness only**, so **the pruned models' predicted letters were not stored, and
  nothing was rescored.**
* From the stored outputs:
  * per level, the paired change split by the item's correct letter;
  * bounds on the pruned predicted-letter counts;
  * a counterfactual: the dense model with a constant per-letter bias added to its stored log-probabilities, fitted to
    each level's accuracy by answer letter.

⟨`letters.json`⟩

**Correct and dense-predicted letters (12,000 items).**
* Correct letters: A 2,912, B 2,942, C 3,046, D 3,100.
* The dense model's predictions: A 3,312, B 3,805, C 2,780, D 2,103. The dense model already under-predicts D.

**Paired loss by the item's correct letter** (points, all categories pooled). The homogeneity test is a χ² test of
lost/gained/unchanged × letter, with 6 df.

| level | pooled | A | B | C | D | homogeneity p | bias-only model's loss | P(pruned lost \| bias model lost) vs P(pruned lost \| bias model kept it right) |
|---|---|---|---|---|---|---|---|---|
| 10 % | +0.26 | −2.30 | −0.41 | +1.28 | +2.29 | 2 × 10⁻¹⁹ | +0.28 | 0.49 vs 0.030 |
| **15 %** | **+2.21** | −4.16 | +0.54 | +4.17 | **+7.84** | 2 × 10⁻⁶³ | **+0.92** | 0.65 vs 0.065 |
| 20 % | +2.13 | −1.72 | −0.37 | +3.91 | +6.39 | 2 × 10⁻²⁶ | +0.78 | 0.62 vs 0.087 |
| 25 % | +1.79 | −0.93 | +1.84 | +2.20 | +3.90 | 8 × 10⁻⁷ | +0.39 | 0.59 vs 0.115 |
| **30 %** | **+1.29** | +2.27 | +2.72 | +0.36 | −0.06 | 0.026 | **−0.13** | 0.52 vs 0.134 |
| 35 % | +1.15 | +6.11 | −0.10 | +0.98 | −2.16 | 2 × 10⁻¹² | −0.09 | 0.61 vs 0.152 |
| 40 % | +3.14 | +12.57 | −2.58 | +2.13 | +0.71 | 5 × 10⁻³⁵ | +0.41 | 0.68 vs 0.203 |

**Answer: in part.**
* **The 15 % jump.** From 5 % to 25 %, the pruned model gains on items whose answer is A and loses on items whose
  answer is C or D. At 15 %, answer-D items lose 7.84 points and answer-A items gain 4.16.
  * This is the pattern of a preference that moves further away from D than the dense model's already is.
  * The best-fitting pure bias (A +1.70, B +1.40, C +0.75 against D) reproduces the accuracy by letter closely. The
    items it would flip are lost by the pruned model 65 % of the time, against 6.5 % for the other items the dense
    model gets right.
  * But by itself the bias produces +0.92 of the +2.21-point loss. The other ≈ 1.3 points also appear at 15 % and
    stay roughly flat to 35 %.
* **The recovery.** By 30 % the fitted bias is back near zero (bias-only loss −0.13), and the letter pattern is nearly
  flat (p = 0.026).
  * From 35 % the bias reverses: items whose answer is A lose 6–25 points.
  * The worst-category loss falls from 2.87 (25 %) to 1.97 (30 %, 35 %) while the pooled loss falls from 1.79 to
    1.15–1.29: what falls away is mostly the letter-preference component.
* **What cannot be settled without rescoring.** The fit uses the dense model's log-probabilities as a stand-in for the
  pruned model's. The pruned models' own letter distribution, which would settle the question, was not stored.
* **For study 2:** store every candidate's four log-probabilities, not only correctness. A letter-preference shift is
  then visible directly.

## P6. A second model family

**Checked on the Hub, metadata only** (no weights downloaded; `config.json` and `README.md` read where allowed;
⟨`candidates.json`⟩, ⟨`accounting.json`⟩).
* Qwen3-8B (`Qwen/Qwen3-8B`) is the post-trained (hybrid thinking) model. Its card names `Qwen/Qwen3-8B-Base` as its
  base, so the matching variant of another family is its **post-trained (instruct)** model.
* MMLU figures are as each card reports them, under that card's protocol. They are not comparable with each other or
  with our 0-shot letter-log-probability scoring.

| candidate | revision to pin | licence, access | parameters | layers, attention | published MMLU (source) | first grid level giving ≥ 10 % of all parameters |
|---|---|---|---|---|---|---|
| **meta-llama/Llama-3.1-8B-Instruct** | `0e9e39f249a16976918f6564b8830bc894c89659` | Llama 3.1 Community License; **gated (manual approval); this account has no access yet** (403 on `config.json`) | 8,030,261,248 | 32, full attention | 69.4, 5-shot macro avg (own card); 68.2, 5-shot (Falcon3 card) | 15 % (13.03 %) |
| **allenai/Olmo-3-7B-Instruct** | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | Apache-2.0, open | 7,298,011,136 | 32; **24 sliding-window (4,096) + 8 full-attention layers** | 69.1 (own card; the same table gives Qwen 3 8B without reasoning 80.4) | 15 % (13.30 %) |
| **ibm-granite/granite-3.3-8b-instruct** | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | Apache-2.0, open | 8,170,864,640 | 40, full attention; tied embeddings | 63.5 (Olmo 3 card's table) | 15 % (14.62 %) |

**Recommendation.**
1. **Llama-3.1-8B-Instruct, first choice.**
   * It is the family most used in the pruning literature that Wanda was measured on.
   * It is full-attention, so cached Wanda works as it is.
   * Its MMLU is close to the middle of the range.
   * It needs access approval from Meta before it can be pinned and hashed.
2. **Olmo-3-7B-Instruct, if open data matters.**
   * Its training data are public, so MMLU contamination can be checked.
   * Its mixed attention layers need an extension of the cached path first: one set of layer arguments per attention
     type, then the mask-identity check against the original path.
   * At our 2,048-token windows a 4,096-token sliding window changes nothing numerically, but the code path differs.
3. **Granite-3.3-8B-Instruct, no code change.**
   * It continues paper A's family (Granite-3.3-2B), so check with the user first: an earlier task's standing rule
     excluded Granite.

**Also checked and not proposed:**
* `google/gemma-2-9b-it`: gated with no access; alternating sliding-window layers; logit soft-capping; 71.3 MMLU for
  the pretrained model.
* `mistralai/Mistral-7B-Instruct-v0.3`: Apache-2.0, but its card reports no MMLU, and it dates from 2024.
* `mistralai/Ministral-8B-Instruct-2410`: Mistral Research License (non-commercial); sliding window.
* `tiiuae/Falcon3-7B-Instruct`: TII Falcon licence.
* `internlm/internlm3-8b-instruct`: needs remote code.

## P7. The base paper's loop (Kodathala & Vunnam, arXiv 2601.09694)

Read from the paper's only version (v1, 14 Jan 2026, HTML). The paper links **no public code**.

**What the loop does** (§3.2–3.5, Algorithm 1, §4.1).
* **Agent inputs, per iteration:**
  * current global sparsity and the target;
  * per-layer profiles sorted by sensitivity z-score: the Wanda score |W| ⊙ ‖X‖₂ reduced to "the 10th percentile of
    active weights", a gradient importance mean |∇W L| over M samples, and the layer's current sparsity, each z-scored
    across layers;
  * current and baseline perplexity;
  * a feedback summary of iteration t − 1 (its reasoning and selections, the sparsity gain, the perplexity change, and
    a qualitative label).
* **Agent output:** JSON with `reasoning`, `stop_pruning` and `layer_decisions` (layer name, additional sparsity
  δ ∈ [0.01, 0.15]).
* **Pruning:** each chosen layer is pruned by δ using its Wanda score.
* **Profiling:** activations on 16 samples; gradients on 8 samples "every third iteration" (§3.2). Algorithm 1 instead
  collects both every iteration.
* **Rollback:** if (PPL_{t+1} − PPL_t)/PPL_t > τ = 0.15 against the parent, the iteration's decisions are discarded,
  the parent is restored, and the agent gets negative feedback. A rollback consumes an iteration (Table 4: 21 = 19 + 2).
* **Perplexity** is measured on 32 samples, which Algorithm 1 implies are calibration (C4) data.
* **Stopping:** at a 50 % target, or when the agent sets `stop_pruning`. No maximum number of iterations is given.
* **Evaluation:**
  * 5-shot MMLU (57 subjects, the "LLM-KICK protocol", full test set), FreebaseQA (20,358 questions) and WikiText-2
    perplexity;
  * one run per model, no seeds, no variance;
  * baselines only 2:4 and 4:8 magnitude pruning (no uniform Wanda);
  * Qwen3-8B: 77.38 → 56.67 at 43.02 %; Qwen3-4B: 71.29 → 44.43 at 45.58 %.
* **Controller:** `gemini-3-flash-preview` at temperature 0.5.
* **Hardware:** one A100 80 GB. No wall time is reported.

**Unspecified or inconsistent; each must be fixed in a registered arm A.**
* **Unspecified:**
  * the prompt, the system prompt and the JSON schema;
  * the history length (§3.3 says t − 1; Algorithm 1 passes the full memory);
  * what a "layer" is (§3.2 says each linear layer);
  * whether δ is absolute or relative;
  * per-row or per-layer ranking;
  * whether masks are cumulative;
  * the denominator of "sparsity";
  * the rollback data and sequence length;
  * the maximum number of iterations;
  * the model variant;
  * the MMLU harness and prompt.
* **Inconsistent:**
  * the reported 43.02 % and 45.58 % do not match Table 4's final 50.73 % and 50.46 % for the same runs;
  * Algorithm 1's rollback test reads "ratio > τ";
  * gradients every iteration vs every third;
  * Table 4's baseline perplexities (22.06, 27.90) are not the WikiText-2 ones (9.72, 13.64);
  * 2:4 and 4:8 are reported at 42.40 % and 45.16 %.

**What a faithful arm A needs in this repository.**
* **The loop exists.** The `paper_agent` method (`capprune/search/loop.py`, `search/controller.py`, defaults in
  `configs/base.yaml`) already resolves the open points one way:
  * the unit is each linear;
  * δ is added to the unit's sparsity;
  * Wanda per output row, cumulative masks;
  * activations every iteration on 16 windows, gradients every 3rd on 8;
  * a C4 validation rollback (32 × 2,048) against the parent at τ = 0.15;
  * history window 1;
  * at most 40 iterations, patience 10.
* **What changes for study 2:**
  * the target (the paper's 50 %);
  * `unstructured_count: floor`;
  * the one-copy reload (P2);
  * the candidates' scoring outputs stored with log-probabilities (P5).
* **A controller model.**
  * The paper's is a preview API model. It may change or be retired, and the earlier Gemini run here stopped when its
    free-tier quota ran out ([`live_gate.md`](live_gate.md)). A paid key and the exact model string per response must
    be recorded.
  * A local controller would share the GPU, which the standing rules forbid while a model is loaded.
  * So arm A needs an API controller, or a second machine.
* **GPU time on Qwen3-8B, estimated from this groundwork's measurements:**
  * **Per iteration:** activation profile about 7 s, gradients about 10 s every third iteration, mask and apply a few
    seconds, the perplexity gate 13.7 s, restore after a rollback ≤ 5 s. About 25–30 s of GPU, plus the controller's
    latency (unmeasured; 10–60 s is typical of an API call).
  * **Search:** the paper's 8B run took 21 iterations, so about 10 min of GPU per seed; 40 iterations would be about
    20 min.
  * **Confirmation:** every accepted checkpoint (19 in the paper's 8B run) scored on the 13,100 sealed items. At
    24.3 ms per item that is about 337 s per checkpoint with perplexity, or **1.8–3.4 h per seed** (19–36 checkpoints).
  * **5-shot at the claim:** about 0.5 h per seed.
  * **Arm A, three seeds: about 7.5–12 h of GPU**, plus 1–3 h of controller waiting.
