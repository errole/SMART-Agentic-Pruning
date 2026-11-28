# Unstructured feasibility plan (2026-11-27)

**Status: EXPLORATORY.** It makes no claim. It decides whether unstructured pruning is worth a larger study.

This plan is committed and pushed **before any model is scored**. That includes the admissibility test on Qwen3-8B,
which is the first time that model is scored. The decision rule (§6) is not changed after scoring starts.

**Standing rules:**
* No sealed confirmation set is opened. Qwen3-8B's sealed rows (`~/cap-prune-runs/confirmatory/qwen3-8b/sealed/confirmation.jsonl`)
  are never read. Its sealed **id list** is read in one place only: the pre-registered admissibility test's split
  check (`screening_split`), exactly as in the confirmatory procedure. This experiment's own code reads the
  registration and the admissibility record, never `sealed/`. No other model's sealed folder is touched.
* Neither SmolLM3 nor Granite is loaded. The code refuses any model outside `unstructured.allowed_models` (Qwen3-1.7B,
  Qwen3-8B).
* No recorded result is changed. Every output goes to new folders: `~/cap-prune-runs/unstructured-feasibility/` and
  `~/cap-prune-runs/confirmatory/qwen3-8b/`.
* Nothing else uses the GPU while models are scored.
* Every test exit code is checked.

**Hardware.** One NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB). Timings are comparable with the
compensation experiment, which ran on the same GPU, and not with the earlier RTX 5090 runs.

## 1. Question

All confirmatory work removed whole MLP channels. On Qwen3-1.7B, that operator stayed within 2 points per category for
13 steps, 5.94 % of the model's parameters (compensation feasibility report). The base paper (Kodathala & Vunnam,
arXiv 2601.09694) instead zeroes individual weights by Wanda score, and the unstructured runs in this repository never
got past setup.

**Does uniform unstructured Wanda pruning keep per-category accuracy to a much higher compression than channel removal
did, on the development model and on the base paper's larger model?**

## 2. Models

| | Qwen3-1.7B (development model) | Qwen3-8B |
|---|---|---|
| repository | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3-8B` |
| variant | the post-trained release (not `-Base`) | the same variant |
| revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, the only revision any earlier run used (the prepared data's tokenizer id `@main` is this snapshot) | `b968826d9c46dd6066d109eabc6255188de91218`, the Hub's `main` on 2026-11-27 |
| weight files (SHA-256) | `model-00001-of-00002`: `169ad53e…`; `model-00002-of-00002`: `912becff…` | five files, each equal to the Hub's LFS SHA-256: `31d6a825…`, `5991236c…`, `c5185c47…`, `b5ee7de7…`, `20c2d636…` |
| licence | Apache-2.0 | Apache-2.0 (`LICENSE` in the snapshot, SHA-256 `832dd9e0…`); not gated |
| parameters | 1,720,574,976 (tied input and output embeddings) | 8,190,735,360 (untied) |
| bf16 load on the RTX PRO 6000 | 3.4 GB | **8,190,735,360 parameters, all bfloat16, 16.4 GB on `cuda:0`** (load only, no item scored) |

* **Full hashes** are pinned in `configs/unstructured_feasibility.yaml` and `configs/confirmatory_qwen3_8b.yaml`. Every
  run re-hashes the weight files before loading and refuses on any mismatch.
* **Records:** `unstructured-feasibility/qwen3-8b/model.json` and `qwen3-1.7b/model.json` (hashes, licence, load, GPU).
* **Same tokenizer.** `tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt` are byte-identical
  between the two snapshots.
* **Same prepared data.** `capprune prepare` for Qwen3-8B (`~/cap-prune-runs/data-qwen3-8b`) produced the same
  SHA-256 as Qwen3-1.7B's prepared data (`~/cap-prune-runs/data`) for all 11 files. That covers the C4-train
  calibration pool (`72f1b0a0…`) and the C4-validation perplexity windows (`054c0d6c…`). **Both models are
  calibrated and evaluated for perplexity on identical tokens.**

## 3. Data

### Qwen3-1.7B

* **Set:** the compensation-feasibility evaluation set, reused unchanged. It has 12,000 items, 3,000 per MMLU
  category.
  * qid SHA-256: `7b7e7514e13164d2b6d8a439552bc30c99d215005de972991ac56abdce8a3d48`.
  * Items file SHA-256: `a19ab049…`.
* **Comparability.** Results compare directly with that experiment's channel-removal trajectory: 13-step reach,
  102.2 M parameters, 5.94 %.
* **Seen items.** Every item has been seen on Qwen before, so this part is exploratory.

### Qwen3-8B

**1. Sealed first, with the confirmatory procedure** (`capprune confirmatory build-data --config
configs/confirmatory_qwen3_8b.yaml`), before any scoring.
* **Settings:** new split seed **2026112701**; commit `d81c49a` (clean tree).
* **Pool:** the same sources and deduplication as the confirmatory models: 25,432 items after dropping 135
  duplicates and 6 few-shot exemplars.
* **Registration:** `~/cap-prune-runs/confirmatory/qwen3-8b/registration.json`, **SHA-256
  `f96e8ef2739c316cba0c16626043d4f612e4d77106ac5074d32b8a68e4820a48`**.

| set | n | STEM | Humanities | Social Sciences | Other | set SHA-256 |
|---|---|---|---|---|---|---|
| confirmation (sealed, never read here) | 13,100 | 3,600 | 3,000 | 3,000 | 3,500 | `d905fad9e7704f63…` |
| gate | 11,132 | 4,231 | 3,574 | 954 | 2,373 | `ace74db0bb0314b4…` |
| probe | 400 | 100 | 100 | 100 | 100 | `c59bbc5c025b97be…` |
| hold-out | 800 | 200 | 200 | 200 | 200 | `82b4eeea90081b7d…` |

The **screening pool** is gate + probe + hold-out: 12,332 items, and it is the only part this experiment draws from. Its
file is `data-qwen3-8b/mmlu_probe.jsonl`, with SHA-256 `e16a14af…` as recorded in the registration.

**2. Admissibility, before anything else.** The pre-registered test (`capprune confirmatory admissibility`) scores
the dense model 0-shot on the gate pool (11,132 items).
* **Thresholds** (`confirmatory.yaml`): overall accuracy ≥ 0.40, and every category ≥ 0.30.
* **If it fails, the experiment stops.** Qwen3-8B gets no evaluation set and no pruning, and Qwen3-1.7B is not run
  either.
* **Plan check.** The test refuses to run unless this plan is committed unchanged and records the registration's hash.

**3. Evaluation set** (`capprune unstructured build-eval --model qwen3-8b`), drawn only after admissibility passes.
* **Pool:** the screening pool only. Before drawing, the pool file's SHA-256 and its per-category counts are checked
  against the registration.
* **Draw:** per category, the items are sorted by qid. 3,000 are chosen without replacement with
  `numpy.random.default_rng([20261127, crc32(category)])`. **If a category has 3,000 items or fewer, every item is
  used.**
* **Why not 3,000 everywhere.** The seal leaves fewer than 3,000 in two categories: the same category sizes and
  sealed targets hold for every model, whatever the split seed. We decided this with the user on 2026-11-27, before
  any scoring.

| | STEM | Humanities | Social Sciences | Other | all |
|---|---|---|---|---|---|
| screening pool | 4,531 | 3,874 | 1,254 | 2,673 | 12,332 |
| evaluation set | 3,000 | 3,000 | **1,254 (every item)** | **2,673 (every item)** | 9,927 |

* **Precision, stated before scoring.** One category's paired-loss estimate has a standard error of about √(q / n),
  where q is the paired disagreement rate. At q = 9 %, that is:
  * Social Sciences (n = 1,254): **about 0.85 points**;
  * Other (2,673): 0.58 points;
  * STEM and Humanities (3,000): 0.55 points.

  Social Sciences is therefore the most likely category to cross 2 points by chance at a level where its true loss
  is below 2 points. **The reach on Qwen3-8B is conservative:** biased toward a lower reach, so against GO. This
  adds to the upward bias that taking the worst of four estimates already has.
* **Category reported.** Every level reports which category has the largest estimated loss (§7), so the reader can
  see whether Social Sciences is the binding category.
* **Eligibility.** These screening items become seen on Qwen3-8B. A later confirmatory study confirms on the sealed
  13,100 items only, so the model stays eligible.

## 4. The operator: uniform unstructured Wanda, as published

**Definition** (Sun, Liu, Bair & Kolter, *A Simple and Effective Pruning Approach for Large Language Models*, ICLR
2024):
* **Score.** S_ij = |W_ij| · ‖X_j‖₂, where X_j is input feature j over every calibration token.
* **Comparison group.** Each output row. In every row of every linear, the int(columns × s) lowest-scoring weights are
  set to zero.
* **Order.** Transformer blocks are pruned in order. A block's statistics are gathered in one pass with that block
  still dense and every earlier block already pruned.
* **No weight update.**

**Check of `capprune/pruning/wanda.py` against the published code.** We read the official repository
(`locuslab/wanda`: `lib/prune.py::prune_wanda`, `lib/layerwrapper.py::WrappedGPT`, `lib/data.py::get_c4`,
`main.py`).

| published | `sequential_wanda` / `raise_unit` / `unstructured_mask` | |
|---|---|---|
| `W_metric = |W| * sqrt(scaler_row)`, where `scaler_row` is the running mean over samples of ‖x_j‖² | `|W| * sqrt(Σ x_j²)` over all calibration tokens | same ranking (a constant factor per layer) |
| `torch.sort(W_metric, dim=-1, stable=True)`, the lowest per row | `torch.sort(score, dim=1, stable=True)`, the lowest per row | same |
| **`int(W.shape[1] * sparsity_ratio)` weights per row (floor)** | **`int(round(ratio * cols))`** | **changed (below)** |
| `find_layers(layer)`: every `nn.Linear` of the block | units q, k, v, o, gate, up, down of every block | same; embeddings, output head and norms are not pruned |
| hooks on every linear of the block; one pass with the block dense; prune; recompute the block's output as the next block's input | hooks on every linear of block i; forward with blocks < i pruned, stopped after block i; prune block i | same computation. Inputs are recomputed from the embeddings for each block instead of being cached; bf16 rounding can differ slightly |
| `use_variant` off by default | not used | same |

**What we changed:** the per-row count.
* `unstructured_mask` now takes `count="round" | "floor"`.
* `raise_unit` reads `pruning.unstructured_count`, whose default stays `round`, so every existing configuration
  behaves exactly as before.
* This experiment sets `unstructured_count: floor`, the published count. The difference is at most one weight per
  row; at 43 % with 2,048 columns, it is 880 instead of 881.

**Configuration:**
* **Calibration.** 128 windows of 2,048 C4-train tokens (`en/c4-train.00000-of-01024.json.gz`). They are chosen by
  `PreparedData.calibration_indices(seed=0, n=128)` from the prepared 256-window pool.
  * The pool was drawn with the published rule: a random document longer than 2,048 tokens, then a random window.
  * Published Wanda draws its 128 windows directly with `random.seed(0)`, and our windows are not those.
  * Both models use the same 128 windows (identical prepared pool, §2). No benchmark item is used.
* **Levels.** Each level is pruned one-shot from the dense model: 5, 10, 15, 20, 25, 30, 35, 40, 43, 45 and 50 %
  layer sparsity, the same in every linear of every block.
* **The 43 % level** is the base paper's overall sparsity (43.02 %). It is reported as **"uniform Wanda at the same
  overall sparsity, not their method"**: their method is an LLM agent choosing per-layer sparsities, with a
  perplexity rollback.
* **Compression is reported two ways** at every level: the achieved layer sparsity (zeroed ÷ weights in the
  prunable linears), and the **zeroed weights as a share of all model parameters**. The counts below are exact,
  rows × int(cols × s) per linear (`attainability.json`):

| level | 1.7B zeroed | 1.7B layer sparsity | 1.7B share of all | 8B zeroed | 8B layer sparsity | 8B share of all |
|---|---|---|---|---|---|---|
| 5 % | 70,246,400 | 4.98 % | 4.08 % | 346,226,688 | 4.98 % | 4.23 % |
| 10 % | 140,492,800 | 9.97 % | 8.17 % | 693,706,752 | 9.99 % | 8.47 % |
| 15 % | 211,255,296 | 14.99 % | 12.28 % | 1,041,334,272 | 14.99 % | 12.71 % |
| 20 % | 281,501,696 | 19.97 % | 16.36 % | 1,388,814,336 | 20.00 % | 16.96 % |
| 25 % | 352,321,536 | 25.00 % | 20.48 % | 1,736,441,856 | 25.00 % | 21.20 % |
| 30 % | 422,567,936 | 29.98 % | 24.56 % | 2,082,668,544 | 29.98 % | 25.43 % |
| 35 % | 492,814,336 | 34.97 % | 28.64 % | 2,430,148,608 | 34.99 % | 29.67 % |
| 40 % | 563,576,832 | 39.99 % | 32.76 % | 2,777,776,128 | 39.99 % | 33.91 % |
| 43 % | 605,609,984 | 42.97 % | 35.20 % | 2,986,205,184 | 42.99 % | 36.46 % |
| 45 % | 633,823,232 | 44.97 % | 36.84 % | 3,125,256,192 | 45.00 % | 38.16 % |
| 50 % | 704,643,072 | 50.00 % | 40.95 % | 3,472,883,712 | 50.00 % | 42.40 % |

* **Parameter accounting.**
  * Qwen3-1.7B: 1,409,286,144 weights in the prunable linears (81.91 % of all), 311,164,928 tied embedding, 123,904
    norm.
  * Qwen3-8B: 6,945,767,424 prunable (84.80 %), 622,329,856 input embedding, 622,329,856 output head, 308,224 norm.
* **Zeroed weights give no memory or speed gain on standard hardware.** They are still stored and multiplied.
  Savings need a hardware-supported pattern such as 2:4, or sparse kernels, which typically pay off only at much
  higher sparsity. This experiment measures accuracy only.

**Persistence and regeneration** (`capprune/unstructured_feasibility.py::LevelRunner`).
* **Order within a level:**
  1. the masks are computed;
  2. the level's **mask hash** is written to `levels/sNN/mask.json`, **before the level is scored**;
  3. the level is scored;
  4. its **per-item outcomes** are written (`outcomes.npz`, `result.json` with the outcome file's SHA-256)
     **before the next level starts**.
* **Mask hash:** SHA-256 over the sorted unit names and each unit's packed keep-mask bits. It is `ranker.mask_hash`
  with the full digest.
* **What `mask.json` also records:**
  * that every row of every linear has exactly int(cols × s) zeros;
  * that every masked weight is zero in the model;
  * the achieved sparsities per linear type.
* **Regeneration.** Masks are not stored, only their hashes. A level is regenerated after a crash only if its
  regenerated mask hash equals the persisted one, and otherwise the run refuses that level. A level with a verified
  `result.json` is never pruned again.
* **Regeneration check.** On each model, the 5 % level's masks are generated twice before it is scored. The two
  hashes must agree, or the run stops and asks. This shows whether GPU regeneration is exact before any level depends
  on it.

## 5. Scoring

* **Dense and every level** are scored 0-shot on the model's evaluation set.
  * **Scorer:** `pool_signals.LogprobScorer`.
  * **Batches:** 100 items per batch within a category, in a seeded order (`question_order`, seed 2026), with a
    32,768-token budget. The dense model and the pruned model are scored in identical batch layouts.
  * **Score cache:** every batch is persisted to a SQLite score cache before it is used, keyed by mask hash.
* **How the pruned model is held.** It is the masked model (zeros in place), on a working copy of the dense model.
  The dense model itself is never modified and supplies the dense weights for every level.
* **Perplexity:** C4 validation (the prepared 32 × 2,048-token set), for the dense model and at every level.
* **Cross-check, Qwen3-1.7B.** The dense answers are compared item by item with the compensation experiment's dense
  answers on the same items, batches and budget.
* **Order of work:**
  1. this plan, committed and pushed;
  2. admissibility on Qwen3-8B (stop if it fails);
  3. the Qwen3-8B evaluation set;
  4. Qwen3-1.7B: dense and all levels;
  5. Qwen3-8B: dense and all levels;
  6. the analysis.
* **Expected time.** Sequential Wanda here re-runs the earlier blocks for every block: about 18 full forward passes
  of 262,144 tokens per level on Qwen3-8B. That is several minutes per level, plus scoring. Measured times are
  reported (§7).

## 6. Decision rule

**Quantities.**
* **Paired loss.** At level ℓ and category c: L_c(ℓ) = (lost − gained) / n on that category's evaluation items,
  against the dense model. "Lost" means dense right and pruned wrong.
* **Worst-category loss:** W(ℓ) = max over c of L_c(ℓ).
* **Reach:** the highest level ℓ in the grid such that W(ℓ′) ≤ 0.02 (2.0 points) for ℓ′ = ℓ and every lower level
  in the grid. It is expressed as the **zeroed share of all model parameters** at that level, the exact achieved
  count. If W(5 %) > 0.02, the reach is 0.
  * The reach uses point estimates, so it is noisy (§3).
  * The rule is a go/no-go screen for a larger study, not a test.

**The rule is on Qwen3-8B:**
* **GO:** reach ≥ 20 % of all parameters.
* **NO-GO:** reach < 10 %.
* **INCONCLUSIVE:** anything between. It is reported, and the decision is the user's.

**Qwen3-1.7B is the size contrast and is descriptive.**

**Attainability (computed before any scoring; `unstructured-feasibility/attainability.json`).**
* **Best attainable reach on this grid** = share of parameters in prunable layers × 50 % = 84.80 % × 50 % =
  **42.40 %** of all parameters on Qwen3-8B. That is ≥ 20 %, so **GO is attainable.** (Qwen3-1.7B: 81.91 % × 50 % =
  40.95 %.)
* **The rule on the grid,** for Qwen3-8B:

| reach ends at | 5 % | 10 % | 15 % | 20 % | 25 % … 50 % |
|---|---|---|---|---|---|
| share of all parameters | 4.23 % | 8.47 % | 12.71 % | 16.96 % | 21.20 % … 42.40 % |
| verdict | NO-GO | NO-GO | INCONCLUSIVE | INCONCLUSIVE | GO |

  A reach of 0 (W(5 %) > 2 points) is NO-GO. All three outcomes are attainable.

## 7. Also reported (descriptive)

* **Per level and category:**
  * the paired loss with one-sided 95 % Clopper–Pearson union bounds (`population_bounds` at α = 0.05 for each side);
  * dense and pruned accuracy;
  * **which category has the largest estimated loss.**
* **The pre-registered confirmation rule** applied to each evaluation set: per-category IUT, Clopper–Pearson union
  bound, 3 points, α = 0.05, fixed sequence over the levels in ascending sparsity, stopping at the first failure.
  Labelled **"not a certificate"**: the items were not sealed for this, and on Qwen3-1.7B they have been seen.
* **Perplexity relative to dense** at every level.
* **Qwen3-1.7B against channel removal.** The per-level table beside the compensation experiment's channel-removal
  trajectory (mask-only, and the compensated column for reference), matched on share of parameters. Each unstructured
  level is matched to the channel step nearest in share, up to that trajectory's end (18.28 %).
* **The 43 % level on Qwen3-8B.** Its per-category losses, beside the base paper's reported MMLU drop for Qwen3-8B at
  43.02 %: 77.38 → 56.67, **20.71 points**.
  * The comparison differs in method (agent with per-layer sparsities against uniform), shots (5-shot against
    0-shot) and items (MMLU test against this evaluation set).
  * Labelled "uniform Wanda at the same overall sparsity, not their method".
* **One figure:** the worst-category loss against the zeroed share of all parameters, for both models, with the
  2- and 3-point lines. For context, it also shows the Qwen3-1.7B channel-removal trajectory against the share
  removed.
* **Wall time and peak GPU memory per level:** pruning, hashing, the regeneration check, scoring and perplexity; the
  peak includes the dense model and the working copy.
* **Records:** the admissibility result, and the dense cross-check on Qwen3-1.7B.

## 8. Tests (CPU, a toy Qwen3; `tests/test_unstructured_feasibility.py`)

* **Exact sparsity.** Each layer reaches the requested sparsity exactly, per output row: int(cols × s) zeros in
  every row of every linear, at 5, 10, 43 and 50 %. The CPU parameter accounting equals the mask counts.
* **Scores.** They match a direct computation of |W| × ‖X_j‖₂.
  * X is captured from a reference copy with the earlier blocks pruned as the run pruned them.
  * The pruned weights are each row's lowest direct scores.
* **The published count** is floor, and the default count is unchanged.
* **Reproducibility.** Level masks are reproducible from the same seed, and another seed gives other masks.
* **Resume.**
  * A crash after a level's mask hash was persisted, and before its outcomes were, resumes to the same mask hashes as
    an uninterrupted run.
  * Earlier levels are not pruned again.
  * A regenerated hash that differs from the persisted one is refused.
* **Regeneration check:** it generates twice and records it.
* **The draw and the rule.**
  * The evaluation-set draw takes 3,000 where possible and every item otherwise, independent of input order.
  * The reach, worst-category and verdict rules are tested.
* **The analysis end to end** on synthetic run files: it recomputes the reach and the verdict, names the
  largest-loss category at every level, and draws the figure.

## 9. Outputs

* **Run files:** `~/cap-prune-runs/unstructured-feasibility/`:
  * `attainability.json`;
  * `<model>/model.json`, `<model>/eval/` (the evaluation set and the score cache) and `<model>/dense.json` +
    `dense.npz`;
  * `<model>/levels/sNN/` (`mask.json`, `result.json`, `outcomes.npz`);
  * `analysis/` (`analysis.json`, `analysis.md`, `worst_loss.png` / `.pdf`).

  Each file records the commit, the tree state and the GPU. Qwen3-8B's seal and admissibility are in
  `~/cap-prune-runs/confirmatory/qwen3-8b/`.
* **Report:** `docs/unstructured_feasibility_report.md`. It gives the verdict against §6, the §7 tables and the
  figure, with every number citing its run file.
* **Tests:** the full suite, passing with exit code 0, before the pull request. Then the documented backup.

## References

* Sun, M., Liu, Z., Bair, A. and Kolter, J. Z. (2024). A Simple and Effective Pruning Approach for Large Language
  Models. ICLR 2024 (arXiv 2306.11695); code `github.com/locuslab/wanda`.
* Kodathala and Vunnam (2026). LLMs can Compress LLMs: Adaptive Pruning by Agents. arXiv 2601.09694.
