# Experiment 1 — adaptive candidate evaluation (evaluator + replay)

**Question.** Can adaptive allocation of benchmark questions across domains identify a pruned checkpoint that
meets predefined per-domain accuracy-loss limits with less evaluation compute than round-robin early stopping?

This milestone covers the evaluator, the three allocation policies, an offline replay harness, a deterministic
CPU fixture and the bridge from the existing Qwen3 pipeline into the replay. LLM-guided pruning is a later
experiment; nothing here calls an LLM API. Code: [`capprune/adaptive/`](../capprune/adaptive/), tests:
[`tests/test_adaptive.py`](../tests/test_adaptive.py).

## 1. Candidate evaluation

For every benchmark item we record its domain, a stable item id (the MMLU `qid`, e.g. `abstract_algebra/test/60`),
dense correctness and candidate correctness. The paired loss of an item is `dense − candidate`: **+1** if pruning
broke a correct answer, **−1** if it fixed a wrong one, **0** otherwise. Every candidate is compared against the
original dense checkpoint. A domain's loss is `L_d = (K+ − K−) / N_d` over its search pool; its tolerance
`ε_d` is configurable (`0.03` = 3 accuracy points).

**Bounds** ([`bounds.py`](../capprune/adaptive/bounds.py)). Items of a domain are revealed in a fixed, seeded
random order, so the first `n` items are a simple random sample without replacement and the observed `k+` is
Hypergeometric(`N_d`, `K+`, `n`). Inverting the hypergeometric tails gives exact one-sided bounds on `K+` and `K−`;
they combine into

    hi = (upper(K+) − lower(K−)) / N_d        lo = (lower(K+) − upper(K−)) / N_d

with each count bound at half the look's error share. At `n = N_d` both are exact.

**Looks and error spending.** Domain `d` is examined only at planned sizes `m_d, m_d + b_d, m_d + 2b_d, …`
(minimum sample, then fixed batches; the last batch ends exactly at the pool; sizes the budget can never reach are
not planned). `alpha_false_accept` is split equally over domains and then over each domain's planned looks
(`spending: uniform`, or the Lan–DeMets power family `spending: power` with `spending_rho`); the upper bound at
each look gets its share. `alpha_false_reject` is split the same way for the lower bounds.

**What the bounds cover.** For a fixed candidate, with probability at least `1 − α_FA − α_FR` over the random
question order, *every* bound at *every* planned look of *every* domain contains that domain's **full search-pool**
loss. Because this holds simultaneously, it is valid for any allocation policy and any stopping time:
P(false accept) ≤ `α_FA`, P(false reject) ≤ `α_FR`. Not covered: the holdout split or the underlying population
(the holdout is reported separately as a point check), and multiplicity across candidates (each candidate has
its own error budget). Treating `K+` and `K−` separately is conservative: the half-widths add instead of
combining in quadrature.

**Decision rule** ([`evaluator.decide`](../capprune/adaptive/evaluator.py), one function for every policy).
After the minimum sample in every domain: **reject** if any domain's lower bound exceeds its tolerance;
**accept** if every domain's upper bound is at or below its tolerance; otherwise sample again. **Inconclusive**
when no open domain's next batch fits the remaining budget.

## 2. Allocation policies ([`policies.py`](../capprune/adaptive/policies.py))

All policies use the same initial sample, per-domain question order, candidate order, looks, stopping rule and
budget. They only choose which *eligible* domain (open, next batch fits the budget) gets the next batch. Ties
go to the fixed domain order.

| policy | next batch goes to |
|---|---|
| `round_robin` | the next eligible domain after the last one sampled, cycling in the fixed order |
| `widest_interval` | the eligible domain with the largest `hi − lo` |
| `threshold_aware` | the eligible domain with the largest `log P(lo′ > ε)` (tie-break below) |

**Threshold-aware score.** `lo′` is the domain's lower bound at its *next* planned look, with that look's
error share. The probability is over the next batch's (broken, fixed) counts under a Dirichlet-multinomial
predictive built from the revealed counts with a Dirichlet(½, ½, ½) prior. It is computed exactly on the
(broken, fixed) grid, and the log keeps tiny probabilities comparable. Tie-break: larger `(hi − ε)/(hi − lo)`,
meaning the tolerance sits lower in the interval, then the fixed order.

Why only the reject side counts: while two or more domains are open, one batch can settle the checkpoint
decision only by triggering a reject. ACCEPT needs every domain to pass. A domain also passes at the same look
regardless of policy, because its look sizes and question order are fixed. So **an ACCEPT costs exactly the
same under every policy that only samples open domains** (the tests check this). Adaptive allocation saves
questions only by reaching REJECT sooner, or by reaching it at all before the budget runs out.

## 3. Replay

* **Input** ([`cache.py`](../capprune/adaptive/cache.py)) is cached per-item correctness for the dense model and a
  fixed pool of candidates (format below).
* **`ReplaySource`** is built from search-pool items only. It keeps the answers private, returns a domain's next
  items in its fixed order on request, and refuses to go over budget.
* **The policy** receives a `PolicyView`: per-domain revealed counts, bounds and status, plus the static plan.
  It never gets a reference to the source.
* **The scorer** ([`scorer.py`](../capprune/adaptive/scorer.py)) is separate and privileged. It computes each
  candidate's full-pool reference (qualifies iff every `L_d ≤ ε_d`) and labels each decision: correct accept,
  correct reject, false accept, false reject or inconclusive.
* **Reports** ([`replay.py`](../capprune/adaptive/replay.py)): per candidate and policy, the decision, questions,
  prompt tokens and measured eval seconds; per-policy totals; and the budget each policy needs to find its first
  qualifying checkpoint under a latency target. Candidates are tried in the cache order; latency misses are
  skipped unevaluated; the search stops at the first accept.

**Time accounting.** GPU time cannot be measured in replay, so `adaptive-build` times every look-sized batch
(CUDA-synchronised) while scoring the pool in the replay's own question order and layout. Replay then sums the
measured time of exactly the batches a run consumed. If the replay's `seed`, `min_sample` or `batch_size`
differ from the build, GPU time is reported as `n/a`. The report keeps two figures apart:

* **candidate-eval s**: scoring time of the consumed batches.
* **search wall s**: candidate-eval s plus building or loading and benchmarking every considered candidate, plus
  the replay's decision overhead. It is reconstructed from measured parts, not timed end to end.

A fully live sequential runner, which scores only what a policy requests, is not implemented yet.

## 4. Commands (PowerShell, from the repo root, venv active)

```powershell
# tests (25 new; `python -m pytest -q` also runs the existing 43)
python -m pytest tests/test_adaptive.py -q

# A. deterministic CPU fixture (no model, no data)
python -m capprune adaptive-fixture --out .smoke/adaptive-fixture
python -m capprune adaptive-replay  --config configs/adaptive_fixture.yaml

# B. pipeline -> cache -> replay on CPU (tiny random Qwen3, synthetic MMLU-format data)
python -m capprune make-tiny --out .smoke/tiny-qwen3
python -m capprune prepare --config configs/smoke.yaml
python -m capprune adaptive-build  --config configs/adaptive_smoke.yaml
python -m capprune adaptive-replay --config configs/adaptive_smoke.yaml

# C. real pilot: Qwen3-1.7B, four MMLU categories, dense + structured Wanda candidates (GPU)
python -m capprune prepare        --config configs/qwen3_1.7b_structured.yaml    # already done on this machine
python -m capprune adaptive-build  --config configs/adaptive_pilot.yaml --dry-run # checks config/data, loads no model
python -m capprune adaptive-build  --config configs/adaptive_pilot.yaml           # resumable; one file per model
python -m capprune adaptive-replay --config configs/adaptive_pilot.yaml
# replay-only sweeps need no GPU, e.g. a 5-point tolerance:
python -m capprune adaptive-replay --config configs/adaptive_pilot.yaml --set evaluation.tolerance=0.05 --out ~/cap-prune-runs/adaptive/pilot/replay-tol05
```

Outputs are in `out_dir`:

* `summary.md`: the tables.
* `summary.json`: reference, totals and search results.
* `runs.jsonl`: every run with its full per-look trace.
* `runs.csv`: one row per run with final per-domain n, lo, hi and status.

## 5. Replay input format: the files to provide

`adaptive-build` writes this format. A cache produced by another pipeline only needs the same files.

```
<cache_dir>/items.jsonl          one line per item
    {"item_id": "abstract_algebra/test/60", "domain": "STEM", "split": "pool", "tokens": 187}
    split: "pool" = searched by the sequential test, "holdout" = final check only; tokens may be null
<cache_dir>/models/dense.json    {"name": "dense", "role": "dense", "correct": {"<item_id>": 0|1, ...},
                                  "latency": {"decode_ms_per_token": 28.9, ...} | null, "timing": ... | null}
<cache_dir>/models/<name>.json   {"name", "role": "candidate", "order": 0, "correct": {...}, "latency": ...,
                                  "timing": {"layout": {"seed", "min_sample", "batch_size"},
                                             "batch_seconds": {"STEM": [look0, look1, ...], ...},
                                             "setup_seconds", "bench_seconds", "holdout_seconds"} | null,
                                  "info": {...}}
```

Every model needs correctness for every pool and holdout item. `order` sets the search order (ascending).

For the real pilot on this machine, `adaptive-build` creates everything. It uses the prepared MMLU splits in
`~/cap-prune-runs/data`, the cached `Qwen/Qwen3-1.7B` weights, and structured Wanda (`pruning/wanda.py`) at
the sparsities listed in `configs/adaptive_pilot.yaml`. To add your own checkpoints, list them under
`build.candidates`:

* `kind: masks` with a structured mask file saved by a CAP-Prune run (`<run>/masks/<state>.safetensors`).
* `kind: hf_dir` with a pruned Hugging Face checkpoint directory that uses the same tokenizer.

## 6. Smoke-test results (CPU, this repository)

Fixture: 4 domains, pools of 600 / 900 / 500 / 550 items, tolerance 5 points, minimum sample 40, batch 40,
budget 1,000, α_FA = α_FR = 0.05.

| candidate | full-pool reference | round_robin | widest_interval | threshold_aware |
|---|---|---|---|---|
| clear_fail (Other +28 pts) | no | reject, 320 q | reject, 200 q | reject, 200 q |
| near_threshold (Humanities +5.0 pts) | yes, at the limit | inconclusive, 1,000 q | inconclusive, 1,000 q | inconclusive, 1,000 q |
| easy_pass (all 0) | yes | accept, 800 q | accept, 800 q | accept, 800 q |

No false accepts or false rejects. Budget to the first qualifying checkpoint (latency ≤ 0.85 × dense):
round-robin 2,120 questions; widest-interval and threshold-aware 2,000 (−5.7 %). The whole saving comes
from rejecting `clear_fail` faster. Accepting `easy_pass` and exhausting the budget on `near_threshold` cost
the same under every policy.

The tiny-Qwen3 CPU build (dense plus structured Wanda at 50 % and 25 %) runs the real scorer, pruning,
materialisation and benchmark code, and replays with measured batch times. Its decisions are meaningless:
the model is random and pools have 16–64 items.

## 7. Certifying 3 points at the real pool sizes

With the pilot's look plan (100 / 100, budget 3,000, α = 0.05 per side), and assuming revealed counts sit at
their expected values, the smallest look at which a domain's upper bound reaches the tolerance is:

| broken / fixed rate | STEM (N=1261), ε=0.03 | Humanities (N=1881), ε=0.03 | STEM, ε=0.05 | Humanities, ε=0.05 |
|---|---|---|---|---|
| 2 % / 2 % (loss 0) | 600 | 700 | 400 | 400 |
| 4 % / 3 % (+1 pt) | 1,000 | 1,300 | 600 | 700 |
| 6 % / 5 % (+1 pt) | 1,100 | 1,500 | 700 | 900 |
| 10 % / 8 % (+2 pts) | 1,261 (whole pool) | 1,800 | 1,000 | 1,300 |

At 3 points, a candidate is certifiable within 3,000 questions only if it flips about 2 % of answers or fewer
in each direction. Check the measured flip rates in the pilot's reference table and sweep tolerance and budget
in replay; neither requires rebuilding.

## 8. Known limitations

* Per-candidate guarantee only: there is no correction across the candidates tried in a search.
* The bounds cover the search pool. Generalisation to the holdout is checked empirically, not guaranteed.
* The bounds are conservative: union of two count bounds, Bonferroni-style spending. A trinomial inversion or
  a without-replacement betting confidence sequence would be tighter; either fits behind `domain_state`.
* Decode latency from HF eager generation at batch 1 is largely launch-overhead-bound on this GPU. The dense
  run measured about 29 ms/token with a 9–12 ms IQR. Prefill latency (`metric: prefill_ms`) may separate
  candidates better.
