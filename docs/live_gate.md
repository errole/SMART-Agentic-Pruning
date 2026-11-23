# Experiment 2: the capability gate in the live pruning-and-rollback loop

The exact-bound evaluator from experiment 1 ([adaptive_evaluation.md](adaptive_evaluation.md)) becomes the
acceptance gate of the existing Qwen3-1.7B structured-pruning search. It replaces the perplexity rule in
some arms, so the comparisons separate the effect of the gate from the effect of the proposer.

| arm | method | proposer | gate | compares |
|---|---|---|---|---|
| A | `llm_ppl` | LLM (Gemini 3 Flash, paper prompt) | perplexity: roll back if PPL rises > 15 % vs. parent | A vs B: the gate |
| B | `llm_cap` | same LLM, same prompt template and action space | capability, threshold-aware allocation | B vs C: the proposer |
| C | `greedy_cap` | greedy sensitivity heuristic (`HeuristicController`) | same capability gate as B | |
| D | `llm_cap_rr` | same LLM | capability, round-robin allocation | B vs D: allocation only |

All arms start from the same dense checkpoint and share:
* the pruning operator: structured Wanda on MLP channels, in multiples of 64 channels
* the allowed actions: units `L0.mlp`…`L27.mlp`, each step adding between 0.01 and 0.10 sparsity per unit
* the calibration windows (run seed 0)
* the profiling settings
* the proposal budget (`search.max_iterations`)
* the target sparsity: 0.10 of MLP weights

The per-step deployment check is off. It would gate on timing noise; speed is measured once at the end
instead.

## What was already wired and what was added

The existing loop (`search/loop.py`) already did: profile, propose, build a trial from the parent, gate, accept
or roll back, and checkpoint. The perplexity gate, both controllers and checkpoint recovery are reused
unchanged. Added:

* **`gate.kind: capability`** ([`adaptive/live.py`](../capprune/adaptive/live.py)). Each attempt runs
  `evaluator.evaluate_candidate` with a live item source that scores only the batches the gate requests, on the
  trial model. The paired loss is measured against the dense model's cached answers, so losses accumulate over
  steps. Outcomes map as follows:
  * ACCEPT: the trial becomes the parent.
  * REJECT: rolled back (`domain_fail`).
  * INCONCLUSIVE: rolled back (`domain_inconclusive`); the last accepted checkpoint is kept.
  * Perplexity is still measured, but only for the controller's feedback.
* **A fresh question order per attempt.** The seed is `sha256(order_seed, run seed, attempt number)`, fixed
  before the proposal exists, so it can't depend on the feedback that produced the candidate.
* **Rollback verification** (`search.verify_rollback`). An exact float64 digest of every prunable weight is
  checked at the start of each attempt (the parent equals the last accepted checkpoint) and after every
  rollback (restored bit-for-bit, mask fingerprint equal). Results go to the journal as `parent_verified` and
  `rollback_verified`.
* **Per-attempt journal record.** It holds:
  * parent and trial IDs
  * the proposal
  * the gate decision and reason
  * per-category estimate, bounds, counts and status, plus the look-by-look trace
  * questions used and the order seed
  * timings: profiling, controller, trial build, PPL, gate, scoring, rollback, the whole attempt
  * whether the target was reached
  * the exact feedback the controller receives next

  The run's `summary.json` gets `gate_counts`.
* **Controller prompt** (`search/controller.py`). The perplexity arm's prompts are byte-identical to the
  previous version (checked against the committed version once during development). Capability arms get a
  different rule line, see the accepted model's per-category bounds, and receive capability feedback.
* `capprune live-bench`, `capprune live-report`, `capprune gate-design`.

## Data roles (disjoint)

| role | data |
|---|---|
| pruning calibration | C4 train windows (Wanda scores, profiles) |
| gate development | MMLU validation (`gate-design` only) |
| pruning-time evaluation | MMLU test probe split, 40 %: the capability gate. C4 validation: the PPL gate |
| final audit | MMLU test final split, 60 %, 5-shot. FreebaseQA eval. WikiText-2 test |

## Gate settings (locked on development data before any arm ran)

`capprune gate-design --config configs/gate_dev.yaml`:
1. Scores the dense model and uniform structured-Wanda candidates (2, 3 and 5 % sparsity) on MMLU validation.
2. Estimates each category's broken and fixed answer rates.
3. Simulates the gate at the real pruning-time pool sizes over 100 question orders per option. The simulation
   uses counts only, never pruning-time answers.

The locking rule was written into the config beforehand:
* **Tolerance:** 3 points.
* **Budget:** the smallest budget that accepts every 2-point profile at least 80 % of the time.
* **Spending:** uniform, unless power spending adds at least 10 points of acceptance.

Result: **tolerance 3 points, budget 5,614 = the whole pool, uniform spending, minimum sample and batch size 100,
α = 0.05 per side.**
* At 4,000 questions, the high-churn profile (wsp05's STEM: 12 % broken / 6 % fixed, shifted to 2 points) was
  certified 0 % of the time.
* At the full pool, every 2-point profile was certified and every 4-point profile was rejected.

With budget equal to the pool the bound becomes exact on exhaustion, so the live gate cannot return
INCONCLUSIVE. That path is exercised by the CPU smoke test instead.

Reported separately, not used for locking:
* **Locked gate on experiment 1's pruning-time cache:** the 3-point-passing wsp02 is accepted in 50/50 question
  orders (2/50 at the old 3,000 budget); wsp03 (+4.0) in 0/50.
* **Exploratory 5-point table:** in `~/cap-prune-runs/adaptive/gate-dev/design/summary.md`.

**Scope of the bounds: per candidate.** For each trial, with probability at least 1 − α_FA − α_FR over its own
question order, all its bounds cover its loss on the full pruning-time pool. No search-wide error control is
claimed; the selected checkpoint is audited on the held-out final split.

## Commands

```powershell
# CPU smoke test (tiny random Qwen3, scripted mock controller)
python -m pytest tests/test_live_gate.py -q                     # 9 tests, ~80 s
python -m capprune make-tiny --out .smoke/tiny-qwen3
python tests/live_smoke.py                                       # synthetic data, pruning-time key = dense predictions
Start-Process python "tests/mock_controller.py --port 8765 --script live"
python -m capprune supervise   --config configs/live_smoke.yaml
python -m capprune live-bench  --config configs/live_smoke.yaml
python -m capprune live-report --config configs/live_smoke.yaml

# RTX 5090 pilot
python -m capprune adaptive-build --config configs/gate_dev.yaml # gate development cache (~15 min)
python -m capprune gate-design    --config configs/gate_dev.yaml # locks the gate settings
python -m capprune supervise      --config configs/live_gate.yaml   # dense + 4 arms; rerun the same command to resume
python -m capprune live-bench     --config configs/live_gate.yaml
python -m capprune live-report    --config configs/live_gate.yaml   # -> ~/cap-prune-runs/live-gate/_live/report.md
```

Smaller or larger pilots: `--set search.max_iterations=N`, `--set matrix.seeds=[0,1,2]`, or
`--methods greedy_cap` to run only the arm that needs no API.

## Why the smoke test relabels its data

The tiny model is random, so pruning breaks and fixes answers equally often and the paired loss is about 0.
`tests/live_smoke.py` sets the pruning-time answer key to the dense model's own predictions. Every changed
answer then counts as broken, and a proposal's size sets its loss. With tolerance 0.15 and a budget of 360 of
640 questions, the scripted proposals give:
1. ACCEPT (becomes the parent)
2. REJECT, rolled back to 1
3. INCONCLUSIVE after 352 questions, rolled back to 1
4. built from 1 again

All parent and rollback digests match. The threshold-aware reject costs 80 questions, round-robin 128; the two
accepts cost the same under both.

## Results: RTX 5090 pilot, seed 0, 10 proposals per arm (2026-11-21)

Full report: `~/cap-prune-runs/live-gate/_live/report.md`, which includes every attempt.

**Arms B and D are incomplete.** The Gemini free tier allows 20 `gemini-3-flash` requests per day. That quota ran
out during arm B, so B paused after 7 of its 10 attempts with no final audit, and D has not started. Both resume
with the same `supervise` command once the quota resets or a billed key is used. B continues from attempt 8 (the
saved config fingerprint is unchanged), then D runs.

| arm | status | attempts: accept / reject / inconclusive | MLP sparsity (params removed) | gate questions | held-out worst-category loss | held-out overall MMLU loss | FreebaseQA EM (dense 0.209) | WikiText-2 PPL (dense 16.72) | decode / prefill vs. dense |
|---|---|---|---|---|---|---|---|---|---|
| A LLM + PPL gate | complete | 2 / 0 / 0, reached target | 10.4 % (6.4 %) | – | **STEM +18.2** | +14.3 | 0.172 | 18.05 | inconclusive (noisy) |
| B LLM + capability gate | paused, 7/10 | 4 / 3 / 0 | 3.68 % | 19,883 | not audited yet | – | – | – | not measured |
| C greedy + capability gate | complete | 4 / 6 / 0 | 3.72 % (2.3 %) | 29,927 | **STEM +5.2** | +2.8 | 0.189 | 16.99 | inconclusive (noisy) |
| D LLM + capability gate, round-robin | pending | – | – | – | – | – | – | – | – |

* **The gate.** The perplexity gate accepted two steps that raised perplexity by only 3.1 % and 3.7 % but cost
  18 points of held-out STEM accuracy (14 points overall). The capability gate rejected every step that pushed a
  category past 3 points: 6 of C's attempts and 3 of B's.
  * Rejects cost 1,200–3,661 questions.
  * Accepts close to the limit cost up to 5,061, because certifying STEM at +2.5–2.9 means scoring the whole
    STEM pool.
  * With budget = pool, no attempt ended inconclusive.
* **The certificate does not carry over in full.** C's last accepted step was certified at STEM +2.9 on the
  pruning-time pool (0-shot), but STEM lost +5.2 in the held-out audit (5-shot). My first split of that gap used
  only three cells and credited ~1.3 points to the 0-shot-to-5-shot change; the completed 2×2 below supersedes
  it.
* **The proposer.** Under the same capability gate, the LLM reached about the same sparsity as the greedy
  heuristic (3.68 % after 7 attempts vs 3.72 % after 10), with fewer rejections so far (3 of 7 vs 6 of 10).
  * Its first step, larger than greedy's, was rejected (STEM +3.3).
  * It then shrank its steps (increments 0.04 → 0.03 → 0.02 → 0.01) after capability feedback.
  * It is not yet a comparison: B is unfinished and unaudited, and there is one seed.
* **Deployment speed.** Removing 2.3 % (C) or 6.4 % (A) of parameters gave no measurable change. Decode and
  prefill ratios are 0.95–1.03 with intervals that include 1, and the dense timing itself varied 14–17 % (IQR /
  median), so speed is marked inconclusive. Peak memory fell from 3.41 to 3.33 and 3.20 GiB.
* **Time.** C's search took 304 s, of which 224 s was scoring for the gate. A's took 86 s (68 s controller). B's
  controller time (746 s) is mostly waiting on the rate limit.

**Status at 2026-11-21 21:34 EST:** the quota was still exhausted (HTTP 429), so B stays paused at attempt 7 and D
has not started. B's config fingerprint, seed, checkpoint (next attempt 8, state `7139f7e7`) and controller cache
are unchanged. After the quota resets at midnight Pacific:

```powershell
python -m capprune supervise   --config configs/live_gate.yaml                            # resumes B at attempt 8, then D
python -m capprune live-bench  --config configs/live_gate.yaml --methods llm_cap,llm_cap_rr   # new session file
python -m capprune live-report --config configs/live_gate.yaml
```

`live-bench` now writes one `_live/bench-<time>.json` per session and never overwrites the A/C session
(`_live/bench.json`). The report takes each arm from its latest session and compares it with that session's own
interleaved dense reference.

## Gap diagnostic: the greedy checkpoint, matched prompting (2026-11-21)

Separate from the live-arm results. Outputs are in `~/cap-prune-runs/live-gate-diagnostics/greedy_cap_gap/`
(`summary.md`, `summary.json`, `per_item.jsonl`, `provenance.json`, `run.log`); the run artifacts are only read.

```powershell
python -m capprune gap-diagnostic --config configs/live_gate.yaml --arm greedy_cap --out ~/cap-prune-runs/live-gate-diagnostics/greedy_cap_gap
python -m capprune gap-diagnostic --config configs/live_gate.yaml --arm greedy_cap --out ~/cap-prune-runs/live-gate-diagnostics/greedy_cap_gap --reanalyze   # tables only, no GPU
```

**Setup.** Dense Qwen3-1.7B vs. C's saved checkpoint `4bb4502405a1a703` (3.72 % MLP sparsity, mask sha256
checked).
* **Procedure, identical for every cell:** the final audit's. Masked bf16 weights, `max_seq_len` 4096, a batch
  budget of 32,768 tokens, 512-item chunks.
* **Exemplars:** the first five MMLU-dev exemplars of each subject, the same on both splits. No 5-shot prompt
  dropped an exemplar, and no 0-shot prompt exceeds the gate's 2,048 tokens.
* **Exclusions:** five scored items are word-for-word duplicates (question and choices) of exemplars used for a
  sibling subject. They are all in "Other" and none appears in its own prompt; they are excluded from every
  cell.
* **Reuse:** the held-out 5-shot cell is the run's final audit. Its cache keys carry the checkpoint ID for all
  8,428 items, its settings match, the scoring code is unchanged since the runs, and a recomputed 512-item chunk
  agrees item for item.
* **Scored here:** the pool 5-shot cell, which was missing. Also both 0-shot cells: the gate's cache covered
  4,761 of 5,614 pool items in different batches, and the earlier held-out 0-shot run kept no per-item answers.

STEM loss (accuracy points; 95 % paired, category-stratified item bootstrap):

| question split | 0-shot | 5-shot |
|---|---:|---:|
| Pruning-time pool (n = 1,261) | +2.78 [+0.63, +4.92] | +2.30 [+0.08, +4.52] |
| Held-out audit (n = 1,892) | +4.23 [+2.48, +5.97] | +5.18 [+3.38, +6.92] |

| STEM contrast | estimate | 95 % interval |
|---|---:|---|
| pool: 5-shot − 0-shot | −0.48 | [−3.17, +2.22] |
| held-out: 5-shot − 0-shot | +0.95 | [−1.27, +3.17] |
| 0-shot: held-out − pool | +1.45 | [−1.32, +4.20] |
| 5-shot: held-out − pool | +2.88 | [+0.05, +5.68] |
| change in the split gap (5-shot − 0-shot) | +1.43 | [−2.03, +4.86] |

All four cells, all categories:

| cell | overall | STEM | Humanities | Social Sciences | Other | worst |
|---|---:|---:|---:|---:|---:|---|
| pool, 0-shot | +1.37 | +2.78 | +1.44 | +1.14 | +0.08 | STEM |
| pool, 5-shot | +1.85 | +2.30 | +2.34 | +1.06 | +1.45 | Humanities |
| held-out, 0-shot | +1.84 | +4.23 | +2.12 | +0.49 | +0.32 | STEM |
| held-out, 5-shot | +2.82 | +5.18 | +2.05 | +2.17 | +2.25 | STEM |

**Reading it.**
* **Prompting.** Moving from 0-shot to 5-shot does not consistently change the STEM loss: −0.5 points on the
  pool, +1.0 on the held-out split, and both intervals include 0.
* **The split gap remains under matched prompting.** The held-out loss exceeds the pool loss at both shot counts
  (+1.5 at 0-shot, +2.9 at 5-shot; only the 5-shot interval clears 0), and there is no evidence that the gap
  depends on the shot count.
* **Revision of my earlier split.** The ~1.3 points I attributed to prompting came from comparing different items
  and does not hold up.
* **Possible causes.** Repeatedly accepting steps that only just pass on one reused pool (selection) is one
  explanation. Ordinary sampling error is another: as an estimate of population loss, the pool's STEM value
  carries about ±2.1 points of uncertainty on its own. This design cannot separate the two. The held-out split
  has now been inspected, so it counts as diagnostic data.

**Numerics.** About 1 % of predictions change with batch composition alone in bf16:
* rescored dense pool answers match the cached reference on 5,561 of 5,614 items
* the rescored checkpoint matches the gate's own answers on 4,715 of 4,761
* masked and materialised weights agree on 254 of 256 at 0-shot and 256 of 256 at 5-shot

That is why the gate saw STEM +2.9 where this matched rescoring gives +2.78. The earlier quick held-out 0-shot
run (one pass instead of 512-item chunks) gave STEM +3.86 against +4.23 here, so batching alone moves a category
loss by about 0.4 points. In the live gate, dense answers were scored in one pass and trial answers in gate
batches, so this noise enters the paired outcomes.

**Implications for the next gate experiment** (this run's tolerance and prompting are unchanged):
1. A certificate on the pruning-time pool is not a certificate on new items. A held-out 3-point goal needs either
   a margin, a bound on the population rather than on the finite pool, or a fresh confirmation sample for the
   final accept, which also limits repeated selection against one pool.
2. Gating at 0-shot is not what inflates the audit loss, so switching the gate to 5-shot alone would not close
   the gap.
3. Score the dense reference and the trial in the same batch layout, or in higher precision, so that ~1 %
   numeric churn does not enter the paired loss.
4. Isolating selection needs a separate experiment, for example rescoring every accepted checkpoint of the search
   on an untouched split, or giving each attempt its own fresh items.

## Local-controller pilot: Ollama + Qwen2.5-7B-Instruct (2026-11-21, exploratory, one seed)

The controller is `qwen2.5:7b-instruct`, served by Ollama 0.40.16 and installed with winget at the user's
request. Model details: digest `845dbda0ea48…`, Q4_K_M GGUF, 7.6B parameters. It ran with a 32,768-token context,
fully on the GPU (6.6 GB), and processed the full ~1.4–2.0K-token prompts.

The model being pruned is still Qwen3-1.7B. [`configs/live_gate_local.yaml`](../configs/live_gate_local.yaml)
inherits everything else from `live_gate.yaml`: seed 0, 10 proposals, the 3-point gate with a 5,614-question
budget, the action space, and the final audit. Its only controller endpoint is `http://127.0.0.1:11434/v1`.
Outputs go to `~/cap-prune-runs/live-gate-ollama/`. The Gemini run directory was only read (for the greedy
reference arm), and paused arm B there stays resume-safe.

**One real proposal, outside the experiment journals.** `capprune controller-check` sent the exact iteration-2
prompt of Gemini arm B, read from a copy of its controller cache, through the experiment's adapter:
* the JSON was complete on the first try
* it proposed `L1.mlp +0.05`, a known unit inside [0.01, 0.10] and under the 0.875 per-unit cap
* `sanitize` changed nothing

The result is in `~/cap-prune-runs/live-gate-ollama-checks/`.

| arm | valid proposals | accept / reject / inconclusive | MLP sparsity (params removed) | gate questions | held-out worst category | held-out MMLU loss | FreebaseQA (dense 0.209) | WikiText-2 (dense 16.72) | controller s | search s | run wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A local LLM + PPL gate | 10/10 | 10 / 0 / 0 | 3.94 % (2.42 %) | – | STEM +7.61 | +3.57 | 0.179 | 17.42 | 13 | 105 | 484 |
| B local LLM + capability, threshold-aware | 10/10 | 8 / 2 / 0 | 1.15 % (0.71 %) | 37,625 | Other +1.61 | +1.25 | 0.201 | 16.82 | 17 | 451 | 778 |
| D local LLM + capability, round-robin | 10/10 | 7 / 3 / 0 | 1.19 % (0.73 %) | 41,322 | Other +1.88 | +1.33 | 0.202 | 16.87 | 16 | 491 | 814 |
| C greedy reference (Gemini-run dir) | n/a | 4 / 6 / 0 | 3.72 % (2.29 %) | 29,927 | STEM +5.18 | +2.82 | 0.189 | 16.99 | – | 304 | 608 |

The 30 local replies had no JSON repairs, heuristic fallbacks, unknown units or clamped increments. Median
controller latency was 1.1–1.5 s.

**Speed.** Measured in an interleaved session after the Ollama model was unloaded (`ollama ps` was empty). The
dense decode spread was only 3.6 % of the median, so the intervals can be trusted.
* **Decode:** no measurable change for any arm (ratios 0.997–1.006, every interval includes 1).
* **Prefill, A:** measured 4.5 % slower [1.8, 7.9] %. My unconfirmed guess is that irregular per-layer GEMM shapes
  after uneven channel removal are to blame.
* **Prefill, B and D:** inconclusive.
* **Peak memory:** 3.41 GiB dense, 3.33 GiB for A, 3.38 GiB for B and D.

**What the local pilot shows (exploratory, one seed).**
* **The 7B controller can drive the full loop.** All 30 attempts produced valid proposals and all three arms
  finished their audits. It is a timid, non-exploratory pruner, though:
  * A walked through layers in index order, two or three units at a time.
  * B and D put almost every step on `L1.mlp`, one unit at a time.
  * No arm approached the 10 % target.
* **Gate, A vs B (same controller).** The perplexity gate accepted all ten steps (perplexity never rose more
  than 1.2 % per step) and ended at STEM +7.6 held-out, 2.5 times the limit. The capability gate rejected two
  steps that pushed STEM past 3 points on the pool and ended within the limit held-out (worst category +1.6).
  * Part of the difference is the controller: given capability feedback, it took far smaller steps.
  * Unlike the greedy arm, B stayed far from the boundary, so the pool-to-held-out gap (see the gap diagnostic)
    did not push it over.
* **Allocation, B vs D.** Attempts 1–2 were identical trials:
  * the accept cost the same (2,500 questions)
  * threshold-aware rejected the same STEM +3.3 trial with 3,061 questions against round-robin's 4,861 (−37 %)

  After that the trajectories diverged, because the bounds reported after that rejection differed. The totals
  (37,625 vs 41,322 questions, 451 vs 491 s) therefore mix allocation with different proposals. Rejects averaged
  3,132 questions for B against 4,574 for D.

```powershell
winget install --id Ollama.Ollama -e                    # done once, with the user's approval
ollama pull qwen2.5:7b-instruct
python -m capprune supervise        --config configs/live_gate_local.yaml --dry-run
python -m capprune controller-check --config configs/live_gate_local.yaml --method llm_cap --prompts-from ~/cap-prune-runs/live-gate/llm_cap__t10__s0/caches/controller.sqlite --iteration 2 --out ~/cap-prune-runs/live-gate-ollama-checks
python -m capprune supervise        --config configs/live_gate_local.yaml
ollama stop qwen2.5:7b-instruct                         # unload before benchmarking; `ollama ps` should be empty
python -m capprune live-bench       --config configs/live_gate_local.yaml
python -m capprune live-report      --config configs/live_gate_local.yaml   # -> ~/cap-prune-runs/live-gate-ollama/_live/report.md
```
