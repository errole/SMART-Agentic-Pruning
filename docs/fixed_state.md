# Fixed-state controller comparison: Qwen2.5-14B vs 7B, free-form vs shortlist (2026-11-22)

**Questions.** Does Qwen2.5-14B choose better pruning actions than Qwen2.5-7B? Does a shortlist fix 7B's
behaviour in the local pilot, where it chose `L1.mlp` again and again?

**Answer from this experiment: no to both, at fixed states.** The model being pruned is Qwen3-1.7B.
Code: [`capprune/fixed_state.py`](../capprune/fixed_state.py); configs:
[`configs/fixed_state.yaml`](../configs/fixed_state.yaml) and
[`configs/fixed_state_smoke.yaml`](../configs/fixed_state_smoke.yaml).
Outputs are in `~/cap-prune-runs/fixed-state/compare/`. The Gemini run and the earlier Ollama runs were only read.

## Design

**States.** The rule was fixed before any action was scored: the dense model, plus the greedy run's accepted
checkpoints nearest 1 %, 2 % and 3 % MLP sparsity.
* s1 (1.12 %) had been pruned from storage, so it was rebuilt by replaying the run's decisions with its saved
  per-iteration profiles.
* The same replay reproduces the three stored masks (2.42 %, 3.35 %, 3.72 %) bit for bit.

**Same prompt for every controller.** For each state, every controller saw the same:
* layer profiles (16 activation and 8 gradient calibration samples, seed 0)
* perplexity
* capability view: the gate's per-category estimate and bounds for that state against the dense model
* rules and action space

No previous-attempt feedback was given.

**Matched step.** Every action removes exactly 20 blocks × 64 = 1,280 MLP channels (7,864,320 parameters).
* Free-form proposals (`layer_decisions`) were validated with `parse_response` and `sanitize`, then rescaled
  to exactly 20 blocks (largest remainder, ≤ 9 blocks per unit).
* Shortlist proposals return `action_id`.

**Shortlist.** Built from each state's profiles before anything was scored, 9 actions of 4 units × 5 blocks:
* the greedy heuristic's lowest-risk set
* lowest z_sens
* lowest z_grad
* contiguous depth windows

The order is by first layer, and both models got identical text.

**Controllers.** Each was queried 3 times per (state, condition) at temperature 0.5, one model on the GPU at a
time:
* `qwen2.5:7b-instruct`: Q4_K_M, digest `845dbda0ea48`
* `qwen2.5:14b-instruct`: Q4_K_M, digest `7cdf5a0187d5`

Both ran with a 32,768-token context; the largest prompt was 2,183 tokens. The greedy reference is the
shortlist action with the lowest mean `0.5·(z_sens + z_grad)`, the existing heuristic's risk.

**Scoring.** Each distinct (state, action) pair was scored once (52 pairs):
* every shortlist action on every state, plus every distinct free-form action
* the full pruning-time pool, 0-shot, with the same chunking as the dense reference
* the threshold-aware gate (3 points, budget 5,614) replayed with one question order per state

That gives the gate decision, the questions the live gate would score, and the exact pool loss per category.
The holdout was not used.

## Results (exploratory: 4 states × 3 samples)

Mean worst-category loss on the pruning-time pool, in accuracy points (accepts / valid samples in brackets):

| condition | dense | s1 (1.12 %) | s2 (2.42 %) | s3 (3.35 %) | mean | valid | accepts | median latency |
|---|---|---|---|---|---|---|---|---|
| best shortlist action (oracle) | −0.08 | +0.24 | +2.93 | +1.51 | +1.15 | – | 4/4 | – |
| random shortlist pick (mean of 9) | +1.32 | +2.13 | +4.71 | +4.51 | +3.17 | – | 17/36 | – |
| 14B shortlist | +2.70 (3/3) | +2.70 (3/3) | +6.42 (0/3) | +3.89 (0/3) | +3.93 | 12/12 | 6/12 | 1.1 s |
| 7B shortlist | +2.70 (3/3) | +3.38 (1/3) | +4.68 (0/3) | +4.49 (0/3) | +3.81 | 12/12 | 4/12 | 0.6 s |
| 14B free | +3.32 (1/3) | +2.93 (2/3) | +5.05 (0/3) | +3.28 (0/3) | +3.64 | 12/12 | 3/12 | 2.1 s |
| 7B free | +2.86 (3/3) | +2.70 (3/3) | +4.47 (0/3) | +4.78 (0/3) | +3.70 | 12/12 | 6/12 | 1.1 s |
| greedy on the shortlist | +0.95 (1/1) | +3.73 (0/1) | +3.57 (0/1) | +7.61 (0/1) | +3.97 | 4/4 | 1/4 | – |

**Choices.**
* 14B picked A1 (layers 0–3, "lowest z_sens") in 12 of 12 shortlist trials.
* 7B picked A1 in 7 of 12, plus the greedy set or layers 8–11.
* In free-form, both models favoured L0–L2. 7B also chose mid layers (L11–L13, L12/L13/L17).
* With the fixed step stated, neither model proposed a single unit: every free-form proposal used 3–5 units.

| question | comparison | per-state difference in mean worst loss (A − B) | verdict |
|---|---|---|---|
| Does 14B beat 7B with the same shortlist? | 14B vs 7B, shortlist | 0.00, −0.69, +1.74, −0.61 (mean +0.11) | **No.** Better on 2 states, worse on 1, tied on dense; accepts 6/12 vs 4/12 |
| Does 7B improve when given the shortlist? | 7B shortlist vs free | −0.16, +0.69, +0.21, −0.29 (mean +0.11) | **No.** Split 2/2; fewer accepts (4/12 vs 6/12) |
| Does either LLM beat greedy at equal removal? | each LLM condition vs greedy | e.g. 14B shortlist +1.74, −1.03, +2.85, −3.73 | **No consistent winner.** Every comparison splits 2/2; all are worse than a random pick on average |
| Does any gain justify the added time? | 14B vs 7B latency | 14B about 2× slower (1.1–2.1 s vs 0.6–1.1 s per decision) | **No gain to pay for.** Controller time is small either way next to 20–40 s of gate scoring per attempt |

## What the comparisons support

* **Model size.** The clean test is 14B vs 7B on the same shortlist: same options, same information, so only
  the chooser differs. It shows no model-size effect. Both models follow the prompt's framing that a negative
  z_sens is safer; 14B does so deterministically.
* **How candidates are offered.** The clean test is 7B shortlist vs 7B free-form: same model, different
  interface. It shows no improvement in action quality. What removed the single-unit fixation was the explicit
  fixed-step rule, which both conditions had.
* **Headroom.** The best shortlist action is accepted on every state and averages +1.15 points against +3.17
  for a random pick. There is a lot to gain from choosing well; no controller here captures it.
* **Post-hoc observation (a hypothesis, not a result).** The late-layer actions (layers 20–27) and especially
  "lowest z_grad" (A8, layers 22–25) passed on all four states: −0.08, +0.24, +2.93, +2.22. On this model the
  prompt's sensitivity statistic (z_sens) appears to point the wrong way for STEM. Testing whether controllers
  improve when given per-action capability estimates (for example, results of earlier trials) needs a new
  pre-registered comparison.

**Live search: not run.** You asked for it only if the fixed-state comparison showed a promising improvement,
and it did not. Everything above is fixed-state action quality on the pruning-time pool, not live-search
performance.

## Commands

```powershell
ollama pull qwen2.5:14b-instruct
python -m pytest tests/test_live_gate.py -q -k fixed_state                       # CPU smoke (mock controllers)
python -m capprune fixed-state prepare       --config configs/fixed_state.yaml   # GPU: states, profiles, prompts, shortlists
python -m capprune fixed-state record-models --config configs/fixed_state.yaml
python -m capprune fixed-state query --config configs/fixed_state.yaml --model qwen2.5:7b-instruct
ollama stop qwen2.5:7b-instruct
python -m capprune fixed-state query --config configs/fixed_state.yaml --model qwen2.5:14b-instruct
ollama stop qwen2.5:14b-instruct
python -m capprune fixed-state score         --config configs/fixed_state.yaml   # GPU: 52 state-action pairs, ~25 min
python -m capprune fixed-state report        --config configs/fixed_state.yaml
```

Outputs:
* `states.json`: states, profiles, prompts, shortlists, replay checks
* `controller_models.json` and `loaded-*.json`: digests, quantization, runtime context
* `proposals-<model>.jsonl`: every reply, outcome, latency and raw text
* `controller-<model>.sqlite`
* `scores.jsonl`: one record per state–action pair
* `summary.json` and `report.md`: per-state tables
