# Matched-compression signal test (2026-11-23, exploratory)

> **Follow-up:** the same test on the ranker's 13 ordinary parents, and a parent-level comparison with these 3 stall
> parents, is in [`matched_signals_ordinary.md`](matched_signals_ordinary.md). On ordinary parents, KL does not beat
> gradient rank 1 where the gate accepts it. The KL advantage reported here comes from the stall parents.

**Question.** At matched compression and with the same number of questions, does KL(dense ‖ pruned) pick actions that
keep held-out flip loss lower than the gate's net-flip statistic does? If so, from which sample size?

**Answer (descriptive; 3 parents × 11 actions).**
* **Yes, at every sample size tested, and therefore from the smallest one (100 items per category).** KL's top-1
  choice has a pooled mean regret of **0.27–0.36 points**. The net-flip statistic's is **0.56–0.88 points**.
  * Per draw, KL is better in 42–63 % of draws, ties in 30–57 %, and is worse in at most 7 %.
* **Both beat the references.** A random equal-size action has an expected regret of 1.56 points. The gradient's own
  rank 1 has 2.13 points.
* **The advantage is in the choice, not in the ranking or the pass/fail calls.**
  * At n ≥ 300, net flip ranks the 11 actions better than KL does (pooled Spearman 0.58–0.61 vs 0.49–0.50).
  * KL's pass/fail calls at 3 points, mapped to the net-flip scale, are no better than the majority-class rate.

**Main caveat.** The target, half-2 net flip loss per category, has a median standard error of 1.31 points. That is
larger than the 0.3–0.5-point regret gaps, so the "best action" is itself partly noise. The gaps are averaged over only 3
parents, and the 20 draws per size overlap heavily (400 of 565–891 half-1 items).

* Code: [`capprune/matched_signals.py`](../capprune/matched_signals.py) · tests
  [`tests/test_matched_signals.py`](../tests/test_matched_signals.py)
* Plan: [`matched_signal_plan.md`](matched_signal_plan.md), committed in `5077db4` before any scoring.
  `matched-signals register` recorded the plan's SHA-256 (`49685fb4…`, LF) and the split's (`26afbba9…`) before
  scoring; score and analyze refuse to run if either changes.
* Preceded by the screen-then-confirm protocol ([`screen_confirm_protocol.md`](screen_confirm_protocol.md), commit
  `15171d3`).
* Outputs: `~/cap-prune-runs/matched-signals/`. It holds `registration.json`, `split.json`, `manifest.json` (provenance
  and verification), `logprobs.npz`, `scores.sqlite` (answer cache), `analysis.json` (every draw), `report.md` (all
  tables), `score.log` and `analyze.log`.

## Design

* **Candidates.** The stall parents of the three live runs, each with its full 11-action shortlist of equal-size actions
  (1,280 MLP channels each). That gives 33 candidates at matched compression within each parent.
  * A: attempt 16, parent `e65e9461`, 11.16 % MLP sparsity.
  * B: attempt 21, parent `4ac58c5a`, 12.65 %.
  * C: attempt 10, parent `05e68e07`, 5.21 %.
* **Verification, before any scoring.** Each candidate mask was rebuilt from three things:
  * the frozen parent, whose hash matches the journal's `parent_mask_hash`;
  * the action;
  * the stalled attempt's persisted profile.

  Then:
  * Every rebuilt mask is in the stall diagnostic's answer cache at the diagnostic's order seed.
  * The live-gated actions also equal the journals' `trial_mask_hash`:
    * A: rank 1 (attempt 16);
    * B: ranks 1 and 2 (attempts 20 and 21);
    * C: ranks 1 and 2 (attempts 9 and 10).
  * The parent weights were restored bit-exactly after each candidate.
* **Split.** The 5,214-item gate pool, split within each category: 2,608 items in half 1 and 2,606 in half 2. Half-1
  sizes: Humanities 891, Other 571, STEM 581, Social Sciences 565.
* **Scoring.** Dense and all 33 candidates were scored on the whole pool in the pool-signals layout:
  * identical batches per category: seeded order, 100 per batch, a 32,768-token budget, 0-shot, masked bf16;
  * the full-vocabulary log-probabilities of the four letters were saved.
* **Reuse.** The dense model and A's rank 2 (`9be4536b…`) were reused from pool-signals (10,428 answers). The
  provenance matched: pool hash, protocol, a clean commit `d9c0ff6`, and scoring code unchanged since. A re-score of
  the first dense batch of each category reproduced the cache exactly (max |Δ log p| = 0).
* **Cost.** 166,848 item-model evaluations, 1,339 GPU-seconds.
* **Signals** on half-1 subsamples of n ∈ {100, 200, 300, 400} per category, 20 seeded draws each:
  * S0: net flip loss.
  * S2: KL over A–D.
  * S1c: Δ log p(correct) over A–D, on the items the dense model answers correctly.
* **Target.** Net flip loss per category on all of half 2. The choice uses the worst category.

## Results

### Regret of each signal's top-1 choice

Regret is in points of half-2 worst-category net flip loss above the parent's best action; the oracle's is 0.

| n per category | S0 net flip | S2 KL | S1c | gradient rank 1 | random (expected) | KL better / tie / worse than S0 |
|---|---|---|---|---|---|---|
| 100 | 0.88 | **0.36** | 0.37 | 2.13 | 1.56 | 63 / 30 / 7 % |
| 200 | 0.65 | **0.30** | 0.39 | 2.13 | 1.56 | 57 / 40 / 3 % |
| 300 | 0.56 | **0.27** | 0.37 | 2.13 | 1.56 | 42 / 57 / 2 % |
| 400 | 0.64 | **0.27** | 0.38 | 2.13 | 1.56 | 53 / 47 / 0 % |
| all of half 1 (supplementary) | 0.72 | 0.27 | 0.27 | 2.13 | 1.56 | |

Per parent, KL's regret minus S0's, at n = 100 / 200 / 300 / 400 (points):

| parent | KL − S0 regret | note |
|---|---|---|
| A | −0.66 / −0.12 / −0.14 / 0.00 | By n = 400 both pick layers 24–27, the half-2 best (+2.70). |
| B | −0.71 / −0.84 / −0.66 / −0.78 | KL picks layers 24–27 (+2.70, the best). S0 often prefers a random action that looks better on half 1. |
| C | −0.18 / −0.10 / −0.06 / −0.34 | Every action passes, with losses of −0.35 to +2.58 points. KL's regret (about 0.8) is close to gradient rank 1 (0.91). |

### Ranking: Spearman correlation with half-2 net flip, per category, pooled

| n | S0 | KL | S1c |
|---|---|---|---|
| 100 | +0.41 | +0.48 | **+0.51** |
| 200 | +0.50 | +0.50 | **+0.56** |
| 300 | **+0.58** | +0.50 | +0.56 |
| 400 | **+0.61** | +0.49 | +0.57 |

* KL's ranking does not improve with n beyond 100.
* KL's errors are systematic, not noise. It over-penalises actions in layers 4–11 and the random actions that touch
  them. In every parent these are among the highest-KL actions but have only moderate held-out flip loss (for example C rank 8: KL 0.20,
  half-2 loss −0.35).
* KL is reliable at the low end, which is where the choice is made: the 24–27 / 20–23 blocks.

### Pass/fail call on half 2 at 3 points

Pooled truth pass rate is 42 % (majority-class accuracy 58 %). Per parent:
* A: 18 % pass (majority 82 %);
* B: 9 % (91 %);
* C: 100 % (100 %).

S2 and S1c are mapped to the net-flip scale by a line fitted on the other two parents.

| n | S0 | KL (mapped) | S1c (mapped) |
|---|---|---|---|
| 100 | 75 % | 79 % | 82 % |
| 200 | 77 % | 76 % | 84 % |
| 300 | 80 % | 77 % | 83 % |
| 400 | 83 % | 77 % | 85 % |

Per parent, no signal beats the majority-class rate except S0 for A at n ≥ 300 (85–89 % vs 82 %).
* The mapped KL is worse than "always fail" for A and B (68–78 %).
* The mapped KL is worse than "always pass" for C (84–87 %).
* The line fitted on two parents does not transfer to the third.

## Interpretation

* **For choosing among equal-size actions, KL is the better measurement at every question budget tried.** It roughly
  halves the net-flip statistic's regret, with almost no draws where it does worse.
  * The reason is variance. At 100–400 items per category, net flip on 1,280-channel steps is a difference of two
    small counts. KL uses every item's whole answer distribution.
  * S1c behaves like KL for the choice and ranks slightly better than KL. For pass/fail calls it is the best of the
    three signals pooled, but still does not reliably beat the majority class per parent.
* **KL is not a substitute for the flip statistic as a pass/fail test.** Its relation to flip loss differs between
  parents and between layer groups, so a KL threshold is not a 3-point test. This fits the screen-then-confirm split:
  * use KL, or S1c, to choose the step;
  * keep a flip-based test, certified on fresh items, to decide whether to stop.
* **The gradient's top-1 action is the worst reference.** Its regret (2.13 points) is above a random equal-size
  action's (1.56). This repeats the stall diagnostics' finding that the gradient ranking does not order these actions
  by capability loss.

## Limits

* Exploratory and descriptive:
  * 3 parents;
  * the half-1 subsamples and the half-2 target come from the same 5,214-item pool used during pruning (no audit sets);
  * there are no confidence intervals across parents.
* The target noise (median 1.31-point standard error per category) is larger than the regret gaps.
* The draws are not independent across n.

## Provenance notes

* **Verification at the first launch.** The first `score` launch stopped at verification, before scoring anything; its
  log is `score_attempt1_verification_failed.log`.
  * The cause was a lookup error: the check searched only the diagnostic's `pairs.sqlite`. The diagnostic had
    replayed the live-gated rank (A 1, B 2, C 2) from the run's `caches/eval.sqlite`, so that rank's answers were
    only there.
  * The fix (commit `6c8d633`) looks in the cache the diagnostic actually read. Both key counts are recorded per
    candidate in `manifest.json`. The plan was not changed.
* **`clean_tree: false` in the scoring manifest.** During scoring, edits were uncommitted to `README.md`,
  `configs/results_registry.yaml` and `capprune/results_index.py`. These are index files, not scoring code. The
  scoring code was committed at `6c8d633`. The analysis ran from a clean tree (`35db4e6`).

## Commands

```powershell
python -m capprune matched-signals register --config configs/matched_signals.yaml   # split + plan hashes
python -m capprune matched-signals score    --config configs/matched_signals.yaml   # GPU, ~23 min
python -m capprune matched-signals analyze  --config configs/matched_signals.yaml   # CPU, seconds
```
