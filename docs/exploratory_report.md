# Exploratory analyses and bound development (2026-11-25)

**Everything in this document is EXPLORATORY.** It uses only the confirmatory study's already-scored confirmation
outcomes (`*.outcomes.npz`, SHA-256 checked against the confirmation records), on CPU. It does no new scoring and no
new pruning, and it changes no recorded confirmatory result ([`confirmatory_report.md`](confirmatory_report.md),
`batches/*` and `study/*` are untouched).

The bound development is **method development on seen data: it selects a method, it is not a claim.** Its plan,
[`bound_development_plan.md`](bound_development_plan.md), was committed and pushed (`231c2a6`) before any candidate
bound was computed on real data.

## 1. Bound development and the paper-B go/no-go

**Verdict: B does not proceed.** The rule is in §7 of the plan; its two conditions are reported separately.

| condition | result |
|---|---|
| 1. Simulation: the best valid bound needs ≥ 25 % fewer items per category than the CP union bound, at a 1-point true loss, q = 6–9 %, 80 % power | **holds**: 1,050 / 1,200 / 1,300 / 1,500 items, against 2,050 / 2,350 / 2,650 / 2,950 (0.49–0.51×) |
| 2. Real data: on arm (a)'s frozen orders, at full n and 3 points, it certifies ≥ 1 more checkpoint than the CP union bound in ≥ 4 of 6 model × seed pairs (ceiling 5) | **does not hold**: 0 of 6 |

Sources: `~/cap-prune-runs/bound-dev/simulation.json` and `real_data.json` (`.md` renderings alongside), both at
commit `78d994f` with a clean tree.

**Validity.** Coverage of the one-sided 95 % upper bound over the plan's 90-point grid: n ∈ {500, 1,000, 3,000},
q ∈ 4–15 %, true loss 0–3 points.

| method | worst coverage | valid |
|---|---|---|
| CP union bound (reference) | 0.9943 (exact) | yes |
| (ii) exact unconditional score test on the trinomial | 0.9502 (exact) | yes |
| (iii) betting confidence sequence | 0.9647 (simulated, 10,000 replicates) | yes |
| normal (Wald), reference only | 0.9260 (exact) | **no**: cannot be adopted |

The exact test's boundary size at the real-data sizes (n = 3,000, 3,500, 3,600; q = 4.0–15.0 %) is at most 0.0494.

**The best valid bound** is the exact score test, chosen from the simulation alone: mean n at a 1-point loss of
1,262.5, against the betting sequence's 3,662.5.

**Sample size.** At 80 % power for one category at 3 points:
* the exact test needs about half the CP union bound's items at true losses of 0, 1.0 and 1.5 points (0.48–0.53×);
* the betting sequence needs more than the CP union bound (1.05–1.84×).

**Real data** (all 18 runs, 3 points, full sets):
* **Arm (a)'s orders, where condition 2 is counted:** the exact test certifies the same checkpoint as the CP union
  bound in all six pairs. At the first checkpoint past each CP claim it misses as follows:
  * **Narrowly, in three pairs:**
    * SmolLM3 seed 0: Social Sciences +2.17 points, Z = −1.51 against the critical −1.65;
    * SmolLM3 seed 1: Social Sciences +2.33, Z = −1.21;
    * Granite seed 2: STEM +2.00 (Z = −1.59 against −1.66) and Other +2.14 (Z = −1.50).
  * **Out of reach, in three pairs.** The estimate itself exceeds 3 points:
    * SmolLM3 seed 2: Social Sciences +3.20;
    * Granite seed 0: Other +3.60;
    * Granite seed 1: Other +4.20.
* **Observed after computing.** The plan's ceiling of 5 of 6 counted only SmolLM3 seed 2 as out of reach, so the
  effective ceiling on these data was 3 of 6.
* **Arms (b) and (c):** the exact test gains a checkpoint only in Granite's arm (c), seeds 0 and 1:
  * seed 0: 62.9 → 78.6 M;
  * seed 1: 78.6 → 94.4 M.

  Condition 2 does not count these.
* **Betting** certifies one checkpoint fewer than the CP union bound in every SmolLM3 run, and the same in Granite.

**What this means for B.** In simulation the exact test halves the confirmation items needed. On these trajectories,
though, the checkpoints just past the CP claims either have estimates above 3 points or miss narrowly. So the tighter
bound would not have extended any arm-(a) claim.

## 2. Certification frontier

The pre-registered procedure at limits of 2–5 points and 500 / 1,000 / 2,000 / full items per category. The
subsamples are the amendment-3 S4 draws: 50 seeded, category-stratified per size.

The full table is in `~/cap-prune-runs/exploratory/frontier.json`, rendered in `exploratory.md` and
[`paper/workshop/tables/frontier.md`](paper/workshop/tables/frontier.md); the figure is
`paper/workshop/figures/certification_frontier.png`.

Median certified parameters removed, % of the dense model, arm (b); arm (a) is identical:

| model | limit | 500 | 1,000 | 2,000 | full |
|---|---|---|---|---|---|
| SmolLM3-3B | 2 points | 0 | 0 | 0.61 | 0.61 |
| SmolLM3-3B | 3 points | 0 | 0.61 | 1.23 | 1.84 |
| SmolLM3-3B | 5 points | 0.61 | 1.84 | 2.46 | 3.07 |
| Granite-3.3-2B | 2 points | 0 | 0 | 0.62 | 1.24 |
| Granite-3.3-2B | 3 points | 0 | 1.24 | 1.86 | 1.86 |
| Granite-3.3-2B | 5 points | 1.24 | 1.86 | 2.48 | 2.48 |

* Granite's arm (c) reaches 2.48 % at 3 points and 4.35 % at 5 points, at full n.
* **Nothing is certifiable at 500 items per category at 2–3 points.**
* **Check:** the computation reproduces every recorded 3-point claim, S3 (5 points, full n) and S4 (3 points,
  subsamples) before writing.

## 3. Loss curves

Per category, the figures show the paired loss and one-sided 95 % bounds at every frozen checkpoint, for every arm
and seed, with each arm's screen stop and confirmation stop: `~/cap-prune-runs/exploratory/loss_curves.json`, and the
figures `paper/workshop/figures/loss_curves_smollm3-3b.png` and `loss_curves_granite-3.3-2b.png`.

What ends confirmation:
* **SmolLM3:** Social Sciences in every run (plus Other in seed 2).
* **Granite:** Other in every run, with STEM and Social Sciences in some.

## 4. C1 mechanism

`~/cap-prune-runs/exploratory/c1_mechanism.json`. **Two models × four categories are far too few for inference.**

| model | category | gate pool | mean gap, tested (points) | share of gate decisions using the whole pool | item uses per pool item |
|---|---|---|---|---|---|
| SmolLM3 | STEM | 4,231 | −0.39 | 4 % | 33.5 |
| SmolLM3 | Humanities | 3,574 | +0.14 | 0 % | 24.1 |
| SmolLM3 | Social Sciences | 954 | +0.76 | 39 % | 47.7 |
| SmolLM3 | Other | 2,373 | −0.28 | 11 % | 33.2 |
| Granite | STEM | 4,231 | +0.14 | 0 % | 16.6 |
| Granite | Humanities | 3,574 | −0.71 | 0 % | 19.2 |
| Granite | Social Sciences | 954 | −0.10 | 34 % | 42.7 |
| Granite | Other | 2,373 | +0.80 | 12 % | 23.5 |

Social Sciences has the smallest pool and the most reuse in both models. Its gap is positive in SmolLM3 and slightly
negative in Granite, so these data show no consistent relation between gap and pool size or reuse.

## Commands

```bash
python -m capprune exploratory --config configs/confirmatory.yaml           # frontier, loss curves, C1 mechanism
python -m capprune bound-dev simulate --config configs/confirmatory.yaml    # data-free: validity, sizes, condition 1
python -m capprune bound-dev apply --config configs/confirmatory.yaml       # the scored outcomes: condition 2, verdict
python -m capprune paper-a --config configs/confirmatory.yaml --out docs/paper/workshop
```

All four ran on CPU (`CUDA_VISIBLE_DEVICES=-1`).

**Simulation provenance.** The simulation was run twice:
* first at `d62e412`, recorded with an unclean tree because unrelated files were being edited while it ran;
* again at `78d994f` with a clean tree.

The two results are identical; the second is the record.
