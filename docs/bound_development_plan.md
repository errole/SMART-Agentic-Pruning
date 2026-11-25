# Bound development plan (2026-11-25)

**Status: method development, EXPLORATORY.** This plan selects a confirmation bound for a possible follow-up study
(paper B). It makes no claim. It is committed and pushed **before any candidate bound below is computed on real
confirmation data.**

**What has already been seen.** The confirmatory study (PR #7, [`confirmatory_report.md`](confirmatory_report.md))
scored every frozen checkpoint of all 18 runs on the two models' sealed confirmation sets. It reported each checkpoint's
per-category estimates and Clopper–Pearson union bounds. Those per-item outcomes (`*.outcomes.npz`) are the real data
of §6, so any result on them is **method development on seen data: it selects a method, it is not a claim.**

The new candidates have not been computed on them. While the code was written, only synthetic inputs were used, to
check timing and the boundary size of the implementations at n = 500–3,000.

**Fixed, as in the confirmatory study:**
* α = 0.05, one-sided, per category;
* the intersection-union test (IUT) across the four categories: a checkpoint passes only if every category certifies;
* fixed-sequence testing over each run's frozen order, stopping at the first failure;
* the 3-point limit;
* the full confirmation sets (STEM 3,600, Humanities 3,000, Social Sciences 3,000, Other 3,500 items per model).

**Only the per-category decision changes.**

## 1. The quantity and the decision

For one checkpoint and one category with n items:
* **counts:** k⁺ items lost (dense right, checkpoint wrong), k⁻ gained (the reverse), the rest agree;
* **model:** (k⁺, k⁻, n − k⁺ − k⁻) is trinomial with probabilities (p⁺, p⁻, 1 − p⁺ − p⁻);
* **paired loss:** Δ = p⁺ − p⁻;
* **disagreement rate:** q = p⁺ + p⁻.

Every candidate is a level-α test of H0: Δ ≥ tol. **Certifying** means rejecting H0. Equivalently, the candidate's
one-sided (1 − α) upper bound on Δ is below tol. The bound is the inversion of the test, and only the test at the
limit is needed.

## 2. Candidates

All are implemented in `capprune/bounds.py` at the commit of this plan.

**(i) Reference: CP union bound (pre-registered).**
* Certify iff cp_upper(k⁺, n, α/2) − cp_lower(k⁻, n, α/2) ≤ tol. This is `screen_confirm.certifies`.
* It is the reference every candidate is compared with; it is not a candidate.

**(ii) Exact unconditional score test on the trinomial.**
* **Statistic:** Z = (d − tol) / √(ṽ / n), with d = (k⁺ − k⁻)/n and ṽ = 2g̃ + tol − tol². Here g̃ is the restricted MLE of
  p⁻ under p⁺ − p⁻ = tol, which has a closed form:
  * g̃ = (B + √(B² + 8nC)) / (4n);
  * B = D(1 − tol) − 2 tol k⁻ − 2 tol (n − D);
  * C = k⁻ tol (1 − tol);
  * D = k⁺ + k⁻.

  This is the score statistic for the paired difference (Nam 1997; Tango 1998).
* **Critical value:** z*(n, tol, α) is the largest z, found by bisection to 10⁻⁹ on [−10, 0], with
  max over the nuisance grid of P(Z ≤ z) ≤ α. The probabilities are exact trinomial sums on the boundary p⁺ − p⁻ = tol
  (exact unconditional test; Sidik 2003).
* **Nuisance grid:** 400 values of p⁻, quadratically spaced over [0, (1 − tol)/2), so the full range of q from tol to
  1, dense near 0. Added to them are the boundary points of q ∈ {4, 6, 7, 8, 9, 10, 12, 15 %}.
* **Mass ignored:** only cells outside ±9 standard deviations of each margin, under 10⁻¹⁵.
* **Certify** iff Z ≤ z*.

**(iii) Betting confidence sequence for a bounded mean** (Waudby-Smith & Ramdas 2024).
* **Data:** paired differences X = +1 (lost), −1 (gained) or 0 (agree), in [−1, 1], with Y = (X + 1)/2 in [0, 1] and
  μ₀ = (1 + tol)/2.
* **Capital:** K_t = ∏_{i≤t} (1 + λ_i (μ₀ − Y_i)).
* **Bets:** predictable plug-in, λ_i = min(√(2 log(1/α) / (σ̂²_{i−1} · i · log(1 + i))), c/(1 − μ₀)), with c = 1/2. Here
  μ̂_t = (1/2 + Σ_{i≤t} Y_i)/(t + 1), σ̂²_t = (1/4 + Σ_{i≤t} (Y_i − μ̂_i)²)/(t + 1), and σ̂²_0 = 1/4.
* **Certify** iff max_{t≤n} K_t ≥ 1/α (Ville's inequality).
* **Item order:** in each category, a seeded permutation of the qid-sorted items,
  `numpy.random.default_rng([20261125, crc32(category)])`. The same order is used for every checkpoint.

**(ref) Normal (Wald) bound, for reference only.**
* Certify iff d + z_{1−α} √((k⁺ + k⁻)/n − d²) / √n ≤ tol.
* It is not finite-sample valid. It is shown for reference and **cannot be adopted**, whatever its simulated coverage.

## 3. Validity requirement

* **Coverage** at true (p⁺, p⁻) = 1 − P(certify at tol = Δ). That is the probability that the one-sided upper bound is
  at least the true loss.
* **Grid:** 90 points.
  * n ∈ {500, 1,000, 3,000};
  * total disagreement q ∈ {4, 6, 8, 10, 12, 15 %};
  * true loss Δ ∈ {0, 1.0, 1.5, 2.0, 3.0 points};
  * p⁺ = (q + Δ)/2 and p⁻ = (q − Δ)/2;
  * Δ = 3 points is the case of a true loss exactly at the limit.
* **Count-based methods** ((i), (ii), normal):
  * the exact coverage, from summing the trinomial over the decision region, must be ≥ 0.95 at every grid point
    (to 10⁻⁹);
  * a Monte Carlo check with 10,000 seeded replicates per point is reported alongside.
* **Betting (iii)** depends on the item order, so it is simulated. Its coverage over 10,000 seeded replicates must be
  ≥ 0.95 − 3 SE, with SE = √(0.95 · 0.05 / 10,000) ≈ 0.0022, at every grid point.
* **Extra check for (ii):** its exact size P(certify at 3 points) on the boundary must be ≤ 0.05 at q = 4.0, 4.1, …,
  15.0 % for n = 3,000, 3,500 and 3,600, the real-data sizes.
* **Seeds:** `default_rng([20261125, i])` for grid point i.
* **A candidate failing any of these is not valid and cannot be adopted.**

## 4. Sample size

* **Definition:** the smallest n per category, a multiple of 50 in [50, 20,000] found by bisection, at which one
  category's test certifies 3 points with probability ≥ 0.80.
* **Conditions:** true losses of 0, 1.0 and 1.5 points; q ∈ {6, 7, 8, 9 %}.
* **Power:** exact for count-based methods. For betting, 4,000 seeded replicates per n, with
  `default_rng([20261126, n, round(1000 q), round(10000 Δ)])`.
* **Reported:** the n for every method, and the ratio to the CP union bound's n.

## 5. Selecting the best valid bound (data-free)

* **The best valid bound** is the valid candidate, among (ii) and (iii), with the smallest mean n (§4) over
  q ∈ {6, 7, 8, 9 %} at a true loss of 1.0 point.
* **Exclusions:** the normal bound cannot be adopted, and the CP union bound is the reference.
* **If no candidate is valid,** B does not proceed.

This selection uses the simulation only. It is fixed before §6 is computed.

## 6. Application to the scored confirmation data (after §3–§5)

* **Data:** for every one of the 18 runs, every checkpoint of its frozen order is scored, from its `*.outcomes.npz`.
  The per-category decision is computed with each method: (i), each valid candidate, and the normal bound (labelled
  "reference only, not valid").
* **Procedure:** the IUT and the fixed sequence at 3 points and full n give each run's certified checkpoint and
  parameters removed.
* **Reported:** per model × seed × arm, together with each method's per-category statistic at the first checkpoint
  past the CP claim.
* **Label:** everything here is "method development on seen data: it selects a method, it is not a claim".

## 7. The go/no-go rule for paper B

**The rule first written in the request is untestable.** It required the best valid bound to certify at least one more
checkpoint than the CP union bound in at least 4 of the 6 model × seed pairs on arm (b)'s orders. From PR #7's results:
* in model 2 seeds 0 and 1, arm (b)'s whole frozen order (3 and 4 checkpoints) is already certified, so there is no
  checkpoint left to certify;
* in model 1 seed 2, arm (b) fails at a checkpoint whose Social Sciences estimate is +3.20 points. No valid upper bound
  can certify a checkpoint whose estimate exceeds the limit.

At most 3 of 6 pairs could gain, so that rule was a no-go by construction. **On 2026-11-25, before any candidate was
computed on real data, the rule was replaced by the following.**

**B proceeds only if BOTH conditions hold for the same best valid bound (§5):**

1. **Simulation (data-free).** At a true loss of 1.0 point and total disagreement of 6–9 %, it needs at least 25 %
   fewer confirmation items per category than the CP union bound to certify 3 points with 80 % power.
   * Operationally: n_best ≤ 0.75 × n_CP at each of q = 6, 7, 8 and 9 % (§4).
2. **Real data.** On arm (a)'s frozen orders (the unscreened gradient trajectory), at full n and a 3-point limit, it
   certifies at least one more checkpoint than the CP union bound in at least 4 of the 6 model × seed pairs.
   * **Ceiling: 5 of 6.** Model 1 seed 2's first failing step (checkpoint 5, Social Sciences estimate +3.20) is shared
     by all three arms and cannot be certified.

**Both conditions are reported separately, whichever way the verdict goes.**

## 8. Outputs and tests

* `~/cap-prune-runs/bound-dev/simulation.json` and `simulation.md`: §3–§5, the best valid bound, and condition 1.
* `~/cap-prune-runs/bound-dev/real_data.json` and `real_data.md`: §6, condition 2, and the verdict.
* `tests/test_bounds.py` (CPU):
  * the reference reproduces `screen_confirm.certifies`;
  * the restricted MLE matches a brute-force maximization;
  * the exact test's size is ≤ α on its boundary;
  * exact and Monte Carlo certification probabilities agree;
  * the betting test's coverage at the boundary;
  * the normal bound's size exceeds α somewhere (why it cannot be adopted);
  * the go/no-go logic.
* **Defects:** if an implementation defect is found after this commit, the fix is recorded here as a dated note before
  anything is rerun.

## 9. What this is not

* This is not a claim about either model and not part of the confirmatory study.
* If B proceeds, the selected bound is registered before it is used, and confirmed prospectively on a third model with
  freshly sealed data.

## References

The bibliographic details below are written from memory. Check them before citing in a paper.

* Nam, J. (1997). Establishing equivalence of two treatments and sample size requirements in matched-pairs design.
  *Biometrics* 53, 1422–1430.
* Tango, T. (1998). Equivalence test and confidence interval for the difference in proportions for the paired-sample
  design. *Statistics in Medicine* 17, 891–908.
* Sidik, K. (2003). Exact unconditional tests for testing non-inferiority in matched-pairs design. *Statistics in
  Medicine* 22, 265–278.
* Waudby-Smith, I. and Ramdas, A. (2024). Estimating means of bounded random variables by betting. *JRSS B* 86, 1–27.
