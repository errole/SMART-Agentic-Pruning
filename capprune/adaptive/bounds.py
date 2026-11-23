"""Exact finite-pool bounds on a domain's paired accuracy loss, and error spending.

What is being bounded
---------------------
A domain's *search pool* holds N benchmark items.  For one candidate every item
has a paired outcome  x = dense_correct - candidate_correct  in {+1, 0, -1}:

    K+ = number of items with x = +1   (pruning broke a correct answer)
    K- = number of items with x = -1   (pruning fixed a wrong answer)
    L  = (K+ - K-) / N                 (the domain's full-pool accuracy loss)

Items are revealed in a seeded uniformly random order, so the first n revealed
items are a simple random sample *without replacement* from the pool and the
observed count k+ follows Hypergeometric(N, K+, n) (likewise k-).

One-sided count bounds by inverting hypergeometric tails
--------------------------------------------------------
    upper_count(k) = max{K : P_K(X <= k) > delta}
    lower_count(k) = min{K : P_K(X >= k) > delta} = N - upper_count(n - k)

``P_K(X <= k)`` is non-increasing in K, so the maximum is found by bisection.
Each bound misses the true K with probability <= delta, exactly (no normal
approximation).  At n = N they collapse to the observed count.

Loss bounds (union bound over the two counts)
---------------------------------------------
    hi = (upper_count(k+) - lower_count(k-)) / N      each count bound at delta_hi / 2
    lo = (lower_count(k+) - upper_count(k-)) / N      each count bound at delta_lo / 2

so P(hi < L) <= delta_hi and P(lo > L) <= delta_lo.  Treating the two counts
separately is conservative (the half-widths add instead of combining in
quadrature); an exact trinomial inversion would be tighter.

Error spending across looks
---------------------------
A domain is examined only at pre-planned sample sizes n_1 < n_2 < ... < n_J
(the initial minimum sample, then fixed-size batches).  A domain's error
budget alpha_d is split over its J planned looks; see :func:`spend`.  A union
bound over every (domain, look) pair makes all bounds hold *simultaneously*,
so they remain valid under any rule for choosing which domain to sample next
and when to stop.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

_LOGFACT = np.zeros(1)


def _logfact(n_max: int) -> np.ndarray:
    """log(i!) for i = 0..n_max (table grown on demand)."""
    global _LOGFACT
    if len(_LOGFACT) <= n_max:
        size = max(n_max + 1, 2 * len(_LOGFACT))
        _LOGFACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, size, dtype=np.float64)))])
    return _LOGFACT


def hypergeom_cdf(k: int, N: int, K: int, n: int) -> float:
    """P(X <= k) for X ~ Hypergeometric(population N, K successes, n draws without replacement)."""
    x_min, x_max = max(0, n - (N - K)), min(n, K)
    if k < x_min:
        return 0.0
    if k >= x_max:
        return 1.0
    lf = _logfact(N)
    x = np.arange(x_min, k + 1)
    logp = (lf[K] - lf[x] - lf[K - x]) + (lf[N - K] - lf[n - x] - lf[N - K - n + x]) - (lf[N] - lf[n] - lf[N - n])
    return float(np.exp(logp).sum())


@lru_cache(maxsize=None)
def upper_count(k: int, n: int, N: int, delta: float) -> int:
    """Largest K with P_K(X <= k) > delta: a (1 - delta) upper confidence bound on K."""
    _check(k, n, N, delta)
    lo, hi = k, k + (N - n)  # P_k(X <= k) = 1 > delta, so the answer is >= k
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom_cdf(k, N, mid, n) > delta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def lower_count(k: int, n: int, N: int, delta: float) -> int:
    """Smallest K with P_K(X >= k) > delta: a (1 - delta) lower confidence bound on K.

    By symmetry (count the zeros instead of the ones) this is N - upper_count(n - k).
    """
    _check(k, n, N, delta)
    return N - upper_count(n - k, n, N, delta)


def _check(k: int, n: int, N: int, delta: float) -> None:
    if not (0 <= k <= n <= N and N >= 1):
        raise ValueError(f"need 0 <= k <= n <= N, got k={k} n={n} N={N}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def loss_bound_counts(N: int, n: int, k_plus: int, k_minus: int, delta_hi: float, delta_lo: float) -> tuple[int, int]:
    """(lo, hi) bounds on K+ - K- (multiply by 1/N for the loss).

    P(hi < K+ - K-) <= delta_hi and P(lo > K+ - K-) <= delta_lo.
    """
    hi = upper_count(k_plus, n, N, delta_hi / 2) - lower_count(k_minus, n, N, delta_hi / 2)
    lo = lower_count(k_plus, n, N, delta_lo / 2) - upper_count(k_minus, n, N, delta_lo / 2)
    return lo, hi


# ---------------------------------------------------------------------------- error spending
def spend(look_sizes: list[int], alpha: float, scheme: str = "uniform", rho: float = 1.0) -> list[float]:
    """Split one domain's error budget ``alpha`` over its planned looks.

    uniform : alpha / J at every look (Bonferroni over the J planned looks).
    power   : Lan-DeMets power family on the information fraction t_j = n_j / n_J:
              alpha * (t_j**rho - t_{j-1}**rho).  rho > 1 keeps more error for late looks
              (conservative early, O'Brien-Fleming-like); rho < 1 spends more early.
    The returned amounts are positive and sum to alpha (up to rounding).
    """
    J = len(look_sizes)
    if J == 0:
        return []
    if scheme == "uniform":
        return [alpha / J] * J
    if scheme == "power":
        n_max = look_sizes[-1]
        cum = [(n / n_max) ** rho for n in look_sizes]
        out = [alpha * (c - p) for c, p in zip(cum, [0.0] + cum[:-1])]
        if min(out) <= 0:
            raise ValueError("power spending produced a non-positive amount; look sizes must increase")
        return out
    raise ValueError(f"unknown spending scheme {scheme!r} (uniform | power)")


def logsumexp(a: np.ndarray) -> float:
    if a.size == 0:
        return -math.inf
    m = float(np.max(a))
    if m == -math.inf:
        return -math.inf
    return m + math.log(float(np.exp(a - m).sum()))
