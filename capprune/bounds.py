"""Confirmation bounds for the per-category paired loss: method development (docs/bound_development_plan.md).

CPU only. Every candidate is a level-alpha test of H0: Delta >= tol, where Delta = p_lost - p_gained is the paired loss
of a checkpoint against the dense model in one category (lost: dense right, checkpoint wrong; gained: the reverse).
"Certify" means rejecting H0, i.e. the one-sided (1 - alpha) upper bound on Delta is below the limit. The bound is the
inversion of the test; only the test is needed for certification.

* ``cp_union``: the pre-registered reference, cp_upper(lost, n, alpha/2) - cp_lower(gained, n, alpha/2) <= tol.
* ``exact_score``: an exact unconditional test on the trinomial (lost, agree, gained): the restricted-MLE score
  statistic for the paired difference (Nam 1997; Tango 1998), with the critical value taken as the supremum of the
  exact null distribution over the nuisance parameter on the boundary Delta = tol (Sidik 2003).
* ``betting``: a one-sided betting confidence sequence for a bounded mean (Waudby-Smith & Ramdas 2024), on the paired
  differences X = lost - gained in [-1, 1], predictable plug-in bets, certified if the capital ever reaches 1/alpha.
* ``normal``: the Wald bound d + z sqrt(v / n), for reference only (not finite-sample valid; cannot be adopted).
"""
from __future__ import annotations

import math
import zlib
from functools import lru_cache
from statistics import NormalDist

import numpy as np

from .population_regate import population_bounds

METHODS = ("cp_union", "exact_score", "betting", "normal")
ADOPTABLE = ("exact_score", "betting")  # candidates; cp_union is the reference, normal is shown for reference only


# ============================================================================ helpers
@lru_cache(maxsize=8)
def _log_factorials(n: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))])


def trinomial_logpmf(n: int, kp: np.ndarray, km: np.ndarray, pl: float, pg: float) -> np.ndarray:
    """log P(lost = kp, gained = km) for n items with P(lost) = pl and P(gained) = pg (arrays broadcast)."""
    lf = _log_factorials(n)
    kp = np.asarray(kp)
    km = np.asarray(km)
    ka = n - kp - km
    out = lf[n] - lf[kp] - lf[km] - lf[np.clip(ka, 0, n)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = out + np.where(kp > 0, kp * math.log(pl) if pl > 0 else -np.inf, 0.0)
        out = out + np.where(km > 0, km * math.log(pg) if pg > 0 else -np.inf, 0.0)
        pa = 1.0 - pl - pg
        out = out + np.where(ka > 0, ka * math.log(pa) if pa > 0 else -np.inf, 0.0)
    return np.where(ka >= 0, out, -np.inf)


def _window(n: int, p: float, width: float = 9.0) -> tuple[int, int]:
    sd = math.sqrt(max(n * p * (1 - p), 0.0))
    return max(0, int(math.floor(n * p - width * sd - 2))), min(n, int(math.ceil(n * p + width * sd + 2)))


def _grid(n: int, pl: float, pg: float) -> tuple[np.ndarray, np.ndarray]:
    """The (lost, gained) cells carrying all but a negligible (< 1e-15) share of the trinomial mass."""
    a0, a1 = _window(n, pl)
    b0, b1 = _window(n, pg)
    kp, km = np.meshgrid(np.arange(a0, a1 + 1), np.arange(b0, b1 + 1), indexing="ij")
    keep = kp + km <= n
    return kp[keep], km[keep]


# ============================================================================ (i) the reference: CP union bound
def cp_union_upper(n: int, kp: int, km: int, alpha: float) -> float:
    return population_bounds(n, int(kp), int(km), alpha, alpha)[1]


def cp_union_certifies(n: int, kp: int, km: int, alpha: float, tol: float) -> bool:
    """Identical to ``screen_confirm.certifies`` (the pre-registered per-category decision)."""
    return cp_union_upper(n, kp, km, alpha) <= tol + 1e-12


@lru_cache(maxsize=64)
def _cp_vectors(n: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    from .population_regate import cp_lower, cp_upper

    up = np.array([cp_upper(k, n, alpha / 2) for k in range(n + 1)])
    lo = np.array([cp_lower(k, n, alpha / 2) for k in range(n + 1)])
    return up, lo


def cp_union_region(n: int, kp: np.ndarray, km: np.ndarray, alpha: float, tol: float) -> np.ndarray:
    up, lo = _cp_vectors(n, alpha)
    return up[kp] - lo[km] <= tol + 1e-12


# ============================================================================ reference only: the normal (Wald) bound
def normal_upper(n: int, kp, km, alpha: float):
    z = NormalDist().inv_cdf(1 - alpha)
    d = (np.asarray(kp) - np.asarray(km)) / n
    v = np.maximum((np.asarray(kp) + np.asarray(km)) / n - d * d, 0.0)
    return d + z * np.sqrt(v / n)


def normal_region(n: int, kp, km, alpha: float, tol: float) -> np.ndarray:
    return normal_upper(n, kp, km, alpha) <= tol + 1e-12


# ============================================================================ (ii) exact unconditional score test
def restricted_gained(n: int, kp, km, tol: float):
    """MLE of p_gained under p_lost - p_gained = tol (closed form of the quadratic score equation)."""
    kp = np.asarray(kp, dtype=np.float64)
    km = np.asarray(km, dtype=np.float64)
    d = kp + km
    b = d * (1 - tol) - 2 * tol * km - 2 * tol * (n - d)
    c = km * tol * (1 - tol)
    g = (b + np.sqrt(b * b + 8 * n * c)) / (4 * n)
    return np.clip(g, 0.0, (1 - tol) / 2)


def score_z(n: int, kp, km, tol: float):
    """(d - tol) / sqrt(var / n) with var = p_lost + p_gained - tol^2 at the restricted MLE (0 where var = 0)."""
    g = restricted_gained(n, kp, km, tol)
    var = 2 * g + tol - tol * tol
    d = (np.asarray(kp, dtype=np.float64) - np.asarray(km, dtype=np.float64)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, (d - tol) / np.sqrt(var / n), 0.0)
    return z


def nuisance_grid(tol: float, points: int = 400, extra_q: tuple[float, ...] = ()) -> np.ndarray:
    """p_gained values on the boundary p_lost - p_gained = tol: quadratic spacing over [0, (1 - tol)/2) (dense near 0),
    plus the boundary points of any listed disagreement rates q = 2 p_gained + tol."""
    top = (1 - tol) / 2 * (1 - 1e-9)
    g = top * (np.arange(points) / (points - 1)) ** 2
    extra = [(q - tol) / 2 for q in extra_q if q > tol]
    return np.unique(np.concatenate([g, extra]))


def _boundary_tables(n: int, tol: float, grid: np.ndarray):
    tabs = []
    for g in grid:
        pl, pg = g + tol, g
        kp, km = _grid(n, pl, pg)
        w = np.exp(trinomial_logpmf(n, kp, km, pl, pg))
        tabs.append((score_z(n, kp, km, tol), w))
    return tabs


def size_at(n: int, tol: float, z_crit: float, grid: np.ndarray) -> np.ndarray:
    """Exact P(Z <= z_crit) at every boundary point of ``grid``."""
    return np.array([w[z <= z_crit].sum() for z, w in _boundary_tables(n, tol, grid)])


@lru_cache(maxsize=4096)
def exact_score_critical(n: int, tol: float, alpha: float, points: int = 400) -> float:
    """The largest z (to 1e-9) with sup over the boundary grid of P(Z <= z) <= alpha. Certify iff Z_obs <= z*."""
    grid = nuisance_grid(tol, points, extra_q=(0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15))
    tabs = _boundary_tables(n, tol, grid)

    def worst(zc: float) -> float:
        return max(float(w[z <= zc].sum()) for z, w in tabs)

    lo, hi = -10.0, 0.0
    if worst(lo) > alpha:
        return -math.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if worst(mid) <= alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return lo


def exact_score_region(n: int, kp, km, alpha: float, tol: float) -> np.ndarray:
    return score_z(n, kp, km, tol) <= exact_score_critical(n, float(tol), float(alpha))


def exact_score_certifies(n: int, kp: int, km: int, alpha: float, tol: float) -> bool:
    return bool(exact_score_region(n, np.array([kp]), np.array([km]), alpha, tol)[0])


# ============================================================================ (iii) betting confidence sequence
def betting_certifies(x: np.ndarray, alpha: float, tol: float, c: float = 0.5) -> np.ndarray:
    """Rows of ``x`` are sequences of paired differences in {-1, 0, 1} (lost = +1, gained = -1). Certified (H0: mean >= tol
    rejected) iff the capital prod(1 + lambda_i (mu0 - Y_i)) reaches 1/alpha at some time, with Y = (x + 1)/2 in [0, 1],
    mu0 = (1 + tol)/2 and predictable plug-in bets lambda_i = min(sqrt(2 log(1/alpha) / (s2_{i-1} i log(1 + i))),
    c / (1 - mu0)) (Waudby-Smith & Ramdas 2024, one-sided)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = (x + 1.0) / 2.0
    mu0 = (1.0 + tol) / 2.0
    t = np.arange(1, y.shape[1] + 1, dtype=np.float64)
    mu_hat = (0.5 + np.cumsum(y, axis=1)) / (t + 1)
    s2 = (0.25 + np.cumsum((y - mu_hat) ** 2, axis=1)) / (t + 1)
    s2_prev = np.concatenate([np.full((y.shape[0], 1), 0.25), s2[:, :-1]], axis=1)
    lam = np.sqrt(2 * math.log(1 / alpha) / (s2_prev * t * np.log1p(t)))
    lam = np.minimum(lam, c / (1 - mu0))
    logk = np.cumsum(np.log1p(lam * (mu0 - y)), axis=1)
    return logk.max(axis=1) >= math.log(1 / alpha)


def category_order(qids: list[str], category: str, seed: int = 20261125) -> np.ndarray:
    """The pre-specified item order for the betting sequence: a seeded permutation of the category's qid-sorted items."""
    rng = np.random.default_rng([seed, zlib.crc32(category.encode())])
    return rng.permutation(len(qids))


# ============================================================================ one decision for any count-based method
def region(method: str, n: int, kp, km, alpha: float, tol: float) -> np.ndarray:
    if method == "cp_union":
        return cp_union_region(n, np.asarray(kp), np.asarray(km), alpha, tol)
    if method == "exact_score":
        return exact_score_region(n, kp, km, alpha, tol)
    if method == "normal":
        return normal_region(n, kp, km, alpha, tol)
    raise ValueError(f"{method} is not count-based")


def exact_certify_probability(method: str, n: int, pl: float, pg: float, alpha: float, tol: float) -> float:
    """P(certify at ``tol``) for a count-based method, by summing the trinomial over its decision region."""
    kp, km = _grid(n, pl, pg)
    w = np.exp(trinomial_logpmf(n, kp, km, pl, pg))
    return float(w[region(method, n, kp, km, alpha, tol)].sum())


def simulate_certify(method: str, n: int, pl: float, pg: float, alpha: float, tol: float, reps: int,
                     rng: np.random.Generator, chunk: int = 2000) -> float:
    """Monte Carlo P(certify at ``tol``): count-based methods draw the trinomial counts; betting draws item sequences."""
    hits = 0
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        if method == "betting":
            u = rng.random((r, n))
            x = np.where(u < pl, 1.0, np.where(u < pl + pg, -1.0, 0.0))
            hits += int(betting_certifies(x, alpha, tol).sum())
        else:
            cnt = rng.multinomial(n, [pl, pg, 1 - pl - pg], size=r)
            hits += int(region(method, n, cnt[:, 0], cnt[:, 1], alpha, tol).sum())
        done += r
    return hits / reps
