"""Confirmation bounds for the per-category paired loss: method development (docs/bound_development_plan.md).

CPU only. Every candidate is a level-alpha test of H0: Delta >= tol, where Delta = p_lost - p_gained is the paired loss
of a checkpoint against the dense model in one category (lost: dense right, checkpoint wrong; gained: the reverse).
"Certify" means rejecting H0, i.e. the one-sided (1 - alpha) upper bound on Delta is below the limit. The bound is the
inversion of the test; only the test is needed for certification.

* ``cp_union``: the pre-registered reference, cp_upper(lost, n, alpha/2) - cp_lower(gained, n, alpha/2) <= tol.
* ``exact_score``: an exact unconditional test on the trinomial (lost, agree, gained): the restricted-MLE score
  statistic for the paired difference (Nam 1997; Tango 1998), with the critical value taken as the supremum of the
  exact null distribution over the nuisance parameter on the boundary Delta = tol (Sidik 2003). v1 (the 400-point
  grid, used by bound-dev's recorded results) is exact only up to its grid; v2 (``exact_score_critical_v2``, the
  registered fix of 2026-11-27) refines the grid and proves the size <= alpha at every nuisance value.
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


def cp_union_region(n: int, kp: np.ndarray, km: np.ndarray, alpha: float, tol: float) -> np.ndarray:
    """The reference decision on many cells: the same cp_upper / cp_lower values as ``population_bounds`` (cached),
    computed only for the counts that occur."""
    from .population_regate import cp_lower, cp_upper

    kp, km = np.asarray(kp), np.asarray(km)
    up = {int(k): cp_upper(int(k), n, alpha / 2) for k in np.unique(kp)}
    lo = {int(k): cp_lower(int(k), n, alpha / 2) for k in np.unique(km)}
    u = np.vectorize(up.__getitem__, otypes=[float])(kp) if kp.size else np.zeros(0)
    v = np.vectorize(lo.__getitem__, otypes=[float])(km) if km.size else np.zeros(0)
    return u - v <= tol + 1e-12


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


# ============================================================================ (ii) v2: the registered grid fix
# docs/bound_development_plan.md, section 8, note of 2026-11-27. The v1 critical value above (400-point grid) is kept
# unchanged: bound-dev's recorded results were computed with it.
V2_EXTRA_Q = (0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15)
V2_PER_UNIT = 8            # uniform grid spacing 1 / (8 n) in p_gained, added to the v1 grid
V2_MARGIN = 1e-4           # z* keeps the grid supremum of the size at or below alpha - 1e-4
V2_WINDOW_SLACK = 1e-12    # bound on the trinomial mass outside the +-9 sd windows (< 1e-15), with room


def nuisance_grid_v2(n: int, tol: float, per_unit: int = V2_PER_UNIT) -> np.ndarray:
    """The v1 grid plus a uniform grid of spacing 1/(per_unit n) over [0, (1 - tol)/2), both ends included."""
    top = (1 - tol) / 2 * (1 - 1e-9)
    m = int(math.ceil(top * per_unit * n)) + 1
    return np.unique(np.concatenate([nuisance_grid(tol, 400, extra_q=V2_EXTRA_Q), np.linspace(0.0, top, m)]))


def _cells(n: int, tol: float, g: float) -> tuple[np.ndarray, np.ndarray]:
    """(Z, probability) of every retained cell on the boundary point p_gained = g."""
    pl, pg = g + tol, g
    kp, km = _grid(n, pl, pg)
    keep = kp - km < n * tol  # Z < 0 exactly when d < tol: the cells that can ever be in the rejection region
    kp, km = kp[keep], km[keep]
    return score_z(n, kp, km, tol), np.exp(trinomial_logpmf(n, kp, km, pl, pg))


def size_point(n: int, tol: float, z_crit: float, g: float) -> float:
    """Exact P(Z <= z_crit) at the boundary point p_gained = g (all but < 1e-15 of the mass)."""
    z, w = _cells(n, tol, float(g))
    return float(w[z <= z_crit].sum())


def fisher_slope_bound(n: int, tol: float, a: float, b: float) -> float:
    """An upper bound on |dS/dg| over [a, b] for S(g) = P_g(R), any fixed region R, on the boundary
    (p_lost, p_gained, p_agree) = (g + tol, g, 1 - tol - 2g). S' = Cov(U, 1_R) with U the score of g, so
    |S'| <= sqrt(n I(g) / 4), I(g) = 1/(g + tol) + 1/g + 4/(1 - tol - 2g); and |S'| <= E|U| <= 4n everywhere."""
    pa = 1 - tol - 2 * b
    if a <= 0 or pa <= 0:
        return 4.0 * n
    info = 1 / (a + tol) + 1 / a + 4 / pa  # each term is at its largest over [a, b]
    return min(4.0 * n, 0.5 * math.sqrt(n * info))


def interval_size_bound(n: int, tol: float, a: float, b: float, sa: float, sb: float) -> float:
    """An upper bound on S over [a, b] from its values at the ends, the smaller of two rigorous bounds:
    * S itself is Lipschitz with ``fisher_slope_bound``: max S <= (S(a) + S(b) + L (b - a)) / 2;
    * theta = 2 arcsin(sqrt(S)) has |theta'| = |S'| / sqrt(S (1 - S)) <= sqrt(n I(g)) (since |S'| <= sqrt(n I S (1 - S))),
      so max theta <= (theta(a) + theta(b) + sqrt(n I_max) (b - a)) / 2 and max S <= sin(max theta / 2)^2."""
    s_bound = 0.5 * (sa + sb + fisher_slope_bound(n, tol, a, b) * (b - a))
    pa = 1 - tol - 2 * b
    if a <= 0 or pa <= 0:
        return s_bound
    info = 1 / (a + tol) + 1 / a + 4 / pa
    ta, tb = 2 * math.asin(math.sqrt(min(max(sa, 0.0), 1.0))), 2 * math.asin(math.sqrt(min(max(sb, 0.0), 1.0)))
    th = 0.5 * (ta + tb + math.sqrt(n * info) * (b - a))
    t_bound = 1.0 if th >= math.pi else math.sin(th / 2) ** 2
    return min(s_bound, t_bound)


class _Band:
    """Per grid point: the mass with Z < z_lo, and the sorted Z values in [z_lo, z_hi] with cumulative mass, so the
    size at any z in [z_lo, z_hi] is one lookup."""

    def __init__(self, n: int, tol: float, grid: np.ndarray, z_lo: float, z_hi: float):
        self.z_lo, self.z_hi = z_lo, z_hi
        self.rows = [self._row(n, tol, g) for g in grid]

    def _row(self, n, tol, g):
        z, w = _cells(n, tol, float(g))
        below = float(w[z < self.z_lo].sum())
        sel = (z >= self.z_lo) & (z <= self.z_hi)
        o = np.argsort(z[sel], kind="stable")
        return below, z[sel][o], np.cumsum(w[sel][o])

    def add(self, n, tol, gs):
        self.rows += [self._row(n, tol, g) for g in gs]

    def sizes(self, zc: float) -> np.ndarray:
        out = np.empty(len(self.rows))
        for i, (below, zs, cw) in enumerate(self.rows):
            j = int(np.searchsorted(zs, zc, side="right"))
            out[i] = below + (cw[j - 1] if j else 0.0)
        return out


def exact_score_critical_v2(n: int, tol: float, alpha: float, per_unit: int = V2_PER_UNIT, margin: float = V2_MARGIN,
                            max_rounds: int = 8, max_evals: int = 2_000_000) -> dict:
    """The registered fix of candidate (ii)'s critical value, with a certificate that its size is at most alpha at
    every nuisance value, on and off the grid.

    1. Grid: ``nuisance_grid_v2`` (v1 grid + uniform spacing 1/(per_unit n)).
    2. z* is the largest z (bisection to 1e-9) with the exact size at every grid point <= alpha - margin.
    3. Certificate over the continuum [0, (1 - tol)/2): every interval between adjacent grid points is bounded by
       ``interval_size_bound`` + slack (a Lipschitz bound from the Fisher information of the boundary curve); an
       interval whose bound exceeds alpha is split at its midpoint (branch and bound). A point with S > alpha is added
       to the grid and z* recomputed.
    Returns z*, the grid size, the largest size found at any evaluated point, and the certified bound."""
    z1 = exact_score_critical(n, tol, alpha)  # more nuisance points can only lower z*: z1 bounds it from above
    z_hi, z_lo = z1 + 1e-9, z1 - 0.75
    top = (1 - tol) / 2 * (1 - 1e-9)
    grid = list(nuisance_grid_v2(n, tol, per_unit))
    band = _Band(n, tol, np.array(grid), z_lo, z_hi)
    rounds = []
    for rnd in range(max_rounds):
        lo, hi = z_lo, z_hi
        if band.sizes(lo).max() > alpha - margin:
            raise RuntimeError("z* is below the band; widen z_lo")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if band.sizes(mid).max() <= alpha - margin:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-9:
                break
        z = lo
        gs = np.array(grid)
        order = np.argsort(gs)
        gs = gs[order]
        sz = band.sizes(z)[order]
        # ---- branch and bound over every interval of the grid
        evals, worst_bound, worst_found, found_over = 0, 0.0, float(sz.max()), []
        stack = [(float(gs[i]), float(gs[i + 1]), float(sz[i]), float(sz[i + 1])) for i in range(len(gs) - 1)]
        if gs[-1] < top:
            stack.append((float(gs[-1]), top, float(sz[-1]), size_point(n, tol, z, top)))
        certified_intervals = 0
        while stack:
            a, b, sa, sb = stack.pop()
            bound = interval_size_bound(n, tol, a, b, sa, sb) + V2_WINDOW_SLACK
            if bound <= alpha:
                certified_intervals += 1
                worst_bound = max(worst_bound, bound)
                continue
            m = 0.5 * (a + b)
            if m <= a or m >= b:
                raise RuntimeError(f"interval [{a}, {b}] cannot be split further")
            sm = size_point(n, tol, z, m)
            evals += 1
            worst_found = max(worst_found, sm)
            if sm > alpha:
                found_over.append(m)
                continue
            if evals > max_evals:
                raise RuntimeError("certificate budget exhausted")
            stack += [(a, m, sa, sm), (m, b, sm, sb)]
        rounds.append({"round": rnd, "z": z, "grid_points": len(grid), "grid_max_size": float(sz.max()),
                       "bnb_evaluations": evals, "certified_intervals": certified_intervals,
                       "points_over_alpha": len(found_over), "largest_size_found": worst_found,
                       "certified_sup_bound": worst_bound})
        if not found_over:
            return {"n": n, "tol": tol, "alpha": alpha, "z": z, "z_v1": z1, "per_unit": per_unit, "margin": margin,
                    "grid_points": len(grid), "grid_max_size": float(sz.max()), "largest_size_found": worst_found,
                    "certified_sup_bound": worst_bound, "certified": True, "rounds": rounds}
        grid += found_over
        band.add(n, tol, found_over)
    raise RuntimeError(f"no certified critical value after {max_rounds} rounds")


@lru_cache(maxsize=64)
def exact_score_critical_v2_z(n: int, tol: float, alpha: float) -> float:
    return float(exact_score_critical_v2(n, tol, alpha)["z"])


def exact_score_v2_region(n: int, kp, km, alpha: float, tol: float) -> np.ndarray:
    return score_z(n, kp, km, tol) <= exact_score_critical_v2_z(n, float(tol), float(alpha))


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
