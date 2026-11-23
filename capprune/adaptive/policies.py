"""Allocation policies: which eligible domain receives the next batch.

Every policy sees only a :class:`~capprune.adaptive.evaluator.PolicyView`
(revealed counts, current bounds, the static plan).  Ties are always broken by
the fixed domain order in the settings.

round_robin
    Cycle through the domains in the fixed order, skipping domains that are
    not eligible (already settled, or their next batch does not fit the budget).

widest_interval
    The eligible domain with the widest current loss interval (hi - lo).

threshold_aware
    The eligible domain whose next batch is most likely to settle the
    checkpoint decision.  While two or more domains are open, one batch can
    settle the decision only by triggering REJECT (ACCEPT needs every open
    domain to pass), so the score is

        score_d = log P( lo_d' > tol_d )

    where lo_d' is domain d's lower loss bound at its *next* planned look
    (exact hypergeometric bound with that look's error share), and the
    probability is over the next batch's (broken, fixed) counts under the
    Dirichlet-multinomial predictive from the revealed counts with a
    Dirichlet(1/2, 1/2, 1/2) prior.  The log keeps tiny probabilities
    comparable instead of underflowing to ties.
    Tie-breaks: (1) larger (hi - tol) / (hi - lo), i.e. the tolerance sits
    lower inside the current interval, so the domain looks more likely to be
    over its limit; (2) fixed domain order.
    Rationale: with fixed per-domain question orders and fixed looks, a domain
    passes at the same look whatever the policy does, so an ACCEPT costs the
    same under every policy that only samples open domains.  Questions are
    saved by reaching a REJECT sooner, hence the focus on the lower bound.
"""
from __future__ import annotations

import math

import numpy as np

from .bounds import logsumexp, lower_count, upper_count
from .evaluator import PolicyView, within


class RoundRobin:
    name = "round_robin"

    def choose(self, view: PolicyView) -> str:
        order = [s.name for s in view.states]
        start = (order.index(view.history[-1]) + 1) % len(order) if view.history else 0
        for i in range(len(order)):
            d = order[(start + i) % len(order)]
            if d in view.eligible:
                return d
        raise AssertionError("no eligible domain")


class WidestInterval:
    name = "widest_interval"

    def choose(self, view: PolicyView) -> str:
        best, best_w = None, -math.inf
        for s in view.states:  # fixed order; strict '>' keeps the earliest domain on ties
            if s.name in view.eligible and round(s.hi - s.lo, 12) > best_w:
                best, best_w = s.name, round(s.hi - s.lo, 12)
        return best


class ThresholdAware:
    name = "threshold_aware"

    def choose(self, view: PolicyView) -> str:
        best, best_key = None, None
        for s in view.states:
            if s.name not in view.eligible:
                continue
            key = (round(log_p_reject_next(s, view), 9), round((s.hi - s.tolerance) / (s.hi - s.lo), 12))
            if best_key is None or key > best_key:  # strict '>' keeps the earliest domain on ties
                best, best_key = s.name, key
        return best


def log_p_reject_next(s, view: PolicyView) -> float:
    """log P(the domain's lower bound exceeds its tolerance after its next batch)."""
    nxt = view.plan[s.name].looks[s.look + 1]
    N, n2, b = s.pool_size, nxt.n, nxt.n - s.n
    d2 = nxt.delta_lo / 2  # each of the two count bounds behind the lower loss bound
    x = np.arange(b + 1)
    lo_plus = np.array([lower_count(s.k_plus + i, n2, N, d2) for i in x])     # non-decreasing in i
    hi_minus = np.array([upper_count(s.k_minus + j, n2, N, d2) for j in x])   # non-decreasing in j
    reject = ~within(lo_plus[:, None] - hi_minus[None, :], N, s.tolerance)
    logp = dirichlet_multinomial_grid(b, s.k_plus + 0.5, s.k_minus + 0.5, (s.n - s.k_plus - s.k_minus) + 0.5)
    return logsumexp(logp[reject & np.isfinite(logp)])


def dirichlet_multinomial_grid(b: int, a_plus: float, a_minus: float, a_zero: float) -> np.ndarray:
    """log P(x_plus = i, x_minus = j) for a batch of b items; -inf where i + j > b."""
    lg = math.lgamma
    i = np.arange(b + 1)
    lf = np.array([lg(v + 1) for v in range(b + 1)])
    gp = np.array([lg(v + a_plus) for v in range(b + 1)]) - lg(a_plus) - lf
    gm = np.array([lg(v + a_minus) for v in range(b + 1)]) - lg(a_minus) - lf
    g0 = np.array([lg(v + a_zero) for v in range(b + 1)]) - lg(a_zero) - lf
    A = a_plus + a_minus + a_zero
    const = lg(b + 1) + lg(A) - lg(b + A)
    rest = b - i[:, None] - i[None, :]
    out = np.full((b + 1, b + 1), -np.inf)
    ok = rest >= 0
    out[ok] = (const + gp[:, None] + gm[None, :] + g0[np.clip(rest, 0, b)])[ok]
    return out


POLICIES = {p.name: p for p in (RoundRobin, WidestInterval, ThresholdAware)}


def make_policy(name: str):
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name]()
