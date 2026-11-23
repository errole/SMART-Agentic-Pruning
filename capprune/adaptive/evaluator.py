"""Sequential evaluation of one pruned candidate against the dense checkpoint.

Flow (identical for every allocation policy)
--------------------------------------------
1. Plan: for every domain d, the planned look sizes are the minimum sample m_d,
   then m_d + b_d, m_d + 2 b_d, ... (b_d = predefined batch size), the last one
   capped at the pool size N_d and nothing beyond what the budget could ever
   allow.  Each look gets its share of the error budget (``bounds.spend``).
2. Initial sample: the first m_d items of every domain's fixed question order.
3. Loop: compute every domain's loss bounds at its current look and apply the
   shared decision rule :func:`decide`:
       REJECT  if some domain's lower bound exceeds its tolerance
       ACCEPT  if every domain's upper bound is at or below its tolerance
   otherwise the *policy* picks one eligible domain (status open, next batch
   fits the remaining budget) and that domain's next batch is revealed.
   No eligible domain left -> INCONCLUSIVE.

Only step 3's choice of domain differs between policies.  The policy receives
a :class:`PolicyView` built from revealed results only; it never holds a
reference to the item source.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .bounds import loss_bound_counts, spend

ACCEPT, REJECT, INCONCLUSIVE = "accept", "reject", "inconclusive"
PASS, FAIL, OPEN = "pass", "fail", "open"  # per-domain status at the current look
_EPS = 1e-9


def within(count_diff: int | float, N: int, tolerance: float) -> bool:
    """(count_diff / N) <= tolerance, robust to float rounding of tolerance * N."""
    return count_diff <= tolerance * N + _EPS


# ---------------------------------------------------------------------------- settings and plan
@dataclass(frozen=True)
class Settings:
    domains: tuple[str, ...]          # fixed order: round-robin order and tie-break order
    tolerance: dict[str, float]       # max tolerated accuracy loss per domain (0.03 = 3 points)
    min_sample: dict[str, int]        # questions per domain before any decision
    batch_size: dict[str, int]        # questions per later look of that domain
    budget: int                       # max questions per candidate (all domains together)
    alpha_false_accept: float         # error budget of all upper bounds (controls false accepts)
    alpha_false_reject: float         # error budget of all lower bounds (controls false rejects)
    spending: str = "uniform"         # uniform | power  (see bounds.spend)
    spending_rho: float = 1.0
    seed: int = 0                     # seeds every domain's fixed question order

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        domains = tuple(d["domains"])

        def per_domain(key, typ):
            v = d[key]
            if isinstance(v, dict):
                missing = set(domains) - set(v)
                if missing:
                    raise ValueError(f"{key}: no value for domains {sorted(missing)}")
                return {k: typ(v[k]) for k in domains}
            return {k: typ(v) for k in domains}

        return cls(domains=domains, tolerance=per_domain("tolerance", float),
                   min_sample=per_domain("min_sample", int), batch_size=per_domain("batch_size", int),
                   budget=int(d["budget"]), alpha_false_accept=float(d["alpha_false_accept"]),
                   alpha_false_reject=float(d["alpha_false_reject"]), spending=str(d.get("spending", "uniform")),
                   spending_rho=float(d.get("spending_rho", 1.0)), seed=int(d.get("seed", 0)))

    def to_dict(self) -> dict:
        return {"domains": list(self.domains), "tolerance": self.tolerance, "min_sample": self.min_sample,
                "batch_size": self.batch_size, "budget": self.budget,
                "alpha_false_accept": self.alpha_false_accept, "alpha_false_reject": self.alpha_false_reject,
                "spending": self.spending, "spending_rho": self.spending_rho, "seed": self.seed}


@dataclass(frozen=True)
class Look:
    n: int            # domain sample size at this look
    delta_hi: float   # error allotted to the upper loss bound at this look
    delta_lo: float   # error allotted to the lower loss bound at this look


@dataclass(frozen=True)
class DomainPlan:
    name: str
    pool_size: int
    tolerance: float
    looks: tuple[Look, ...]


@dataclass(frozen=True)
class Plan:
    """Static, data-free schedule shared by every policy and candidate."""
    domains: tuple[DomainPlan, ...]
    budget: int

    def __getitem__(self, name: str) -> DomainPlan:
        for dp in self.domains:
            if dp.name == name:
                return dp
        raise KeyError(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dp.name for dp in self.domains)


def look_sizes(N: int, m: int, b: int, n_cap: int) -> list[int]:
    """m, m+b, m+2b, ... ; the final batch is shortened to end exactly at the pool size N;
    sizes above n_cap (unreachable within the budget) are not planned."""
    sizes, n = [], m
    while n <= n_cap:
        sizes.append(n)
        if n >= N:
            break
        n = min(n + b, N)
    return sizes


def make_plan(s: Settings, pool_sizes: dict[str, int]) -> Plan:
    if not 0 < s.alpha_false_accept < 1 or not 0 < s.alpha_false_reject < 1:
        raise ValueError("alpha_false_accept and alpha_false_reject must be in (0, 1)")
    if set(pool_sizes) != set(s.domains):
        raise ValueError(f"cache domains {sorted(pool_sizes)} != configured domains {sorted(s.domains)}")
    total_min = sum(s.min_sample.values())
    if total_min > s.budget:
        raise ValueError(f"budget {s.budget} is smaller than the minimum sample over all domains ({total_min})")
    D = len(s.domains)
    plans = []
    for d in s.domains:
        N, m, b = int(pool_sizes[d]), s.min_sample[d], s.batch_size[d]
        if not 1 <= m <= N:
            raise ValueError(f"min_sample for {d} must be in [1, {N}], got {m}")
        if b < 1:
            raise ValueError(f"batch_size for {d} must be >= 1")
        n_cap = min(N, s.budget - (total_min - m))  # the most this domain could ever receive
        sizes = look_sizes(N, m, b, n_cap)
        a_hi = spend(sizes, s.alpha_false_accept / D, s.spending, s.spending_rho)
        a_lo = spend(sizes, s.alpha_false_reject / D, s.spending, s.spending_rho)
        looks = tuple(Look(n, h, l) for n, h, l in zip(sizes, a_hi, a_lo))
        plans.append(DomainPlan(d, N, s.tolerance[d], looks))
    return Plan(tuple(plans), s.budget)


def question_order(item_ids: list[str], seed: int, domain: str) -> list[str]:
    """The fixed random order in which a domain's items are revealed (same for every policy)."""
    ids = sorted(item_ids)
    rng = np.random.default_rng([int(seed), zlib.crc32(domain.encode("utf-8"))])
    return [ids[i] for i in rng.permutation(len(ids))]


# ---------------------------------------------------------------------------- state and decision rule
@dataclass(frozen=True)
class DomainState:
    """Everything a policy may know about one domain: computed from revealed items only."""
    name: str
    pool_size: int
    tolerance: float
    n: int              # items revealed so far
    k_plus: int         # revealed items where pruning broke a correct answer
    k_minus: int        # revealed items where pruning fixed a wrong answer
    look: int           # index of the current look in the plan (0 = initial minimum sample)
    lo: float           # lower bound on the full-pool loss
    hi: float           # upper bound on the full-pool loss
    status: str         # pass (hi <= tol) | fail (lo > tol) | open
    next_batch: int     # size of this domain's next planned batch (0 = no further look planned)

    @property
    def estimate(self) -> float:
        return (self.k_plus - self.k_minus) / self.n


def domain_state(dp: DomainPlan, n: int, k_plus: int, k_minus: int, look: int) -> DomainState:
    L = dp.looks[look]
    if L.n != n:
        raise AssertionError(f"{dp.name}: look {look} plans n={L.n} but {n} items were revealed")
    N = dp.pool_size
    lo_c, hi_c = loss_bound_counts(N, n, k_plus, k_minus, L.delta_hi, L.delta_lo)
    if not within(lo_c, N, dp.tolerance):
        status = FAIL
    elif within(hi_c, N, dp.tolerance):
        status = PASS
    else:
        status = OPEN
    nxt = dp.looks[look + 1].n - n if look + 1 < len(dp.looks) else 0
    return DomainState(dp.name, N, dp.tolerance, n, k_plus, k_minus, look, lo_c / N, hi_c / N, status, nxt)


def decide(states: tuple[DomainState, ...]) -> str | None:
    """THE decision rule, shared by every allocation policy.

    REJECT if any domain's lower bound exceeds its tolerance; ACCEPT if every
    domain's upper bound is at or below its tolerance; otherwise keep testing.
    (lo <= hi always, so the two cannot hold at once.)
    """
    if any(s.status == FAIL for s in states):
        return REJECT
    if all(s.status == PASS for s in states):
        return ACCEPT
    return None


@dataclass(frozen=True)
class PolicyView:
    """The only input a policy receives."""
    states: tuple[DomainState, ...]   # every domain, in the fixed domain order
    eligible: tuple[str, ...]         # open domains whose next batch fits the remaining budget
    budget_left: int
    history: tuple[str, ...]          # domains chosen at earlier steps
    plan: Plan                        # static schedule and error spending (no data)

    def state(self, name: str) -> DomainState:
        for s in self.states:
            if s.name == name:
                return s
        raise KeyError(name)


class Policy(Protocol):
    name: str

    def choose(self, view: PolicyView) -> str: ...


class ItemSource(Protocol):
    """Reveals a domain's next items in its fixed order. Items: (item_id, dense_correct, cand_correct)."""
    pool_sizes: dict[str, int]

    def reveal(self, domain: str, count: int) -> list[tuple[str, int, int]]: ...


class PolicyError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- the sequential test
@dataclass
class CandidateRun:
    candidate: str
    policy: str
    decision: str
    questions: int
    steps: int                                   # batches chosen by the policy (after the initial sample)
    final: dict[str, dict]                       # per-domain state at the end
    looks_reached: dict[str, int]                # index of the last look per domain
    revealed: dict[str, list[str]]               # item ids revealed per domain, in order
    trace: list[dict] = field(default_factory=list)
    decision_seconds: float = 0.0                # CPU time in bounds + decision rule + policy

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "revealed"} | {
            "revealed_counts": {d: len(v) for d, v in self.revealed.items()}}


def evaluate_candidate(source: ItemSource, policy: Policy, plan: Plan, candidate: str = "") -> CandidateRun:
    names = plan.names
    counts = {d: [0, 0, 0] for d in names}  # n, k_plus, k_minus
    look = {d: 0 for d in names}
    revealed: dict[str, list[str]] = {d: [] for d in names}
    used = 0

    def reveal(d: str, size: int) -> None:
        nonlocal used
        if used + size > plan.budget:
            raise PolicyError(f"revealing {size} more items would exceed the budget {plan.budget}")
        items = source.reveal(d, size)
        if len(items) != size:
            raise RuntimeError(f"source returned {len(items)} items for {d}, expected {size}")
        c = counts[d]
        for item_id, dense_ok, cand_ok in items:
            x = int(dense_ok) - int(cand_ok)
            c[0] += 1
            c[1] += x == 1
            c[2] += x == -1
            revealed[d].append(item_id)
        used += size

    for dp in plan.domains:  # identical initial minimum sample for every policy
        reveal(dp.name, dp.looks[0].n)

    trace: list[dict] = []
    history: tuple[str, ...] = ()
    spent = 0.0
    chosen = None
    while True:
        t0 = time.perf_counter()
        states = tuple(domain_state(plan[d], *counts[d], look[d]) for d in names)
        decision = decide(states)
        trace.append({"step": len(history), "chosen": chosen, "used": used,
                      "domains": {s.name: [s.n, round(s.lo, 6), round(s.hi, 6), s.status] for s in states}})
        if decision is None:
            left = plan.budget - used
            eligible = tuple(s.name for s in states if s.status == OPEN and 0 < s.next_batch <= left)
            if not eligible:
                decision = INCONCLUSIVE
            else:
                chosen = policy.choose(PolicyView(states, eligible, left, history, plan))
                if chosen not in eligible:
                    raise PolicyError(f"policy {policy.name} chose {chosen!r}, not one of {eligible}")
        spent += time.perf_counter() - t0
        if decision is not None:
            break
        reveal(chosen, next(s.next_batch for s in states if s.name == chosen))
        look[chosen] += 1
        history += (chosen,)

    final = {s.name: {"n": s.n, "k_plus": s.k_plus, "k_minus": s.k_minus, "estimate": s.estimate,
                      "lo": s.lo, "hi": s.hi, "status": s.status, "look": s.look} for s in states}
    return CandidateRun(candidate=candidate, policy=policy.name, decision=decision, questions=used,
                        steps=len(history), final=final, looks_reached=dict(look), revealed=revealed,
                        trace=trace, decision_seconds=spent)
