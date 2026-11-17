"""Budgeted, uncertainty-aware domain testing (the core CAP-Prune addition).

Statistical model
-----------------
Every probe question q in domain d is answered by the dense model (once,
cached) and by the candidate.  The paired outcome is one of

    lost   = dense correct, candidate wrong
    agree  = same correctness
    gained = dense wrong, candidate correct

With a Dirichlet(prior + counts) posterior over the three outcome
probabilities, the domain's accuracy loss relative to the dense model is

    L_d = p_lost - p_gained          (cumulative loss vs. the original model)

and its full posterior is obtained by Monte-Carlo draws.  Pairing removes the
between-question variance, so far fewer questions are needed than comparing
two independent accuracies.

Domain-loss limits (per Monte-Carlo draw)
-----------------------------------------
    V_d = (L_d > eps_abs(rho))  or  (L_d - mean_d' L_d' > eps_rel  and  L_d > eps_rel_min)

``eps_abs`` follows a *loss budget schedule*: it grows linearly with sparsity
progress (eps_final * rho/target, never below ``floor``) so damage cannot be
front-loaded.  The second clause bounds *imbalance* - one domain being hurt
much more than the rest - which is exactly what an average perplexity hides.

Decision (sequential, with early stopping) - rule "per_domain" (default)
-----------------------------------------------------------------------
    every domain has P(V_d) <= accept_max          -> PASS (conclusive)
    some domain has  P(V_d) >= reject_min, n_d >= min_fail_n -> FAIL (early stop)
    otherwise keep testing; when the step budget is spent -> INCONCLUSIVE
INCONCLUSIVE retains the last accepted model (proposal: "inconclusive cases
retain the last accepted model").

PASS therefore means: for each domain separately, the (1 - accept_max) upper
credible bound of its loss is inside the limit.  FAIL needs positive evidence
against one specific domain.  The earlier family-wise rule (thresholds on
P(any V_d)) is kept as rule "familywise" for comparison only: with 8 domains
P(any V_d) is inflated by sampling noise alone.  In simulation with the real
probe-pool sizes it wrongly rejects ~22 % of harmless steps at the current
defaults (per_domain: <= 8 %), and ~96 % at the original defaults (limit floor
0.02, 800 questions per step), which were infeasible for any rule.

Allocation policies (all share the minimum coverage and stopping rule, so they
differ only in *where* questions go):
    balanced     equal split                                  (equal testing)
    loss_ranked  proportional to observed loss                (observed-loss allocation)
    neyman       proportional to posterior sd                 (statistical uncertainty policy)
    cap          sd_d * (alpha + 4 P(V_d)(1 - P(V_d)))        (uncertainty x undecidedness)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

PASS, FAIL, INCONCLUSIVE = "pass", "fail", "inconclusive"


@dataclass
class Tally:
    lost: int = 0
    agree: int = 0
    gained: int = 0

    @property
    def n(self) -> int:
        return self.lost + self.agree + self.gained

    def add(self, dense_correct: bool, cand_correct: bool) -> None:
        if dense_correct and not cand_correct:
            self.lost += 1
        elif cand_correct and not dense_correct:
            self.gained += 1
        else:
            self.agree += 1


@dataclass
class Limits:
    eps_abs: float
    eps_rel: float
    eps_rel_min: float


@dataclass
class TestResult:
    decision: str
    used: int
    rounds: int
    early_stop: bool
    p_any: float
    domains: dict[str, dict]
    tallies: dict[str, dict]
    limits: dict
    allocations: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def loss_limit(rho: float, target: float, eps_final: float, floor: float, schedule: str = "linear") -> float:
    if schedule == "constant" or target <= 0:
        return eps_final
    return max(floor, eps_final * min(1.0, rho / target))


def posterior_draws(tallies: dict[str, Tally], domains: list[str], prior: tuple[float, float, float],
                    S: int, rng: np.random.Generator) -> np.ndarray:
    """Returns an [S, D] array of draws of L_d = p_lost - p_gained."""
    a0 = np.asarray(prior, dtype=np.float64)
    out = np.empty((S, len(domains)))
    for j, d in enumerate(domains):
        t = tallies.get(d, Tally())
        p = rng.dirichlet(a0 + np.array([t.lost, t.agree, t.gained], dtype=np.float64), size=S)
        out[:, j] = p[:, 0] - p[:, 2]
    return out


def summarize(draws: np.ndarray, domains: list[str], lim: Limits) -> tuple[dict[str, dict], float]:
    mean_all = draws.mean(axis=1, keepdims=True)
    V = (draws > lim.eps_abs) | ((draws - mean_all > lim.eps_rel) & (draws > lim.eps_rel_min))
    p_any = float(V.any(axis=1).mean())
    out = {}
    for j, d in enumerate(domains):
        col = draws[:, j]
        out[d] = {
            "mean": float(col.mean()),
            "sd": float(col.std()),
            "lo90": float(np.quantile(col, 0.05)),
            "hi90": float(np.quantile(col, 0.95)),
            "p_violate": float(V[:, j].mean()),
        }
    return out, p_any


def allocate(policy: str, domains: list[str], summary: dict[str, dict], tallies: dict[str, Tally],
             available: dict[str, int], n: int, round_idx: int, alpha: float) -> dict[str, int]:
    if n <= 0:
        return {}
    live = [d for d in domains if available.get(d, 0) > 0]
    if not live:
        return {}
    if policy == "balanced":
        w = {d: 1.0 for d in live}
    elif policy == "loss_ranked":
        w = {d: max(summary[d]["mean"], 0.0) + 1e-3 for d in live}
    elif policy == "neyman":
        w = {d: summary[d]["sd"] + 1e-6 for d in live}
    elif policy == "cap":
        # uncertainty x closeness to the decision boundary: p(1-p) peaks where the domain's verdict is
        # undecided, so settled domains (clearly safe or clearly damaged) stop consuming budget.
        w = {d: (summary[d]["sd"] + 1e-6) * (alpha + 4 * summary[d]["p_violate"] * (1 - summary[d]["p_violate"]))
             for d in live}
    else:
        raise ValueError(policy)
    return _largest_remainder(w, available, n, rotate=round_idx)


def _largest_remainder(w: dict[str, float], available: dict[str, int], n: int, rotate: int = 0) -> dict[str, int]:
    """Proportional integer allocation honouring per-domain availability.

    Ties in the fractional remainders are broken by a rotating order so that
    'balanced' does not systematically favour the alphabetically first domains.
    """
    alloc = {d: 0 for d in w}
    remaining = n
    active = {d: v for d, v in w.items() if available.get(d, 0) > 0}
    while remaining > 0 and active:
        tot = sum(active.values())
        quotas = {d: remaining * v / tot for d, v in active.items()}
        base = {d: int(np.floor(q)) for d, q in quotas.items()}
        left = remaining - sum(base.values())
        keys = sorted(active)
        rank = {d: (i - rotate) % len(keys) for i, d in enumerate(keys)}
        order = sorted(active, key=lambda d: (-round(quotas[d] - base[d], 9), rank[d]))
        for d in order[:left]:
            base[d] += 1
        progressed = 0
        for d in list(active):
            cap = available[d] - alloc[d]
            give = min(base[d], cap)
            alloc[d] += give
            progressed += give
            if alloc[d] >= available[d]:
                active.pop(d)
        remaining -= progressed
        if progressed == 0:
            break
    return {d: c for d, c in alloc.items() if c > 0}


def sequential_test(
    *,
    domains: list[str],
    pool: dict[str, list[dict]],
    dense_correct: dict[str, bool],
    evaluate: Callable[[list[dict]], dict[str, bool]],
    policy: str,
    step_budget: int,
    min_per_domain: int,
    round_size: int,
    limits: Limits,
    accept_max: float,
    reject_min: float,
    rule: str = "per_domain",
    min_fail_n: int = 0,
    prior: tuple[float, float, float],
    mc_samples: int,
    alpha: float,
    seed: int,
    iteration: int,
    on_round: Callable[[dict], None] | None = None,
) -> TestResult:
    """Run one budgeted sequential test of a candidate model. Deterministic given seed/iteration."""
    order: dict[str, list[dict]] = {}
    for j, d in enumerate(domains):
        rng = np.random.default_rng([int(seed), int(iteration), j, 31337])
        idx = rng.permutation(len(pool[d]))
        order[d] = [pool[d][i] for i in idx]
    cursor = {d: 0 for d in domains}
    tallies = {d: Tally() for d in domains}
    used, rnd = 0, 0
    allocations = []
    budget = max(0, int(step_budget))
    per = min(min_per_domain, budget // max(1, len(domains)))
    alloc = {d: min(per, len(order[d])) for d in domains}
    summary: dict[str, dict] = {}
    p_any = 1.0
    decision = INCONCLUSIVE
    while True:
        qs = []
        for d, c in alloc.items():
            take = order[d][cursor[d] : cursor[d] + c]
            cursor[d] += len(take)
            qs += take
        if qs:
            res = evaluate(qs)
            for q in qs:
                tallies[q["domain"]].add(dense_correct[q["qid"]], res[q["qid"]])
            used += len(qs)
        allocations.append({"round": rnd, "alloc": alloc})
        draws = posterior_draws(tallies, domains, prior, mc_samples, np.random.default_rng([int(seed), int(iteration), rnd, 4242]))
        summary, p_any = summarize(draws, domains, limits)
        if on_round:
            on_round({"round": rnd, "used": used, "p_any": p_any, "alloc": alloc})
        if rule == "familywise":
            ok, bad = p_any <= accept_max, p_any >= reject_min
        elif rule == "per_domain":
            ok = all(v["p_violate"] <= accept_max for v in summary.values())
            bad = any(summary[d]["p_violate"] >= reject_min and tallies[d].n >= min_fail_n for d in domains)
        else:
            raise ValueError(f"unknown testing.decision_rule {rule!r}")
        if bad:
            decision = FAIL
            break
        if ok:
            decision = PASS
            break
        available = {d: len(order[d]) - cursor[d] for d in domains}
        left = budget - used
        if left <= 0 or sum(available.values()) == 0:
            decision = INCONCLUSIVE
            break
        rnd += 1
        alloc = allocate(policy, domains, summary, tallies, available, min(round_size, left), rnd, alpha)
        if not alloc:
            decision = INCONCLUSIVE
            break
    return TestResult(
        decision=decision,
        used=used,
        rounds=rnd + 1,
        early_stop=decision != INCONCLUSIVE and used < budget,
        p_any=p_any,
        domains=summary,
        tallies={d: asdict(t) for d, t in tallies.items()},
        limits=asdict(limits),
        allocations=allocations,
    )
