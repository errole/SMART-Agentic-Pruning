"""The capability gate inside the live pruning-and-rollback loop.

For each attempt the search loop builds a trial checkpoint (parent masks +
one proposal) and calls :meth:`CapabilityGate.evaluate`.  The gate runs the
same sequential test as the replay experiment (``evaluator.evaluate_candidate``
with the configured allocation policy), but its item source scores each
requested batch on the trial model *on demand* - nothing outside the requested
items is ever scored.

* Paired loss is measured against the ORIGINAL dense model (cached dense answers
  on the pruning-time pool), so every step is judged on cumulative loss.
* Question order: a fresh random order for every attempt, derived only from
  (gate.order_seed, run seed, attempt number) - fixed before the proposal exists,
  so it cannot depend on the feedback that produced the candidate.
* Scope of the bounds: per candidate.  For one trial, with probability at least
  1 - alpha_false_accept - alpha_false_reject over its question order, all of its
  bounds cover its full-pool losses.  No error control across the attempts of a
  search is claimed.
"""
from __future__ import annotations

import hashlib
import time

import torch

from .evaluator import ACCEPT, INCONCLUSIVE, REJECT, Settings, evaluate_candidate, make_plan, question_order
from .policies import make_policy


def attempt_order_seed(base_seed: int, run_seed: int, attempt: int) -> int:
    """Question-order seed of one attempt: depends on nothing but these three numbers."""
    h = hashlib.sha256(f"capability-gate:{base_seed}:{run_seed}:{attempt}".encode()).digest()
    return int.from_bytes(h[:4], "little")


class LiveSource:
    """Reveals a domain's next items by scoring them on the trial model (``evaluate``), in a fixed order."""

    def __init__(self, pool: dict[str, list[dict]], dense_correct: dict[str, bool], evaluate, seed: int,
                 budget: int, device: torch.device | None = None):
        self.pool_sizes = {d: len(rows) for d, rows in pool.items()}
        self._rows = {r["qid"]: r for rows in pool.values() for r in rows}
        self._order = {d: question_order([r["qid"] for r in rows], seed, d) for d, rows in pool.items()}
        self._cursor = {d: 0 for d in pool}
        self._dense = dense_correct
        self._evaluate = evaluate
        self._budget = int(budget)
        self._dev = device
        self.used = 0
        self.seconds = 0.0  # scoring time of the requested batches only

    def reveal(self, domain: str, count: int) -> list[tuple[str, int, int]]:
        if self.used + count > self._budget:
            raise RuntimeError(f"gate budget exceeded: {self.used} + {count} > {self._budget}")
        c = self._cursor[domain]
        ids = self._order[domain][c : c + count]
        if len(ids) != count:
            raise IndexError(f"{domain}: only {len(self._order[domain]) - c} items left")
        _sync(self._dev)
        t0 = time.perf_counter()
        res = self._evaluate([self._rows[i] for i in ids])
        _sync(self._dev)
        self.seconds += time.perf_counter() - t0
        self._cursor[domain] = c + count
        self.used += count
        return [(i, int(self._dense[i]), int(res[i])) for i in ids]


def _sync(dev) -> None:
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize(dev)


class CapabilityGate:
    """Configuration (``gate.capability``) + one sequential test per attempt."""

    def __init__(self, rc: dict, probe_rows: list[dict]):
        g = rc["gate"]["capability"]
        self.field = g.get("domain_field", "category")
        self.domains = list(g["domains"])
        self.pool = {d: [r for r in probe_rows if r[self.field] == d] for d in self.domains}
        empty = [d for d, v in self.pool.items() if not v]
        if empty:
            raise ValueError(f"gate.capability: no pruning-time items for domains {empty} (field '{self.field}')")
        self.settings = Settings.from_dict({**g, "seed": 0})  # seed is replaced per attempt
        self.plan = make_plan(self.settings, {d: len(v) for d, v in self.pool.items()})
        self.policy_name = g.get("policy", "threshold_aware")
        make_policy(self.policy_name)  # validate early
        self.base_seed = int(g.get("order_seed", 0))
        self.total_budget = g.get("total_budget")

    @property
    def budget(self) -> int:
        return self.settings.budget

    def describe(self) -> dict:
        """What the controller is told about the gate (identical for every capability arm)."""
        return {"kind": "capability", "domains": self.domains, "tolerance": self.settings.tolerance,
                "budget": self.settings.budget}

    def evaluate(self, attempt: int, run_seed: int, dense_correct: dict, score_fn, cand_id: str, device=None):
        seed = attempt_order_seed(self.base_seed, run_seed, attempt)
        src = LiveSource(self.pool, dense_correct, score_fn, seed, self.settings.budget, device)
        t0 = time.perf_counter()
        run = evaluate_candidate(src, make_policy(self.policy_name), self.plan, cand_id)
        wall = time.perf_counter() - t0
        return run, {"order_seed": seed, "scoring_seconds": src.seconds, "gate_seconds": wall,
                     "decision_seconds": run.decision_seconds}


def outcome_of(decision: str) -> tuple[str, list[str]]:
    """Map a gate decision onto the loop's accept / rollback outcome and reason."""
    if decision == ACCEPT:
        return "accept", []
    if decision == REJECT:
        return "rollback", ["domain_fail"]
    if decision == INCONCLUSIVE:
        return "rollback", ["domain_inconclusive"]
    raise ValueError(decision)


def domain_summary(run) -> dict[str, dict]:
    return {d: {"n": v["n"], "est": round(v["estimate"], 5), "lo": round(v["lo"], 5), "hi": round(v["hi"], 5),
                "status": v["status"], "broken": v["k_plus"], "fixed": v["k_minus"]} for d, v in run.final.items()}


def feedback_note(run, settings: Settings) -> tuple[str, str]:
    """(domain note, assessment) shown to the controller after a capability-gated attempt."""
    tol = settings.tolerance
    per = "; ".join(f"{d} {100 * v['estimate']:+.1f} pts [{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}]"
                    for d, v in run.final.items())
    note = (f"{run.decision.upper()} after {run.questions} questions. Accuracy loss vs. original model per category "
            f"(estimate [lower, upper bound]): {per}. Limit: "
            + ", ".join(f"{d} {100 * tol[d]:.1f}" for d in run.final) + " pts.")
    if run.decision == ACCEPT:
        worst = max(run.final, key=lambda d: run.final[d]["hi"])
        a = (f"Good - every category certified within its limit (largest upper bound: {worst} "
             f"{100 * run.final[worst]['hi']:+.1f} pts).")
    elif run.decision == REJECT:
        bad = max((d for d in run.final if run.final[d]["status"] == "fail"),
                  key=lambda d: run.final[d]["lo"] - tol[d])
        a = (f"Poor - {bad} accuracy loss is above its limit (lower bound {100 * run.final[bad]['lo']:+.1f} pts > "
             f"{100 * tol[bad]:.1f}); avoid units likely tied to {bad} and use smaller increments.")
    else:
        open_ = [d for d in run.final if run.final[d]["status"] == "open"]
        a = ("Uncertain - the question budget ran out before every category could be certified "
             f"(unresolved: {', '.join(open_)}); take a smaller, safer step.")
    return note, a


# ---------------------------------------------------------------------------- rollback verification
@torch.no_grad()
def weights_digest(state) -> list[float]:
    """Exact fingerprint of every prunable weight (float64 sums of values and of absolute values).

    Weights are always written as dense * mask, so a parent restored after a rollback is
    bit-identical to the parent before the attempt and the digests match exactly.
    """
    out = []
    for u in state.units:
        for part in sorted(u.linears):
            w = u.linears[part].weight
            out.append(float(w.double().sum()))
            out.append(float(w.double().abs().sum()))
    return out
