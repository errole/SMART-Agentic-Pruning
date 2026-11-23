"""Scores policy decisions against the full-pool reference.

The scorer is privileged: it reads every cached answer.  It is kept apart from
the replay source and the policies, which only ever see revealed items.

Reference decision for a candidate: it *qualifies* iff every domain's full
search-pool loss L_d = (K+ - K-) / N_d is at or below that domain's tolerance
-- exactly the quantity the sequential bounds cover.  The holdout losses are
reported next to it as an independent check and never enter any decision.
"""
from __future__ import annotations

from .cache import ReplayCache
from .evaluator import ACCEPT, REJECT, Settings, within

OUTCOMES = ("correct_accept", "correct_reject", "false_accept", "false_reject", "inconclusive")


def paired_counts(dense: dict[str, int], cand: dict[str, int], ids: list[str]) -> tuple[int, int]:
    broken = sum(1 for i in ids if dense[i] == 1 and cand[i] == 0)
    fixed = sum(1 for i in ids if dense[i] == 0 and cand[i] == 1)
    return broken, fixed


def reference(cache: ReplayCache, s: Settings) -> dict[str, dict]:
    out = {}
    dense = cache.dense.correct
    for c in cache.candidates:
        doms = {}
        for d in s.domains:
            pool, hold = cache.pool[d], cache.holdout.get(d, [])
            kp, km = paired_counts(dense, c.correct, pool)
            rec = {"N": len(pool), "broken": kp, "fixed": km, "loss": (kp - km) / len(pool),
                   "within": within(kp - km, len(pool), s.tolerance[d]),
                   "dense_acc": sum(dense[i] for i in pool) / len(pool),
                   "cand_acc": sum(c.correct[i] for i in pool) / len(pool),
                   "holdout_n": len(hold), "holdout_loss": None, "holdout_within": None}
            if hold:
                hp, hm = paired_counts(dense, c.correct, hold)
                rec["holdout_loss"] = (hp - hm) / len(hold)
                rec["holdout_within"] = within(hp - hm, len(hold), s.tolerance[d])
            doms[d] = rec
        hw = [r["holdout_within"] for r in doms.values()]
        out[c.name] = {"qualifies": all(r["within"] for r in doms.values()), "domains": doms,
                       "max_loss": max(r["loss"] for r in doms.values()),
                       "holdout_qualifies": None if None in hw else all(hw),
                       "holdout_max_loss": None if None in hw else max(r["holdout_loss"] for r in doms.values())}
    return out


def outcome(decision: str, qualifies: bool) -> str:
    if decision == ACCEPT:
        return "correct_accept" if qualifies else "false_accept"
    if decision == REJECT:
        return "false_reject" if qualifies else "correct_reject"
    return "inconclusive"


def per_policy(rows: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for r in rows:
        a = agg.setdefault(r["policy"], {"candidates": 0, "questions": 0, "tokens": 0, "eval_seconds": 0.0,
                                         "decision_seconds": 0.0, **{o: 0 for o in OUTCOMES},
                                         "accept": 0, "reject": 0})
        a["candidates"] += 1
        a["questions"] += r["questions"]
        a["tokens"] = None if a["tokens"] is None or r["tokens"] is None else a["tokens"] + r["tokens"]
        a["eval_seconds"] = (None if a["eval_seconds"] is None or r["eval_seconds"] is None
                             else a["eval_seconds"] + r["eval_seconds"])
        a["decision_seconds"] += r["decision_seconds"]
        a[r["outcome"]] += 1
        if r["decision"] in ("accept", "reject"):
            a[r["decision"]] += 1
    return agg
