"""Paired answer-distribution signals and the selection rule shared by the evidence and the live loop.

The matched-signal studies (``matched_signals``) and the live loop's KL probe after a rank-1 rejection
(``gradient_live``, protocol amendment 2) use exactly the same functions:

* ``item_terms``   per-item terms from the four letters' log-probabilities of the dense model and a candidate
                   (renormalised over A-D): flip (lost - gained), KL(dense || candidate), the correct answer's
                   log-probability drop, total variation, and the dense-correct mask;
* ``signal_means`` per category: S0 net flip loss, S2 KL, S1c (log-prob drop on dense-correct items), TV;
* ``pick_action``  THE selection rule: the action with the lowest worst-category (maximum over categories) value of
                   the signal; ties go to the lower gradient rank (``choose``).
"""
from __future__ import annotations

import numpy as np


def renorm(lp: np.ndarray) -> np.ndarray:
    """Log-probabilities renormalised over the four letters (rows)."""
    z = lp.astype(np.float64) - lp.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def item_terms(lp_dense: np.ndarray, lp_cand: np.ndarray, answer: np.ndarray, c_dense: np.ndarray,
               c_cand: np.ndarray) -> dict:
    """Per-item terms whose means are the signals: flip x in {-1,0,1}, KL and TV over A-D, and the renormalised
    correct-answer log-prob change (kept with a dense-correct mask for S1c)."""
    a, b = renorm(np.asarray(lp_dense)), renorm(np.asarray(lp_cand))
    answer = np.asarray(answer, dtype=int)
    idx = np.arange(len(answer))
    pa, pb = np.exp(a), np.exp(b)
    return {"flip": np.asarray(c_dense).astype(int) - np.asarray(c_cand).astype(int), "kl": (pa * (a - b)).sum(1),
            "dlogp": a[idx, answer] - b[idx, answer], "tv": 0.5 * np.abs(pa - pb).sum(1),
            "dense_correct": np.asarray(c_dense).astype(bool)}


def signal_means(t: dict, sel: np.ndarray) -> dict:
    """S0, S2 (KL), S1c and TV over the selected items (S1c only over the dense-correct ones)."""
    dc = sel & t["dense_correct"]
    return {"S0": float(t["flip"][sel].mean()), "S2": float(t["kl"][sel].mean()),
            "S1c": float(t["dlogp"][dc].mean()) if dc.any() else float("nan"), "TV": float(t["tv"][sel].mean())}


def choose(scores: list[float], ranks: list[int]) -> int:
    """Index of the action with the lowest score; ties -> lower gradient rank."""
    return min(range(len(scores)), key=lambda i: (round(scores[i], 12), ranks[i]))


def worst_category(v: dict, signal: str) -> float:
    """The worst (largest) per-category value of ``signal``; ``v[category][signal]``."""
    return float(np.nanmax([v[k][signal] for k in sorted(v)]))


def pick_action(vals: list[dict], ranks: list[int], signal: str = "S2") -> int:
    """THE selection rule (matched-signal evidence and the live KL probe): index of the action whose worst-category
    value of ``signal`` is lowest, ties to the lower gradient rank. ``vals[j][category][signal]``."""
    return choose([worst_category(v, signal) for v in vals], ranks)
