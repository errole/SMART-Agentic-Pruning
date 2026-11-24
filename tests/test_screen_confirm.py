"""Screen-then-confirm amendment 1: intersection-union test per checkpoint + fixed-sequence testing over checkpoints.

Family-wise error (FWER) = P(the procedure passes at least one checkpoint whose true paired loss is >= 3 points in some
category), simulated on synthetic paired counts. It must stay <= alpha = 0.05, including when true losses sit exactly at
the 3-point boundary (the least favourable null)."""
from __future__ import annotations

import numpy as np
import pytest

ALPHA, TOL, CATS = 0.05, 0.03, ("STEM", "Humanities", "Social Sciences", "Other")


def _draw(rng, n, q, mu):
    """Paired lost / gained counts for a category with disagreement rate q and true loss mu = p+ - p-."""
    pp, pm = (q + mu) / 2, (q - mu) / 2
    kp, km, _ = rng.multinomial(n, [pp, pm, 1 - pp - pm])
    return int(n), int(kp), int(km)


def _fwer(truth: list[dict[str, tuple[float, float]]], n: int, sims: int, seed: int) -> tuple[float, int]:
    """``truth[k][category] = (q, mu)`` for checkpoint k (already in ascending parameters-removed order)."""
    from capprune.screen_confirm import fixed_sequence

    bad = [any(mu >= TOL - 1e-12 for _, mu in t.values()) for t in truth]
    rng = np.random.default_rng(seed)
    errors = 0
    for _ in range(sims):
        cps = [{"id": k, "params_removed": k, "counts": {c: _draw(rng, n, *t[c]) for c in CATS}} for k, t in enumerate(truth)]
        res = fixed_sequence(cps, ALPHA, TOL)
        passed = [r["id"] for r in res["tested"] if r["pass"]]
        errors += any(bad[k] for k in passed)
    return errors / sims, errors


def _upper99(x: int, n: int) -> float:
    from capprune.population_regate import cp_upper

    return cp_upper(x, n, 0.01)


@pytest.mark.parametrize("name,truth,n", [
    # one category exactly at 3 points after two clean checkpoints (the others never disagree, so they always pass):
    # the procedure reaches the boundary checkpoint every time - the least favourable null for the IUT
    ("one category at the boundary", [{c: (0.0, 0.0) for c in CATS}] * 2
     + [{**{c: (0.0, 0.0) for c in CATS}, "STEM": (0.10, 0.03)}] * 2, 400),
    # only losses (p- = 0) exactly at the boundary
    ("losses only, at the boundary", [{**{c: (0.0, 0.0) for c in CATS}, "Other": (0.03, 0.03)}], 600),
    # every category exactly at 3 points
    ("all categories at the boundary", [{c: (0.08, 0.03) for c in CATS}] * 3, 800),
    # a non-monotone truth: the first checkpoint is bad, later ones are fine
    ("first bad, later good", [{**{c: (0.0, 0.0) for c in CATS}, "Humanities": (0.12, 0.031)}]
     + [{c: (0.0, 0.0) for c in CATS}] * 3, 500),
    # a realistic run: losses grow with compression and cross 3 points at checkpoint 4
    ("growing losses", [{c: (q, mu) for c in CATS} for q, mu in ((0.02, 0.0), (0.04, 0.005), (0.06, 0.01),
                                                              (0.08, 0.03), (0.10, 0.045))], 3000),
])
def test_iut_fixed_sequence_controls_familywise_error(name, truth, n):
    fwer, errors = _fwer(truth, n, sims=3000, seed=20261123)
    assert fwer <= ALPHA, f"{name}: FWER {fwer:.4f}"
    assert _upper99(errors, 3000) <= ALPHA + 0.01, f"{name}: FWER upper 99 % bound {_upper99(errors, 3000):.4f}"


def test_procedure_has_power_and_stops_at_the_first_failure():
    from capprune.screen_confirm import fixed_sequence, iut_test

    clean = {c: (3000, 30, 30) for c in CATS}           # 2 % disagreement, zero loss, 3,000 items per category
    r = iut_test(clean, ALPHA, TOL)
    assert r["pass"] and all(r["upper_bonferroni_descriptive"][c] >= r["upper"][c] for c in CATS)
    fail_one = {**clean, "STEM": (3000, 150, 30)}       # 4-point estimated loss in one category
    assert not iut_test(fail_one, ALPHA, TOL)["pass"]
    seq = [{"id": "a", "params_removed": 1, "counts": clean}, {"id": "b", "params_removed": 2, "counts": fail_one},
           {"id": "c", "params_removed": 3, "counts": clean}]
    out = fixed_sequence(seq, ALPHA, TOL)
    assert out["claim"] == "a" and [t["id"] for t in out["tested"]] == ["a", "b"] and out["untested"] == ["c"]
    assert fixed_sequence(seq[1:], ALPHA, TOL)["claim"] is None
    with pytest.raises(ValueError):
        fixed_sequence(list(reversed(seq)), ALPHA, TOL)     # the order is fixed by parameters removed


def test_decision_delta_per_rule():
    from capprune.screen_confirm import decision_delta

    assert decision_delta("bonferroni", 0.05, 4) == pytest.approx(0.0125)
    assert decision_delta("iut_fixed_sequence", 0.05, 4) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        decision_delta("other", 0.05, 4)
