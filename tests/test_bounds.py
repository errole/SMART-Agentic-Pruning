"""Candidate confirmation bounds (docs/bound_development_plan.md): CPU only, synthetic data only."""
import numpy as np
import pytest

from capprune import bounds as B
from capprune.screen_confirm import certifies


def test_reference_reproduces_the_preregistered_decision():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(50, 600))
        kp, km = (int(v) for v in rng.integers(0, n // 5, size=2))
        for tol in (0.02, 0.03, 0.05):
            assert B.cp_union_certifies(n, kp, km, 0.05, tol) == certifies(n, kp, km, 0.05, tol)
    kp, km = np.meshgrid(np.arange(0, 40), np.arange(0, 40), indexing="ij")
    reg = B.cp_union_region(300, kp.ravel(), km.ravel(), 0.05, 0.03)
    assert all(reg[i] == certifies(300, int(a), int(b), 0.05, 0.03) for i, (a, b) in enumerate(zip(kp.ravel(), km.ravel())))


def test_restricted_mle_matches_brute_force():
    n, tol = 800, 0.03
    gs = np.linspace(1e-9, (1 - tol) / 2 - 1e-9, 400001)
    for kp, km in [(40, 25), (12, 0), (0, 0), (3, 30), (150, 20), (0, 9)]:
        ll = (kp * np.log(gs + tol) + km * np.log(gs) + (n - kp - km) * np.log(1 - tol - 2 * gs))
        assert float(B.restricted_gained(n, kp, km, tol)) == pytest.approx(float(gs[ll.argmax()]), abs=5e-6)


@pytest.mark.parametrize("n", [200, 700])
def test_exact_score_size_is_at_most_alpha_on_its_boundary(n):
    tol, alpha = 0.03, 0.05
    z = B.exact_score_critical(n, tol, alpha)
    qs = np.arange(0.031, 0.40, 0.0025)  # well beyond the grid the critical value was computed on
    sizes = [B.exact_certify_probability("exact_score", n, (q + tol) / 2, (q - tol) / 2, alpha, tol) for q in qs]
    assert max(sizes) <= alpha + 1e-6
    assert max(sizes) > 0.03  # and it is not trivially conservative
    assert z < 0


def test_exact_score_size_off_the_nuisance_grid_at_n_3000():
    """The critical value is the supremum over a 400-point nuisance grid, so the test is exact only up to the grid's
    resolution (docs/bound_development_plan.md section 8, note of 2026-11-25). At n = 3,000 and a 3-point limit, the
    size between grid points near p_gained = 0.479 (98.8 % disagreement) exceeds 0.05 by about 7e-6. Inside the data
    range (4-15 % disagreement) it stays below 0.05. If the grid is refined or the nuisance range restricted, this test
    records the change."""
    n, tol, alpha = 3000, 0.03, 0.05
    z = B.exact_score_critical(n, tol, alpha)
    off = np.arange(0.4780, 0.4810, 0.0001)
    sizes = B.size_at(n, tol, z, off)
    assert sizes.max() > alpha  # the documented excess
    assert sizes.max() - alpha < 2e-5  # and its size
    data = (np.arange(0.040, 0.15001, 0.0005) - tol) / 2
    assert B.size_at(n, tol, z, data).max() <= alpha


def test_exact_and_monte_carlo_certification_probabilities_agree():
    n, q, d = 500, 0.08, 0.01
    pl, pg = (q + d) / 2, (q - d) / 2
    for m in ("cp_union", "exact_score", "normal"):
        ex = B.exact_certify_probability(m, n, pl, pg, 0.05, 0.03)
        mc = B.simulate_certify(m, n, pl, pg, 0.05, 0.03, 20000, np.random.default_rng(3))
        assert abs(ex - mc) <= 4 * np.sqrt(max(ex * (1 - ex), 1e-4) / 20000), (m, ex, mc)


def test_betting_is_valid_at_the_boundary_and_has_power_below_it():
    rng = np.random.default_rng(5)
    q, tol = 0.08, 0.03
    at_limit = B.simulate_certify("betting", 500, (q + tol) / 2, (q - tol) / 2, 0.05, tol, 4000, rng)
    assert at_limit <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 4000)
    far_below = B.simulate_certify("betting", 3000, q / 2, q / 2, 0.05, tol, 1000, rng)
    assert far_below > 0.5
    # one fixed sequence: certification is a deterministic function of the sequence
    x = np.where(np.random.default_rng(1).random((1, 2000)) < 0.03, 1.0, 0.0)
    assert B.betting_certifies(x, 0.05, tol)[0] == B.betting_certifies(x.copy(), 0.05, tol)[0]


def test_normal_bound_is_not_finite_sample_valid():
    n, q, tol = 500, 0.04, 0.03
    size = B.exact_certify_probability("normal", n, (q + tol) / 2, (q - tol) / 2, 0.05, tol)
    assert size > 0.05


def test_category_order_is_seeded_and_a_permutation():
    qids = [f"q{i}" for i in range(50)]
    a, b = B.category_order(qids, "STEM"), B.category_order(qids, "STEM")
    assert (a == b).all() and sorted(a.tolist()) == list(range(50))
    assert not (a == B.category_order(qids, "Other")).all()
