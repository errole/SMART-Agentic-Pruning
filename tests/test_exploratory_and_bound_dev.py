"""Exploratory analyses and the bound-development driver on synthetic outcomes (CPU)."""
import numpy as np
import pytest

from capprune import bound_dev as bd
from capprune.confirmatory import CATS, sequence_claim


def _synthetic_oc(seed=0, k=6, per_cat=400, step=0.006):
    """A run whose checkpoints lose ``step`` more accuracy each, per category (dense accuracy 60 %)."""
    rng = np.random.default_rng(seed)
    cats = np.repeat(np.array(CATS), per_cat)
    qids = np.array([f"q{i:05d}" for i in range(len(cats))])
    dense = rng.random(len(cats)) < 0.6
    rows = []
    for i in range(k):
        flip_lose = rng.random(len(cats)) < 0.04 + step * (i + 1)
        flip_gain = rng.random(len(cats)) < 0.04
        cand = np.where(dense, ~flip_lose, flip_gain)
        rows.append(cand)
    return {"qids": qids, "category": cats, "dense": np.tile(dense, (k, 1)), "cand": np.array(rows),
            "position": np.arange(1, k + 1), "params_removed": np.arange(1, k + 1) * 1000,
            "materialized_parameters_removed": np.arange(1, k + 1) * 1000}


def test_reference_sequence_reproduces_the_preregistered_claim():
    for s in range(3):
        oc = _synthetic_oc(s)
        r = bd.run_sequence("cp_union", oc)
        pos, params = sequence_claim(oc, 0.05, 0.03)
        assert (r["claim_position"], r["parameters_removed"]) == (pos, params)


def test_exact_score_certifies_at_least_as_far_as_the_reference_here():
    oc = _synthetic_oc(1)
    assert bd.run_sequence("exact_score", oc)["certified_checkpoints"] >= bd.run_sequence("cp_union", oc)["certified_checkpoints"]


def test_betting_sequences_follow_the_preregistered_item_order():
    oc = _synthetic_oc(2)
    a = bd._sequences(oc, 0, oc["category"], oc["qids"])
    b = bd._sequences(oc, 0, oc["category"], oc["qids"])
    assert all((a[c] == b[c]).all() for c in CATS)
    assert all(len(a[c]) == 400 and set(np.unique(a[c])) <= {-1.0, 0.0, 1.0} for c in CATS)


def test_selection_and_condition_1():
    def sizes(ns):
        return {"0.010": {f"{q:.2f}": {"n": n} for q, n in zip(bd.SIZE_Q, ns)}}

    S = {"cp_union": sizes([2000, 2400, 2600, 3000]), "exact_score": sizes([1000, 1200, 1300, 1500]),
         "betting": sizes([1800, 2000, 2400, 2800]), "normal": sizes([900, 1100, 1200, 1400])}
    valid = {"exact_score": {"meets_requirement": True}, "betting": {"meets_requirement": True}}
    best = bd.select_best(valid, S)
    assert best["best"] == "exact_score"
    assert bd.condition_1("exact_score", S)["holds"] is True
    assert bd.condition_1("betting", S)["holds"] is False  # 0.9 / 0.83 / 0.92 / 0.93 of CP
    valid["exact_score"]["meets_requirement"] = False
    assert bd.select_best(valid, S)["best"] == "betting"
    valid["betting"]["meets_requirement"] = False
    assert bd.select_best(valid, S)["best"] is None and bd.condition_1(None, S)["holds"] is False


def test_n_needed_is_the_first_multiple_of_50_reaching_the_target():
    r = bd.n_needed("exact_score", 0.08, 0.0)
    assert r["n"] % 50 == 0 and r["power_at_n"] >= 0.8
    assert r["power_at_n_minus_50"] is None or r["power_at_n_minus_50"] < 0.8


def test_loss_curve_and_frontier_on_a_synthetic_model():
    from capprune.exploratory import check_frontier, frontier, loss_curves

    oc = _synthetic_oc(3, k=4, per_cat=600)
    pos, params = sequence_claim(oc, 0.05, 0.03)
    s5 = sequence_claim(oc, 0.05, 0.05)
    tested = [{"position": int(p), "pass": (pos is not None and p <= pos)} for p in oc["position"][: (pos or 0) + 1]]
    run = {"run_id": "ppl_top1__t00__s0", "method": "ppl_top1", "seed": 0, "batch": "x", "oc": oc,
           "confirm": {"claim_position": pos, "confirmed_parameters_removed": params, "dense_parameters": 10000, "tested": tested},
           "report": {"stop_reason": "proposal_cap", "secondary": {"S3": {"parameters_removed": s5[1]}, "S4": {}}}}
    M = {"label": "m", "model_key": "mk", "runs": [run]}
    sec = {"s4_sizes": [100, 200], "s4_draws": 5, "s4_seed": 1}
    F = frontier([M], sec)
    assert F["models"][0]["by_arm"]["ppl_top1"]["0.03"]["full"]["median"] == params
    assert check_frontier(F, [M]) == []
    # more items or a looser limit never certify less, in the median
    by = F["models"][0]["runs"][0]["by_tolerance"]
    assert by["0.05"]["full"]["values"][0] >= by["0.03"]["full"]["values"][0] >= by["0.02"]["full"]["values"][0]
    LC = loss_curves([M])
    cp = LC["models"][0]["runs"][0]["checkpoints"][0]["categories"]["STEM"]
    assert cp["lower"] <= cp["estimate"] <= cp["upper"]
    assert LC["models"][0]["runs"][0]["screen_stop_position"] == 4


@pytest.mark.parametrize("method", ["cp_union", "exact_score", "normal"])
def test_decide_reports_the_statistic_of_each_count_based_method(method):
    oc = _synthetic_oc(4, k=2)
    r = bd.decide(method, oc, 0)
    assert set(r["categories"]) == set(CATS) and isinstance(r["pass"], bool)
