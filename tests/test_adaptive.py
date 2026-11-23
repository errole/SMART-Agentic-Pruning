"""Adaptive candidate evaluation: bounds, shared decision rule, budget, information hiding, replay, build."""
from __future__ import annotations

import dataclasses
import json
import math
import shutil

import numpy as np
import pytest

import capprune.adaptive.evaluator as ev
from capprune.adaptive.bounds import (hypergeom_cdf, loss_bound_counts, lower_count, spend, upper_count)
from capprune.adaptive.cache import BudgetExceeded, ModelRecord, ReplayCache, ReplaySource
from capprune.adaptive.evaluator import (ACCEPT, FAIL, INCONCLUSIVE, OPEN, PASS, REJECT, DomainState, Plan,
                                         PolicyView, Settings, evaluate_candidate, look_sizes, make_plan)
from capprune.adaptive.fixture import write_fixture
from capprune.adaptive.policies import POLICIES, dirichlet_multinomial_grid, make_policy
from capprune.adaptive.replay import run_replay, search
from capprune.adaptive.scorer import outcome, per_policy, reference
from conftest import ROOT

POLS = list(POLICIES)
FIXTURE_SETTINGS = dict(domains=["STEM", "Humanities", "Social Sciences", "Other"], tolerance=0.05, min_sample=40,
                        batch_size=40, budget=1000, alpha_false_accept=0.05, alpha_false_reject=0.05)


# ----------------------------------------------------------------------------- bounds
def _pmf(x, N, K, n):
    return math.comb(K, x) * math.comb(N - K, n - x) / math.comb(N, n)


def test_hypergeom_cdf_matches_exact():
    for N, K, n in [(20, 7, 9), (50, 0, 10), (50, 50, 10), (60, 13, 60), (35, 20, 1)]:
        for k in range(-1, n + 2):
            exact = sum(_pmf(x, N, K, n) for x in range(0, min(k, n) + 1) if x <= K and n - x <= N - K)
            assert hypergeom_cdf(k, N, K, n) == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("N", [12, 25])
@pytest.mark.parametrize("delta", [0.02, 0.2])
def test_count_bounds_have_exact_coverage_and_never_cross(N, delta):
    """For every true K and sample size n: P(upper < K) <= delta and P(lower > K) <= delta, computed exactly."""
    for n in range(1, N + 1):
        U = {k: upper_count(k, n, N, delta) for k in range(n + 1)}
        Lo = {k: lower_count(k, n, N, delta) for k in range(n + 1)}
        assert all(Lo[k] <= U[k] for k in U)
        for K in range(N + 1):
            ks = range(max(0, n - (N - K)), min(n, K) + 1)
            miss_hi = sum(_pmf(k, N, K, n) for k in ks if U[k] < K)
            miss_lo = sum(_pmf(k, N, K, n) for k in ks if Lo[k] > K)
            assert miss_hi <= delta + 1e-12 and miss_lo <= delta + 1e-12, (n, K)
    assert upper_count(4, N, N, delta) == 4 == lower_count(4, N, N, delta)  # full pool -> exact


def test_loss_bounds_are_exact_on_the_full_pool_and_ordered():
    assert loss_bound_counts(200, 200, 17, 5, 0.01, 0.01) == (12, 12)
    lo, hi = loss_bound_counts(200, 80, 7, 2, 0.01, 0.01)
    assert lo < 5 / 80 * 200 < hi


def test_spending_sums_to_alpha():
    sizes = [40, 80, 120, 160, 175]
    for scheme, rho in (("uniform", 1.0), ("power", 1.0), ("power", 3.0), ("power", 0.5)):
        a = spend(sizes, 0.0125, scheme, rho)
        assert len(a) == len(sizes) and min(a) > 0 and sum(a) == pytest.approx(0.0125)
    assert spend(sizes, 0.01, "power", 3.0)[0] < spend(sizes, 0.01, "uniform")[0]  # conservative early


def test_look_schedule_caps_at_pool_and_budget():
    assert look_sizes(100, 30, 30, 100) == [30, 60, 90, 100]   # last batch ends exactly at the pool
    assert look_sizes(100, 30, 30, 75) == [30, 60]              # never plan beyond what the budget allows
    plan = make_plan(Settings.from_dict({**FIXTURE_SETTINGS, "budget": 400}), {"STEM": 600, "Humanities": 900,
                                                                               "Social Sciences": 500, "Other": 550})
    assert all(dp.looks[-1].n == 400 - 3 * 40 for dp in plan.domains)  # others keep their minimum sample
    total_hi = sum(L.delta_hi for dp in plan.domains for L in dp.looks)
    assert total_hi == pytest.approx(0.05)


def test_dirichlet_multinomial_grid_is_a_distribution():
    g = dirichlet_multinomial_grid(30, 2.5, 1.5, 27.5)
    assert np.exp(g[np.isfinite(g)]).sum() == pytest.approx(1.0)
    assert np.isneginf(g[20, 20])


# ----------------------------------------------------------------------------- fixture replay
@pytest.fixture(scope="module")
def fixture_cache(tmp_path_factory):
    return ReplayCache.load(write_fixture(tmp_path_factory.mktemp("adaptive_fixture")))


@pytest.fixture(scope="module")
def fixture_rows(fixture_cache):
    return run_replay(fixture_cache, Settings.from_dict(FIXTURE_SETTINGS), POLS)


def test_fixture_reference_is_as_designed(fixture_cache):
    ref = reference(fixture_cache, Settings.from_dict(FIXTURE_SETTINGS))
    assert not ref["clear_fail"]["qualifies"] and ref["clear_fail"]["domains"]["Other"]["loss"] > 0.25
    nt = ref["near_threshold"]["domains"]["Humanities"]
    assert nt["loss"] == pytest.approx(0.05) and ref["near_threshold"]["qualifies"]
    assert ref["easy_pass"]["qualifies"] and ref["easy_pass"]["max_loss"] == 0.0


def test_fixture_decisions_for_every_policy(fixture_rows):
    by = {(r["policy"], r["candidate"]): r for r in fixture_rows}
    for p in POLS:
        assert by[(p, "clear_fail")]["decision"] == REJECT
        assert by[(p, "easy_pass")]["decision"] == ACCEPT
        nt = by[(p, "near_threshold")]
        assert nt["decision"] == INCONCLUSIVE
        left = 1000 - nt["questions"]
        assert left < 40  # the budget is spent: no further batch fits
        assert nt["final"]["Humanities"]["status"] == OPEN and nt["final"]["Humanities"]["n"] < 900
    agg = per_policy(fixture_rows)
    assert all(a["false_accept"] == 0 and a["false_reject"] == 0 for a in agg.values())
    # the clearly failing domain is found faster by the adaptive policies than by round-robin
    rr = by[("round_robin", "clear_fail")]["questions"]
    assert by[("threshold_aware", "clear_fail")]["questions"] < rr
    assert by[("widest_interval", "clear_fail")]["questions"] < rr


def test_threshold_aware_goes_straight_to_the_failing_domain(fixture_rows):
    by = {(r["policy"], r["candidate"]): r for r in fixture_rows}
    assert by[("threshold_aware", "clear_fail")]["trace"][1]["chosen"] == "Other"
    assert by[("round_robin", "clear_fail")]["trace"][1]["chosen"] == "STEM"  # fixed order, Other is last


def test_identical_initial_sample_and_question_order(fixture_rows, fixture_cache):
    plan = make_plan(Settings.from_dict(FIXTURE_SETTINGS), fixture_cache.pool_sizes)
    for name in ("clear_fail", "near_threshold", "easy_pass"):
        runs = [r for r in fixture_rows if r["candidate"] == name]
        assert len({json.dumps(r["trace"][0], sort_keys=True) for r in runs}) == 1  # same initial look
        for d in plan.names:
            seqs = []
            for p in POLS:
                src = ReplaySource(fixture_cache, name, 0, 10 ** 6)
                run = evaluate_candidate(src, make_policy(p), plan, name)
                seqs.append(run.revealed[d])
            longest = max(seqs, key=len)
            assert all(s == longest[: len(s)] for s in seqs)  # every policy reads the same order


# ----------------------------------------------------------------------------- shared rule, budget (randomised)
def _random_cache(rng: np.random.Generator) -> tuple[ReplayCache, Settings]:
    D = int(rng.integers(2, 5))
    doms = [f"d{j}" for j in range(D)]
    items, dense = [], {}
    for d in doms:
        for i in range(int(rng.integers(40, 260))):
            iid = f"{d}/{i}"
            items.append({"item_id": iid, "domain": d, "split": "pool", "tokens": 100})
            dense[iid] = int(rng.random() < rng.uniform(0.3, 0.8))
    cands = []
    for c in range(4):
        loss = {d: float(rng.choice([0.0, 0.03, 0.08, 0.3])) if c else 0.0 for d in doms}
        corr = {}
        for it in items:  # dense-correct items break with prob 0.02 + loss, dense-wrong items get fixed with prob 0.02
            iid, dc = it["item_id"], dense[it["item_id"]]
            corr[iid] = (1 - dc) if rng.random() < 0.02 + loss[it["domain"]] * dc else dc
        cands.append(ModelRecord(f"c{c}", "candidate", c, corr, None, None, {}))
    cache = ReplayCache(items, ModelRecord("dense", "dense", -1, dense, None, None, {}), cands)
    m = {d: int(rng.integers(5, 30)) for d in doms}
    s = Settings.from_dict(dict(domains=doms, tolerance=float(rng.uniform(0.02, 0.15)), min_sample=m,
                                batch_size={d: int(rng.integers(5, 40)) for d in doms},
                                budget=int(sum(m.values()) + rng.integers(0, sum(cache.pool_sizes.values()))),
                                alpha_false_accept=0.05, alpha_false_reject=0.1, seed=int(rng.integers(0, 100))))
    return cache, s


def _check_rule(run: dict, s: Settings) -> None:
    """The reported decision is exactly what the shared rule says about the final per-domain bounds."""
    final = run["final"]
    statuses = {d: v["status"] for d, v in final.items()}
    for d, v in final.items():
        tol = s.tolerance[d]
        assert statuses[d] == (FAIL if v["lo"] > tol + 1e-9 else PASS if v["hi"] <= tol + 1e-9 else OPEN)
    if any(x == FAIL for x in statuses.values()):
        assert run["decision"] == REJECT
    elif all(x == PASS for x in statuses.values()):
        assert run["decision"] == ACCEPT
    else:
        assert run["decision"] == INCONCLUSIVE
    for step in run["trace"][:-1]:  # every earlier look: no decision yet
        st = [v[3] for v in step["domains"].values()]
        assert FAIL not in st and not all(x == PASS for x in st)


def test_all_policies_share_the_rule_and_never_exceed_the_budget():
    rng = np.random.default_rng(2026)
    for _ in range(25):
        cache, s = _random_cache(rng)
        rows = run_replay(cache, s, POLS)
        for r in rows:
            assert r["questions"] <= s.budget
            _check_rule(r, s)
        for c in cache.candidates:  # ACCEPT is policy-invariant: same looks -> same cost
            acc = [r for r in rows if r["candidate"] == c.name and r["decision"] == ACCEPT]
            if acc:
                assert len(acc) == len(POLS)
                assert len({json.dumps({d: v["n"] for d, v in r["final"].items()}) for r in acc}) == 1


def test_decision_rule_is_one_shared_function(monkeypatch, fixture_cache):
    calls = []
    real = ev.decide
    monkeypatch.setattr(ev, "decide", lambda states: calls.append(1) or real(states))
    plan = make_plan(Settings.from_dict(FIXTURE_SETTINGS), fixture_cache.pool_sizes)
    for p in POLS:
        n0 = len(calls)
        run = evaluate_candidate(ReplaySource(fixture_cache, "clear_fail", 0, 1000), make_policy(p), plan)
        assert len(calls) - n0 == len(run.trace)  # consulted at every look, by every policy


def test_initial_sample_only_budget_gives_identical_decisions(fixture_cache):
    s = Settings.from_dict({**FIXTURE_SETTINGS, "budget": 160})
    rows = run_replay(fixture_cache, s, POLS)
    for name in ("clear_fail", "near_threshold", "easy_pass"):
        assert len({(r["decision"], r["questions"]) for r in rows if r["candidate"] == name}) == 1


def test_replay_source_enforces_budget_and_order(fixture_cache):
    src = ReplaySource(fixture_cache, "easy_pass", 0, 50)
    a = src.reveal("STEM", 30)
    with pytest.raises(BudgetExceeded):
        src.reveal("Other", 21)
    b = src.reveal("STEM", 20)
    assert len({i for i, _, _ in a + b}) == 50 and src.used == 50


# ----------------------------------------------------------------------------- information hiding
class _ForbiddenAccess(Exception):
    pass


def _reachable(obj, seen=None):
    """Objects reachable from a policy view through containers and dataclass fields."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _reachable(getattr(obj, f.name), seen)
    elif isinstance(obj, (tuple, list, set, frozenset)):
        for x in obj:
            yield from _reachable(x, seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _reachable(k, seen)
            yield from _reachable(v, seen)


class _SpyPolicy:
    """Wraps a policy and inspects everything it is handed."""

    def __init__(self, inner, revealed_counts):
        self.inner, self.name, self.counts, self.views = inner, inner.name, revealed_counts, 0

    def choose(self, view: PolicyView) -> str:
        self.views += 1
        for o in _reachable(view):
            if isinstance(o, (ReplaySource, ReplayCache, ModelRecord)):
                raise _ForbiddenAccess(type(o).__name__)
            assert isinstance(o, (PolicyView, DomainState, Plan, ev.DomainPlan, ev.Look, str, int, float, tuple))
        for s in view.states:  # the view describes revealed items only
            assert s.n == self.counts()[s.name]
        return self.inner.choose(view)


def test_policies_receive_only_revealed_results(fixture_cache):
    plan = make_plan(Settings.from_dict(FIXTURE_SETTINGS), fixture_cache.pool_sizes)
    for p in POLS:
        for name in ("clear_fail", "near_threshold", "easy_pass"):
            src = ReplaySource(fixture_cache, name, 0, 1000)
            spy = _SpyPolicy(make_policy(p), lambda: dict(src._cursor))
            evaluate_candidate(src, spy, plan, name)
            assert spy.views > 0


def _unchanged_when_hidden_answers_flip(cache: ReplayCache, cand, make, s: Settings) -> bool:
    """Run, then flip every pool answer that run never revealed, and run again: identical?"""
    plan = make_plan(s, cache.pool_sizes)
    src = ReplaySource(cache, cand.name, s.seed, s.budget)
    run = evaluate_candidate(src, make(src), plan, cand.name)
    seen = {i for v in run.revealed.values() for i in v}
    hidden = [i for v in cache.pool.values() for i in v if i not in seen]
    assert hidden
    dense = {**cache.dense.correct, **{i: 1 - cache.dense.correct[i] for i in hidden}}
    alt_c = {**cand.correct, **{i: 1 - cand.correct[i] for i in hidden[::2]}}
    alt = ReplayCache(cache.items, dataclasses.replace(cache.dense, correct=dense), [dataclasses.replace(cand, correct=alt_c)])
    src2 = ReplaySource(alt, cand.name, s.seed, s.budget)
    run2 = evaluate_candidate(src2, make(src2), plan, cand.name)
    return run2.trace == run.trace and run2.decision == run.decision


def test_unrevealed_results_cannot_change_any_choice(fixture_cache):
    s = Settings.from_dict(FIXTURE_SETTINGS)
    for p in POLS:
        for c in fixture_cache.candidates:
            assert _unchanged_when_hidden_answers_flip(fixture_cache, c, lambda src: make_policy(p), s)


class _Cheater:
    """Peeks at the source's private answers: prefers the domain with the most unrevealed broken answers."""
    name = "cheater"

    def __init__(self, src):
        self.src = src

    def choose(self, view):
        s = self.src
        return max(view.eligible, key=lambda d: sum(s._dense[i] - s._cand[i] for i in s._order[d][s._cursor[d]:]))


def test_the_hidden_answer_check_catches_a_cheating_policy(fixture_cache):
    s = Settings.from_dict(FIXTURE_SETTINGS)
    assert not all(_unchanged_when_hidden_answers_flip(fixture_cache, c, _Cheater, s) for c in fixture_cache.candidates)


# ----------------------------------------------------------------------------- validity (simulation)
def test_error_rates_stay_below_alpha_over_random_question_orders():
    """A domain truly 2 points over its limit is (almost) never accepted; one 2 points under is never rejected."""
    rng = np.random.default_rng(5)
    N = 300
    items = [{"item_id": f"a/{i}", "domain": "a", "split": "pool", "tokens": 1} for i in range(N)]
    dense = {f"a/{i}": int(i % 5 != 0) for i in range(N)}

    def cand(loss_items):
        c = dict(dense)
        right = [i for i in dense if dense[i] == 1]
        for i in rng.permutation(right)[: 15 + loss_items]:
            c[str(i)] = 0
        for i in rng.permutation([i for i in dense if dense[i] == 0])[:15]:
            c[str(i)] = 1
        return c

    over = ModelRecord("over", "candidate", 0, cand(21), None, None, {})    # loss 0.07 vs tolerance 0.05
    under = ModelRecord("under", "candidate", 1, cand(9), None, None, {})   # loss 0.03
    cache = ReplayCache(items, ModelRecord("dense", "dense", -1, dense, None, None, {}), [over, under])
    fa = fr = 0
    trials = 60
    for seed in range(trials):
        s = Settings.from_dict(dict(domains=["a"], tolerance=0.05, min_sample=30, batch_size=30, budget=270,
                                    alpha_false_accept=0.2, alpha_false_reject=0.2, seed=seed))
        rows = run_replay(cache, s, ["threshold_aware"])
        fa += rows[0]["decision"] == ACCEPT
        fr += rows[1]["decision"] == REJECT
    assert fa / trials <= 0.2 and fr / trials <= 0.2


# ----------------------------------------------------------------------------- scorer and search
def test_outcome_labels():
    assert outcome(ACCEPT, True) == "correct_accept" and outcome(ACCEPT, False) == "false_accept"
    assert outcome(REJECT, False) == "correct_reject" and outcome(REJECT, True) == "false_reject"
    assert outcome(INCONCLUSIVE, True) == "inconclusive"


def test_search_to_first_qualifier_respects_latency_target(fixture_rows, fixture_cache):
    s = Settings.from_dict(FIXTURE_SETTINGS)
    ref = reference(fixture_cache, s)
    order = ["clear_fail", "near_threshold", "easy_pass"]
    by = {(r["policy"], r["candidate"]): r["questions"] for r in fixture_rows}
    res = search(fixture_rows, fixture_cache, ref, POLS, {"max_ratio_vs_dense": 0.85}, order)
    for p in POLS:
        assert res[p]["found"] == "easy_pass" and res[p]["truly_qualifies"]
        assert res[p]["questions"] == sum(by[(p, c)] for c in order)
    res = search(fixture_rows, fixture_cache, ref, POLS, {"max_value": 7.0}, order)  # easy_pass is too slow
    assert all(r["found"] is None and r["skipped_latency"] == ["easy_pass"] for r in res.values())
    res = search(fixture_rows, fixture_cache, ref, POLS, None, order)
    assert res["round_robin"]["found"] == "easy_pass"


def test_cli_fixture_and_replay(tmp_path):
    from capprune.cli import main

    cache = tmp_path / "cache"
    assert main(["adaptive-fixture", "--out", str(cache)]) == 0
    assert main(["adaptive-replay", "--config", str(ROOT / "configs" / "adaptive_fixture.yaml"),
                 "--cache", str(cache), "--out", str(tmp_path / "out")]) == 0
    md = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "| clear_fail | no | reject" in md and "What the bounds cover" in md
    rows = [json.loads(l) for l in open(tmp_path / "out" / "runs.jsonl", encoding="utf-8")]
    assert len(rows) == 9 and all(r["questions"] <= 1000 for r in rows)


# ----------------------------------------------------------------------------- the Qwen3 pipeline bridge (tiny model, CPU)
def test_build_cache_from_pipeline_and_replay(smoke_assets, tmp_path):
    from capprune.adaptive.build import build
    from capprune.adaptive.config import load_adaptive
    from capprune.config import load_config
    from capprune.model import build_units, load_model
    from capprune.pruning.masks import PruneState

    pcfg = load_config(ROOT / "configs" / "smoke.yaml", ["pruning.mode=structured"])
    units = build_units(load_model(pcfg), "structured", [])
    st = PruneState(units, "structured")
    for u in units[:2]:
        st.masks[u.name][:64] = False
    st.save(tmp_path / "mask.safetensors")
    shutil.copytree(ROOT / ".smoke" / "tiny-qwen3", tmp_path / "copy")

    cands = ("[{name: wsp50, kind: wanda, sparsity: 0.5}, {name: m2, kind: masks, path: '%s'},"
             " {name: same, kind: hf_dir, path: '%s'}]" % ((tmp_path / "mask.safetensors").as_posix(),
                                                         (tmp_path / "copy").as_posix()))
    cfg = load_adaptive(ROOT / "configs" / "adaptive_smoke.yaml",
                        [f"cache_dir={(tmp_path / 'cache').as_posix()}", f"build.candidates={cands}"])
    build(cfg)
    cache = ReplayCache.load(tmp_path / "cache")
    assert [c.name for c in cache.candidates] == ["wsp50", "m2", "same"]
    assert cache.pool_sizes == {"STEM": 64, "Humanities": 32, "Social Sciences": 16, "Other": 16}
    s = Settings.from_dict(cfg["evaluation"])
    ref = reference(cache, s)
    assert all(v["loss"] == 0 for v in ref["same"]["domains"].values())  # an identical checkpoint loses nothing
    assert cache.candidate("m2").info["achieved_sparsity"] == pytest.approx(2 / 4 * 64 / 256)  # 2 of 4 MLPs, 64 of 256 channels
    assert cache.candidate("wsp50").info["parameters"] < cache.dense.info["parameters"]
    rows = run_replay(cache, s, POLS)
    assert all(r["eval_seconds"] is not None and r["tokens"] > 0 for r in rows)  # measured layout matches
    for r in rows:
        _check_rule(r, s)
    cfg2 = load_adaptive(ROOT / "configs" / "adaptive_smoke.yaml", ["evaluation.seed=1"])
    rows2 = run_replay(cache, Settings.from_dict(cfg2["evaluation"]), ["round_robin"])
    assert rows2[0]["eval_seconds"] is None  # a different question order is not what was timed
