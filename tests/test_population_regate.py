"""Population-level re-gating: exact binomial bounds, coverage at the nominal rate, error spending, deterministic
re-decisions on smoke runs (CPU)."""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pytest

from conftest import ROOT
from live_smoke import make_live_smoke_data
from test_recovery import cli

from capprune.population_regate import (binom_cdf, cp_lower, cp_upper, look_deltas, population_bounds, rule_params,
                                        total_spent)

RULE = ROOT / "docs" / "population_gate_rule.md"
GL = ROOT / "configs" / "gradient_live_smoke.yaml"


def test_exact_binomial_bounds_match_closed_forms_and_brute_force():
    for n in (1, 7, 30, 60):
        for p in (0.01, 0.2, 0.5, 0.93):
            for k in range(0, n + 1, max(1, n // 6)):
                brute = sum(math.comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(k + 1))
                assert abs(binom_cdf(k, n, p) - brute) < 1e-10
    for n, d in ((50, 0.05), (1161, 1e-5)):
        assert abs(cp_upper(0, n, d) - (1 - d ** (1 / n))) < 1e-9       # P_p(X = 0) = (1-p)^n = d
        assert abs(cp_lower(n, n, d) - d ** (1 / n)) < 1e-9
        k = n // 10
        assert abs(binom_cdf(k, n, cp_upper(k, n, d)) - d) < 1e-9      # the inversion is exact
        assert cp_lower(k, n, d) < k / n < cp_upper(k, n, d)


POPS = [(0.06, 0.04), (0.03, 0.0), (0.10, 0.07), (0.05, 0.05), (0.15, 0.12)]


@pytest.mark.parametrize("n", [100, 400, 1200])
def test_bound_coverage_at_the_nominal_rate(n):
    """For populations with KNOWN loss mu = p+ - p-, P(UB < mu) <= delta and P(LB > mu) <= delta."""
    rng = np.random.default_rng(n)
    delta, draws = 0.10, 4000
    se = 3 * math.sqrt(delta * (1 - delta) / draws)
    for pp, pm in POPS:
        mu = pp - pm
        k = rng.multinomial(n, [pp, pm, 1 - pp - pm], size=draws)
        b = np.array([population_bounds(n, int(a), int(c), delta, delta) for a, c, _ in k])
        miss_hi, miss_lo = float(np.mean(b[:, 1] < mu - 1e-12)), float(np.mean(b[:, 0] > mu + 1e-12))
        assert miss_hi <= delta + se and miss_lo <= delta + se, (pp, pm, n, miss_hi, miss_lo)


def test_sequential_coverage_with_uniform_spending_over_looks():
    """The rule's look-level spending: with alpha / J at each of J planned looks, P(UB < mu at ANY look) <= alpha."""
    rng = np.random.default_rng(1)
    looks = list(range(100, 1201, 100))
    alpha, draws, pp, pm = 0.20, 2000, 0.07, 0.04
    d = alpha / len(looks)
    x = rng.choice([1, -1, 0], p=[pp, pm, 1 - pp - pm], size=(draws, looks[-1]))
    kp, km = np.cumsum(x == 1, axis=1), np.cumsum(x == -1, axis=1)
    miss = 0
    for i in range(draws):
        miss += any(population_bounds(n, int(kp[i, n - 1]), int(km[i, n - 1]), d, d)[1] < pp - pm - 1e-12 for n in looks)
    assert miss / draws <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / draws)


def test_error_spending_sums_to_at_most_alpha():
    P = rule_params(RULE)
    sizes = {"STEM": 1161, "Humanities": 1781, "Social Sciences": 1130, "Other": 1142}  # the real label-set pools
    deltas = look_deltas(sizes, P)
    hi, lo = total_spent(deltas, P)
    assert hi <= P["alpha_false_accept"] + 1e-12 and lo <= P["alpha_false_reject"] + 1e-12
    assert hi > 0.999 * P["alpha_false_accept"]  # uniform spending uses the whole budget, no more
    assert all(v["looks"][0] == P["min_sample"] and v["looks"][-1] == sizes[c] for c, v in deltas.items())


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("popgate")
    make_live_smoke_data(tmp / "data-live")
    base = ["--set", f"experiment.data_dir={(tmp / 'data-live').as_posix()}", "--set", "reliability.restart_backoff_s=[0.2]"]
    for name, extra in (("top1", ["gate.capability.tolerance=0.6", "ranked_search.max_proposals=4"]),
                        ("top2", ["ranked_search.fallback_k=2"])):
        rc, out = cli(["supervise", "--config", str(GL), "--set", f"experiment.runs_dir={(tmp / name).as_posix()}", *base,
                       *[x for e in extra for x in ("--set", e)]])
        assert rc == 0, out
    return tmp


def test_redecisions_are_deterministic_and_replay_the_live_gate(smoke):
    from capprune.config import data_dir, load_config
    from capprune.data.prepare import PreparedData
    from capprune.model import tokenizer_id
    from capprune.population_regate import _load_cpu_model, regate_run

    cfg = load_config(GL, [f"experiment.data_dir={(smoke / 'data-live').as_posix()}"])
    data = PreparedData(data_dir(cfg), tokenizer_id(cfg))
    P = {**rule_params(RULE), "min_sample": 16, "batch_size": 16}  # the smoke gate's reveal schedule
    for name in ("top1", "top2"):
        rd = smoke / name / "gradient_top1__t00__s0"
        rc = json.loads((rd / "run.json").read_text())["config"]
        rc["_config_dir"] = str(ROOT / "configs")
        _, units = _load_cpu_model(rc)
        a, b = regate_run(rd, P, data, units, None), regate_run(rd, P, data, units, None)
        assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
        att = a["attempts"]
        assert att and all(r["counts_match_journal"] and r["live_decision_reproduced"] for r in att)
        assert all(c["matches_journal"] for c in a["mask_checks"] if "matches_journal" in c)
        assert all(r["population"]["questions"] <= r["live_questions"] for r in att)
        if name == "top2":
            assert any(r["rank"] == 2 for r in att)  # the fallback candidate is re-decided too


def test_rule_must_be_registered_and_unchanged(tmp_path):
    from capprune.config import load_config
    from capprune.population_regate import check_rule, register

    rule = tmp_path / "rule.md"
    shutil.copy(RULE, rule)
    cfg = load_config(ROOT / "configs" / "population_regate.yaml",
                      [f"population_regate.rule_file={rule.as_posix()}", f"population_regate.out_dir={(tmp_path / 'out').as_posix()}"])
    register(cfg)
    P, reg = check_rule(cfg)
    assert P["tolerance"] == 0.03 and reg["sha256_lf_normalised"]
    rule.write_text(rule.read_text(encoding="utf-8").replace('"tolerance": 0.03', '"tolerance": 0.04'), encoding="utf-8")
    with pytest.raises(SystemExit):
        check_rule(cfg)
