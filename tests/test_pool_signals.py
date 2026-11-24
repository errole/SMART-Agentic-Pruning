"""Phase 2 building blocks on CPU: the log-probability scorer reproduces the gate scorer, and the signals behave."""
from __future__ import annotations

import numpy as np

from conftest import ROOT


def test_logprob_scorer_matches_the_gate_scorer(smoke_assets):
    from capprune.config import data_dir
    from capprune.data.prepare import PreparedData
    from capprune.eval.mmlu import MMLUScorer
    from capprune.model import load_model, load_tokenizer, tokenizer_id
    from capprune.pool_signals import LogprobScorer

    cfg = smoke_assets
    data = PreparedData(data_dir(cfg), tokenizer_id(cfg))
    tok, model = load_tokenizer(cfg), load_model(cfg)
    rows = data.mmlu_probe[:60]
    ref = MMLUScorer(model, tok, data.mmlu_fewshot, 0, 256, {}, "k", 2048).score(rows)
    got, budget = LogprobScorer(model, tok, data.mmlu_fewshot, 0, 256, 2048).score(rows, 2048)
    assert budget == 2048 and set(got) == set(ref)
    for q, r in ref.items():
        assert got[q]["pred"] == r["pred"] and got[q]["correct"] == int(r["correct"])
        assert len(got[q]["logp"]) == 4 and max(got[q]["logp"]) <= 0.0
        assert int(np.argmax(got[q]["logp"])) == r["pred"]


def test_signals_and_correlations():
    from capprune.pool_signals import corr, signals

    rng = np.random.default_rng(0)
    lp = np.log(rng.dirichlet(np.ones(4), size=200))
    ans = rng.integers(0, 4, size=200)
    c = (lp.argmax(1) == ans).astype(np.int8)
    same = signals(lp, lp, ans, c, c)
    assert same["S0"] == 0 and same["S0b"] == 0 and abs(same["S1"]) < 1e-12 and abs(same["S2"]) < 1e-12
    worse = lp.copy()
    worse[np.arange(200), ans] -= 1.0  # the pruned model is less sure of the right answer
    cw = (worse.argmax(1) == ans).astype(np.int8)
    s = signals(lp, worse, ans, c, cw)
    assert s["S1"] > 0.9 and s["S2"] > 0 and s["S0"] >= 0
    x = np.arange(7.0)
    assert abs(corr(x, 2 * x + 1)["pearson"] - 1) < 1e-12 and abs(corr(x, x ** 3)["spearman"] - 1) < 1e-12
    assert abs(corr(x, -x)["spearman"] + 1) < 1e-12 and corr(x, np.ones(7))["pearson"] is None
