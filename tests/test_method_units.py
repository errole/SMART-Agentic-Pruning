"""Unit tests for pruning operations, the domain-testing statistics and the controller."""
import math

import numpy as np
import pytest
import torch

from capprune.config import load_config, run_config
from capprune.model import build_units, load_model
from capprune.pruning.masks import DenseStore, PruneState
from capprune.pruning.wanda import nm_mask, raise_unit, structured_prune_count, unstructured_mask
from capprune.search.controller import (HeuristicController, LLMController, ParseError, parse_response, sanitize)
from capprune.search.testing import FAIL, PASS, Limits, allocate, loss_limit, sequential_test
from capprune.utils.cache import KVCache
from capprune.utils.runtime import ControllerUnavailable
from conftest import SMOKE, endpoint_override, free_port


# ----------------------------------------------------------------------------- pruning
def test_unstructured_per_row_and_monotone():
    torch.manual_seed(0)
    s = torch.rand(8, 40)
    m1 = unstructured_mask(s, 0.25)
    assert ((~m1).sum(1) == 10).all()
    s2 = s.masked_fill(~m1, -1.0)
    m2 = unstructured_mask(s2, 0.5) & m1
    assert ((~m2).sum(1) == 20).all() and not (m2 & ~m1).any()  # superset of pruned weights


def test_nm_pattern_is_exact():
    s = torch.rand(6, 32)
    m = nm_mask(s, 2, 4)
    assert (m.view(6, 8, 4).sum(-1) == 2).all()


def test_structured_counts_are_aligned():
    assert structured_prune_count(0.30, 6144, 64, 0.875) % 64 == 0
    assert structured_prune_count(0.99, 6144, 64, 0.875) == int(0.875 * 6144 / 64) * 64
    assert structured_prune_count(0.001, 6144, 64, 0.875) == 0


@pytest.mark.parametrize("mode", ["unstructured", "nm", "structured"])
def test_masks_apply_save_load_and_materialize(smoke_assets, tmp_path, mode):
    from capprune.pruning.materialize import materialize

    cfg = load_config(SMOKE, [f"pruning.mode={mode}"])
    rc = run_config(cfg, "cap_prune", 0.2, 0)
    model = load_model(rc)
    units = build_units(model, mode, rc["pruning"]["eligible"])
    dense = DenseStore(units)
    st = PruneState(units, mode)
    act = {u.name: torch.rand((u.linears["down"] if mode == "structured" else u.linears["w"]).weight.shape[1]) + 0.1
           for u in units}
    for u in units[:5]:
        assert raise_unit(st, u.name, 0.3 if mode != "nm" else 0.5, act, rc)
    st.apply(dense)
    rho = st.global_sparsity()
    assert 0 < rho < 0.5
    st.save(tmp_path / "m.safetensors")
    st2 = PruneState(units, mode)
    st2.load(tmp_path / "m.safetensors")
    assert st2.fingerprint() == st.fingerprint() and abs(st2.global_sparsity() - rho) < 1e-12
    ids = torch.randint(5, 200, (2, 24))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
        m2, info = materialize(model, st, rc)  # nm on CPU falls back to dense - logits must still match
        out = m2(input_ids=ids).logits
    assert torch.allclose(ref, out, atol=1e-4), f"materialized {mode} model differs from masked model"
    if mode == "structured":
        assert info["channels_removed"] > 0
        assert sum(p.numel() for p in m2.parameters()) < sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- statistics
def _pool(domains, n, acc_dense, acc_cand, seed=0):
    rng = np.random.default_rng(seed)
    pool, dense, cand = {}, {}, {}
    for d in domains:
        pool[d] = [{"qid": f"{d}/{i}", "domain": d} for i in range(n)]
        for q in pool[d]:
            dc = rng.random() < acc_dense
            dense[q["qid"]] = dc
            # candidate: keeps dense answer except it loses (acc_dense - acc_cand[d]) of correct ones
            lose = rng.random() < max(0.0, (acc_dense - acc_cand[d]) / acc_dense)
            cand[q["qid"]] = dc and not lose
    return pool, dense, cand


def _run(policy, acc_cand, budget=800, seed=0):
    domains = list(acc_cand)
    pool, dense, cand = _pool(domains, 400, 0.6, acc_cand, seed)
    return sequential_test(domains=domains, pool=pool, dense_correct=dense,
                           evaluate=lambda qs: {q["qid"]: cand[q["qid"]] for q in qs}, policy=policy,
                           step_budget=budget, min_per_domain=30, round_size=120,
                           limits=Limits(0.08, 0.06, 0.02), accept_max=0.2, reject_min=0.8,
                           prior=(0.5, 4.0, 0.5), mc_samples=3000, alpha=0.25, seed=seed, iteration=1)


def test_harmless_step_passes_and_harmful_step_fails_early():
    safe = _run("cap", {f"d{i}": 0.6 for i in range(6)})
    assert safe.decision == PASS
    bad = _run("cap", {**{f"d{i}": 0.6 for i in range(5)}, "d5": 0.3})
    assert bad.decision == FAIL and bad.early_stop and bad.used < 800
    assert max(bad.domains, key=lambda d: bad.domains[d]["mean"]) == "d5"


def test_cap_allocation_targets_damaged_domain_and_keeps_min_coverage():
    res = _run("cap", {**{f"d{i}": 0.6 for i in range(5)}, "d5": 0.52}, budget=800)
    n = {d: t["lost"] + t["agree"] + t["gained"] for d, t in res.tallies.items()}
    assert min(n.values()) >= 30  # every domain was checked
    assert n["d5"] == max(n.values())  # most questions went to the possibly damaged domain


def test_policies_respect_budget_and_are_deterministic():
    acc = {f"d{i}": 0.6 - 0.03 * (i == 2) for i in range(6)}
    for pol in ("balanced", "loss_ranked", "neyman", "cap"):
        a, b = _run(pol, acc, budget=500), _run(pol, acc, budget=500)
        assert a.used <= 500 and a.to_json() == b.to_json()


def test_loss_limit_schedule():
    assert loss_limit(0.0, 0.4, 0.10, 0.02) == 0.02
    assert math.isclose(loss_limit(0.2, 0.4, 0.10, 0.02), 0.05)
    assert loss_limit(0.5, 0.4, 0.10, 0.02) == 0.10


def test_allocate_largest_remainder_exact():
    summ = {d: {"mean": 0.0, "sd": 0.05, "p_violate": 0.1} for d in "abc"}
    a = allocate("balanced", list("abc"), summ, {}, {"a": 100, "b": 100, "c": 1}, 31, 0, 0.25)
    assert sum(a.values()) == 31 and a["c"] == 1


# ----------------------------------------------------------------------------- controller
def test_parse_response_variants():
    good = '{"reasoning": "r", "stop_pruning": false, "layer_decisions": [{"layer": "L1.q_proj", "additional_sparsity": 0.05}]}'
    assert parse_response(good)["layer_decisions"][0]["layer"] == "L1.q_proj"
    assert parse_response("<think>hmm {not json}</think>\n```json\n" + good + "\n```")["reasoning"] == "r"
    assert parse_response('{"stop_pruning": "true", "layer_decisions": []}')["stop_pruning"] is True
    with pytest.raises(ParseError):
        parse_response("I would prune layer 3.")


def test_sanitize_clamps_ignores_unknown_and_prevents_overshoot(smoke_assets):
    cfg = load_config(SMOKE)
    rc = run_config(cfg, "cap_prune", 0.05, 0)
    model = load_model(rc)
    st = PruneState(build_units(model, "unstructured", rc["pruning"]["eligible"]), "unstructured")
    names = [u.name for u in st.units]
    parsed = {"layer_decisions": [{"layer": n, "additional_sparsity": 0.5} for n in names] +
                                 [{"layer": "L99.q_proj", "additional_sparsity": 0.1}]}
    dec, notes = sanitize(parsed, st, rc, target=0.05)
    assert all(d <= 0.15 for _, d in dec)
    assert any("unknown" in n for n in notes) and any("scaled" in n for n in notes)
    assert sum(st.share(n) * d for n, d in dec) <= 0.05 + rc["search"]["target_tolerance"] + 1e-9


def _llm(cfg_over, tmp_path):
    cfg = load_config(SMOKE, cfg_over)
    rc = run_config(cfg, "cap_prune", 0.2, 0)
    return LLMController(rc, KVCache(tmp_path / "ctrl.sqlite"))


def test_llm_controller_retries_repairs_and_replays(tmp_path):
    from mock_controller import serve

    port = free_port()
    srv, state = serve(port, fail_first=2, garbage_first=1)
    try:
        c = _llm([endpoint_override(port), "controller.retries=4"], tmp_path)
        user = "name | share | z_sens | z_grad | sparsity\nL0.q_proj | 0.1000 | -1.00 | -1.00 | 0.000"
        parsed, meta = c.propose(1, "system", user)
        assert parsed["layer_decisions"][0]["layer"] == "L0.q_proj" and not meta["replayed"]
        calls = state.calls
        parsed2, meta2 = c.propose(1, "system", user)  # resume: replayed from cache, no new call
        assert meta2["replayed"] and parsed2 == parsed and state.calls == calls
    finally:
        srv.shutdown()


def test_llm_controller_failover_and_unavailable(tmp_path):
    from mock_controller import serve

    port = free_port()
    srv, _ = serve(port)
    try:
        dead = free_port()
        eps = (f"controller.endpoints=[{{name: dead, api: openai, base_url: 'http://127.0.0.1:{dead}/v1'}}, "
               f"{{name: mock, api: openai, base_url: 'http://127.0.0.1:{port}/v1'}}]")
        c = _llm([eps], tmp_path)
        _, meta = c.propose(1, "s", "L0.q_proj | 0.1 | -1.0 | -1.0 | 0.000")
        assert meta["endpoint"] == "mock"
    finally:
        srv.shutdown()
    c = _llm([endpoint_override(free_port())], tmp_path / "x")
    with pytest.raises(ControllerUnavailable):
        c.propose(2, "s", "u")
    # a permanent error (missing API key) is not retried for minutes: fails fast
    import time
    eps = "controller.endpoints=[{name: k, api: openai, base_url: 'http://127.0.0.1:9/v1', api_key_env: CAPPRUNE_TEST_NO_SUCH_KEY}]"
    c = _llm([eps, "controller.retries=6", "controller.backoff_s=[30]"], tmp_path / "y")
    t0 = time.time()
    with pytest.raises(ControllerUnavailable):
        c.propose(3, "s", "u")
    assert time.time() - t0 < 5


def test_model_switch_guard(tmp_path):
    eps = "controller.endpoints=[{name: other, api: openai, base_url: 'http://x/v1', model: other-model}]"
    with pytest.raises(ValueError):
        _llm([eps], tmp_path)


def test_heuristic_controller_adapts_step():
    h = HeuristicController({"controller": {"heuristic": {}}})
    prof = [{"name": f"u{i}", "share": 0.05, "z_s": i - 10.0, "z_g": 0.0, "sparsity": 0.0} for i in range(20)]
    ctx = {"target": 0.4, "rho": 0.0, "profiles": prof, "max_unit": 0.9, "mode": "unstructured", "nm": (2, 4)}
    p0 = h.propose(ctx, [])
    p1 = h.propose(ctx, [{"outcome": "rollback", "decisions": [("u0", 0.05)]}])
    assert p0["layer_decisions"][0]["layer"] == "u0"
    assert "u0" not in [d["layer"] for d in p1["layer_decisions"]]  # tabu after rollback
    assert len(p1["layer_decisions"]) < len(p0["layer_decisions"])  # smaller step after rollback


def test_gate_operating_characteristics_at_real_config_defaults():
    """The defaults in configs/base.yaml must let harmless steps through and stop harmful ones.
    (The smoke config loosens the limits, so only this test guards the real settings.)"""
    import yaml
    from conftest import ROOT

    t = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))["testing"]
    pool_n = dict(mathematics=425, physical=377, life=838, computing=223, history=372, law=1510, social=1231, business=641)
    doms = list(pool_n)

    def rate(flip, hurt, eps, rule, n_trials=40):
        out = []
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            pool, dense, cand = {}, {}, {}
            for d in doms:
                pool[d] = [{"qid": f"{d}/{i}", "domain": d} for i in range(pool_n[d])]
                loss = hurt if d == "computing" else 0.0
                for q in pool[d]:
                    dc = rng.random() < 0.55
                    u = rng.random()
                    c = (not (u < (flip / 2 + loss) / 0.55)) if dc else (u < (flip / 2) / 0.45)
                    dense[q["qid"]], cand[q["qid"]] = dc, c
            r = sequential_test(domains=doms, pool=pool, dense_correct=dense,
                                evaluate=lambda qs: {q["qid"]: cand[q["qid"]] for q in qs}, policy="cap",
                                step_budget=t["step_budget"], min_per_domain=t["min_per_domain"],
                                round_size=t["round_size"], limits=Limits(eps, t["imbalance_limit"], t["imbalance_min_loss"]),
                                accept_max=t["accept_max_violation"], reject_min=t["reject_min_violation"], rule=rule,
                                min_fail_n=t["min_fail_n"], prior=tuple(t["prior"]), mc_samples=1500, alpha=t["alpha"],
                                seed=seed, iteration=1)
            out.append(r.decision)
        return out.count(PASS) / n_trials, out.count(FAIL) / n_trials

    floor, final20 = t["loss_limit"]["floor"], t["loss_limit"]["final"][0.2]
    assert t["decision_rule"] == "per_domain"
    p, f = rate(0.04, 0.0, floor, "per_domain")       # harmless early step
    assert p >= 0.85 and f <= 0.08, (p, f)
    f_per_domain = f
    p, f = rate(0.10, 0.0, final20, "per_domain")     # harmless step near the 20 % target
    assert p >= 0.85 and f <= 0.08, (p, f)
    p, f = rate(0.04, 0.10, floor, "per_domain")      # one domain truly 10 points down
    assert f >= 0.80 and p <= 0.05, (p, f)
    p, f = rate(0.04, 0.0, floor, "familywise")       # the old rule wrongly rejects far more harmless steps
    assert f >= f_per_domain + 0.10, (f, f_per_domain)
