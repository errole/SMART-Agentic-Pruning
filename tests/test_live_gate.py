"""Live capability gate inside the pruning-and-rollback loop (tiny Qwen3, CPU, scripted mock controller)."""
from __future__ import annotations

import json

import pytest

from capprune.adaptive.live import attempt_order_seed
from capprune.config import load_config
from capprune.utils.runtime import ConfigError
from conftest import ROOT, endpoint_override, free_port
from live_smoke import LIVE_SMOKE, make_live_smoke_data
from mock_controller import LIVE_SCRIPT, serve
from test_recovery import cli


def _attempts(run_dir):
    last = {}
    for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") == "iteration":
            last[r["iteration"]] = r
    return [last[k] for k in sorted(last)]


@pytest.fixture(scope="module")
def live_runs(tmp_path_factory):
    """Dense + all four arms under the supervisor, then live-bench and live-report."""
    tmp = tmp_path_factory.mktemp("live")
    data = make_live_smoke_data(tmp / "data-live")
    port = free_port()
    srv, _ = serve(port, script=LIVE_SCRIPT)
    sets = ["--set", endpoint_override(port), "--set", f"experiment.runs_dir={(tmp / 'runs').as_posix()}",
            "--set", f"experiment.data_dir={data.as_posix()}", "--set", "reliability.restart_backoff_s=[0.2]"]
    try:
        rc, out = cli(["supervise", "--config", str(LIVE_SMOKE), *sets])
        assert rc == 0, out
    finally:
        srv.shutdown()
    rc, out = cli(["live-bench", "--config", str(LIVE_SMOKE), *sets])
    assert rc == 0, out
    rc, out = cli(["live-report", "--config", str(LIVE_SMOKE), *sets])
    assert rc == 0, out
    return tmp / "runs"


@pytest.mark.parametrize("arm", ["llm_cap", "llm_cap_rr"])
def test_accept_reject_inconclusive_and_both_rollback_paths(live_runs, arm):
    att = _attempts(live_runs / f"{arm}__t50__s0")
    assert [a["gate_decision"] for a in att] == ["accept", "reject", "inconclusive", "accept"]
    first = att[0]["trial_id"]
    # the accepted trial became the parent; after the reject AND after the inconclusive the next attempt
    # starts from that same parent, verified bit-for-bit
    assert [a["parent_id"] for a in att] == ["dense", first, first, first]
    assert all(a["parent_verified"] for a in att)
    assert att[1]["rollback_verified"] is True and att[2]["rollback_verified"] is True
    assert att[2]["gate"]["questions"] > 360 - 16  # inconclusive = the budget ran out
    assert att[1]["gate"]["domains"]["Other"]["status"] == "fail"
    seeds = [a["gate"]["order_seed"] for a in att]
    assert len(set(seeds)) == 4  # a fresh question order for every candidate evaluation
    summary = json.loads((live_runs / f"{arm}__t50__s0" / "final" / "summary.json").read_text())
    assert summary["gate_counts"] == {"accept": 2, "reject": 1, "inconclusive": 1, "noop": 0}


def test_only_allocation_differs_between_threshold_aware_and_round_robin(live_runs):
    b, d = _attempts(live_runs / "llm_cap__t50__s0"), _attempts(live_runs / "llm_cap_rr__t50__s0")
    assert [x["trial_id"] for x in b] == [x["trial_id"] for x in d]          # same proposals, same trials
    assert [x["gate"]["order_seed"] for x in b] == [x["gate"]["order_seed"] for x in d]
    for i in (0, 3):  # an ACCEPT costs the same under both allocation policies
        assert b[i]["gate"]["questions"] == d[i]["gate"]["questions"]
    assert b[1]["gate"]["questions"] < d[1]["gate"]["questions"]  # threshold-aware reaches the reject sooner


def test_ppl_arm_keeps_the_paper_gate(live_runs):
    att = _attempts(live_runs / "llm_ppl__t50__s0")
    assert all(a["gate"]["kind"] == "ppl" for a in att)
    assert all(a["prompt_variant"] == "ppl" for a in att)
    cap = _attempts(live_runs / "llm_cap__t50__s0")
    assert all(a["prompt_variant"] == "capability" for a in cap)
    assert cap[1]["feedback"]["domain_note"] and att[0]["feedback"]["domain_note"] is None


def test_greedy_arm_uses_the_heuristic_with_the_same_gate(live_runs):
    att = _attempts(live_runs / "greedy_cap__t50__s0")
    assert att and all(a["controller"]["source"] == "heuristic" and a["gate"]["kind"] in ("capability", "ppl", "noop")
                       for a in att)
    assert any(a["gate"]["kind"] == "capability" for a in att)


def test_bench_and_report(live_runs):
    sessions = sorted((live_runs / "_live").glob("bench-*.json"))
    assert len(sessions) == 1  # one file per benchmark session; earlier sessions are never overwritten
    bench = json.loads(sessions[0].read_text())
    arms = {m["arm"]: m for m in bench["models"]}
    assert set(arms) == {"dense", "greedy_cap", "llm_ppl", "llm_cap", "llm_cap_rr"}
    assert arms["llm_cap"]["removed_parameters"] > 0
    assert arms["llm_cap"]["decode_ratio"]["verdict"] in ("faster", "slower", "inconclusive")
    md = (live_runs / "_live" / "report.md").read_text(encoding="utf-8")
    for arm in ("greedy_cap", "llm_ppl", "llm_cap", "llm_cap_rr"):
        assert f"`{arm}`" in md
    assert "Scope of the capability bounds: per candidate" in md
    assert "| 4/4" in md  # valid LLM proposals per arm (scripted mock: every reply valid)


def test_report_reference_arm_from_another_runs_dir(live_runs, tmp_path):
    from capprune.config import load_config
    from capprune.live_report import collect

    cfg = load_config(LIVE_SMOKE, [f"experiment.runs_dir={live_runs.as_posix()}",
                                   f"experiment.data_dir={(live_runs.parent / 'data-live').as_posix()}",
                                   f"live_report.reference_arms=[{{method: greedy_cap, runs_dir: '{live_runs.as_posix()}', label: ref}}]"])
    R = collect(cfg)
    own = next(a for a in R["arms"] if a["arm"] == "greedy_cap")
    ref = next(a for a in R["arms"] if a["arm"] == "greedy_cap (ref)")
    assert ref["reference"] == "ref" and ref["rho"] == own["rho"] and ref["audit"] == own["audit"]
    assert ref["bench_session"].startswith("ref: ")  # its speed comes from its own session and dense reference


def _prompt_db(path, iteration=2):
    from capprune.utils.cache import KVCache

    user = (f"Iteration {iteration} (at most 10).\nCurrent global sparsity: 0.0000.\n\nUnit profiles (sorted by z_sens, safest first):\n"
            "name | share | z_sens | z_grad | sparsity\n" + "\n".join(f"L{i}.mlp | 0.0357 | -1.00 | -1.00 | 0.000" for i in range(28)))
    kv = KVCache(path)
    kv.put(f"iter{iteration:06d}/a0", {"system": "Respond with ONLY a JSON object.", "user": user,
                                        "meta": {"model": "some-model"}, "parsed": {}, "raw": ""})
    kv.close()
    return path


@pytest.mark.parametrize("script,ok", [(None, True), ({2: [{"layer": "L99.mlp", "additional_sparsity": 0.5}]}, False)])
def test_controller_check_validates_one_real_proposal(tmp_path, script, ok):
    from capprune.controller_check import check

    db = _prompt_db(tmp_path / "run" / "controller.sqlite")
    before = db.read_bytes()
    port = free_port()
    srv, _ = serve(port, script=script)
    try:
        cfg = load_config(ROOT / "configs" / "live_gate_local.yaml", [endpoint_override(port)])
        res = check(cfg, "llm_cap", db, 2, tmp_path / "out")
    finally:
        srv.shutdown()
    assert res["ok"] is ok
    assert res["checks"]["json_complete_first_try"]
    if not ok:
        assert res["checks"]["unknown_units"] == ["L99.mlp"] and res["checks"]["increments_outside_range"]
    assert db.read_bytes() == before and not (tmp_path / "run" / "controller.sqlite-wal").exists()  # source untouched


def test_gap_diagnostic_on_a_finished_arm(live_runs, tmp_path):
    from capprune.gap_diagnostic import duplicate_items

    data = live_runs.parent / "data-live"
    out = tmp_path / "gap"
    rc, log = cli(["gap-diagnostic", "--config", str(LIVE_SMOKE), "--set", f"experiment.runs_dir={live_runs.as_posix()}",
                   "--set", f"experiment.data_dir={data.as_posix()}", "--arm", "greedy_cap", "--out", str(out),
                   "--bootstrap", "200"])
    assert rc == 0, log
    s = json.loads((out / "summary.json").read_text())
    assert set(s["table"]) == {"pool|0", "pool|5", "holdout|0", "holdout|5"}
    prov = s["provenance"]
    assert prov["audit_reused"] is True  # the final audit's cache keys carry the checkpoint and match item by item
    assert all(v["agree"] == v["items"] for v in prov["audit_recompute_check"].values())
    items = [json.loads(l) for l in (out / "per_item.jsonl").read_text().splitlines()]
    assert {r["split"] for r in items} == {"pool", "holdout"} and all(
        set(r) >= {"dense_0", "pruned_0", "dense_5", "pruned_5"} for r in items)
    for cell in s["table"].values():  # paired loss = dense accuracy - pruned accuracy on the same items
        assert abs(cell["overall"] - (cell["dense_acc"] - cell["pruned_acc"])) < 1e-12
    rows = {"pool": [{"qid": "a", "question": "Q ", "choices": ["x", "y"]}, {"qid": "b", "question": "Q", "choices": ["x", "z"]}]}
    assert duplicate_items(rows, [{"question": "Q", "choices": ["x ", "y"]}]) == ["a"]  # question AND choices must match


def test_fixed_state_comparison_end_to_end(live_runs, tmp_path):
    """Frozen states from a finished greedy run; two mock controllers (one valid, one invalid/abstaining)."""
    from capprune.fixed_state import normalize_free

    port = free_port()
    srv, _ = serve(port)
    cfg = ROOT / "configs" / "fixed_state_smoke.yaml"
    sets = ["--set", endpoint_override(port), "--set", f"experiment.data_dir={(live_runs.parent / 'data-live').as_posix()}",
            "--set", f"fixed_state.state_source={(live_runs / 'greedy_cap__t50__s0').as_posix()}",
            "--set", f"fixed_state.out_dir={(tmp_path / 'fs').as_posix()}"]
    try:
        for phase in (["prepare"], ["query", "--model", "mock-a"], ["query", "--model", "mock-bad"], ["score"], ["report"]):
            rc, out = cli(["fixed-state", phase[0], "--config", str(cfg), *sets, *phase[1:]])
            assert rc == 0, out
    finally:
        srv.shutdown()
    info = json.loads((tmp_path / "fs" / "states.json").read_text())
    assert [s["name"] for s in info["states"]][0] == "dense" and len(info["states"]) >= 2
    assert all(c["stored_mask_equals_replay"] for c in info["mask_replay_checks"])  # rebuilt states == stored masks
    scores = [json.loads(l) for l in (tmp_path / "fs" / "scores.jsonl").read_text().splitlines()]
    assert len({(r["state"], r["action_key"]) for r in scores}) == len(scores)  # every pair scored once
    assert len({r["channels"] for r in scores}) == 1  # identical realised removal for every action
    props = [json.loads(l) for f in ("proposals-mock-a.jsonl", "proposals-mock-bad.jsonl")
             for l in (tmp_path / "fs" / f).read_text().splitlines()]
    assert {r["outcome"] for r in props} == {"valid", "abstain", "invalid"}
    s = json.loads((tmp_path / "fs" / "summary.json").read_text())
    assert s["summary"]["greedy shortlist"]["valid"] == len(info["states"])
    # free-form normalisation: proportional, exact total, per-unit cap respected, unreachable -> None
    prof = [{"name": f"L{i}.mlp", "sparsity": 0.0} for i in range(4)]
    geo = {"B": 96, "maxb": 9, "cap_blocks": 84}
    assert normalize_free([("L0.mlp", 0.1), ("L1.mlp", 0.1), ("L2.mlp", 0.05)], prof, geo, 20) == {"L0.mlp": 8, "L1.mlp": 8, "L2.mlp": 4}
    assert sum(normalize_free([("L0.mlp", 0.1), ("L1.mlp", 0.1), ("L2.mlp", 0.1)], prof, geo, 20).values()) == 20
    assert normalize_free([("L0.mlp", 0.1), ("L1.mlp", 0.1)], prof, geo, 20) is None  # 2 units x 9 blocks < 20


def test_ranker_splits_by_trajectory_uses_only_prescoring_features_and_charges_probes(live_runs, tmp_path):
    import copy

    import numpy as np

    from capprune import ranker

    cfg = ROOT / "configs" / "ranker_smoke.yaml"
    sets = ["--set", f"experiment.data_dir={(live_runs.parent / 'data-live').as_posix()}",
            "--set", f"ranker.out_dir={(tmp_path / 'rk').as_posix()}", "--set", "ranker.dev_source=null",
            "--set", "ranker.trajectories=4", "--set", "gate.capability.tolerance=0.5"]
    for phase in ("collect", "evaluate"):
        rc, out = cli(["ranker", phase, "--config", str(cfg), *sets])
        assert rc == 0, out
    D = json.loads((tmp_path / "rk" / "decisions.json").read_text())
    E = json.loads((tmp_path / "rk" / "evaluation.json").read_text())
    decs = D["decisions"]
    # 1) trajectory-level split: every non-dense parent mask belongs to exactly one trajectory; the shared dense
    #    parent is its own group; every fold tests one whole trajectory that is absent from its training data
    owner = {}
    for d in decs:
        if d["group"] == "new":
            assert owner.setdefault(d["parent_mask"], d["trajectory"]) == d["trajectory"]
    d0 = {d["parent_mask"] for d in decs if d["group"] == "D0"}
    assert len(d0) == 1 and not (d0 & set(owner))
    folds = [f for f in E["folds"] if f.startswith("T")]
    assert folds and all({r["trajectory"] for r in E["results"] if r["fold"] == f} == {f} for f in folds)
    bad = copy.deepcopy(decs)
    new = [d for d in bad if d["group"] == "new"]
    if len(new) >= 2:  # a parent mask on both sides must be refused
        new[1]["parent_mask"], new[1]["trajectory"] = new[0]["parent_mask"], "T99"
        (tmp_path / "rk2").mkdir()
        (tmp_path / "rk2" / "decisions.json").write_text(json.dumps({**D, "decisions": bad}))
        from capprune.config import load_config
        c2 = load_config(cfg, [f"ranker.out_dir={(tmp_path / 'rk2').as_posix()}"])
        with pytest.raises(RuntimeError, match="both sides"):
            ranker.evaluate(c2)
    # 2) features exist before scoring: the pre-scoring ranking ignores labels and probe answers entirely
    assert not set(ranker.FEATURES) & {"label", "worst", "probe", "probe_worst", "gate"}
    test_d = next(d for d in decs if d["group"] == "new")
    train = [d for d in decs if d["trajectory"] != test_d["trajectory"]]
    model = ranker.fit_selector(train, D["domains"], False, [1.0])
    before = ranker._rank("ridge", test_d, model, D["domains"])
    scrambled = copy.deepcopy(test_d)
    for c in scrambled["candidates"]:
        c["label"] = {k: np.random.default_rng(0).random() for k in c["label"]}
        c["probe"] = {k: 0.9 for k in c["probe"]}
        c["worst"] = c["probe_worst"] = 0.5
    assert ranker._rank("ridge", scrambled, model, D["domains"]) == before
    assert ranker._rank("gradient", scrambled, None, D["domains"]) == ranker._rank("gradient", test_d, None, D["domains"])
    # 3) probe cost: every candidate is probed, chosen or not, and the probe questions are charged
    for r in E["results"]:
        if r["selector"] in ("probe", "ridge+probe"):
            assert r["probe_q"] == r["n_candidates"] * D["probe_questions"]
            assert r["questions"] > r["probe_q"] and r["probe_s"] > 0
        elif r["selector"] in ("greedy", "gradient", "ridge"):
            assert r["probe_q"] == 0


# ----------------------------------------------------------------------------- units
def test_question_order_seed_depends_only_on_the_attempt():
    assert attempt_order_seed(0, 0, 3) == attempt_order_seed(0, 0, 3)
    assert len({attempt_order_seed(0, 0, t) for t in range(1, 50)}) == 49


def test_capability_gate_config_is_validated():
    with pytest.raises(ConfigError):
        load_config(ROOT / "configs" / "live_gate.yaml", ["gate.kind=capability", "testing.policy=cap"])
    with pytest.raises(ConfigError):
        load_config(ROOT / "configs" / "live_gate.yaml", ["gate.kind=capability", "gate.capability.policy=best"])
    load_config(ROOT / "configs" / "live_gate.yaml", ["gate.kind=capability"])


def test_gate_design_on_fixture(tmp_path):
    from capprune.adaptive.config import load_adaptive
    from capprune.adaptive.fixture import write_fixture
    from capprune.adaptive.gate_design import design

    cache = write_fixture(tmp_path / "dev")
    cfg = load_adaptive(ROOT / "configs" / "gate_dev.yaml", [
        f"cache_dir={cache.as_posix()}", f"out_dir={(tmp_path / 'out').as_posix()}", "design.orders=6",
        "design.validation_cache=null", "design.options.budget=[1500, 2550]", "design.min_sample=40", "design.batch_size=40",
        "design.pool_sizes={STEM: 600, Humanities: 900, Social Sciences: 500, Other: 550}"])
    p = design(cfg)
    s = json.loads((p.parent / "summary.json").read_text())
    assert s["locked"]["budget"] in (1500, 2550) and s["locked"]["spending"] in ("uniform", "power")
    for key, res in s["results"].items():  # the shifted 4-point profiles are never accepted at a 3-point limit
        assert all(res[n]["accept"] == 0 for n in s["fail_profiles"])
