"""Top-2 fallback, the stricter (margin) gate, and the boundary/calibration pipeline (tiny Qwen3, CPU).

At the smoke tolerance (15 points) the first gradient action is accepted and the second rejected; with top-2 the
parent is restored and its rank-2 action becomes the next proposal (it is rejected too: the tiny random model's
rank-2 actions are worse).  The rescue path is exercised with the CAPPRUNE_TEST_FORCE_REJECT hook.
"""
from __future__ import annotations

import json

import pytest

from conftest import ROOT
from live_smoke import make_live_smoke_data
from test_recovery import cli

CFG = ROOT / "configs" / "gradient_live_smoke.yaml"
BND = ROOT / "configs" / "gradient_boundary_smoke.yaml"
RID = "gradient_top1__t00__s0"


def _sets(tmp, name, extra=()):
    return ["--set", f"experiment.runs_dir={(tmp / name).as_posix()}",
            "--set", f"experiment.data_dir={(tmp / 'data-live').as_posix()}",
            "--set", f"gradient_live.out_dir={(tmp / name / '_analysis').as_posix()}",
            "--set", "reliability.restart_backoff_s=[0.2]", *[x for e in extra for x in ("--set", e)]]


def _attempts(run_dir):
    last = {}
    for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") == "iteration":
            last[r["iteration"]] = r
    return [last[k] for k in sorted(last)]


def _S(run_dir):
    from capprune.utils.checkpoint import CheckpointStore

    return CheckpointStore(run_dir).load_latest()[0]["search"]


def _run(tmp, name, extra=(), env=None):
    rc, out = cli(["supervise", "--config", str(CFG), *_sets(tmp, name, extra)], env=env)
    assert rc == 0, out
    return _attempts(tmp / name / RID), _S(tmp / name / RID)


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    t = tmp_path_factory.mktemp("top2")
    make_live_smoke_data(t / "data-live")
    return t


@pytest.fixture(scope="module")
def runs(tmp):
    top1 = _run(tmp, "top1")
    top2 = _run(tmp, "top2", ["ranked_search.fallback_k=2"])
    return top1, top2


def test_fallback_restores_the_parent_and_tries_rank2_of_the_same_shortlist(tmp, runs):
    from capprune.adaptive.live import attempt_order_seed

    (a1, s1), (a2, s2) = runs
    assert [a["gate_decision"] for a in a1] == ["accept", "reject"] and s1["stop_reason"] == "candidate_rejected"
    # identical to top-1 until the first failure
    for x, y in zip(a1, a2):
        assert (x["cand_id"], x["gate_decision"], x["gate"]["questions"]) == (y["cand_id"], y["gate_decision"], y["gate"]["questions"])
    trig, fb = a2[1], a2[2]
    assert trig["rank"] == 1 and trig["fallback_triggered"] and trig["rollback_verified"] is True
    assert fb["rank"] == 2 and fb["fallback_of"] == 2 and fb["profile_attempt"] == 2
    assert fb["parent_id"] == trig["parent_id"] and fb["parent_mask_hash"] == trig["parent_mask_hash"]
    assert fb["parent_verified"] is True  # the restored parent is bit-identical to the accepted checkpoint
    assert fb["shortlist"] == trig["shortlist"] and fb["chosen"]["action_key"] == trig["shortlist"][1]["action_key"]
    assert fb["gate"]["order_seed"] == attempt_order_seed(0, 0, 3) != trig["gate"]["order_seed"]
    assert not (tmp / "top2" / RID / "profiles" / "iter-000003.safetensors").exists()  # the parent's profile is reused
    assert fb["timings"]["profile_s"] < 5
    # both candidates count against the caps; the run ends when the top-2 are exhausted
    assert s2["stop_reason"] == "fallback_exhausted" and s2["state_id"] == a2[0]["cand_id"] and s2["rollbacks"] == 2
    assert s2["questions"] == sum(a["gate"]["questions"] for a in a2) and s2["iteration"] - 1 == len(a2) == 3
    f = s2["fallback"]
    assert (f["triggers"], f["rescued"], f["failed"]) == (1, 0, 1) and f["extra_questions"] == fb["gate"]["questions"]
    assert f["extra_evaluations"] == 2 * fb["gate"]["questions"]


def test_rank2_rescue_path(tmp):
    att, S = _run(tmp, "rescue", ["ranked_search.fallback_k=2", "gate.capability.tolerance=0.85",
                                  "ranked_search.max_proposals=4"], env={"CAPPRUNE_TEST_FORCE_REJECT": "2"})
    assert att[1]["forced_by_test_hook"] and att[1]["fallback_triggered"]
    assert att[2]["rank"] == 2 and att[2]["gate_decision"] == "accept" and att[2]["rescued"]
    assert att[3]["rank"] == 1 and att[3]["parent_id"] == att[2]["cand_id"]  # the rescued trial is the next parent
    assert S["fallback"]["rescued"] == 1 and S["fallback"]["triggers"] == 1


def test_caps_count_both_candidates(tmp, runs):
    (a1, _), (a2, _) = runs
    q2 = a2[0]["gate"]["questions"] + a2[1]["gate"]["questions"]
    # proposal cap reached at the triggering attempt: the queued rank-2 action is never gated
    att, S = _run(tmp, "cap-prop", ["ranked_search.fallback_k=2", "ranked_search.max_proposals=2"])
    assert len(att) == 2 and S["stop_reason"] == "proposal_cap" and S["fallback"]["blocked_by_cap"] == 1
    # question cap below the next initial sample: stop before the fallback
    att, S = _run(tmp, "cap-q0", ["ranked_search.fallback_k=2", f"ranked_search.question_cap={q2 + 10}"])
    assert len(att) == 2 and S["stop_reason"] == "question_cap" and S["fallback"]["blocked_by_cap"] == 1
    assert S["budget_stop"]["when"] == "before_attempt"
    # question cap inside the fallback attempt: a budget stop, not a fallback failure
    assert a2[2]["gate"]["questions"] > 64 + 8
    att, S = _run(tmp, "cap-q1", ["ranked_search.fallback_k=2", f"ranked_search.question_cap={q2 + 64 + 8}"])
    assert att[-1]["rank"] == 2 and att[-1]["gate_decision"] == "budget_stop" and att[-1]["gate"]["questions"] == 64
    assert S["stop_reason"] == "question_cap" and S["questions"] == q2 + 64 <= q2 + 72
    assert S["fallback"]["failed"] == 0 and S["fallback"]["extra_questions"] == 64


def test_stricter_threshold(tmp, runs):
    from capprune.config import load_config, run_config
    from capprune.gradient_live import make_gate

    (a1, _), _ = runs
    att, S = _run(tmp, "strict", ["ranked_search.gate_margin=0.05"])
    a = att[0]
    assert a["gate_margin"] == 0.05 and all(abs(v - 0.10) < 1e-12 for v in a["gate"]["tolerance"].values())
    assert a["trial_mask_hash"] == a1[0]["trial_mask_hash"] and a["gate"]["order_seed"] == a1[0]["gate"]["order_seed"]
    # the same trial on the same question order: stricter can only turn an ACCEPT into a REJECT/INCONCLUSIVE
    assert a1[0]["gate_decision"] == "accept"
    if a["gate_decision"] == "accept":
        assert all(v["hi"] <= 0.10 + 1e-12 for v in a["gate"]["domains"].values())
    else:
        assert any(v["hi"] > 0.10 for v in a["gate"]["domains"].values())
    rc = run_config(load_config(CFG), "gradient_top1", 0.0, 0)
    pool = [{"qid": f"q{i}", "category": c} for i, c in enumerate(["STEM", "Humanities", "Social Sciences", "Other"] * 30)]
    g = make_gate(rc, pool, margin=0.05)
    assert all(abs(v - 0.10) < 1e-12 for v in g.settings.tolerance.values())
    with pytest.raises(ValueError):
        make_gate(rc, pool, margin=0.15)


def test_calibration_rule():
    from capprune.gradient_boundary import apply_rule

    def st(name, meets, dec):  # dec: decisions at margins 0, 0.5, 1.0
        return {"name": name, "meets_audit": meets, "gate": {k: {"decision": d} for k, d in zip(("0.0", "0.5", "1.0"), dec)}}

    rows = [st("s1", True, ["accept"] * 3), st("s2", True, ["accept", "accept", "reject"]),
            st("s3", False, ["accept", "accept", "reject"]), st("s4", False, ["accept", "reject", "reject"])]
    r = apply_rule(rows, [0.0, 0.005, 0.01])
    assert r["chosen_margin_pts"] == 1.0 and r["conclusive"] and r["support"]["states_passing_and_meeting_audit"] == 1
    assert r["support"]["states_blocked_that_meet_audit"] == 1
    r = apply_rule([st("s3", False, ["accept"] * 3)], [0.0, 0.005, 0.01])
    assert r["chosen_margin_pts"] is None and not r["conclusive"]


def test_boundary_pipeline(tmp, runs):
    """Recover states (saved / replayed, verified vs. the journal), refuse to score without a plan, then score,
    calibrate and reproduce the live decision at margin 0."""
    (a1, _), _ = runs
    sets = _sets(tmp, "top1")
    for phase in ("freeze", "diagnose", "audit-data", "audit-materialized"):
        rc, out = cli(["gradient-live", phase, "--config", str(CFG), *sets])
        assert rc == 0, (phase, out)
    A = tmp / "top1" / "_analysis"
    r2key = a1[1]["shortlist"][1]["action_key"]
    b = ["--set", f"experiment.data_dir={(tmp / 'data-live').as_posix()}",
         "--set", f"experiment.runs_dir={(tmp / 'top1').as_posix()}",
         "--set", f"gradient_boundary.out_dir={(tmp / 'bnd').as_posix()}",
         "--set", f"gradient_boundary.source_run={(tmp / 'top1' / RID).as_posix()}",
         "--set", f"gradient_boundary.diagnostic={(A / 'diagnostic').as_posix()}",
         "--set", f"gradient_boundary.anchors=[{{frozen: '{(A / 'frozen' / 'frozen.json').as_posix()}', audit: '{(A / 'audit_materialized').as_posix()}'}}]",
         "--set", f"gradient_boundary.rank2={{attempt: 2, rank: 2, action_key: '{r2key}'}}",
         "--set", f"gradient_boundary.audit_inventory={(A / 'audit_data' / 'inventory.json').as_posix()}",
         "--set", f"gradient_boundary.dense_reference=['{(A / 'audit_materialized' / 'scores.sqlite').as_posix()}']"]
    rc, out = cli(["gradient-boundary", "recover", "--config", str(BND), *b])
    assert rc == 0, out
    st = json.loads((tmp / "bnd" / "states.json").read_text())
    assert [s["name"] for s in st["states"]] == ["attempt 1", "attempt 2 rank 2"]
    assert st["states"][0]["state_id"] == a1[0]["cand_id"] and st["states"][0]["mask_hash"] == a1[0]["trial_mask_hash"]
    assert any(c.get("mask_hash_in_diagnostic_cache") for c in st["checks"])
    rc, out = cli(["gradient-boundary", "audit", "--config", str(BND), *b])  # no plan yet -> refused
    assert rc != 0 and "plan" in out
    for phase in ("margin-plan", "audit", "calibrate"):
        rc, out = cli(["gradient-boundary", phase, "--config", str(BND), *b])
        assert rc == 0, (phase, out)
    B = json.loads((tmp / "bnd" / "boundary.json").read_text())
    anchor, cand = B["states"]
    assert anchor["cost"]["materialized_evaluations_run"] == 0 and anchor["cost"]["answers_reused"] > 0  # reused, provenance ok
    assert cand["cost"]["materialized_evaluations_run"] > 0 and B["rank2_vs_parent"]["parent"] == anchor["state_id"]
    assert B["dense_reference"]["sources"]
    plan = json.loads((tmp / "bnd" / "calibration_plan.json").read_text())
    cal = json.loads((tmp / "bnd" / "calibration.json").read_text())
    assert plan["recorded_at"] < (tmp / "bnd" / "boundary.json").stat().st_mtime
    assert all(r["live_decision_reproduced_at_margin_0"] for r in cal["states"])
    assert len(cal["table"]) == 3 and "chosen_margin_pts" in cal
