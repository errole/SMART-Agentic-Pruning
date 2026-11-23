"""Gradient-only top-1 selection in the live loop (tiny Qwen3, CPU, no controller).

The live-smoke data's pruning-time key is the dense model's own prediction, so every changed answer counts as
broken and the cumulative loss grows with each accepted step: with a 15-point tolerance the first gradient-ranked
action is accepted and the second is rejected.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from conftest import ROOT
from live_smoke import make_live_smoke_data
from test_recovery import cli

CFG = ROOT / "configs" / "gradient_live_smoke.yaml"
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


def _latest(run_dir):
    from capprune.utils.checkpoint import CheckpointStore

    return CheckpointStore(run_dir).load_latest()[0]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gl")
    make_live_smoke_data(tmp / "data-live")
    rc, out = cli(["supervise", "--config", str(CFG), *_sets(tmp, "plain")])
    assert rc == 0, out
    return tmp


def test_ranking_acceptance_and_rollback(base):
    rd = base / "plain" / RID
    att = _attempts(rd)
    assert [a["gate_decision"] for a in att] == ["accept", "reject"]
    # ranking: the logged shortlist is in the offline gradient order (mean z_grad, then generator index) and the
    # gated action is its first entry
    for a in att:
        sl = a["shortlist"]
        assert [s["rank"] for s in sl] == list(range(1, len(sl) + 1))
        assert all(x["z_g_mean"] <= y["z_g_mean"] for x, y in zip(sl, sl[1:]))  # exact tie order: unit test below
        assert a["chosen"]["action_key"] == sl[0]["action_key"] and a["chosen"]["z_g_mean"] == min(s["z_g_mean"] for s in sl)
        assert a["proposed_channels"] == a["actual_channels"] == 64 and a["channels_match"]
        assert a["parent_verified"] is True
    # acceptance: the accepted trial is the next parent
    assert att[1]["parent_id"] == att[0]["cand_id"]
    assert att[1]["parent_sparsity"] == att[0]["rho_candidate"]
    assert (rd / "masks" / f"{att[0]['cand_id']}.safetensors").exists()
    # rollback: the rejected trial is undone bit-for-bit and the run ends at its parent (a candidate failure)
    assert att[1]["rollback_verified"] is True and att[1]["outcome"] == "rollback"
    assert any(v["status"] == "fail" for v in att[1]["gate"]["domains"].values())
    S = _latest(rd)
    assert S["state_id"] == att[0]["cand_id"] and S["search"]["rho"] == att[0]["rho_candidate"]
    assert S["search"]["stop_reason"] == "candidate_rejected" and S["search"]["rollbacks"] == 1
    assert S["search"]["questions"] == sum(a["gate"]["questions"] for a in att) == att[-1]["questions_cumulative"]
    # paired scoring: every gate batch holds dense and trial answers for the same questions; orders differ per attempt
    db = sqlite3.connect(rd / "caches" / "eval.sqlite")
    pairs = [json.loads(v) for k, v in db.execute("SELECT k, v FROM kv WHERE k LIKE 'pair|%'")]
    db.close()
    assert pairs and all(set(p["d"]) == set(p["t"]) for p in pairs)
    assert sum(len(p["t"]) for p in pairs) == S["search"]["questions"]
    assert len({a["gate"]["order_seed"] for a in att}) == len(att)


def test_rank_ties_break_by_generator_index():
    from capprune.gradient_live import rank_gradient

    c = [{"features": {"z_g_mean": z}} for z in (0.3, -0.5, -0.5 + 1e-15, -0.7, 0.3)]
    assert rank_gradient(c) == [3, 1, 2, 0, 4]


def test_cumulative_question_cap_is_a_budget_stop(base):
    plain = _attempts(base / "plain" / RID)
    q1 = plain[0]["gate"]["questions"]
    first_look = 4 * 16                                  # min_sample x 4 categories
    assert plain[1]["gate"]["questions"] > first_look + 16  # uncapped, attempt 2 needs more than one more batch
    cap = q1 + first_look + 8                            # the initial sample fits, the next 16-question batch does not
    rc, out = cli(["supervise", "--config", str(CFG), *_sets(base, "cap", [f"ranked_search.question_cap={cap}"])])
    assert rc == 0, out
    rd = base / "cap" / RID
    att = _attempts(rd)
    assert att[0]["cand_id"] == plain[0]["cand_id"] and att[0]["gate"]["questions"] == q1  # cap not binding yet
    a = att[1]
    assert a["gate_decision"] == "budget_stop" and a["outcome"] == "budget_stop"
    assert a["gate"]["questions"] == first_look and a["questions_cumulative"] == q1 + first_look <= cap
    assert a["rollback_verified"] is True
    assert all(v["lo"] is not None for v in a["gate"]["domains"].values())  # bounds at the last completed look
    S = _latest(rd)["search"]
    assert S["stop_reason"] == "question_cap" and S["rollbacks"] == 0  # a budget stop, not a candidate failure
    assert S["gate_counts"]["budget_stop"] == 1 and S["gate_counts"]["reject"] == 0
    assert S["budget_stop"]["when"] == "during_attempt"
    # a cap below the next attempt's initial sample stops the run before that attempt starts
    rc, out = cli(["supervise", "--config", str(CFG), *_sets(base, "cap0", [f"ranked_search.question_cap={q1 + 10}"])])
    assert rc == 0, out
    rd0 = base / "cap0" / RID
    assert len(_attempts(rd0)) == 1
    S0 = _latest(rd0)["search"]
    assert S0["stop_reason"] == "question_cap" and S0["budget_stop"]["when"] == "before_attempt"
    assert S0["questions"] == q1


def test_kill_and_resume_replays_identically(base, tmp_path):
    faults = {"CAPPRUNE_FAULTS": "ranked_after_profile@2,mid_ranked_gate@6,ranked_after_masks@1,ranked_after_build@2",
              "CAPPRUNE_FAULT_STATE": str(tmp_path / "faults.json")}
    rc, out = cli(["supervise", "--config", str(CFG), *_sets(base, "faults", ["reliability.max_restarts=8"])], env=faults)
    assert rc == 0, out
    rd = base / "faults" / RID
    fired = json.loads((tmp_path / "faults.json").read_text())["fired"]
    assert len(fired) == 4, fired
    starts = [json.loads(l) for l in (rd / "journal.jsonl").read_text().splitlines() if '"type": "start"' in l]
    assert len(starts) >= 5
    key = lambda a: (a["iteration"], a["cand_id"], a["gate_decision"], a["gate"]["questions"],  # noqa: E731
                     json.dumps(a["gate"]["domains"], sort_keys=True), json.dumps(a["shortlist"], sort_keys=True))
    assert [key(a) for a in _attempts(rd)] == [key(a) for a in _attempts(base / "plain" / RID)]
    assert _latest(rd)["state_id"] == _latest(base / "plain" / RID)["state_id"]


def test_freeze_diagnose_audit_report(base):
    sets = _sets(base, "plain")
    for phase in ("audit-data", "freeze", "diagnose", "audit", "report"):
        rc, out = cli(["gradient-live", phase, "--config", str(CFG), *sets])
        assert rc == 0, (phase, out)
    A = base / "plain" / "_analysis"
    att = _attempts(base / "plain" / RID)
    fr = json.loads((A / "frozen" / "frozen.json").read_text())
    assert fr["state_id"] == att[0]["cand_id"] and fr["mask_sha256"] == fr["run_mask_sha256"]
    d = json.loads((A / "diagnostic" / "diagnostic.json").read_text())
    assert d["stalled_attempt"] == 2 and d["n_candidates"] == len(att[1]["shortlist"])
    assert [c["action_key"] for c in d["candidates"]] == [s["action_key"] for s in att[1]["shortlist"]]
    assert d["live_decision_reproduced"] is True  # same gate, same order: the top-1 replays the live decision
    assert d["candidates"][0]["replayed_from_run_cache"] == att[1]["gate"]["questions"]
    assert d["recommendation"]["rule"] in ("top-k fallback", "ranking signals", "smaller step", "undetermined")
    a = json.loads((A / "audit" / "audit.json").read_text())
    s = a["sets"]["fresh"]
    assert s["stats"]["overall"]["n"] == s["questions"] and set(s["by_category"]) <= set(s["stats"])
    assert a["size"]["parameters_removed"] == 64 * 3 * 64  # 64 channels x (gate + up + down) x hidden 64
    assert abs(a["size"]["mlp_sparsity"] - fr["mlp_sparsity"]) < 1e-12
    assert (A / "report.md").read_text(encoding="utf-8").startswith("# Gradient-only selector")
    # re-freezing is idempotent; the run's own files were only read
    rc, out = cli(["gradient-live", "freeze", "--config", str(CFG), *sets])
    assert rc == 0, out


def test_proposal_cap_without_a_stall_probes_the_next_attempt(base):
    """Every attempt accepted -> the run ends at the proposal cap (a budget stop, no rollback); the diagnostic then
    gates the frozen checkpoint's full shortlist offline as the next live attempt would have seen it."""
    sets = _sets(base, "nostall", ["gate.capability.tolerance=0.6", "ranked_search.max_proposals=2"])
    rc, out = cli(["supervise", "--config", str(CFG), *sets])
    assert rc == 0, out
    rd = base / "nostall" / RID
    att = _attempts(rd)
    assert [a["gate_decision"] for a in att] == ["accept", "accept"]
    S = _latest(rd)["search"]
    assert S["stop_reason"] == "proposal_cap" and S["rollbacks"] == 0 and S["state_id"] == att[1]["cand_id"]
    for phase in ("freeze", "diagnose"):
        rc, out = cli(["gradient-live", phase, "--config", str(CFG), *sets])
        assert rc == 0, (phase, out)
    d = json.loads((base / "nostall" / "_analysis" / "diagnostic" / "diagnostic.json").read_text())
    assert d["mode"] == "next_attempt" and d["stalled_attempt"] == 3 and d["live_decision_reproduced"] is None
    assert d["parent"] == att[1]["cand_id"] and d["n_candidates"] >= 1
    from capprune.adaptive.live import attempt_order_seed
    assert d["order_seed"] == attempt_order_seed(0, 0, 3)  # the order the live loop would have used next
    if d["first_pass_rank"] == 1:
        assert d["recommendation"]["rule"] == "no stall"


def test_stall_rule():
    from capprune.gradient_audit import summarize_diagnostic

    def row(rank, dec, fail=(), open_=()):
        return {"rank": rank, "decision": dec, "questions": 100, "confirmed_violations": list(fail), "unresolved": list(open_)}

    last = {"gate_decision": "reject", "gate": {"decision": "reject", "questions": 100}}
    r = summarize_diagnostic([row(1, "reject", ["STEM"]), row(2, "accept"), row(3, "reject", ["STEM"])], last, 3)
    assert r["recommendation"]["rule"] == "top-k fallback" and r["first_pass_rank"] == 2
    r = summarize_diagnostic([row(i, "reject", ["STEM"]) for i in range(1, 5)] + [row(5, "accept")], last, 3)
    assert r["recommendation"]["rule"] == "ranking signals" and r["n_pass"] == 1
    r = summarize_diagnostic([row(i, "reject", ["Other"]) for i in range(1, 4)], last, 3)
    assert r["recommendation"]["rule"] == "smaller step" and r["blocking_domains"] == {"Other": {"confirmed": 3, "unresolved": 0}}
    r = summarize_diagnostic([row(1, "reject", ["STEM"]), row(2, "inconclusive", open_=["STEM"])], last, 3)
    assert r["recommendation"]["rule"] == "undetermined"
    # no live stall: a passing top-1 at the frozen checkpoint means none of the three stall rules applies
    r = summarize_diagnostic([row(1, "accept"), row(2, "reject", ["STEM"])], last, 3, "next_attempt", "proposal_cap")
    assert r["recommendation"]["rule"] == "no stall" and r["live_decision_reproduced"] is None
    # ... but a failing top-1 there is the first stall, found offline, and the rules apply
    r = summarize_diagnostic([row(1, "reject", ["STEM"]), row(2, "accept")], last, 3, "next_attempt", "proposal_cap")
    assert r["recommendation"]["rule"] == "top-k fallback"
