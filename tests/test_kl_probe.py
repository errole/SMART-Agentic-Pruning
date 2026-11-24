"""Protocol amendment 2 in the live loop (tiny Qwen3, CPU): after a rank-1 rejection a KL probe over ranks 2-11 on the
probe items picks the fallback, KL's choice is gated, top-1 resumes on accept and the run stops on a second rejection;
a counterfactual log scores the KL / S1c / TV picks and gradient rank 2 on a hold-out slice. Probe items count against
the question cap; probe and counterfactual answers are persisted before use and replay after a crash."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from conftest import ROOT
from live_smoke import make_live_smoke_data
from test_recovery import cli

CFG = ROOT / "configs" / "gradient_live_smoke.yaml"
RID = "gradient_top1__t00__s0"
KL = ["ranked_search.fallback_policy=kl_probe", "ranked_search.kl_probe={per_domain: 10, batch_size: 10}"]
CF = ["ranked_search.counterfactual={enabled: true, holdout_per_domain: 20, batch_size: 10}"]


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


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    t = tmp_path_factory.mktemp("klprobe")
    data = make_live_smoke_data(t / "data-live")
    # the sealed confirmation set (stand-in): the prepared final MMLU split, which screening must never read
    sealed = [json.loads(l)["qid"] for l in (data / "mmlu_final.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    (t / "sealed.json").write_text(json.dumps(sealed), encoding="utf-8")
    return t


def _run(tmp, name, extra=(), env=None, ok=True):
    sets = ["--set", f"experiment.runs_dir={(tmp / name).as_posix()}",
            "--set", f"experiment.data_dir={(tmp / 'data-live').as_posix()}",
            "--set", f"gradient_live.out_dir={(tmp / name / '_analysis').as_posix()}",
            "--set", "reliability.restart_backoff_s=[0.2]",
            "--set", f"ranked_search.sealed_confirmation=['{(tmp / 'sealed.json').as_posix()}']",
            *[x for e in extra for x in ("--set", e)]]
    rc, out = cli(["supervise", "--config", str(CFG), *sets], env=env)
    if ok:
        assert rc == 0, out
    return _attempts(tmp / name / RID), _S(tmp / name / RID)


@pytest.fixture(scope="module")
def base(tmp):
    return _run(tmp, "base", KL + CF)


# ---------------------------------------------------------------------------------------------------------- the policy
def test_kl_probe_after_rank1_rejection_then_stop_after_second_rejection(tmp, base):
    from capprune.signals import pick_action

    att, S = base
    assert [a["gate_decision"] for a in att[:2]] == ["accept", "reject"]
    trig, kl = att[1], att[2]
    assert trig["rank"] == 1 and trig["fallback_triggered"] and trig["kl_probe"] is None
    p = kl["kl_probe"]
    n_short = len(trig["shortlist"])
    assert p["ranks"] == list(range(2, min(11, n_short) + 1)) and p["selection_rule"] == "capprune.signals.pick_action"
    # the gated action is the probe's KL pick, recomputed here with the shared rule from the recorded per-category values
    vals = [c["by_category"] for c in p["candidates"]]
    assert p["choice_rank"] == p["ranks"][pick_action(vals, p["ranks"], "S2")] == p["picks"]["S2"] == kl["rank"]
    assert kl["chosen"]["action_key"] == trig["shortlist"][kl["rank"] - 1]["action_key"]
    assert kl["parent_mask_hash"] == trig["parent_mask_hash"] and kl["parent_verified"] is True
    assert kl["profile_attempt"] == 2 and kl["fallback_of"] == 2
    assert not (tmp / "base" / RID / "profiles" / "iter-000003.safetensors").exists()  # the parent's profile is reused
    # a second rejection (or an inconclusive result) ends the run: no further proposal
    assert kl["gate_decision"] in ("reject", "inconclusive") and len(att) == 3
    assert S["stop_reason"] == "fallback_exhausted" and S["fallback"]["failed"] == 1 and S["fallback"]["triggers"] == 1


def test_tie_break_goes_to_the_lower_gradient_rank(base):
    from capprune import matched_signals, signals

    assert matched_signals.pick_action is signals.pick_action  # the evidence and the live loop share one rule
    v = {"a": {"S2": 0.1}, "b": {"S2": 0.3}}
    tie = [v, {"a": {"S2": 0.3}, "b": {"S2": 0.2}}, {"a": {"S2": 0.25}, "b": {"S2": 0.3}}]   # worst values 0.3, 0.3, 0.3
    assert signals.pick_action(tie, [2, 3, 4], "S2") == 0 and signals.pick_action(tie, [5, 3, 4], "S2") == 1
    assert signals.pick_action([v, {"a": {"S2": 0.29}, "b": {"S2": 0.0}}], [2, 3], "S2") == 1  # lowest worst category wins
    att, _ = base
    p = att[2]["kl_probe"]
    for s in ("S0", "S1c", "TV"):  # every recorded pick follows the same rule (the smoke's net-flip values tie)
        w = [c["worst"][s] for c in p["candidates"]]
        best = min(round(x, 12) for x in w)
        assert p["picks"][s] == min(r for r, x in zip(p["ranks"], w) if round(x, 12) == best)


def test_resume_top1_after_kl_accept(tmp):
    att, S = _run(tmp, "resume", KL + ["gate.capability.tolerance=0.85", "ranked_search.max_proposals=4"],
                  env={"CAPPRUNE_TEST_FORCE_REJECT": "2"})
    assert att[1]["forced_by_test_hook"] and att[1]["fallback_triggered"]
    kl = att[2]
    assert kl["kl_probe"] is not None and kl["rank"] == kl["kl_probe"]["choice_rank"] >= 2
    assert kl["gate_decision"] == "accept" and kl["rescued"]
    nxt = att[3]
    assert nxt["rank"] == 1 and nxt["kl_probe"] is None and nxt["parent_id"] == kl["cand_id"]  # top-1 resumes
    assert S["fallback"]["rescued"] == 1 and S["kl_probe"]["probes"] == 1


# ---------------------------------------------------------------------------------------------------------- budget
def test_cap_accounting_includes_probe_items(tmp, base):
    att, S = base
    kl = att[2]
    charge = kl["kl_probe"]["cost"]["paired_items_charged"]
    assert charge == len(kl["kl_probe"]["ranks"]) * 40 == kl["kl_probe_items_charged"]
    assert S["questions"] == sum(a["gate"]["questions"] for a in att) + charge == kl["questions_cumulative"]
    assert S["kl_probe"]["paired_items_charged"] == charge and S["budget"]["kl_probe_paired_items"] == charge
    assert S["counterfactual"]["charged_to_cap"] is False and S["budget"]["counterfactual_paired_items"] > 0  # reported, not charged
    q12 = att[0]["gate"]["questions"] + att[1]["gate"]["questions"]
    # the probe does not fit under the cap: a budget stop before anything is scored, not a fallback failure
    a2, S2 = _run(tmp, "cap-probe", KL + CF + [f"ranked_search.question_cap={q12 + charge - 1}"])
    assert len(a2) == 3 and a2[2]["outcome"] == "budget_stop" and a2[2]["gate_decision"] == "none"
    assert S2["stop_reason"] == "question_cap" and S2["budget_stop"]["when"] == "before_kl_probe"
    assert S2["questions"] == q12 and S2.get("kl_probe") is None and S2["fallback"]["blocked_by_cap"] == 1
    con = sqlite3.connect((tmp / "cap-probe" / RID / "caches" / "eval.sqlite").as_posix())
    assert con.execute("SELECT COUNT(*) FROM kv WHERE k LIKE 'kl_probe|%'").fetchone()[0] == 0
    con.close()
    # the probe fits but the gate's next look does not: the cap stops the KL choice's gate
    a3, S3 = _run(tmp, "cap-gate", KL + CF + [f"ranked_search.question_cap={q12 + charge + 64 + 8}"])
    assert a3[2]["gate_decision"] == "budget_stop" and a3[2]["gate"]["questions"] == 64
    assert S3["stop_reason"] == "question_cap" and S3["questions"] == q12 + charge + 64
    assert S3["kl_probe"]["paired_items_charged"] == charge and S3["fallback"]["failed"] == 0


# ---------------------------------------------------------------------------------------------------------- crash safety
def test_replay_after_forced_crash_mid_probe(tmp, base):
    att, S = base
    fs = tmp / "faults.json"
    a2, S2 = _run(tmp, "crash", KL + CF, env={"CAPPRUNE_FAULTS": "mid_kl_probe@2,mid_counterfactual@1",
                                             "CAPPRUNE_FAULT_STATE": str(fs)})
    fired = json.loads(fs.read_text())["fired"]
    assert "mid_kl_probe@2" in fired and "mid_counterfactual@1" in fired  # both crashes happened
    p, q = att[2]["kl_probe"], a2[2]["kl_probe"]
    assert q["choice_rank"] == p["choice_rank"] and q["picks"] == p["picks"]
    assert [c["by_category"] for c in q["candidates"]] == [c["by_category"] for c in p["candidates"]]
    # crash 1 (2nd probe batch) and crash 2 (1st counterfactual batch, after the probe finished): the third pass replays
    # the whole probe and the persisted counterfactual batch from the cache and scores only the rest
    assert q["cost"]["replayed"] == p["cost"]["items"] and q["cost"]["scored"] == 0
    assert a2[2]["counterfactual"]["cost"]["replayed"] == 10
    assert a2[2]["counterfactual"]["by_rank"] == att[2]["counterfactual"]["by_rank"]
    assert [(a["cand_id"], a["gate_decision"], a["gate"]["questions"]) for a in a2] == \
           [(a["cand_id"], a["gate_decision"], a["gate"]["questions"]) for a in att]
    assert S2["questions"] == S["questions"] and S2["kl_probe"]["paired_items_charged"] == S["kl_probe"]["paired_items_charged"]


# ---------------------------------------------------------------------------------------------------------- data roles
def _qids_in_cache(run_dir, prefix):
    con = sqlite3.connect((run_dir / "caches" / "eval.sqlite").as_posix())
    out = set()
    for k, v in con.execute("SELECT k, v FROM kv WHERE k LIKE ?", (prefix + "|%",)):
        rec = json.loads(v)
        out |= set(rec.get("t") or rec.get("items") or {})
    con.close()
    return out


def test_counterfactual_never_touches_confirmation_or_gate_items(tmp, base):
    att, _ = base
    rd = tmp / "base" / RID
    sp = json.loads((rd / "screening_split.json").read_text(encoding="utf-8"))
    sealed = set(json.loads((tmp / "sealed.json").read_text(encoding="utf-8")))
    gate, probe = set(sp["gate_qids"]), {q for v in sp["probe"].values() for q in v}
    hold = {q for v in sp["holdout"].values() for q in v}
    assert hold and probe and not (hold & gate) and not (hold & probe) and not (probe & gate)
    assert not ((gate | probe | hold) & sealed)
    cf = _qids_in_cache(rd, "counterfactual")
    assert cf == hold                                   # every counterfactual answer is a hold-out item, and only those
    assert _qids_in_cache(rd, "kl_probe") == probe      # the probe reads only probe items
    gate_seen = set()
    con = sqlite3.connect((rd / "caches" / "eval.sqlite").as_posix())
    for (v,) in con.execute("SELECT v FROM kv WHERE k LIKE 'pair|%'"):
        gate_seen |= set(json.loads(v)["t"])
    con.close()
    assert gate_seen <= gate and not (gate_seen & (hold | sealed | probe))
    c = att[2]["counterfactual"]
    assert set(c["picks"]) == {"KL", "S1c", "TV", "gradient rank 2"} and c["picks"]["gradient rank 2"] == 2
    assert set(c["by_rank"]) == {str(r) for r in set(c["picks"].values())}
    assert c["charged_to_cap"] is False and c["cost"]["paired_items"] == len(c["by_rank"]) * len(hold)


def test_screening_split_refuses_overlap_with_the_sealed_set(tmp):
    from capprune.config import data_dir, load_config, run_config
    from capprune.data.prepare import PreparedData
    from capprune.gradient_live import screening_split
    from capprune.model import tokenizer_id

    rc = run_config(load_config(CFG), "gradient_top1", 0.0, 0)
    rc["experiment"]["data_dir"] = str(tmp / "data-live")
    rc["ranked_search"]["counterfactual"] = {"enabled": True, "holdout_per_domain": 20}
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    s = screening_split(rc, data)
    bad = tmp / "sealed-bad.json"
    bad.write_text(json.dumps([s["holdout"][0]["qid"], s["probe"][0]["qid"]]), encoding="utf-8")
    rc["ranked_search"]["sealed_confirmation"] = [str(bad)]
    with pytest.raises(RuntimeError, match="sealed confirmation"):
        screening_split(rc, data)
    rc["ranked_search"]["sealed_confirmation"] = [str(tmp / "sealed.json")]
    ok = screening_split(rc, data)
    assert ok["info"]["sealed_confirmation_qids"] > 0 and len(ok["holdout"]) == 20 * len(ok["domains"])
    # the hold-out comes out of the gate's items; without the counterfactual the gate pool is unchanged
    rc["ranked_search"]["counterfactual"] = {}
    plain = screening_split(rc, data)
    assert {r["qid"] for r in plain["gate"]} == {r["qid"] for r in ok["gate"]} | {r["qid"] for r in ok["holdout"]}
    assert np.isclose(len(plain["gate"]) - len(ok["gate"]), len(ok["holdout"]))
