"""Materialised-checkpoint audit and the extended (replayed + continued) gradient-only search, tiny Qwen3 on CPU.

The "original" run is the smoke pilot with a 60-point tolerance and a 2-proposal cap, so it ends at the proposal cap
with every attempt accepted (like the real pilot).  The extension replays it in a separate run directory and must
reproduce it before continuing; the original's files must not change.
"""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from conftest import ROOT
from live_smoke import make_live_smoke_data
from test_recovery import cli

BASE = ROOT / "configs" / "gradient_live_smoke.yaml"
EXT = ROOT / "configs" / "gradient_live_ext_smoke.yaml"
RID = "gradient_top1__t00__s0"


def _orig_sets(tmp, extra=()):
    return ["--set", f"experiment.runs_dir={(tmp / 'orig').as_posix()}",
            "--set", f"experiment.data_dir={(tmp / 'data-live').as_posix()}",
            "--set", f"gradient_live.out_dir={(tmp / 'orig' / '_analysis').as_posix()}",
            "--set", "gate.capability.tolerance=0.6", "--set", "ranked_search.max_proposals=2",
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


def _tree_hash(d):
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file() and not p.name.endswith(("-shm", "-wal"))}


@pytest.fixture(scope="module")
def orig(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("glx")
    make_live_smoke_data(tmp / "data-live")
    sets = _orig_sets(tmp)
    rc, out = cli(["supervise", "--config", str(BASE), *sets])
    assert rc == 0, out
    for phase in ("audit-data", "freeze", "audit"):
        rc, out = cli(["gradient-live", phase, "--config", str(BASE), *sets])
        assert rc == 0, (phase, out)
    fr = json.loads((tmp / "orig" / "_analysis" / "frozen" / "frozen.json").read_text())
    assert fr["stop_reason"] == "proposal_cap" and fr["accepts"] == 2
    return tmp, fr


def _ext_sets(tmp, fr, name, extra=(), journal=None):
    vp = (f"{{journal: '{(journal or tmp / 'orig' / RID / 'journal.jsonl').as_posix()}', "
          f"frozen: '{(tmp / 'orig' / '_analysis' / 'frozen' / 'frozen.json').as_posix()}', attempts: {fr['attempts']}, "
          f"state_id: '{fr['state_id']}', mask_sha256: '{fr['mask_sha256']}'}}")
    return ["--set", f"experiment.runs_dir={(tmp / name).as_posix()}",
            "--set", f"experiment.data_dir={(tmp / 'data-live').as_posix()}",
            "--set", f"gradient_live.out_dir={(tmp / name / '_analysis').as_posix()}",
            "--set", f"gradient_live.compare_with={(tmp / 'orig' / '_analysis').as_posix()}",
            "--set", f"ranked_search.verify_prefix={vp}", "--set", "reliability.restart_backoff_s=[0.2]",
            *[x for e in extra for x in ("--set", e)]]


def test_materialized_audit_reproduces_the_original_and_counts_parameters(orig):
    tmp, fr = orig
    A = tmp / "orig" / "_analysis"
    before = _tree_hash(A / "audit")
    sets = _orig_sets(tmp, ["gradient_live.audit_materialized.sets=[fresh]",
                            f"gradient_live.audit_materialized.expect={{state_id: '{fr['state_id']}', mask_sha256: '{fr['mask_sha256']}'}}"])
    rc, out = cli(["gradient-live", "audit-materialized", "--config", str(BASE), *sets])
    assert rc == 0, out
    r = json.loads((A / "audit_materialized" / "audit_materialized.json").read_text())
    sz = r["size"]
    assert sz["count_matches_masks"] and sz["intermediate_sizes_match"] and sz["kept_weights_identical_to_masked"]
    assert sz["parameters_removed"] == sz["expected_from_masks"] == sz["mlp_channels_removed"] * 3 * 64 > 0
    s = r["sets"]["fresh"]
    m = s["original_audit_match"]  # same ids, order, batches and token budget as the original (masked) audit
    assert m["batches"] > 0 and m["batches_same_ids"] == m["batches"]
    assert m["dense_equal"] == m["dense_items"] == s["questions"] and m["masked_equal"] == m["masked_items"]
    a = s["agreement"]["overall"]
    assert a["n"] == s["questions"] and 0 <= a["pred_agree"] <= a["n"]
    assert set(s["stats"]["materialized"]) >= set(s["by_category"]) | {"overall", "conclusion"}
    assert all("ub_simultaneous" in s["stats"]["materialized"][c] for c in s["by_category"])
    assert r["item_model_evaluations_run"] == {k: s["questions"] for k in ("dense", "masked", "materialized")}
    assert _tree_hash(A / "audit") == before  # the original audit is preserved; the verification is written separately
    # a wrong expected checkpoint id is refused before anything is scored
    bad = _orig_sets(tmp, ["gradient_live.audit_materialized.expect={state_id: 'deadbeef'}"])
    rc, out = cli(["gradient-live", "audit-materialized", "--config", str(BASE), *bad])
    assert rc != 0 and "expected" in out


def test_extension_replays_the_prefix_and_leaves_the_original_untouched(orig):
    tmp, fr = orig
    before = _tree_hash(tmp / "orig")
    sets = _ext_sets(tmp, fr, "ext")
    rc, out = cli(["gradient-live", "register", "--config", str(EXT), *sets])
    assert rc == 0, out
    reg = json.loads((tmp / "ext" / "_analysis" / "registration.json").read_text())
    assert reg["caps"]["max_proposals"] == 4 and reg["caps"]["question_cap"] == 100000 and not reg["launched_before_registration"]
    rc, out = cli(["supervise", "--config", str(EXT), *sets])
    assert rc == 0, out
    o, e = _attempts(tmp / "orig" / RID), _attempts(tmp / "ext" / RID)
    assert len(e) >= len(o) + 1
    for a, b in zip(o, e):  # the replayed prefix is identical, attempt by attempt
        assert (a["cand_id"], a["gate_decision"], a["gate"]["questions"], a["gate"]["order_seed"]) == \
               (b["cand_id"], b["gate_decision"], b["gate"]["questions"], b["gate"]["order_seed"])
        assert b["prefix_check"]["ok"] is True
    assert e[len(o) - 1]["prefix_check"]["final_mask_verified"] is True
    assert e[len(o)]["prefix_check"] is None and e[len(o)]["parent_id"] == fr["state_id"]
    S = _latest(tmp / "ext" / RID)["search"]
    assert S["prefix"]["ok"] and S["prefix"]["verified_through"] == len(o)
    assert S["stop_reason"] in ("candidate_rejected", "candidate_inconclusive", "proposal_cap", "question_cap")
    assert S["evaluations"]["paired_questions"] == S["questions"]
    assert S["evaluations"]["item_model_evaluations_run"] == 2 * S["questions"]  # fresh run dir: no cache replays
    assert all(a["params_removed"] == round(a["rho_after"] * 4 * 256) * 3 * 64 for a in e)  # 4 units x 256 ch, hidden 64
    # freeze -> diagnose -> materialised audit -> report, on the extension only
    for phase in ("audit-data", "freeze", "diagnose", "audit-materialized", "report"):
        rc, out = cli(["gradient-live", phase, "--config", str(EXT), *sets])
        assert rc == 0, (phase, out)
    rep = (tmp / "ext" / "_analysis" / "report.md").read_text(encoding="utf-8")
    assert "| | original pilot | extended run |" in rep and "this run: incremental" in rep
    # registration cannot be rewritten after the launch with different caps
    rc, out = cli(["gradient-live", "register", "--config", str(EXT), *sets, "--set", "ranked_search.max_proposals=9"])
    assert rc != 0
    assert _tree_hash(tmp / "orig") == before  # the original run and its analysis were only read


def test_prefix_mismatch_stops_before_extending(orig, tmp_path):
    tmp, fr = orig
    o = _attempts(tmp / "orig" / RID)
    lines = (tmp / "orig" / RID / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    bad = []
    for line in lines:  # pretend the original chose a different (lower-ranked) action at its last attempt
        r = json.loads(line)
        if r.get("type") == "iteration" and r["iteration"] == len(o):
            r["chosen"]["action_key"] = r["shortlist"][-1]["action_key"] + "x"
        bad.append(json.dumps(r))
    j = tmp_path / "journal.jsonl"
    j.write_text("\n".join(bad) + "\n", encoding="utf-8")
    sets = _ext_sets(tmp, fr, "ext-bad", journal=j)
    rc, out = cli(["supervise", "--config", str(EXT), *sets])
    assert rc == 0, out
    S = _latest(tmp / "ext-bad" / RID)["search"]
    assert S["stop_reason"] == "prefix_mismatch" and S["prefix"]["verified_through"] == len(o) - 1
    e = _attempts(tmp / "ext-bad" / RID)
    assert len(e) == len(o) and e[-1]["gate_decision"] == "none"  # nothing was gated at the mismatching attempt


def test_extended_caps_include_the_prefix(orig):
    tmp, fr = orig
    o = _attempts(tmp / "orig" / RID)
    q_pre = sum(a["gate"]["questions"] for a in o)
    # a question cap below the prefix stops the replay itself: the prefix counts toward the cap
    rc, out = cli(["supervise", "--config", str(EXT), *_ext_sets(tmp, fr, "cap-pre", [f"ranked_search.question_cap={q_pre - 1}"])])
    assert rc == 0, out
    S = _latest(tmp / "cap-pre" / RID)["search"]
    assert S["stop_reason"] == "question_cap" and S["questions"] <= q_pre - 1 and S["accepts"] < len(o)
    # a cap that leaves less than the first look after the prefix stops before the first new attempt
    rc, out = cli(["supervise", "--config", str(EXT), *_ext_sets(tmp, fr, "cap-post", [f"ranked_search.question_cap={q_pre + 10}"])])
    assert rc == 0, out
    S = _latest(tmp / "cap-post" / RID)["search"]
    assert S["stop_reason"] == "question_cap" and S["budget_stop"]["when"] == "before_attempt" and S["questions"] == q_pre
    assert S["prefix"]["ok"] and S["accepts"] == len(o)
    # the proposal cap counts the replayed attempts too
    rc, out = cli(["supervise", "--config", str(EXT), *_ext_sets(tmp, fr, "cap-prop", [f"ranked_search.max_proposals={len(o) + 1}"])])
    assert rc == 0, out
    e = _attempts(tmp / "cap-prop" / RID)
    S = _latest(tmp / "cap-prop" / RID)["search"]
    assert len(e) == len(o) + 1
    assert S["stop_reason"] == ("proposal_cap" if e[-1]["gate_decision"] == "accept" else S["stop_reason"])
    shutil.rmtree(tmp / "cap-pre", ignore_errors=True)
