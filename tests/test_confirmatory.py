"""The confirmatory pipeline on CPU (tiny random Qwen3, live-smoke data): sealing, the split rules, the refusals, and one
end-to-end pass seal -> screen (arms a, b, c) -> freeze -> confirm -> report. Numbers are meaningless; the mechanics are
what is tested."""
from __future__ import annotations

import json
import shutil
import sqlite3

import numpy as np
import pytest

from conftest import ROOT
from live_smoke import make_live_smoke_data
from test_recovery import cli

CFG = ROOT / "configs" / "confirmatory_smoke.yaml"
CATS = ("STEM", "Humanities", "Social Sciences", "Other")


def _rows(n_per=60, sources=("mmlu/test", "mmlu-cf/validation")):
    out = []
    for c in CATS:
        for i in range(n_per):
            s = sources[i % len(sources)] if i % 3 else sources[0]
            out.append({"qid": f"{c[:3]}/{s}/{i:04d}", "category": c, "domain": c, "subject": "x", "source": s,
                        "question": f"question {c} {i}?", "choices": ["a", "b", "c", f"d{i}"], "answer": i % 4})
    return out


# ---------------------------------------------------------------------------------------------------------- pure rules
def test_dedupe_drops_duplicate_texts_and_fewshot_exemplars():
    from capprune.confirmatory import dedupe

    rows = _rows(5)
    dup = {**rows[1], "qid": "dup", "question": rows[1]["question"].upper() + "  "}   # same text up to case / spacing
    shot = {**rows[2], "qid": "shot-copy"}
    kept, info = dedupe(rows + [dup], [shot])
    assert info == {"input": len(rows) + 1, "kept": len(rows) - 1, "duplicates_dropped": 1, "fewshot_exemplars_dropped": 1}
    assert "dup" not in {r["qid"] for r in kept} and rows[2]["qid"] not in {r["qid"] for r in kept}


def test_seal_split_is_disjoint_stratified_seeded_and_reports_shortfalls():
    from capprune.confirmatory import seal_split

    rows = _rows(60)
    target = {"STEM": 30, "Humanities": 30, "Social Sciences": 50, "Other": 30}
    reserve = {c: 20 for c in CATS}
    conf, screen, alloc = seal_split(rows, target, reserve, 11)
    ids_c, ids_s = {r["qid"] for r in conf}, {r["qid"] for r in screen}
    assert not ids_c & ids_s and ids_c | ids_s == {r["qid"] for r in rows}
    assert alloc["Social Sciences"]["confirmation"] == 40 and alloc["Social Sciences"]["shortfall"] == 10  # 60 - 20 reserve
    for c in CATS:
        a = alloc[c]
        n = a["confirmation"]
        assert sum(1 for r in conf if r["category"] == c) == n
        for s, k in a["by_source"].items():  # proportional to the source sizes within the category (largest remainder)
            assert abs(k - n * a["source_sizes"][s] / a["available"]) < 1
    again, _, _ = seal_split(list(reversed(rows)), target, reserve, 11)
    assert {r["qid"] for r in again} == ids_c                                   # deterministic, order-independent
    other, _, _ = seal_split(rows, target, reserve, 12)
    assert {r["qid"] for r in other} != ids_c                                   # seeded


def test_power_grows_with_n_and_falls_with_the_true_loss():
    from capprune.confirmatory import power_at

    assert power_at(3000, 0.08, 0.0) > power_at(1000, 0.08, 0.0) > power_at(300, 0.08, 0.0)
    assert power_at(3000, 0.08, 0.0) > power_at(3000, 0.08, 0.015)
    assert power_at(3000, 0.08, 0.03) <= 0.06                                    # at the boundary: about alpha or less


# ---------------------------------------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def study(tmp_path_factory):
    from capprune import confirmatory as cf
    from capprune.config import load_config
    from capprune.confirmatory import _read_jsonl

    t = tmp_path_factory.mktemp("confirmatory")
    data = make_live_smoke_data(t / "data-live")
    out = t / "study"
    sets = [f"experiment.runs_dir={(out / 'runs').as_posix()}", f"experiment.data_dir={data.as_posix()}",
            f"gradient_live.out_dir={(out / '_analysis').as_posix()}", "reliability.restart_backoff_s=[0.2]",
            f"confirmatory.out_dir={out.as_posix()}",
            f"ranked_search.sealed_confirmation=['{(out / 'sealed' / 'confirmation_qids.json').as_posix()}']",
            f"ranked_search.split_registration={(out / 'registration.json').as_posix()}"]
    cfg = load_config(CFG, sets)
    rows = [{**r, "source": "mmlu/test" if i % 2 else "mmlu-cf/validation"} for i, r in enumerate(_read_jsonl(data / "mmlu_probe.jsonl"))]
    fewshot = _read_jsonl(data / "mmlu_fewshot.jsonl")
    reg = cf.build_data(cfg, rows_override=(rows, fewshot, {"smoke": True}))
    return {"t": t, "out": out, "data": data, "sets": sets, "cfg": cfg, "reg": json.loads(reg.read_text(encoding="utf-8"))}


def _cli(study, *args, env=None):
    return cli([*args, "--config", str(CFG), *[x for s in study["sets"] for x in ("--set", s)]], env=env)


def test_sealing_writes_disjoint_sets_and_no_confirmation_item_reaches_the_data_dir(study):
    from capprune.confirmatory import _read_jsonl, normalize

    reg, out, data = study["reg"], study["out"], study["data"]
    S = reg["sets"]
    assert [S[k]["n"] for k in ("confirmation", "gate", "probe", "holdout")] and sum(S[k]["n"] for k in S) == reg["dedupe"]["kept"]
    assert all(S["confirmation"]["per_category"][c] == 20 for c in CATS) and not study["reg"]["shortfalls"]
    assert all(S["probe"]["per_category"][c] == 10 and S["holdout"]["per_category"][c] == 10 for c in CATS)
    conf = {r["qid"] for r in _read_jsonl(out / "sealed" / "confirmation.jsonl")}
    pool = _read_jsonl(data / "mmlu_probe.jsonl")
    assert not conf & {r["qid"] for r in pool}
    assert not {normalize(r) for r in _read_jsonl(out / "sealed" / "confirmation.jsonl")} & {normalize(r) for r in pool}
    assert _read_jsonl(data / "mmlu_final.jsonl") == [] and _read_jsonl(data / "mmlu_dev.jsonl") == []
    from capprune.confirmatory import build_data
    with pytest.raises(SystemExit, match="sealed already"):                    # sealing happens once
        build_data(study["cfg"])


def test_confirm_refuses_before_the_batch_is_frozen(study):
    from capprune.confirmatory import confirm

    with pytest.raises(SystemExit, match="not frozen"):
        confirm(study["cfg"], "pilot")


def test_a_run_refuses_to_start_if_the_split_does_not_match_the_registration(study):
    from capprune.config import data_dir, run_config
    from capprune.data.prepare import PreparedData
    from capprune.gradient_live import screening_split
    from capprune.model import tokenizer_id

    rc = run_config(study["cfg"], "cap_top1", 0.0, 0)
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    assert screening_split(rc, data)["info"]["split_registration_verified"]
    bad = json.loads((study["out"] / "registration.json").read_text(encoding="utf-8"))
    bad["sets"]["holdout"]["sha256"] = "0" * 64
    p = study["t"] / "bad-registration.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    rc["ranked_search"]["split_registration"] = str(p)
    with pytest.raises(RuntimeError, match="does not match the sealed registration"):
        screening_split(rc, data)


@pytest.fixture(scope="module")
def pilot(study):
    rc, out = _cli(study, "supervise")
    assert rc == 0, out
    for phase in ("freeze", "confirm", "report"):
        rc, out = _cli(study, "confirmatory", phase, "--batch", "pilot")
        assert rc == 0, (phase, out)
    b = study["out"] / "batches" / "pilot"
    return {"confirm": json.loads((b / "confirm.json").read_text(encoding="utf-8")),
            "report": json.loads((b / "report.json").read_text(encoding="utf-8")), "dir": b}


def test_pilot_end_to_end(study, pilot):
    from capprune.screen_confirm import iut_test

    runs = {r["method"]: r for r in pilot["confirm"]["runs"]}
    assert set(runs) == {"ppl_top1", "cap_top1", "cap_klprobe"}
    for m, r in runs.items():
        o = json.loads((pilot["dir"] / f"{r['run_id']}.order.json").read_text(encoding="utf-8"))
        pr = [c["params_removed"] for c in o["order"]]
        assert pr == sorted(pr) and r["frozen_checkpoints"] == len(o["order"])
        # fixed sequence: every tested checkpoint passed except possibly the last; nothing after a failure was tested
        assert all(x["pass"] for x in r["tested"][:-1])
        if r["tested"] and not r["tested"][-1]["pass"]:
            assert r["claim_position"] == (r["tested"][-2]["position"] if len(r["tested"]) > 1 else None)
        for x in r["tested"]:
            assert iut_test({c: tuple(v) for c, v in x["counts"].items()}, 0.05, 0.15)["pass"] == x["pass"]
            assert x["count_matches_masks"] and sum(v[0] for v in x["counts"].values()) == 80
    assert runs["ppl_top1"]["tested"] and all(x["pool_loss"] is None for x in runs["ppl_top1"]["tested"])
    rows = {r["method"]: r for r in pilot["report"]["runs"]}
    assert rows["cap_klprobe"]["cost"]["questions_charged_to_cap"] >= rows["cap_klprobe"]["cost"]["screening_gate_questions"]
    assert (pilot["dir"] / "report.md").exists()


def test_screening_never_read_the_sealed_items(study, pilot):
    conf = set(json.loads((study["out"] / "sealed" / "confirmation_qids.json").read_text(encoding="utf-8")))
    for rd in (study["out"] / "runs").glob("*__t00__s0"):
        con = sqlite3.connect((rd / "caches" / "eval.sqlite").as_posix())
        seen = set()
        for (v,) in con.execute("SELECT v FROM kv WHERE k LIKE 'pair|%' OR k LIKE 'kl_probe|%' OR k LIKE 'counterfactual|%'"):
            seen |= set(json.loads(v)["t"])
        con.close()
        assert not seen & conf, rd.name
    # the confirm phase is the only reader, and only after the freeze
    assert json.loads((study["out"] / "batches" / "pilot" / "frozen.json").read_text())["frozen_at"] < \
           (study["out"] / "confirm" / "scores.sqlite").stat().st_mtime


def test_supervisor_counts_the_heartbeat_of_a_worker_launched_through_a_launcher():
    """Regression (found in the pilot): on Windows the venv's python.exe is a launcher, so the worker that writes the
    heartbeat has a different pid from the launched process; the supervisor must still see its heartbeats."""
    from capprune.supervisor import heartbeat_ts

    started = 1000.0
    assert heartbeat_ts({"pid": 7, "ts": 1500.0, "started": 1001.0}, 7, started) == 1500.0     # same pid
    assert heartbeat_ts({"pid": 8, "ts": 1500.0, "started": 1001.0}, 7, started) == 1500.0     # launcher child
    assert heartbeat_ts({"pid": 8, "ts": 990.0, "started": 900.0}, 7, started) == 0.0         # an earlier, killed worker


# ---------------------------------------------------------------------------------------------- amendment 3 (S1-S4)
def _outcomes(seed=0, k=4, per=30, flip=(0.02, 0.05, 0.08, 0.2)):
    """Synthetic stored outcomes: k checkpoints in frozen order, ``per`` items per category, growing loss."""
    rng = np.random.default_rng(seed)
    n = per * len(CATS)
    cats = np.repeat(np.array(CATS), per)
    dense = np.tile(rng.random(n) < 0.6, (k, 1))
    cand = dense.copy()
    for i in range(k):
        cand[i] = dense[i] & ~(rng.random(n) < flip[i])
    return {"qids": np.array([f"q{i:04d}" for i in range(n)]), "category": cats, "dense": dense, "cand": cand,
            "position": np.arange(1, k + 1), "params_removed": np.arange(1, k + 1) * 100,
            "materialized_parameters_removed": np.arange(1, k + 1) * 100}


def test_counts_and_sequence_claim_match_the_protocol_functions():
    from capprune.confirmatory import counts_from_outcomes, sequence_claim
    from capprune.screen_confirm import fixed_sequence

    oc = _outcomes()
    seq = [{"id": int(oc["position"][i]), "params_removed": int(oc["params_removed"][i]),
            "counts": counts_from_outcomes(oc["dense"][i], oc["cand"][i], oc["category"])} for i in range(4)]
    claims = set()
    for tol in (0.05, 0.10, 0.15, 0.30, 0.60):
        fs = fixed_sequence(seq, 0.05, tol)
        pos, params = sequence_claim(oc, 0.05, tol)
        assert pos == fs["claim"] and params == (100 * fs["claim"] if fs["claim"] else 0)
        claims.add(pos)
    assert len(claims) > 1                                                # the tolerances lead to different claims
    c = counts_from_outcomes(oc["dense"][3], oc["cand"][3], oc["category"])
    assert all(c[k][0] == 30 and c[k][2] == 0 for k in CATS)              # this synthetic checkpoint only loses items
    sub = counts_from_outcomes(oc["dense"][3], oc["cand"][3], oc["category"], np.array([0, 1, 30, 60, 61, 90]))
    assert [sub[k][0] for k in CATS] == [2, 1, 2, 1]


def test_s4_draws_are_seeded_stratified_and_without_replacement():
    from capprune.confirmatory import s4_draws, s4_run, summarize

    oc = _outcomes()
    d1 = s4_draws(oc["qids"], oc["category"], [5, 10], 6, 20261124)
    d2 = s4_draws(oc["qids"][::-1], oc["category"][::-1], [5, 10], 6, 20261124)   # same items, other storage order
    for n in (5, 10):
        assert len(d1[n]) == 6
        for idx in d1[n]:
            assert len(idx) == 4 * n and len(set(idx.tolist())) == 4 * n
            assert all((oc["category"][idx] == c).sum() == n for c in CATS)
        # draws are defined on qids, not on storage positions
        assert [sorted(oc["qids"][i].tolist()) for i in d1[n]] == [sorted(oc["qids"][::-1][i].tolist()) for i in d2[n]]
        assert len({tuple(sorted(i.tolist())) for i in d1[n]}) > 1           # draws differ
    with pytest.raises(ValueError):
        s4_draws(oc["qids"], oc["category"], [31], 1, 0)
    r = s4_run(oc, d1, 0.05, 0.30)
    assert set(r) == {"5", "10", "full"} and r["full"]["n"] == 1 and len(r["5"]["values"]) == 6
    s = summarize([1, 2, 3, 4])
    assert s["median"] == 2.5 and s["q25"] == 1.75 and s["q75"] == 3.25


def test_study_statistics():
    from capprune.confirmatory_study import across_seeds, bootstrap_mean_ci, sign_test

    a = across_seeds({0: {"ppl_top1": 10, "cap_top1": 12, "cap_klprobe": 12}, 1: {"ppl_top1": 5, "cap_top1": 9, "cap_klprobe": None},
                      2: {"ppl_top1": 7, "cap_top1": 8, "cap_klprobe": 9}})
    assert a["b_minus_a"]["per_seed"] == {0: 2, 1: 4, 2: 1} and a["b_minus_a"]["consistent_sign"]
    assert a["c_minus_b"]["seeds"] == 2 and not a["c_minus_b"]["consistent_sign"] and a["cap_klprobe"]["undefined"] == 1
    assert sign_test(4, 0) == pytest.approx(0.125) and sign_test(0, 0) == 1.0
    lo, hi = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], 2000, 0)
    assert lo < 2.5 < hi and bootstrap_mean_ci([1.0], 10, 0) == (None, None)


def test_pilot_scores_every_frozen_checkpoint_and_keeps_the_primary_decision(study, pilot):
    from capprune.confirmatory import load_outcomes, sequence_claim
    from capprune.screen_confirm import fixed_sequence
    from capprune.utils.io import sha256_file

    for r in pilot["confirm"]["runs"]:
        assert r["amendment"] == 3
        scored = r["tested"] + r["scored_after_failure"]
        assert [x["position"] for x in scored] == list(range(1, r["frozen_checkpoints"] + 1))   # every checkpoint
        # the decision uses only the prefix up to the first failure
        fs = fixed_sequence([{"id": x["position"], "params_removed": x["params_removed"],
                              "counts": {c: tuple(v) for c, v in x["counts"].items()}} for x in scored], 0.05, 0.15)
        assert fs["claim"] == r["claim_position"] and len(fs["tested"]) == len(r["tested"])
        npz = pilot["dir"] / r["outcomes_file"]
        assert sha256_file(npz) == r["outcomes_sha256"]
        oc = load_outcomes(npz)
        assert oc["cand"].shape == (r["frozen_checkpoints"], 80) and sequence_claim(oc, 0.05, 0.15)[0] == r["claim_position"]
    for r in pilot["report"]["runs"]:
        s = r["secondary"]
        assert "post hoc" in s["label"] and set(s["S4"]) == {"5", "10", "full"} and len(s["S4"]["5"]["values"]) == 4
        assert s["S1"]["overshoot_parameters"] == s["S1"]["screened_final_parameters_removed"] - r["confirmed_parameters_removed"]
        assert s["S3"]["parameters_removed"] >= r["confirmed_parameters_removed"]                  # 25 >= 15 points
        assert s["S4"]["full"]["values"] == [r["confirmed_parameters_removed"]]


def test_reconfirming_a_pre_amendment_batch_keeps_its_records_and_refuses_a_changed_claim(study, pilot):
    from capprune.confirmatory import confirm

    b = pilot["dir"]
    keep = {p.name: p.read_bytes() for p in b.iterdir() if p.is_file()}
    old = json.loads((b / "confirm.json").read_text(encoding="utf-8"))
    pre = {k: v for k, v in old.items() if k != "amendment"}
    try:
        # a changed claim: refused, nothing written
        bad = json.loads(json.dumps(pre))
        bad["runs"][0]["claim_position"] = 99
        (b / "confirm.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SystemExit, match="differs from the recorded one"):
            confirm(study["cfg"], "pilot")
        assert json.loads((b / "confirm.json").read_text(encoding="utf-8"))["runs"][0]["claim_position"] == 99
        assert not (b / "superseded").exists()
        # the same claims: re-confirmed from the cache, the earlier records moved to superseded/
        (b / "confirm.json").write_text(json.dumps(pre), encoding="utf-8")
        confirm(study["cfg"], "pilot")
        new = json.loads((b / "confirm.json").read_text(encoding="utf-8"))
        assert new["amendment"] == 3 and new["cost"]["items_scored"] == 0                          # all replayed
        assert [r["claim_position"] for r in new["runs"]] == [r["claim_position"] for r in old["runs"]]
        assert "amendment" not in json.loads((b / "superseded" / "confirm.json").read_text(encoding="utf-8"))
        assert (b / "superseded" / "report.json").exists()
    finally:
        shutil.rmtree(b / "superseded", ignore_errors=True)
        for n, data in keep.items():
            (b / n).write_bytes(data)


def test_study_summary_from_the_cli(study, pilot):
    sd = study["t"] / "study-out"
    models = "confirmatory.study.models=[{label: smoke, out_dir: '" + study["out"].as_posix() + "', batches: [pilot]}]"
    rc, out = _cli(study, "confirmatory", "study", "--set", f"confirmatory.study.out_dir={sd.as_posix()}", "--set", models)
    assert rc == 0, out
    S = json.loads((sd / "study.json").read_text(encoding="utf-8"))
    M = S["models"][0]
    conf = {r["method"]: r for r in pilot["confirm"]["runs"]}
    assert {a: M["primary"][a]["per_seed"]["0"] for a in conf} == {a: r["confirmed_parameters_removed"] for a, r in conf.items()}
    assert set(M["secondary"]) == {"all seeds", "excluding the pilot"}
    assert all(x["post_hoc"] for x in M["per_run_secondary"])
    assert M["c1"]["social_sciences_gate_pool"] == study["reg"]["sets"]["gate"]["per_category"]["Social Sciences"]
    assert S["c6"]["all"]["primary_KL_vs_rank2"]["rejections"] == len(S["c6"]["rows"])
    assert "## C6" in (sd / "study.md").read_text(encoding="utf-8")


def _synthetic_run(method, seed, confirmed, n_tested, scored, cf_logs=(), charged=0, gate_s=0.0, kl_s=0.0):
    s4 = {"500": {"median": confirmed, "q25": confirmed, "q75": confirmed, "values": [confirmed, confirmed]},
          "full": {"median": confirmed, "q25": confirmed, "q75": confirmed, "values": [confirmed]}}
    final = scored[-1]["params"] if scored else 0
    return {"run_id": f"{method}__t00__s{seed}", "method": method, "seed": seed, "batch": "main", "post_hoc_secondary": False,
            "confirmed_parameters_removed": confirmed, "dense_parameters": 1000, "n_tested": n_tested, "claim_position": None,
            "scored": [{"position": i + 1, "estimate": x["est"], "pool_loss": x.get("pool"), "pass": x["pass"]} for i, x in enumerate(scored)],
            "secondary": {"S1": {"overshoot_parameters": final - confirmed}, "S3": {"parameters_removed": confirmed, "claim_position": None},
                          "S4": s4},
            "cost": {"questions_charged_to_cap": charged, "screening_gate_questions": charged, "kl_probe_items": 0,
                     "counterfactual_items": 0, "seconds": {"gate_s": gate_s, "kl_probe_s": kl_s}},
            "stop_reason": "x", "accepts": len(scored), "attempts": len(scored), "wall": {"seconds": 60.0, "attempts": 1},
            "screened_final": {"materialized_parameters_removed": final}, "power_achieved": None, "counterfactual_logs": list(cf_logs)}


def test_study_analysis_c1_c4_c6_and_s2():
    from capprune.confirmatory import _secondary_cfg
    from capprune.confirmatory_study import analyse, render

    u = lambda v: {c: v for c in CATS}  # noqa: E731
    cf = {"picks": {"KL": 5, "gradient rank 2": 2, "TV": 5, "S1c": 3},
          "by_rank": {"5": {"worst": 0.02}, "2": {"worst": 0.06}, "3": {"worst": 0.02}}}
    runs = []
    for s in (0, 1, 2):
        ppl = [{"est": u(0.0), "pass": True, "params": 100}, {"est": u(0.05), "pass": False, "params": 200}]
        gate = [{"est": u(0.01), "pool": u(0.0), "pass": True, "params": 100}, {"est": u(0.04), "pool": u(0.01), "pass": False, "params": 200}]
        runs += [_synthetic_run("ppl_top1", s, 50 if s else 100, 2, ppl),
                 _synthetic_run("cap_top1", s, 100, 2, gate, charged=1000, gate_s=10.0),
                 _synthetic_run("cap_klprobe", s, 100, 2, gate, cf_logs=[cf, cf], charged=1400, gate_s=10.0, kl_s=5.0)]
    M = {"label": "m", "model_key": "mk", "runs": runs, "batches": [], "n_confirmation": 100, "per_category": u(25),
         "gate_pool": u(10), "registration_sha16": "0" * 16, "admissibility": None,
         "confirm_rate": {"items": 1000, "seconds": 10.0, "seconds_per_item": 0.01}}
    R = analyse([M], _secondary_cfg({"confirmatory": {"secondary": {"c1_bootstrap": {"resamples": 500, "seed": 0}}}}),
                {"model_key": "mk", "seed": 0, "label": "seed 0"})
    m = R["models"][0]
    # primary, paired across seeds
    assert m["primary"]["b_minus_a"]["per_seed"] == {0: 0, 1: 50, 2: 50} and not m["primary"]["b_minus_a"]["consistent_sign"]
    # C1 over the tested checkpoints of (b) and (c): gaps +1 and +3 points, 6 runs x 2 checkpoints
    t = m["c1"]["tested_summary"]
    assert t["checkpoints"] == 12 and t["mean_worst_gap"] == pytest.approx(0.02) and m["c1"]["model_criterion_met"] is True
    assert t["per_category_max_gap"]["STEM"] == pytest.approx(0.03) and m["c1"]["social_sciences_gate_pool"] == 10
    # C4: (a) < (b) in seeds 1 and 2; (a)'s endpoint at +5 points in every seed
    assert m["c4"]["a_less_than_b_seeds"] == 2 and m["c4"]["final_exceeds_seeds"] == 3 and m["c4"]["model_criterion_met"]
    assert m["c4"]["per_seed"][0]["first_failing_position"] == 2
    # S2: items = charged + 100 x (1 + 2 tested); seconds = gate + probe + 300 items x 0.01 s
    s2 = {(x["method"], x["seed"]): x["S2"] for x in m["per_run_secondary"]}
    assert s2[("cap_klprobe", 1)]["items"] == 1400 + 300 and s2[("cap_klprobe", 1)]["gpu_seconds"] == pytest.approx(15.0 + 3.0)
    assert s2[("cap_klprobe", 1)]["items_per_confirmed_M"] == pytest.approx(1700 / 1e-4)
    assert set(m["secondary"]) == {"all seeds", "excluding seed 0"}
    assert m["secondary"]["excluding seed 0"]["S1"]["ppl_top1"]["per_seed"] == {1: 150, 2: 150}
    # C6: 6 rejections, KL better than rank 2 in all, TV ties KL, S1c ties KL
    c6 = R["c6"]["all"]
    assert c6["primary_KL_vs_rank2"] == {"rejections": 6, "better": 6, "worse": 0, "ties": 0, "p_two_sided": pytest.approx(2 / 64)}
    assert c6["secondary_TV_vs_KL"]["ties"] == 6 and R["c6"]["criterion_met"]
    assert R["c6"]["excluding_pilot"]["primary_KL_vs_rank2"]["rejections"] == 4
    R.update({"commit": "0" * 40, "clean_tree": True})
    assert "## C6" in render(R)
