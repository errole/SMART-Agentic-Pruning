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
