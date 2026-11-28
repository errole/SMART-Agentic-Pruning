"""Unstructured feasibility (docs/unstructured_feasibility_plan.md section 7): CPU, a toy Qwen3 (random weights)."""
import copy

import numpy as np
import pytest
import torch

from capprune import unstructured_feasibility as uf
from capprune.confirmatory import CATS
from capprune.model import build_units
from capprune.pruning import wanda
from capprune.pruning.masks import DenseStore, PruneState

ELIG = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CFG = {"pruning": {"mode": "unstructured", "max_unit_sparsity": 0.9, "unstructured_count": "floor"}}
LEVELS = [0.05, 0.10, 0.15, 0.43]


def _toy(seed: int = 0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=192, num_hidden_layers=3, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, max_position_embeddings=128, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(c).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    return m


def _pool():
    return torch.randint(0, 97, (16, 32), generator=torch.Generator().manual_seed(123))


def _calib(seed: int):
    """As ``PreparedData.calibration_indices``: the run seed selects the windows of a fixed pool."""
    idx = sorted(np.random.default_rng([int(seed), 7_919]).choice(16, size=8, replace=False).tolist())
    return _pool()[idx]


def _prune(model, s, calib):
    units = build_units(model, "unstructured", ELIG)
    dense = DenseStore(units)
    st = PruneState(units, "unstructured")
    wanda.sequential_wanda(model, st, dense, calib, s, CFG, batch_size=4)
    return units, dense, st


def test_the_published_count_is_floor_and_the_default_is_unchanged():
    assert wanda.unstructured_count(64, 0.43, "floor") == 27 and wanda.unstructured_count(64, 0.43) == 28
    assert wanda.unstructured_count(192, 0.43, "floor") == 82 and wanda.unstructured_count(2048, 0.25, "floor") == 512
    with pytest.raises(ValueError):
        wanda.unstructured_count(64, 0.5, "ceil")


@pytest.mark.parametrize("s", [0.05, 0.10, 0.43, 0.50])
def test_each_layer_reaches_the_requested_sparsity_exactly_per_row(s):
    m = _toy()
    units, _, st = _prune(m, s, _calib(0))
    assert {u.kind for u in units} == set(ELIG) and len(units) == 3 * 7
    for u in units:
        mk = st.masks[u.name]
        rows, cols = mk.shape
        assert torch.all((~mk).sum(dim=1) == int(cols * s)), u.name   # every output row: exactly int(cols x s) pruned
        assert torch.all(u.weight[~mk] == 0)                           # and they are zero in the model
    stats = uf.mask_stats(st, units, s, sum(p.numel() for p in m.parameters()))
    assert stats["per_row_exact"] and stats["masked_weights_not_zero"] == 0
    shapes = {u.name: tuple(u.weight.shape) for u in units}
    acc = uf.accounting(shapes, sum(p.numel() for p in m.parameters()), [s])["levels"][uf.level_name(s)]
    assert acc["zeroed"] == stats["zeroed"]                            # the CPU accounting equals the masks


def test_scores_match_a_direct_computation_of_weight_times_activation_norm(monkeypatch):
    """The activation norms sequential Wanda uses at layer i are those of the model with layers < i pruned and layer i
    dense; the score is |W| x ||X_j||_2; the pruned weights are each row's lowest scores."""
    s = 0.43
    m = _toy()
    dense_model = copy.deepcopy(m)
    seen = {}
    orig = wanda.raise_unit

    def spy(state, name, ratio, act, cfg):
        seen[name] = (state.by_name[name].weight.detach().clone(), act[name].detach().clone())
        return orig(state, name, ratio, act, cfg)

    monkeypatch.setattr(wanda, "raise_unit", spy)
    calib = _calib(0)
    units, _, st = _prune(m, s, calib)
    for li in range(3):
        ref = copy.deepcopy(dense_model)
        runits = {u.name: u for u in build_units(ref, "unstructured", ELIG)}
        for u in units:  # earlier layers pruned exactly as the run pruned them, layer li and later dense
            if u.layer < li:
                runits[u.name].weight.mul_(st.masks[u.name].to(runits[u.name].weight.dtype))
        X, hooks = {}, []
        for name, ru in runits.items():
            if ru.layer == li:
                hooks.append(ru.linears["w"].register_forward_pre_hook(
                    lambda mod, inp, _n=name: X.setdefault(_n, []).append(inp[0].detach().double().reshape(-1, inp[0].shape[-1]))))
        with torch.no_grad():
            ref(input_ids=calib, use_cache=False)
        for h in hooks:
            h.remove()
        for name in [n for n, ru in runits.items() if ru.layer == li]:
            x = torch.cat(X[name])
            norm = x.pow(2).sum(0).sqrt()                               # ||X_j||_2 over every calibration token
            W, sq = seen[name]
            np.testing.assert_allclose(sq.sqrt().double().numpy(), norm.numpy(), rtol=2e-4, atol=1e-6)
            direct = W.double().abs() * norm[None, :]
            got = wanda.linear_scores(W, sq, torch.ones_like(W, dtype=torch.bool)).double()
            np.testing.assert_allclose(got.numpy(), direct.numpy(), rtol=2e-4, atol=1e-7)
            mk = st.masks[name]
            for r in range(W.shape[0]):  # the pruned weights are each row's lowest direct scores
                assert direct[r][~mk[r]].max() <= direct[r][mk[r]].min() * (1 + 1e-4) + 1e-9


def _runner(out, seed=0, crash_hook=None, regen=(), levels=LEVELS):
    m = _toy()
    units = build_units(m, "unstructured", ELIG)
    dense = DenseStore(units)
    qids = [f"q{i:03d}" for i in range(40)]
    cats = [CATS[i % 4] for i in range(40)]

    def score_fn(key, model):  # a deterministic function of the level's mask (the hash) and of the model's weights
        w = float(sum(u.weight.abs().sum() for u in units))
        return ({q: True for q in qids}, {q: bool(int(key[i % 64], 16) % 2) or w < 0 for i, q in enumerate(qids)},
                {"items_scored": len(qids)})

    def ppl_fn(key, model):
        return 10.0 + int(key[:4], 16) / 65536

    calls = {"prune": 0}
    r = uf.LevelRunner(out, m, units, dense, _calib(seed), levels, CFG, qids, cats, score_fn, ppl_fn,
                       sum(p.numel() for p in m.parameters()), batch_size=4, crash_hook=crash_hook, regeneration_check=regen)
    orig = r.prune

    def counted(s):
        calls["prune"] += 1
        return orig(s)

    r.prune = counted
    return r, calls


def _hashes(res):
    return [x["mask_hash"] for x in res]


class _Crash(Exception):
    pass


def test_level_masks_are_reproducible_from_the_same_seed(tmp_path):
    a = _hashes(_runner(tmp_path / "a")[0].run())
    b = _hashes(_runner(tmp_path / "b")[0].run())
    c = _hashes(_runner(tmp_path / "c", seed=1)[0].run())
    assert a == b and len(set(a)) == len(LEVELS)
    assert a != c                                                      # other calibration windows, other masks


def test_resume_after_a_crash_gives_the_same_mask_hashes(tmp_path):
    ref = _hashes(_runner(tmp_path / "ref")[0].run())
    out = tmp_path / "crash"

    def crash(where, s):
        if (where, s) == ("scored", 0.15):
            raise _Crash

    with pytest.raises(_Crash):
        _runner(out, crash_hook=crash)[0].run()
    # before the crash: levels 5 % and 10 % have their outcomes; 15 % only its persisted mask hash
    for n in ("s05", "s10"):
        assert (out / "levels" / n / "result.json").exists() and (out / "levels" / n / "outcomes.npz").exists()
    assert (out / "levels" / "s15" / "mask.json").exists() and not (out / "levels" / "s15" / "result.json").exists()
    r, calls = _runner(out)
    res = r.run()
    assert _hashes(res) == ref
    assert calls["prune"] == 2                                         # 15 % regenerated (hash verified) and 43 %
    assert res[2]["regenerated_after_crash"] is not None and res[0]["regenerated_after_crash"] is None
    # a third invocation re-prunes nothing and returns the persisted records
    r3, calls3 = _runner(out)
    assert _hashes(r3.run()) == ref and calls3["prune"] == 0


def test_a_level_whose_regenerated_hash_differs_is_refused(tmp_path):
    out = tmp_path / "t"

    def crash(where, s):
        if (where, s) == ("mask_persisted", 0.10):
            raise _Crash

    with pytest.raises(_Crash):
        _runner(out, crash_hook=crash)[0].run()
    mp = out / "levels" / "s10" / "mask.json"
    rec = uf.read_json(mp)
    rec["mask_hash"] = "0" * 64
    uf.atomic_write_json(mp, rec)
    with pytest.raises(SystemExit, match="may not be regenerated"):
        _runner(out)[0].run()


def test_the_regeneration_check_generates_twice_and_records_it(tmp_path):
    res = _runner(tmp_path / "g", regen=(0.05,), levels=[0.05, 0.10])[0].run()
    m = uf.read_json(tmp_path / "g" / "levels" / "s05" / "mask.json")
    assert m["regeneration_check"]["match"] and m["regeneration_check"]["second_hash"] == res[0]["mask_hash"]
    assert uf.read_json(tmp_path / "g" / "levels" / "s10" / "mask.json")["regeneration_check"] is None


def test_draw_eval_takes_3000_where_possible_and_every_item_otherwise():
    rows = []
    for c, n in zip(CATS, (40, 25, 12, 30)):
        rows += [{"qid": f"{c[:3]}/{i:03d}", "category": c, "source": "s"} for i in range(n)]
    a, per = uf.draw_eval(rows, 30, 7)
    b, _ = uf.draw_eval(list(reversed(rows)), 30, 7)
    assert a == b                                                      # independent of input order
    assert [per[c]["chosen"] for c in CATS] == [30, 25, 12, 30]
    assert per["Humanities"]["all_items_used"] and per["Social Sciences"]["short_of_target"]
    assert per["Other"]["all_items_used"] and not per["Other"]["short_of_target"]
    assert not per["STEM"]["all_items_used"]
    c2, _ = uf.draw_eval(rows, 30, 8)
    assert {r["qid"] for r in a if r["category"] == "STEM"} != {r["qid"] for r in c2 if r["category"] == "STEM"}


def test_analyze_runs_end_to_end_on_synthetic_run_files(tmp_path):
    """The analysis reads the level records the runner wrote, recomputes the reach and verdict, renders the tables
    (with the largest-loss category per level) and draws the figure."""
    out = tmp_path / "out"
    for key in ("qwen3-1.7b", "qwen3-8b"):
        md = out / key
        r, _ = _runner(md)
        res = r.run()
        (md / "eval").mkdir(parents=True, exist_ok=True)
        uf.atomic_write_json(md / "eval" / "eval_set.json", {"total": 40, "qid_sha256": "x"})
        np.savez_compressed(md / "dense.npz", qids=np.array(r.qids), category=r.cats, dense=np.ones(40, bool))
        uf.atomic_write_json(md / "dense.json", {"outcomes_sha256": uf.sha256_file(md / "dense.npz"), "ppl": 10.0,
                                                 "total_parameters": r.total, "prunable_parameters": 1,
                                                 "accuracy": {c: 1.0 for c in CATS}, "n": {c: 10 for c in CATS}})
        assert len(res) == len(LEVELS)
    uf.atomic_write_json(out / "attainability.json", {"go_attainable": True})
    (tmp_path / "sealed").mkdir()
    uf.atomic_write_json(tmp_path / "sealed" / "admissibility.json", {"admissible": True})
    step = lambda t, w: {"step": t, "params_removed": 1000 * t, "ops": {  # noqa: E731
        "mask": {"worst": w, "ppl_ratio": 1.0, "categories": {c: {"estimate": w if c == "STEM" else 0.0} for c in CATS}},
        "comp": {"worst": w}}}
    uf.atomic_write_json(tmp_path / "chan.json", {"dense_parameters": 100_000, "R_mask": 1, "R_comp": 1,
                                                  "per_step": [step(1, 0.01), step(2, 0.05)]})
    cfg = {"_config_dir": str(tmp_path), "unstructured": {
        "out_dir": str(out), "levels": LEVELS, "base_paper_level": 0.43, "reach_limit": 0.02,
        "rule": {"model": "qwen3-8b", "go_min_share": 0.20, "nogo_below_share": 0.10},
        "confirmation_rule": {"alpha": 0.05, "tolerance": 0.03}, "channel_reference": str(tmp_path / "chan.json"),
        "base_paper": {"source": "s", "model": "Qwen3-8B", "sparsity": 0.4302, "mmlu_dense": 77.38, "mmlu_pruned": 56.67, "shots": 5},
        "models": {"qwen3-1.7b": {"label": "Qwen3-1.7B (development model)", "eval": {"kind": "fixed"}},
                   "qwen3-8b": {"label": "Qwen3-8B", "eval": {"kind": "screening_pool", "sealed_dir": str(tmp_path / "sealed")}}}}}
    md_path = uf.analyze(cfg)
    A = uf.read_json(out / "analysis" / "analysis.json")
    m8 = A["models"]["qwen3-8b"]
    assert A["verdict"] == uf.verdict(m8["reach_share"], cfg["unstructured"]["rule"])
    assert m8["reach_levels"] == uf.reach([x["worst"] for x in m8["levels"]], 0.02)
    assert all(x["worst_category"] in CATS for x in m8["levels"])
    assert A["base_paper_comparison"]["ours"]["name"] == "s43"
    assert (out / "analysis" / "worst_loss.png").exists() and "worst category" in md_path.read_text(encoding="utf-8")


def test_reach_worst_and_verdict():
    assert uf.reach([0.01, 0.02, 0.021, 0.0], 0.02) == 2 and uf.reach([0.03], 0.02) == 0
    rule = {"go_min_share": 0.20, "nogo_below_share": 0.10}
    assert [uf.verdict(x, rule) for x in (0.0, 0.0847, 0.10, 0.1696, 0.20, 0.212)] == \
        ["NO-GO", "NO-GO", "INCONCLUSIVE", "INCONCLUSIVE", "GO", "GO"]
    counts = {"STEM": [100, 5, 1], "Humanities": [100, 6, 2], "Social Sciences": [50, 3, 0], "Other": [100, 0, 0]}
    assert uf.worst_of(counts) == (0.06, "Social Sciences")
