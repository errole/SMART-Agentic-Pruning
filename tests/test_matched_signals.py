"""Matched-compression signal test, CPU building blocks: the split, the subsamples, the signal formulas, the choice."""
from __future__ import annotations

import math

import numpy as np
import pytest

CATS = {"STEM": 1303, "Humanities": 1304, "Social Sciences": 1303, "Other": 1304}


def _pool():
    return {c: [f"{c[:3]}-{i:05d}" for i in range(n)] for c, n in CATS.items()}


# ---------------------------------------------------------------------------------------------------------- split
def test_split_is_disjoint_complete_stratified_and_seeded():
    from capprune.matched_signals import make_split

    pool = _pool()
    s = make_split(pool, 20261123)
    for c, q in pool.items():
        a, b = set(s["half1"][c]), set(s["half2"][c])
        assert not a & b                                   # disjoint
        assert a | b == set(q)                             # complete
        assert len(a) == math.ceil(len(q) / 2) and len(b) == len(q) // 2   # stratified: halves within each category
        assert s["half1"][c] == sorted(a)
    every = [x for h in ("half1", "half2") for c in s[h] for x in s[h][c]]
    assert len(every) == len(set(every)) == sum(CATS.values())  # no item in two categories or two halves
    # deterministic, independent of input order, and seed-dependent
    shuffled = {c: list(reversed(q)) for c, q in reversed(list(pool.items()))}
    assert make_split(shuffled, 20261123) == s
    assert make_split(pool, 20261124)["half1"]["STEM"] != s["half1"]["STEM"]
    # a category's split depends only on its own items (stratum-local rng)
    only = make_split({"STEM": pool["STEM"]}, 20261123)
    assert only["half1"]["STEM"] == s["half1"]["STEM"]


# ---------------------------------------------------------------------------------------------------------- subsamples
def test_subsamples_are_deterministic_without_replacement_and_inside_half1():
    from capprune.matched_signals import make_split, subsample

    s = make_split(_pool(), 20261123)
    h1 = s["half1"]
    a = subsample(h1, 300, 4, 7)
    assert a == subsample(h1, 300, 4, 7)                                     # reproducible
    assert a == subsample({c: h1[c] for c in reversed(list(h1))}, 300, 4, 7)  # order of categories irrelevant
    for c in h1:
        assert len(a[c]) == 300 and len(set(a[c])) == 300                   # without replacement
        assert set(a[c]) <= set(h1[c]) and not set(a[c]) & set(s["half2"][c])  # never touches half 2
    assert subsample(h1, 300, 5, 7)["STEM"] != a["STEM"]                    # draws differ
    assert subsample(h1, 300, 4, 8)["STEM"] != a["STEM"]                    # seed matters
    assert set(subsample(h1, 200, 4, 7)["STEM"]) != set(a["STEM"][:200])   # each n has its own draws
    assert len(subsample({"x": ["a", "b"]}, 5, 0, 7)["x"]) == 2              # capped at the half's size


# ---------------------------------------------------------------------------------------------------------- signals
def _lp(p):
    return np.log(np.asarray(p, float))


def test_signal_formulas_on_hand_computed_items():
    from capprune.matched_signals import item_terms, signal_means

    # 4 items, answer 0 each. dense: right, right, wrong, right; pruned: wrong, right, right, right.
    dense = _lp([[0.7, 0.1, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1], [0.6, 0.2, 0.1, 0.1]])
    pruned = _lp([[0.2, 0.6, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1], [0.5, 0.3, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1]])
    ans = np.zeros(4, int)
    cd, cp = dense.argmax(1) == ans, pruned.argmax(1) == ans
    t = item_terms(dense, pruned, ans, cd.astype(np.int8), cp.astype(np.int8))
    assert list(t["flip"]) == [1, 0, -1, 0]
    m = signal_means(t, np.ones(4, bool))
    assert m["S0"] == pytest.approx(0.0)                                     # (lost - gained) / n = (1 - 1) / 4
    p, q = np.exp(dense), np.exp(pruned)
    kl = (p * np.log(p / q)).sum(1)
    assert m["S2"] == pytest.approx(kl.mean())
    # S1c: only the dense-correct items (0, 1, 3)
    d = np.log(p[:, 0]) - np.log(q[:, 0])
    assert m["S1c"] == pytest.approx(d[[0, 1, 3]].mean())
    assert t["kl"][1] == pytest.approx(0.0) and t["dlogp"][1] == pytest.approx(0.0)
    sub = signal_means(t, np.array([True, False, True, False]))
    assert sub["S0"] == pytest.approx(0.0) and sub["S1c"] == pytest.approx(d[0])
    assert math.isnan(signal_means(t, np.array([False, False, True, False]))["S1c"])  # no dense-correct item


def test_signals_renormalise_over_the_four_letters():
    from capprune.matched_signals import item_terms

    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(4), size=50)
    q = rng.dirichlet(np.ones(4), size=50)
    ans = rng.integers(0, 4, 50)
    c1, c2 = (p.argmax(1) == ans).astype(np.int8), (q.argmax(1) == ans).astype(np.int8)
    full = item_terms(np.log(p), np.log(q), ans, c1, c2)
    # the same distributions carrying only part of the vocabulary's mass (as full-vocab log-softmax letters do)
    leak = item_terms(np.log(p * 0.3), np.log(q * 0.8), ans, c1, c2)
    assert np.allclose(full["kl"], leak["kl"]) and np.allclose(full["dlogp"], leak["dlogp"])
    assert (full["kl"] >= -1e-12).all()
    same = item_terms(np.log(p), np.log(p), ans, c1, c1)
    assert np.allclose(same["kl"], 0) and np.allclose(same["dlogp"], 0) and not same["flip"].any()


# ---------------------------------------------------------------------------------------------------------- choice
def test_choice_regret_spearman_and_calls():
    from capprune.matched_signals import choose, decide, lead_n, spearman

    assert choose([0.02, 0.01, 0.01], [1, 2, 3]) == 1        # lowest; tie -> lower gradient rank
    assert choose([0.01, 0.01, 0.01], [3, 1, 2]) == 1
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3, 4])[0, 1])

    cats = ["a", "b"]
    # three actions; half-2 worst-category losses 0.05, 0.02, 0.04 -> best 0.02 (rank 2)
    target = [{"a": 0.05, "b": 0.01}, {"a": 0.02, "b": 0.00}, {"a": 0.01, "b": 0.04}]
    # S0 (noisy) prefers rank 3; KL prefers rank 2; S1c ties everywhere -> rank 1
    vals = [{k: {"S0": s0, "S2": kl, "S1c": 0.1} for k, s0, kl in zip(cats, r0, r2)}
            for r0, r2 in (((0.03, 0.0), (0.5, 0.1)), ((0.02, 0.025), (0.2, 0.1)), ((0.01, 0.02), (0.1, 0.3)))]
    r = decide([1, 2, 3], vals, target, 0.03, {"S2": (0.0, 0.1), "S1c": (0.0, 0.1)})
    assert r["regret"]["S0"] == pytest.approx(0.04 - 0.02)
    assert r["regret"]["S2"] == pytest.approx(0.0)
    assert r["regret"]["S1c"] == pytest.approx(0.05 - 0.02)
    assert r["regret"]["gradient rank 1"] == pytest.approx(0.03)
    assert r["regret"]["random (expected)"] == pytest.approx((0.05 + 0.02 + 0.04) / 3 - 0.02)
    # truth (worst <= 3 pts): fail, pass, fail. S0 calls: max 0.03 pass, 0.025 pass, 0.02 pass -> 1/3 right
    assert r["call_acc"]["S0"] == pytest.approx(1 / 3)
    # S2 mapped 0.1 * worst KL: 0.05 fail, 0.02 pass, 0.03 pass -> 2/3 right
    assert r["call_acc"]["S2"] == pytest.approx(2 / 3)
    assert math.isnan(r["spearman"]["S1c"])                    # a constant signal has no ranking

    assert lead_n([100, 200, 300, 400], {100: 1, 200: 0.5, 300: 0.4, 400: 0.3}, {100: 0.9, 200: 0.6, 300: 0.5, 400: 0.4}) == 200
    assert lead_n([100, 200], {100: 0.1, 200: 0.5}, {100: 0.2, 200: 0.4}) is None
    assert lead_n([100, 200], {100: 0.2, 200: 0.2}, {100: 0.2, 200: 0.2}) is None  # ties are not a lead


# ---------------------------------------------------------------------------------------------------------- analyze end to end
def test_analyze_end_to_end_on_synthetic_scores(tmp_path):
    """A planted landscape: per parent, action j loses j/10 of the dense-correct items (half 1 and half 2 alike) and its
    KL grows with j, so every signal should pick rank 1 (zero loss) once n is large."""
    import hashlib
    import json

    from conftest import ROOT

    from capprune import matched_signals as ms
    from capprune.utils.io import atomic_write_json, sha256_file

    pool = {c: [f"{c[:3]}-{i:05d}" for i in range(900)] for c in ("Humanities", "Other", "STEM", "Social Sciences")}
    cfg = {"_config_dir": str(ROOT / "configs"),
           "matched_signals": {"plan_file": "../docs/matched_signal_plan.md", "out_dir": str(tmp_path), "runs_root": str(tmp_path)}}
    P, plan_sha = ms.plan(cfg)
    split = ms.make_split(pool, int(P["split_seed"]))
    body = {"split_seed": P["split_seed"], "pool_items": 3600, **split}
    atomic_write_json(tmp_path / "split.json", body)
    reg = {"plan_file": "docs/matched_signal_plan.md", "plan_sha256_lf": plan_sha, "plan_committed_in": "0" * 40,
           "split_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(), "protocol_commit": None}
    atomic_write_json(tmp_path / "registration.json", reg)
    qids = [q for c in sorted(pool) for q in pool[c]]
    rng = np.random.default_rng(0)
    n = len(qids)
    ans = rng.integers(0, 4, n)
    dense = np.full((n, 4), -3.0)
    dense[np.arange(n), ans] = -0.2                       # dense always right
    arrays = {"qid": np.array(qids), "category": np.array([q[:3] for q in qids]), "answer": ans.astype(np.int8),
              "logp__dense": dense.astype(np.float32), "correct__dense": np.ones(n, np.int8)}
    cands = []
    for p in ("A", "B", "C"):
        for j in range(11):
            h = f"{p}{j:02d}"
            lp = dense.copy()
            lost = rng.random(n) < j / 10 * 0.5
            lp[lost, ans[lost]] = -4.0                    # flips the answer on the lost items
            lp[:, (ans + 1) % 4] += -0.01 * j              # small extra KL that grows with j
            arrays[f"logp__{h}"] = lp.astype(np.float32)
            arrays[f"correct__{h}"] = (lp.argmax(1) == ans).astype(np.int8)
            cands.append({"parent": p, "rank": j + 1, "kind": "k", "layers": [j], "z_g_mean": 0.0, "mask_hash": h,
                          "diagnostic_decision": "accept" if j == 0 else "reject"})
    np.savez_compressed(tmp_path / "logprobs.npz", **arrays)
    atomic_write_json(tmp_path / "manifest.json", {
        "registration": reg, "commit": "x" * 40, "clean_tree": True, "reuse": {"used": False}, "candidates": cands,
        "item_model_evaluations_run": {"dense": n}, "answers_reused": {}, "gpu_seconds": {"dense": 1.0}, "wall_s": 1.0,
        "npz_sha256": sha256_file(tmp_path / "logprobs.npz")})
    md = ms.analyze(cfg)
    S = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert md.exists() and "Lead criterion" in md.read_text(encoding="utf-8")
    assert len(S["rows"]) == 3 * (len(P["sizes"]) * int(P["draws"]) + 1)
    for n_ in P["sizes"]:
        r = S["summary"]["regret"][f"{n_}|pooled"]
        assert r["S2"]["mean"] == pytest.approx(0.0) and r["S0"]["mean"] == pytest.approx(0.0, abs=1e-12)
        assert r["gradient rank 1"]["mean"] == pytest.approx(0.0)
        assert r["random (expected)"]["mean"] > 0.1
        assert S["summary"]["spearman"][f"{n_}|pooled"]["S2"]["mean"] > 0.9
    assert S["lead_n"] is None                             # both are perfect here, so KL does not lead
    assert S["truth_base_rates"]["pooled"]["pass_rate"] == pytest.approx(1 / 11)
    # changing the registered split after the fact is refused
    body["half1"]["STEM"] = body["half1"]["STEM"][1:]
    atomic_write_json(tmp_path / "split.json", body)
    with pytest.raises(SystemExit):
        ms.analyze(cfg)
