"""Study 2 groundwork (docs/study2_groundwork.md): CPU only, toy models and synthetic data."""
import copy
import json
import math
import subprocess
import time

import numpy as np
import pytest
import torch

from capprune import bounds as B
from capprune import study2_groundwork as g
from capprune import supervisor as S
from capprune import unstructured_feasibility as uf
from capprune.model import build_units
from capprune.pruning import wanda
from capprune.pruning.masks import DenseStore, LiveDenseStore, PruneState
from capprune.pruning.reload import fingerprint, restore_dense, weight_keys
from capprune.utils import runtime as rt

ELIG = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CFG = {"pruning": {"mode": "unstructured", "max_unit_sparsity": 0.9, "unstructured_count": "floor",
                   "structured": {"channel_multiple": 16, "max_unit_sparsity": 0.875}}}


def _toy(seed: int = 0, layers: int = 3):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=192, num_hidden_layers=layers, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, max_position_embeddings=128, tie_word_embeddings=False)
    m = Qwen3ForCausalLM(c).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    return m


def _calib():
    return torch.randint(0, 97, (12, 32), generator=torch.Generator().manual_seed(5))


# ============================================================================ P1: cached block inputs
@pytest.mark.parametrize("mode,s", [("unstructured", 0.05), ("unstructured", 0.30), ("unstructured", 0.5), ("structured", 0.25)])
def test_cached_wanda_gives_the_original_masks_and_weights(mode, s):
    a = _toy()
    b, c = copy.deepcopy(a), copy.deepcopy(a)
    out = []
    for m, kw, live in ((a, {}, False), (b, {"cache_inputs": True}, False),
                        (c, {"cache_inputs": True, "offload_masks": True}, True)):
        units = build_units(m, mode, ELIG)
        st = PruneState(units, mode, mask_device="cpu" if kw.get("offload_masks") else None)
        dense = LiveDenseStore(units) if live else DenseStore(units)
        wanda.sequential_wanda(m, st, dense, _calib(), s, {**CFG, "pruning": {**CFG["pruning"], "mode": mode}},
                               batch_size=4, **kw)
        out.append((uf.mask_hash(st.masks), units, st))
    assert out[0][0] == out[1][0] == out[2][0]
    assert all(torch.equal(x.linears[k].weight, y.linears[k].weight) and torch.equal(x.linears[k].weight, z.linears[k].weight)
               for x, y, z in zip(out[0][1], out[1][1], out[2][1]) for k in x.linears)
    assert {mk.device.type for mk in out[2][2].masks.values()} == {"cpu"}  # offloaded


def test_cached_wanda_refuses_layers_with_different_attention_masks():
    m = _toy()
    m.config.layer_types = ["full_attention", "sliding_attention", "full_attention"]
    units = build_units(m, "unstructured", ELIG)
    with pytest.raises(NotImplementedError):
        wanda.sequential_wanda(m, PruneState(units, "unstructured"), DenseStore(units), _calib(), 0.1, CFG,
                               batch_size=4, cache_inputs=True)


def test_live_dense_store_refuses_a_second_read():
    units = build_units(_toy(), "unstructured", ELIG)
    live = LiveDenseStore(units)
    live.get(units[0].name, "w")
    with pytest.raises(RuntimeError):
        live.get(units[0].name, "w")
    live.reset()
    live.get(units[0].name, "w")


# ============================================================================ P2: one resident copy, restored from disk
def test_restore_from_disk_returns_the_exact_dense_weights_and_the_checksum_catches_a_change(tmp_path):
    m = _toy(seed=3).to(torch.bfloat16)
    m.save_pretrained(tmp_path, safe_serialization=True)
    units = build_units(m, "unstructured", ELIG)
    assert set(weight_keys(m, units).values()) <= {n for n, _ in m.named_parameters()}
    fp0 = fingerprint(units)
    ref = {u.name: u.weight.detach().clone() for u in units}
    st = PruneState(units, "unstructured", mask_device="cpu")
    wanda.sequential_wanda(m, st, LiveDenseStore(units), _calib(), 0.43, CFG, batch_size=4, cache_inputs=True,
                           offload_masks=True)
    assert fingerprint(units) != fp0
    info = restore_dense(m, units, tmp_path)
    assert info["tensors"] == len(units) and fingerprint(units) == fp0
    assert all(torch.equal(u.weight, ref[u.name]) for u in units)
    with torch.no_grad():  # one changed 16-bit word, or two swapped values, change the checksum
        w = units[4].weight
        w.view(torch.int16)[0, 0] += 1
        assert fingerprint(units) != fp0
        restore_dense(m, units, tmp_path)
        a, b = w[0, 0].clone(), w[0, 1].clone()
        if not torch.equal(a, b):
            w[0, 0], w[0, 1] = b, a
            assert fingerprint(units) != fp0


def test_a_level_from_restored_weights_has_the_hash_of_a_level_from_a_fresh_load(tmp_path):
    m = _toy(seed=4).to(torch.bfloat16)
    m.save_pretrained(tmp_path, safe_serialization=True)
    fresh = copy.deepcopy(m)
    units = build_units(m, "unstructured", ELIG)
    live = LiveDenseStore(units)
    st = PruneState(units, "unstructured", mask_device="cpu")
    wanda.sequential_wanda(m, st, live, _calib(), 0.05, CFG, batch_size=4, cache_inputs=True, offload_masks=True)
    restore_dense(m, units, tmp_path)
    live.reset()
    st = PruneState(units, "unstructured", mask_device="cpu")
    wanda.sequential_wanda(m, st, live, _calib(), 0.30, CFG, batch_size=4, cache_inputs=True, offload_masks=True)
    fu = build_units(fresh, "unstructured", ELIG)
    sf = PruneState(fu, "unstructured")
    wanda.sequential_wanda(fresh, sf, DenseStore(fu), _calib(), 0.30, CFG, batch_size=4)
    assert uf.mask_hash(st.masks) == uf.mask_hash(sf.masks)


def test_score_proxy_batches_are_left_padded_like_the_scorer():
    w = torch.arange(4 * 64).view(4, 64)
    ids, att, pos = g.score_proxy_batch(w, 8, 16, pad_id=-1)
    assert ids.shape == (8, 16) and att[0].all() and (att.sum(1) >= 8).all()
    for j in range(8):
        k = int((att[j] == 0).sum())
        assert (ids[j, :k] == -1).all() and (att[j, k:] == 1).all() and pos[j, -1] == 16 - k - 1


# ============================================================================ P3: the hang timer and the wall clock
class _FakeWorker:
    def __init__(self, exit_after: float | None = None):
        self.pid, self.returncode, self.stopped = 4242, None, False
        self._capprune_launch = "launch-1"
        self.t0, self.exit_after = time.monotonic(), exit_after

    def wait(self, timeout=None):
        if self.stopped or (self.exit_after is not None and time.monotonic() - self.t0 > self.exit_after):
            self.returncode = 0 if not self.stopped else 1
            return self.returncode
        time.sleep(min(timeout or 0.0, 0.02))
        raise subprocess.TimeoutExpired("worker", timeout)


def _sup(tmp_path, hang_timeout: float):
    sup = S.Supervisor.__new__(S.Supervisor)
    sup.root, sup.hang_timeout, sup.grace = tmp_path, hang_timeout, 1.0
    sup._clock, sup._poll_s = time.monotonic, 0.02
    sup._stop_child = lambda p, grace: setattr(p, "stopped", True)
    (tmp_path / "run").mkdir()
    return sup


def _beat(tmp_path, seq: int, ts: float, launch="launch-1"):
    (tmp_path / "run" / "heartbeat.json").write_text(json.dumps(
        {"pid": 9999, "ts": ts, "phase": "search", "detail": f"it {seq}", "started": ts, "launch": launch, "seq": seq}))


def test_a_hang_is_detected_when_the_wall_clock_steps_back(tmp_path, monkeypatch):
    """docs/study2_groundwork.md P3: at d77b9f3 an NTP correction stepped the clock back 5 h ten seconds after the hung
    worker's last heartbeat; with wall-clock arithmetic the hang would have been seen 5 h later."""
    sup = _sup(tmp_path, hang_timeout=0.4)
    real = time.time()
    _beat(tmp_path, 1, real)
    monkeypatch.setattr(S.time, "time", lambda: real - 18_000.0)  # the step, after the last heartbeat
    t0 = time.monotonic()
    assert sup._watch({"id": "run"}, _FakeWorker()) == 124
    assert time.monotonic() - t0 < 5.0


def test_a_forward_clock_step_does_not_kill_a_worker_that_is_progressing(tmp_path, monkeypatch):
    import threading

    sup = _sup(tmp_path, hang_timeout=0.4)
    real = time.time()
    monkeypatch.setattr(S.time, "time", lambda: real + 18_000.0)
    stop = threading.Event()

    def beats():
        i = 0
        while not stop.is_set():
            i += 1
            _beat(tmp_path, i, real)  # the same (stale-looking) wall time, but a new write each time
            time.sleep(0.05)

    th = threading.Thread(target=beats)
    th.start()
    try:
        assert sup._watch({"id": "run"}, _FakeWorker(exit_after=1.5)) == 0
    finally:
        stop.set()
        th.join()


def test_another_launch_heartbeat_is_not_progress(tmp_path):
    sup = _sup(tmp_path, hang_timeout=0.3)
    _beat(tmp_path, 1, time.time(), launch="an-earlier-launch")
    assert sup._watch({"id": "run"}, _FakeWorker()) == 124
    assert S.heartbeat_signature({"launch": "x", "seq": 1, "ts": 1.0}, 1, 0.0, "y") is None
    assert S.heartbeat_signature({"launch": "x", "seq": 1, "ts": 1.0}, 1, 0.0, "x") is not None
    assert S.heartbeat_signature({"pid": 7, "ts": 1500.0, "started": 1001.0}, 7, 1000.0, "x") is not None  # old worker


def test_the_heartbeat_keeps_writing_after_the_wall_clock_steps_back(tmp_path, monkeypatch):
    hb = rt.Heartbeat(tmp_path, min_interval_s=0.05)
    hb.beat("search", "a")
    first = json.loads((tmp_path / "heartbeat.json").read_text())
    real = time.time()
    monkeypatch.setattr(rt.time, "time", lambda: real - 18_000.0)
    time.sleep(0.08)
    hb.beat(None, "b")
    second = json.loads((tmp_path / "heartbeat.json").read_text())
    assert second["seq"] == first["seq"] + 1 and second["detail"] == "b"


def test_the_supervisor_sleep_is_monotonic(tmp_path, monkeypatch):
    sup = _sup(tmp_path, hang_timeout=1.0)
    monkeypatch.setattr(S.time, "time", lambda: 0.0)  # a frozen (or stepped) wall clock does not stretch the backoff
    t0 = time.monotonic()
    sup._sleep(0.2)
    assert 0.15 < time.monotonic() - t0 < 2.0


# ============================================================================ P4: the exact test's grid fix
@pytest.mark.parametrize("n,tol", [(300, 0.03), (300, 0.02)])
def test_exact_test_v2_is_certified_and_its_size_is_at_most_alpha_off_the_grid(n, tol):
    r = B.exact_score_critical_v2(n, tol, 0.05)
    assert r["certified"] and r["certified_sup_bound"] <= 0.05 and r["grid_max_size"] <= 0.05 - B.V2_MARGIN
    assert r["z"] <= r["z_v1"] + 1e-9
    top = (1 - tol) / 2 * (1 - 1e-9)
    rng = np.random.default_rng(11)
    pts = np.concatenate([rng.uniform(0, top, 300), rng.uniform(0.95 * top, top, 200)])
    assert max(B.size_point(n, tol, r["z"], x) for x in pts) <= 0.05


def test_the_slope_bound_holds_against_finite_differences():
    n, tol, z = 400, 0.03, B.exact_score_critical(400, 0.03, 0.05)
    for gg in (0.002, 0.03, 0.2, 0.45, 0.48):
        h = 1e-6
        d = (B.size_point(n, tol, z, gg + h) - B.size_point(n, tol, z, gg - h)) / (2 * h)
        assert abs(d) <= B.fisher_slope_bound(n, tol, gg - h, gg + h)
        sa, sb = B.size_point(n, tol, z, gg - 1e-3), B.size_point(n, tol, z, gg + 1e-3)
        mids = [B.size_point(n, tol, z, x) for x in np.linspace(gg - 1e-3, gg + 1e-3, 21)]
        assert max(mids) <= B.interval_size_bound(n, tol, gg - 1e-3, gg + 1e-3, sa, sb) + 1e-12


def test_the_v1_excess_at_n_3000_is_gone_with_v2():
    """The note of 2026-11-25: with v1 the size between grid points near p_gained = 0.479 exceeds 0.05 by about 7e-6."""
    n, tol = 3000, 0.03
    r = B.exact_score_critical_v2(n, tol, 0.05)
    off = np.arange(0.4780, 0.4810, 0.0001)
    assert B.size_at(n, tol, r["z"], off).max() <= 0.05
    assert r["certified"] and r["certified_sup_bound"] <= 0.05


# ============================================================================ P5: letters
def test_chi_square_tail_for_even_degrees_of_freedom():
    assert g.chi2_sf_even(3.0, 2) == pytest.approx(math.exp(-1.5))
    assert g.chi2_sf_even(12.5916, 6) == pytest.approx(0.05, abs=1e-4)  # the 95 % point of chi-square(6)


def test_the_change_table_and_a_planted_letter_bias_is_recovered():
    rng = np.random.default_rng(0)
    n = 4000
    answer = rng.integers(0, 4, n)
    logp = rng.normal(0, 1, (n, 4))
    logp[np.arange(n), answer] += 1.2
    dense = logp.argmax(1) == answer
    cand = (logp + np.array([0.8, 0.0, 0.0, 0.0])).argmax(1) == answer  # a pure shift toward A
    t = g.change_table(answer, dense, cand)
    assert t.sum() == n and t[0, 1] > 0 and t[0, 0] == 0  # answer-A items can only be gained
    assert g.homogeneity(t)["p"] < 1e-6
    r = g.bias_shift(logp, answer, dense, cand)
    assert r["bias"][0] - max(r["bias"][1:3]) == pytest.approx(0.8, abs=0.1)
    assert r["pruned_lost_given_model_lost"] > 0.9 and r["loss_model"] == pytest.approx(r["loss_pruned"], abs=0.005)
