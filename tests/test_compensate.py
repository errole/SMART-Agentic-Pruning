"""Weight compensation (docs/compensation_feasibility_plan.md section 7): CPU, toy MLPs."""
import numpy as np
import pytest
import torch
from torch import nn

from capprune import compensate as cp


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_refit_is_a_no_op_when_no_channel_is_removed():
    g = _rng(0)
    W = torch.tensor(g.standard_normal((6, 10)))
    H = torch.tensor(g.standard_normal((50, 10)))
    C = H.T @ H
    out, info = cp.refit(W, C, torch.ones(10, dtype=torch.bool))
    assert torch.equal(out, W.to(torch.float64)) and info["refit"] is False and info["lambda"] == 0.0


def test_refit_matches_lstsq_with_a_tiny_ridge():
    g = _rng(1)
    T, N, d = 400, 24, 7
    H = g.standard_normal((T, N))
    W = g.standard_normal((d, N))
    keep = np.ones(N, dtype=bool)
    keep[[2, 5, 11, 17, 20]] = False
    out, info = cp.refit(torch.tensor(W), torch.tensor(H.T @ H), torch.tensor(keep), ridge_fraction=1e-12)
    X, *_ = np.linalg.lstsq(H[:, keep], H @ W.T, rcond=None)  # H_K X ~ H W^T
    np.testing.assert_allclose(out.numpy(), X.T, rtol=1e-7, atol=1e-9)
    assert info["refit"] and info["kept"] == N - 5


def test_refit_recovers_the_output_exactly_when_removed_channels_are_combinations_of_kept_ones():
    g = _rng(2)
    T, nk, nr, d = 300, 16, 6, 5
    HK = g.standard_normal((T, nk))
    HR = HK @ g.standard_normal((nk, nr))  # every removed channel is a linear combination of kept ones
    perm = g.permutation(nk + nr)
    H = np.concatenate([HK, HR], axis=1)[:, perm]
    keep = np.concatenate([np.ones(nk, bool), np.zeros(nr, bool)])[perm]
    W = g.standard_normal((d, nk + nr))
    out, _ = cp.refit(torch.tensor(W), torch.tensor(H.T @ H), torch.tensor(keep), ridge_fraction=1e-13)
    recon = H[:, keep] @ out.numpy().T
    np.testing.assert_allclose(recon, H @ W.T, rtol=1e-6, atol=1e-7)
    rel = cp.relative_error(torch.tensor(W), torch.tensor(H.T @ H), torch.tensor(keep), out)
    assert rel < 1e-10


def test_rank_deficient_gram_does_not_crash_and_the_ridge_is_applied():
    g = _rng(3)
    T, N, d = 8, 30, 4  # fewer tokens than kept channels: C[K, K] is singular
    H = g.standard_normal((T, N))
    W = g.standard_normal((d, N))
    keep = np.ones(N, dtype=bool)
    keep[:5] = False
    C = H.T @ H
    with pytest.raises(RuntimeError):  # without the ridge the Cholesky factorization fails
        cp.refit(torch.tensor(W), torch.tensor(C), torch.tensor(keep), ridge_fraction=0.0)
    out, info = cp.refit(torch.tensor(W), torch.tensor(C), torch.tensor(keep), ridge_fraction=0.01)
    lam = 0.01 * np.mean(np.diag(C[np.ix_(keep, keep)]))
    assert info["lambda"] == pytest.approx(lam) and lam > 0
    want = W @ C[:, keep] @ np.linalg.inv(C[np.ix_(keep, keep)] + lam * np.eye(keep.sum()))
    np.testing.assert_allclose(out.numpy(), want, rtol=1e-8, atol=1e-10)
    c = cp.conditioning(torch.tensor(C), torch.tensor(keep), lam)
    assert c["cond"] == float("inf") or c["cond"] > 1e8
    assert np.isfinite(c["cond_ridge"])


def test_gram_in_chunks_equals_one_pass_and_hidden_uses_all_channels():
    g = _rng(4)
    x = torch.tensor(g.standard_normal((64, 6)), dtype=torch.float32)
    wg = torch.tensor(g.standard_normal((10, 6)), dtype=torch.float32)
    wu = torch.tensor(g.standard_normal((10, 6)), dtype=torch.float32)
    h = cp.hidden(x, wg, wu, nn.SiLU())
    assert h.shape == (64, 10)
    C1 = torch.zeros((10, 10), dtype=torch.float64)
    cp.gram_update(C1, h)
    C2 = torch.zeros((10, 10), dtype=torch.float64)
    for part in torch.split(h, 16):
        cp.gram_update(C2, part)
    torch.testing.assert_close(C1, C2)


# ---------------------------------------------------------------------------- the trajectory: persistence and resume
class _Toy(nn.Module):
    """A residual stack of gated MLPs (no norm): x <- x + down(silu(gate(x)) * up(x))."""

    def __init__(self, d=8, n=16, layers=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            b = nn.Module()
            b.gate_proj, b.up_proj, b.down_proj = nn.Linear(d, n, bias=False), nn.Linear(d, n, bias=False), nn.Linear(n, d, bias=False)
            self.blocks.append(b)
        self.act = nn.SiLU()

    def mlp(self, i, x):
        b = self.blocks[i]
        return b.down_proj(self.act(b.gate_proj(x)) * b.up_proj(x))

    def input_of(self, i, x):
        for j in range(i):
            x = x + self.mlp(j, x)
        return x


def _trajectory(tmp, model, dense, masks, crash_hook=None):
    units = {f"L{i}": (b.gate_proj, b.up_proj, b.down_proj) for i, b in enumerate(model.blocks)}
    X0 = torch.tensor(_rng(9).standard_normal((200, 8)), dtype=torch.float32)

    def inputs(name):
        i = int(name[1:])
        with torch.no_grad():
            x = model.input_of(i, X0)
        return list(torch.split(x, 64))

    hashes = [f"mask{t}" for t in range(1, len(masks) + 1)]
    return cp.CompensatedTrajectory(tmp, units, dense, inputs, model.act, masks, hashes, ridge_fraction=0.01,
                                    crash_hook=crash_hook)


def _setup(seed=0):
    model = _Toy(seed=seed)
    dense = {f"L{i}": (b.gate_proj.weight.detach().clone(), b.up_proj.weight.detach().clone(),
                       b.down_proj.weight.detach().clone()) for i, b in enumerate(model.blocks)}
    keep = {f"L{i}": torch.ones(16, dtype=torch.bool) for i in range(3)}
    masks = []
    for drops in ({"L0": [1, 2], "L2": [5]}, {"L1": [0, 3, 4]}, {"L0": [7], "L1": [9]}):
        keep = {k: v.clone() for k, v in keep.items()}
        for u, idx in drops.items():
            keep[u][idx] = False
        masks.append(keep)
    return model, dense, masks


def test_resume_after_a_crash_mid_step_gives_the_same_checkpoint_hashes(tmp_path):
    model, dense, masks = _setup()
    full = _trajectory(tmp_path / "a", model, dense, masks).run(3)
    assert len(set(full)) == 3

    class Crash(Exception):
        pass

    def crash(t, name):
        if t == 2:  # mid-step: after the first (and only partially persisted) refit of step 2
            raise Crash

    model_b, dense_b, _ = _setup()
    with pytest.raises(Crash):
        _trajectory(tmp_path / "b", model_b, dense_b, masks, crash_hook=crash).run(3)
    M = cp.read_json(tmp_path / "b" / "manifest.json")
    assert [s["step"] for s in M["steps"]] == [1]  # step 2 never reached the manifest
    model_c, dense_c, _ = _setup()  # a fresh process: the dense model again; step 1 comes from the persisted weights
    resumed = _trajectory(tmp_path / "b", model_c, dense_c, masks).run(3)
    assert resumed == full
    # and the persisted weights, not a recomputation, define step 1: tampering with them is detected
    f = tmp_path / "b" / "steps" / "step_001.safetensors"
    f.write_bytes(f.read_bytes()[:-1] + bytes([f.read_bytes()[-1] ^ 1]))
    model_d, dense_d, _ = _setup()
    with pytest.raises(RuntimeError):
        _trajectory(tmp_path / "b", model_d, dense_d, masks).restore()


def test_compensation_reduces_the_reconstruction_error_on_the_toy(tmp_path):
    model, dense, masks = _setup(seed=1)
    _trajectory(tmp_path, model, dense, masks).run(3)
    for t in (1, 2, 3):
        rec = cp.read_json(tmp_path / "steps" / f"step_{t:03d}.json")
        for n, d in rec["diagnostics"].items():
            assert d["rel_error_after"] <= d["rel_error_before"] + 1e-12, (t, n, d)
            assert d["refit"] and d["lambda"] > 0


def test_restore_to_an_intermediate_step_uses_its_persisted_weights(tmp_path):
    model, dense, masks = _setup(seed=2)
    traj = _trajectory(tmp_path, model, dense, masks)
    hashes = traj.run(3)
    snap = {}
    for t in (1, 2, 3):
        traj.restore(t)
        assert traj.checkpoint_hash_at(t) == hashes[t - 1]
        snap[t] = [b.down_proj.weight.detach().clone() for b in model.blocks]
    # step 1 touched L0 and L2; step 2 only L1: L0 and L2 keep their step-1 weights at step 2
    assert torch.equal(snap[1][0], snap[2][0]) and torch.equal(snap[1][2], snap[2][2])
    assert not torch.equal(snap[1][1], snap[2][1])
    # removed channels are exactly zero in down_proj; kept rows of gate/up are the dense ones
    traj.restore(3)
    for i, b in enumerate(model.blocks):
        keep = masks[2][f"L{i}"]
        assert torch.count_nonzero(b.down_proj.weight[:, ~keep]) == 0
        torch.testing.assert_close(b.gate_proj.weight[keep], dense[f"L{i}"][0][keep])


def test_decision_rule_reach_and_extension():
    from capprune.compensation_feasibility import extension, reach, verdict

    assert reach([0.01, 0.019, 0.021, 0.0], 0.02) == 2 and reach([0.03], 0.02) == 0 and reach([0.0] * 5, 0.02) == 5
    assert verdict(10, 20) == "GO" and verdict(10, 19) == "INCONCLUSIVE" and verdict(10, 14) == "NO-GO"
    assert verdict(3, 7) == "INCONCLUSIVE" and verdict(3, 8) == "GO"  # the 5-step floor binds when R_mask is small
    assert verdict(0, 5) == "GO" and verdict(0, 4) == "INCONCLUSIVE"  # NO-GO cannot occur at R_mask = 0
    assert extension(19, 40, 60) == {"extend": False, "L": 40, "proceed": True}
    assert extension(20, 40, 60) == {"extend": True, "L": 50, "proceed": True}
    assert extension(29, 40, 60) == {"extend": True, "L": 60, "proceed": True}
    assert extension(30, 40, 60)["proceed"] is False
    # attainability (plan section 5): whenever the rule proceeds, GO is reachable at R_comp = L
    for r in range(0, 30):
        e = extension(r, 40, 60)
        if e["proceed"]:
            assert verdict(r, e["L"]) == "GO", r
