"""Numerical checks of profiling/scoring against independent reference computations."""
import math

import numpy as np
import torch

from capprune.config import load_config, run_config
from capprune.data.prepare import PreparedData
from capprune.config import data_dir
from capprune.eval.mmlu import MMLUScorer
from capprune.eval.perplexity import perplexity
from capprune.model import build_units, decoder_layers, load_model, load_tokenizer, tokenizer_id
from capprune.pruning.masks import DenseStore, PruneState
from capprune.pruning.profile import collect_act_sqnorm, collect_grad_importance, sensitivity, zscores
from capprune.pruning.wanda import raise_unit, sequential_wanda
from conftest import SMOKE


def _setup(mode="unstructured"):
    cfg = load_config(SMOKE, [f"pruning.mode={mode}"])
    rc = run_config(cfg, "cap_prune", 0.2, 0)
    model = load_model(rc)
    units = build_units(model, mode, rc["pruning"]["eligible"])
    return rc, model, units, PruneState(units, mode)


def test_activation_norms_match_reference_and_are_batch_size_invariant(smoke_assets):
    rc, model, units, st = _setup()
    toks = torch.randint(5, 200, (6, 48))
    a1 = collect_act_sqnorm(model, st, toks, {}, initial_bs=1)
    a4 = collect_act_sqnorm(model, st, toks, {}, initial_bs=4)
    name = "L2.down_proj"
    lin, got = st.by_name[name].linears["w"], []
    h = lin.register_forward_pre_hook(lambda m, i: got.append(i[0].detach().reshape(-1, i[0].shape[-1])))
    with torch.no_grad():
        model(input_ids=toks)
    h.remove()
    ref = torch.cat(got).double().pow(2).sum(0)
    assert torch.allclose(a4[name].double(), ref, rtol=1e-4)
    assert all(torch.allclose(a1[k], a4[k], rtol=1e-4) for k in a1)
    # Wanda score definition and percentile sensitivity (paper eq. 1, k = 10)
    W = lin.weight.detach().double()
    S = W.abs() * ref.sqrt()[None, :]
    s_ref = float(np.sort(S.flatten().numpy())[math.ceil(0.10 * S.numel()) - 1])
    assert math.isclose(sensitivity(st, a4, 10.0)[name], s_ref, rel_tol=1e-4)


def test_gradient_importance_matches_autograd_and_leaves_model_clean(smoke_assets):
    rc, model, units, st = _setup()
    toks = torch.randint(5, 200, (3, 40))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    g = collect_grad_importance(model, st, toks)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())  # profiling never changes weights
    name = "L1.q_proj"
    w = st.by_name[name].linears["w"].weight
    w.requires_grad_(True)
    ref = 0.0
    for i in range(3):
        loss = model(input_ids=toks[i : i + 1], labels=toks[i : i + 1]).loss
        (gr,) = torch.autograd.grad(loss, w)
        ref += float(gr.abs().mean())
    w.requires_grad_(False)
    assert math.isclose(g[name], ref / 3, rel_tol=1e-4)
    z = zscores(g)
    assert abs(np.mean(list(z.values()))) < 1e-6 and abs(np.std(list(z.values())) - 1) < 1e-6


def test_gradient_importance_ignores_pruned_weights(smoke_assets):
    rc, model, units, st = _setup()
    dense = DenseStore(units)
    act = collect_act_sqnorm(model, st, torch.randint(5, 200, (4, 32)), {})
    raise_unit(st, "L0.up_proj", 0.5, act, rc)
    st.apply(dense)
    u = st.by_name["L0.up_proj"]
    assert float(u.weight[~st.masks[u.name]].abs().max()) == 0.0
    toks = torch.randint(5, 200, (2, 32))
    g = collect_grad_importance(model, st, toks)
    u.weight.requires_grad_(True)
    ref = 0.0
    for i in range(2):
        (gr,) = torch.autograd.grad(model(input_ids=toks[i : i + 1], labels=toks[i : i + 1]).loss, u.weight)
        ref += float(gr.abs()[st.masks[u.name]].mean())
    u.weight.requires_grad_(False)
    assert math.isclose(g[u.name], ref / 2, rel_tol=1e-4)


def test_sequential_wanda_matches_blockwise_reference(smoke_assets):
    """Reference: classic Wanda - feed calibration through, prune block i using inputs produced by pruned blocks < i."""
    rc, model, units, st = _setup()
    dense = DenseStore(units)
    calib = torch.randint(5, 200, (6, 40))
    sequential_wanda(model, st, dense, calib, 0.5, rc, batch_size=3)
    got = {u.name: st.masks[u.name].clone() for u in units}
    # independent reference on a fresh model
    rc2, ref_model, units2, _ = _setup()
    for li, layer in enumerate(decoder_layers(ref_model)):
        lins = {u.name: u.linears["w"] for u in units2 if u.layer == li}
        sq, hooks = {}, []
        for n, lin in lins.items():
            sq[n] = torch.zeros(lin.weight.shape[1], dtype=torch.float64)
            hooks.append(lin.register_forward_pre_hook(
                lambda m, i, _n=n: sq.__setitem__(_n, sq[_n] + i[0].double().pow(2).reshape(-1, i[0].shape[-1]).sum(0))))
        with torch.no_grad():
            ref_model(input_ids=calib)
        for h in hooks:
            h.remove()
        with torch.no_grad():
            for n, lin in lins.items():
                score = lin.weight.abs().double() * sq[n].sqrt()[None, :]
                k = round(0.5 * score.shape[1])
                idx = torch.sort(score, dim=1, stable=True).indices[:, :k]
                mask = torch.ones_like(score, dtype=torch.bool).scatter_(1, idx, False)
                agree = (mask == got[n]).float().mean().item()
                assert agree > 0.999, (n, agree)  # identical up to float32-vs-float64 ties
                lin.weight.mul_(got[n])
    assert abs(st.global_sparsity() - 0.5) < 1e-3


def test_mmlu_scoring_is_independent_of_batch_composition_and_ppl_matches_reference(smoke_assets):
    rc, model, units, st = _setup()
    tok = load_tokenizer(rc)
    data = PreparedData(data_dir(rc), tokenizer_id(rc))
    rows = data.mmlu_probe[:24]
    big = MMLUScorer(model, tok, data.mmlu_fewshot, 1, 256, {}, "a", 100000).score(rows)
    one = MMLUScorer(model, tok, data.mmlu_fewshot, 1, 256, {}, "b", 1).score(rows)  # one question per batch
    assert {q: r["pred"] for q, r in big.items()} == {q: r["pred"] for q, r in one.items()}
    # only last-position logits are needed: the model must honour logits_to_keep (else memory blows up on GPU)
    out = model(input_ids=torch.randint(5, 200, (2, 16)), logits_to_keep=1)
    assert out.logits.shape[1] == 1
    w = data.ppl_gate
    with torch.no_grad():
        lg = model(input_ids=w).logits.double()
    nll = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), w[:, 1:].reshape(-1))
    assert math.isclose(perplexity(model, w, {}, "p", 2), math.exp(float(nll)), rel_tol=1e-4)
