"""Layer-sensitivity profiling exactly as in Kodathala & Vunnam (2026), Sec. 3.2.

For each unit l:
  s_l = k-th percentile (k=10) of the active entries of |W_l| * ||X_l||_2
  g_l = mean over M samples of |grad_W L_i|, averaged over the unit's active weights
  z^(s)_l, z^(g)_l = z-scores across units (eps = 1e-9)

Statistics are saved per iteration (``profiles/iter-XXXXXX.safetensors``) so a
resumed iteration rebuilds *identical* prompts and masks instead of
re-measuring with slightly different GPU floating-point results.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..utils.gpu import AdaptiveBatcher
from ..utils.io import finalize_tmp
from .masks import PruneState
from .wanda import channel_scores, linear_scores


@torch.no_grad()
def collect_act_sqnorm(model, state: PruneState, tokens: torch.Tensor, batch_store: dict,
                       heartbeat=None, initial_bs: int = 4) -> dict[str, torch.Tensor]:
    dev = next(model.parameters()).device
    sq: dict[str, torch.Tensor] = {}
    hooks = []
    for u in state.units:
        lin = u.linears["down"] if state.mode == "structured" else u.linears["w"]
        sq[u.name] = torch.zeros(lin.weight.shape[1], dtype=torch.float32, device=dev)

        def pre(mod, inp, _n=u.name):
            x = inp[0]
            sq[_n] += x.float().pow(2).reshape(-1, x.shape[-1]).sum(0)

        hooks.append(lin.register_forward_pre_hook(pre))

    def run(batch):
        ids = torch.stack(list(batch)).to(dev)
        # On OOM the partial sums from the failed batch would double count; snapshot and restore.
        snap = {k: v.clone() for k, v in sq.items()}
        try:
            _forward_no_logits(model, ids)
        except BaseException:
            for k in sq:
                sq[k].copy_(snap[k])
            raise
        if heartbeat:
            heartbeat.beat("profile", "activations")
        return [None] * len(batch)

    try:
        AdaptiveBatcher(batch_store, "profile_act_bs", initial_bs).run(list(tokens), run)
    finally:
        for h in hooks:
            h.remove()
    return sq


def _forward_no_logits(model, ids):
    try:
        return model(input_ids=ids, use_cache=False, logits_to_keep=1)
    except TypeError:
        return model(input_ids=ids, use_cache=False)


def _dropout_free(model) -> bool:
    cfg = getattr(model, "config", None)
    vals = [v for k, v in (cfg.to_dict() if cfg is not None else {}).items() if "drop" in k and isinstance(v, (int, float))]
    return all(float(v) == 0.0 for v in vals) and not any(
        isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules())


def collect_grad_importance(model, state: PruneState, tokens: torch.Tensor, heartbeat=None,
                            checkpointing: bool = False) -> dict[str, float]:
    """Mean |grad| of every unit's kept weights over ``tokens`` (one sequence per backward pass).

    ``checkpointing`` (``profiling.gradient_checkpointing``): activation checkpointing during the backward passes, for
    models whose 2,048-token backward pass does not fit next to the dense copy. Hugging Face models only checkpoint in
    train mode, so this runs the passes in train mode and refuses any model with non-zero dropout; the result is the
    same computation (measured bitwise identical on SmolLM3-3B-Base)."""
    dev = next(model.parameters()).device
    params = []
    for u in state.units:
        for lin in u.linears.values():
            lin.weight.requires_grad_(True)
            params.append(lin.weight)
    acc = {u.name: 0.0 for u in state.units}
    if checkpointing:
        if not _dropout_free(model):
            raise RuntimeError("gradient checkpointing runs in train mode: the model must have no active dropout")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    try:
        for i in range(tokens.shape[0]):
            ids = tokens[i : i + 1].to(dev)
            with torch.enable_grad():
                out = model(input_ids=ids, labels=ids, use_cache=False)
                out.loss.backward()
            with torch.no_grad():
                for u in state.units:
                    m = state.masks[u.name]
                    if state.mode == "structured":
                        mm = m.float()
                        tot = (u.linears["gate"].weight.grad.float().abs() * mm[:, None]).sum()
                        tot += (u.linears["up"].weight.grad.float().abs() * mm[:, None]).sum()
                        tot += (u.linears["down"].weight.grad.float().abs() * mm[None, :]).sum()
                        cnt = mm.sum() * u.numel / u.channels
                    else:
                        g = u.linears["w"].weight.grad.float().abs()
                        tot = (g * m).sum()
                        cnt = m.sum()
                    acc[u.name] += float(tot / cnt.clamp_min(1))
            for p in params:
                p.grad = None
            if heartbeat:
                heartbeat.beat("profile", f"gradients {i + 1}/{tokens.shape[0]}")
    finally:
        for p in params:
            p.grad = None
            p.requires_grad_(False)
        if checkpointing:
            model.eval()
            model.gradient_checkpointing_disable()
    n = max(1, tokens.shape[0])
    return {k: v / n for k, v in acc.items()}


@torch.no_grad()
def sensitivity(state: PruneState, act: dict[str, torch.Tensor], percentile: float = 10.0) -> dict[str, float]:
    out = {}
    for u in state.units:
        m = state.masks[u.name]
        if state.mode == "structured":
            s = channel_scores(u, act[u.name], m)
        else:
            s = linear_scores(u.weight, act[u.name], m)
        active = s[m]
        if active.numel() == 0:
            out[u.name] = float("nan")
            continue
        k = max(1, int(math.ceil(percentile / 100.0 * active.numel())))
        out[u.name] = float(torch.kthvalue(active.reshape(-1).float(), k).values)
    return out


def zscores(vals: dict[str, float], eps: float = 1e-9) -> dict[str, float]:
    keys = [k for k, v in vals.items() if v == v]  # drop NaN (fully pruned units)
    a = np.array([vals[k] for k in keys], dtype=np.float64)
    if a.size == 0:
        return {k: 0.0 for k in vals}
    mu, sd = a.mean(), a.std()
    z = {k: float((vals[k] - mu) / (sd + eps)) for k in keys}
    for k in vals:
        z.setdefault(k, float("nan"))
    return z


def save_profile(path: Path, act: dict[str, torch.Tensor], sens: dict[str, float], grad: dict[str, float]) -> None:
    tensors = {f"act::{k}": v.detach().float().cpu().contiguous() for k, v in act.items()}
    names = sorted(sens)
    tensors["sens"] = torch.tensor([sens[k] for k in names], dtype=torch.float64)
    tensors["grad"] = torch.tensor([grad.get(k, float("nan")) for k in names], dtype=torch.float64)
    tmp = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp), metadata={"names": "\n".join(names)})
    finalize_tmp(tmp, path)


def load_profile(path: Path, device) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        names = f.metadata()["names"].split("\n")
    t = load_file(str(path))
    act = {k.split("::", 1)[1]: v.to(device) for k, v in t.items() if k.startswith("act::")}
    sens = {n: float(x) for n, x in zip(names, t["sens"].tolist())}
    grad = {n: float(x) for n, x in zip(names, t["grad"].tolist())}
    return act, sens, grad
