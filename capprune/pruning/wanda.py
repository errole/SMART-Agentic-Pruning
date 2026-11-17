"""Wanda scoring and mask selection for the three supported pruning operations.

* unstructured: per output row, prune the lowest ``|W| * ||X||_2`` (Wanda).
* nm (default 2:4): within every group of M consecutive inputs keep the N
  highest-scoring weights - the pattern NVIDIA sparse tensor cores execute.
* structured: drop whole MLP intermediate channels ranked by
  ``||W_down[:, j]||_2 * ||X_j||_2`` (Wanda-SP style), in multiples of
  ``channel_multiple`` so the shrunken GEMMs stay tensor-core aligned.

All selection uses ``torch.sort(stable=True)`` so masks are a deterministic
function of (weights, activation statistics, ratios) - required for exact
replay after a crash.
"""
from __future__ import annotations

import math

import torch

from ..model import Unit, decoder_layers
from ..utils.log import get_logger
from .masks import DenseStore, PruneState

log = get_logger("wanda")


def linear_scores(w: torch.Tensor, act_sqnorm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    s = w.detach().abs().float() * act_sqnorm.to(w.device).float().clamp_min(0).sqrt()[None, :]
    return s.masked_fill(~mask, -1.0)  # already-pruned weights always rank lowest -> masks stay monotone


def channel_scores(u: Unit, down_in_sqnorm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    wd = u.linears["down"].weight.detach().float()
    s = wd.norm(dim=0) * down_in_sqnorm.to(wd.device).float().clamp_min(0).sqrt()
    return s.masked_fill(~mask, -1.0)


@torch.no_grad()
def unstructured_mask(score: torch.Tensor, ratio: float) -> torch.Tensor:
    rows, cols = score.shape
    k = int(round(ratio * cols))
    mask = torch.ones_like(score, dtype=torch.bool)
    if k <= 0:
        return mask
    idx = torch.sort(score, dim=1, stable=True).indices[:, :k]
    mask.scatter_(1, idx, False)
    return mask


@torch.no_grad()
def nm_mask(score: torch.Tensor, n: int, m: int) -> torch.Tensor:
    rows, cols = score.shape
    if cols % m:
        raise ValueError(f"input dim {cols} not divisible by M={m}")
    g = score.view(rows, cols // m, m)
    idx = torch.sort(g, dim=-1, stable=True).indices[..., : m - n]
    mask = torch.ones_like(g, dtype=torch.bool)
    mask.scatter_(-1, idx, False)
    return mask.view(rows, cols)


@torch.no_grad()
def channel_mask(score: torch.Tensor, n_prune: int) -> torch.Tensor:
    mask = torch.ones_like(score, dtype=torch.bool)
    if n_prune <= 0:
        return mask
    idx = torch.sort(score, stable=True).indices[:n_prune]
    mask[idx] = False
    return mask


def structured_prune_count(ratio: float, channels: int, multiple: int, max_ratio: float) -> int:
    ratio = min(ratio, max_ratio)
    k = int(round(ratio * channels / multiple)) * multiple
    k_max = int(math.floor(max_ratio * channels / multiple)) * multiple
    return max(0, min(k, k_max, channels - multiple))


@torch.no_grad()
def raise_unit(state: PruneState, name: str, new_ratio: float, act: dict[str, torch.Tensor], cfg: dict) -> bool:
    """Raise one unit to ``new_ratio`` (mode-aware). Returns True if its mask changed."""
    u = state.by_name[name]
    mode = state.mode
    old = state.masks[name]
    pcfg = cfg["pruning"]
    if mode == "unstructured":
        r = min(new_ratio, float(pcfg.get("max_unit_sparsity", 0.9)))
        new = unstructured_mask(linear_scores(u.weight, act[name], old), r) & old
    elif mode == "nm":
        if (~old).any():  # already N:M - cannot go further without breaking the pattern
            return False
        n, m = state.nm
        new = nm_mask(linear_scores(u.weight, act[name], old), n, m)
    else:
        scfg = pcfg.get("structured", {})
        k = structured_prune_count(new_ratio, u.channels, int(scfg.get("channel_multiple", 64)),
                                   float(scfg.get("max_unit_sparsity", 0.875)))
        if k <= int((~old).sum().item()):
            return False
        new = channel_mask(channel_scores(u, act[name], old), k) & old
    if torch.equal(new, old):
        return False
    state.masks[name] = new
    return True


# ---------------------------------------------------------------------------- one-shot Wanda baseline
class _Stop(Exception):
    pass


@torch.no_grad()
def sequential_wanda(model, state: PruneState, dense: DenseStore, calib: torch.Tensor, target: float,
                     cfg: dict, batch_size: int = 4, heartbeat=None, order: list[str] | None = None) -> PruneState:
    """Original Wanda: layer by layer, statistics gathered with earlier layers already pruned.

    ``order`` (nm mode only): units sorted from least to most sensitive; units are
    converted to N:M in that order until the global target is reached.
    """
    layers = decoder_layers(model)
    dev = next(model.parameters()).device
    by_layer: dict[int, list[Unit]] = {}
    for u in state.units:
        by_layer.setdefault(u.layer, []).append(u)
    nm_selected: set[str] | None = None
    if state.mode == "nm":
        assert order is not None, "nm Wanda baseline needs a sensitivity order"
        n, m = state.nm
        per_unit = 1 - n / m
        acc, nm_selected = 0.0, set()
        for name in order:
            if acc >= target - 1e-9:
                break
            nm_selected.add(name)
            acc += per_unit * state.share(name)
    for li, layer in enumerate(layers):
        units = by_layer.get(li, [])
        if not units:
            continue
        sq: dict[str, torch.Tensor] = {}
        hooks = []
        for u in units:
            lin = u.linears["down"] if state.mode == "structured" else u.linears["w"]
            sq[u.name] = torch.zeros(lin.weight.shape[1], dtype=torch.float32, device=dev)

            def pre(mod, inp, _n=u.name):
                x = inp[0]
                sq[_n] += x.float().pow(2).reshape(-1, x.shape[-1]).sum(0)

            hooks.append(lin.register_forward_pre_hook(pre))

        def stop(mod, inp, out):
            raise _Stop

        hooks.append(layer.register_forward_hook(stop))
        try:
            for i in range(0, calib.shape[0], batch_size):
                try:
                    model(input_ids=calib[i : i + batch_size].to(dev), use_cache=False)
                except _Stop:
                    pass
                if heartbeat:
                    heartbeat.beat("wanda", f"layer {li} batch {i}")
        finally:
            for h in hooks:
                h.remove()
        for u in units:
            if state.mode == "nm":
                if u.name in nm_selected:
                    raise_unit(state, u.name, 0.5, sq, cfg)
            else:
                raise_unit(state, u.name, target, sq, cfg)
        state.apply(dense, only=[u.name for u in units])
    return state
