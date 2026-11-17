"""Turn a masked model into a *deployable* model whose speed/memory can be measured.

Zeroed weights alone do not make a model faster (one of the proposal's two
motivating problems), so hardware measurements always run on a materialised copy:

* structured: gate/up/down are physically sliced to the kept channels -> smaller dense GEMMs.
* nm (2:4):   N:M units are converted to ``torch.sparse`` semi-structured tensors
              (cuSPARSELt / CUTLASS sparse tensor-core kernels).  If the GPU or the
              PyTorch build lacks these kernels, conversion fails and the result
              records ``nm_backend = "unavailable: <reason>"`` - it is never silently
              reported as a speed-up.
* unstructured: no supported kernel; measured as dense (and labelled as such).
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from ..model import build_units
from .masks import PruneState


@torch.no_grad()
def materialize(model, state: PruneState, cfg: dict) -> tuple[nn.Module, dict]:
    m = copy.deepcopy(model)
    mode = state.mode
    info: dict = {"mode": mode}
    units = build_units(m, mode, cfg["pruning"].get("eligible", []))
    if mode == "structured":
        removed = 0
        for u in units:
            keep = state.masks[u.name].nonzero().flatten().to(u.linears["gate"].weight.device)
            if keep.numel() == u.channels:
                continue
            mlp_parent = _parent_of(m, u.linears["gate"])
            for part, attr in (("gate", "gate_proj"), ("up", "up_proj")):
                old = u.linears[part]
                new = nn.Linear(old.in_features, keep.numel(), bias=old.bias is not None,
                                device=old.weight.device, dtype=old.weight.dtype)
                new.weight.copy_(old.weight[keep])
                if old.bias is not None:
                    new.bias.copy_(old.bias[keep])
                setattr(mlp_parent, attr, new)
            old = u.linears["down"]
            new = nn.Linear(keep.numel(), old.out_features, bias=old.bias is not None,
                            device=old.weight.device, dtype=old.weight.dtype)
            new.weight.copy_(old.weight[:, keep])
            if old.bias is not None:
                new.bias.copy_(old.bias)
            setattr(mlp_parent, "down_proj", new)
            if hasattr(mlp_parent, "intermediate_size"):
                mlp_parent.intermediate_size = keep.numel()
            removed += (u.channels - keep.numel())
        info["channels_removed"] = removed
    elif mode == "nm":
        backend = cfg.get("benchmark", {}).get("nm_backend", "auto")
        converted, err = 0, None
        try:
            from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

            if backend == "cutlass":
                SparseSemiStructuredTensor._FORCE_CUTLASS = True
            for u in units:
                if not (~state.masks[u.name]).any():
                    continue
                lin = u.linears["w"]
                if lin.weight.device.type != "cuda":
                    raise RuntimeError("semi-structured sparsity requires CUDA")
                lin.weight = nn.Parameter(to_sparse_semi_structured(lin.weight.detach().contiguous()),
                                          requires_grad=False)
                converted += 1
            # one smoke forward to surface kernel errors now rather than mid-benchmark
            dev = next(m.parameters()).device
            m(input_ids=torch.zeros((1, 64), dtype=torch.long, device=dev), use_cache=False)
            info["nm_backend"] = "cutlass" if getattr(SparseSemiStructuredTensor, "_FORCE_CUTLASS", False) else "default"
        except Exception as e:  # kernels missing on this GPU / build -> measure masked dense, label honestly
            err = f"unavailable: {type(e).__name__}: {e}"[:300]
            del m
            m = copy.deepcopy(model)
            converted = 0
            info["nm_backend"] = err
        info["units_converted"] = converted
    else:
        info["note"] = "unstructured sparsity has no supported kernel; measured as dense"
    m.eval()
    return m, info


def _parent_of(root: nn.Module, child: nn.Module) -> nn.Module:
    for mod in root.modules():
        for c in mod.children():
            if c is child:
                return mod
    raise RuntimeError("parent module not found")


def param_bytes(model: nn.Module) -> int:
    total = 0
    for p in list(model.parameters()) + list(model.buffers()):
        t = p.data if hasattr(p, "data") else p
        if hasattr(t, "packed") and t.packed is not None:  # SparseSemiStructuredTensor
            total += t.packed.numel() * t.packed.element_size()
            meta = getattr(t, "meta", None)
            if meta is not None:
                total += meta.numel() * meta.element_size()
        else:
            total += t.numel() * t.element_size()
    return total
