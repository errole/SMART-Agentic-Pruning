"""One resident copy of the model: the dense weights of the prunable linears are restored from the pinned weight
files on disk between levels, instead of being kept as a second copy on the GPU.

* ``weight_keys``      the checkpoint key of every prunable linear's weight (``model.layers.3.mlp.up_proj.weight``);
* ``restore_dense``    copies those tensors from the safetensors files into the live weights (dtype and shape checked);
* ``fingerprint``      an exact integer checksum of the prunable weights' bits, taken at load and compared after every
                       restore, so an incomplete or wrong restore is caught before a level is pruned.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..model import Unit


def weight_keys(model, units: list[Unit]) -> dict[tuple[str, str], str]:
    names = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for u in units:
        for part, lin in u.linears.items():
            out[(u.name, part)] = names[id(lin.weight)]
    return out


def _files(snapshot: Path, keys: list[str]) -> dict[Path, list[str]]:
    idx = snapshot / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        by: dict[Path, list[str]] = {}
        for k in keys:
            if k not in wm:
                raise KeyError(f"{k} is not in {idx}")
            by.setdefault(snapshot / wm[k], []).append(k)
        return by
    return {snapshot / "model.safetensors": list(keys)}


@torch.no_grad()
def restore_dense(model, units: list[Unit], snapshot: str | Path) -> dict:
    """Copy every prunable linear's dense weight from the checkpoint files into the live model."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    keys = weight_keys(model, units)
    params = dict(model.named_parameters())
    t0 = time.perf_counter()
    n_bytes = 0
    for f, ks in _files(snapshot, list(keys.values())).items():
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in ks:
                t = h.get_tensor(k)
                w = params[k]
                if t.dtype != w.dtype or tuple(t.shape) != tuple(w.shape):
                    raise RuntimeError(f"{k}: file {t.dtype} {tuple(t.shape)} != model {w.dtype} {tuple(w.shape)}")
                w.data.copy_(t, non_blocking=False)
                n_bytes += t.numel() * t.element_size()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0, "bytes": int(n_bytes), "tensors": len(keys)}


@torch.no_grad()
def fingerprint(units: list[Unit], chunk_rows: int = 2048) -> list[int]:
    """Exact checksum of the prunable weights' bits: per tensor, the sum of its 16-bit words and their sum weighted by
    (flat position mod 65521), in int64 on the tensor's device (deterministic; detects changed or moved values)."""
    out = []
    for u in units:
        for part in sorted(u.linears):
            w = u.linears[part].weight.detach()
            if w.element_size() != 2:
                raise ValueError("fingerprint expects 16-bit weights")
            v = w.view(torch.int16).reshape(w.shape[0], -1)
            cols = v.shape[1]
            s1 = torch.zeros((), dtype=torch.int64, device=w.device)
            s2 = torch.zeros((), dtype=torch.int64, device=w.device)
            for r in range(0, v.shape[0], chunk_rows):
                blk = v[r : r + chunk_rows].to(torch.int64)
                pos = (torch.arange(r * cols, (r + blk.shape[0]) * cols, device=w.device, dtype=torch.int64) % 65521)
                s1 += blk.sum()
                s2 += (blk.reshape(-1) * pos).sum()
            out += [int(s1.item()), int(s2.item())]
    return out
