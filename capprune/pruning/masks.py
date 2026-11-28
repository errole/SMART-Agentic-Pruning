"""Pruning state = one boolean keep-mask per unit (True = keep).

The original dense weights of all prunable units are kept in memory
(``DenseStore``), so moving between any two states - accepting a candidate,
rolling back, or resuming from a checkpoint - is just ``W = W_dense * mask``.
Masks are monotone: weights pruned in an accepted state stay pruned.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file as np_load
from safetensors.numpy import save_file as np_save

from ..model import Unit


class DenseStore:
    def __init__(self, units: list[Unit], device: torch.device | str | None = None):
        self.w: dict[tuple[str, str], torch.Tensor] = {}
        for u in units:
            for part, lin in u.linears.items():
                t = lin.weight.detach()
                self.w[(u.name, part)] = t.clone() if device is None else t.to(device, copy=True)

    def get(self, unit: str, part: str) -> torch.Tensor:
        return self.w[(unit, part)]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.w.values())


class LiveDenseStore:
    """The dense weights ARE the live weights, for one-shot pruning from a freshly restored dense model (one resident
    copy; ``pruning/reload.py``). Valid only while a unit has not been applied since the last ``reset``, so a second
    read of the same unit is refused."""

    def __init__(self, units: list[Unit]):
        self.by_name = {u.name: u for u in units}
        self.used: set[tuple[str, str]] = set()

    def reset(self) -> None:
        self.used.clear()

    def get(self, unit: str, part: str) -> torch.Tensor:
        if (unit, part) in self.used:
            raise RuntimeError(f"{unit}.{part} was already pruned since the dense weights were restored")
        self.used.add((unit, part))
        return self.by_name[unit].linears[part].weight.detach()

    def nbytes(self) -> int:
        return 0


class PruneState:
    def __init__(self, units: list[Unit], mode: str, nm: tuple[int, int] = (2, 4), mask_device=None):
        """``mask_device``: where the keep-masks are created (default: each unit's weight device). ``"cpu"`` with
        ``sequential_wanda(..., cache_inputs=True, offload_masks=True)`` keeps the masks off the GPU."""
        self.units = units
        self.by_name = {u.name: u for u in units}
        self.mode = mode
        self.nm = nm
        self.masks: dict[str, torch.Tensor] = {}
        for u in units:
            dev = mask_device if mask_device is not None else next(iter(u.linears.values())).weight.device
            shape = (u.channels,) if mode == "structured" else tuple(u.weight.shape)
            self.masks[u.name] = torch.ones(shape, dtype=torch.bool, device=dev)
        self.total_eligible = sum(u.numel for u in units)

    # ------------------------------------------------------------------ accounting
    def unit_pruned_weights(self, name: str) -> int:
        u = self.by_name[name]
        m = self.masks[name]
        if self.mode == "structured":
            pruned_ch = int((~m).sum().item())
            return pruned_ch * u.numel // u.channels
        return int((~m).sum().item())

    def unit_sparsity(self, name: str) -> float:
        return self.unit_pruned_weights(name) / self.by_name[name].numel

    def all_unit_sparsity(self) -> dict[str, float]:
        return {u.name: round(self.unit_sparsity(u.name), 6) for u in self.units}

    def global_sparsity(self) -> float:
        return sum(self.unit_pruned_weights(u.name) for u in self.units) / self.total_eligible

    def share(self, name: str) -> float:
        """Fraction of all eligible weights that live in this unit."""
        return self.by_name[name].numel / self.total_eligible

    # ------------------------------------------------------------------ copy / diff
    def clone(self) -> "PruneState":
        s = PruneState.__new__(PruneState)
        s.units, s.by_name, s.mode, s.nm, s.total_eligible = self.units, self.by_name, self.mode, self.nm, self.total_eligible
        s.masks = {k: v.clone() for k, v in self.masks.items()}
        return s

    def changed_units(self, other: "PruneState") -> list[str]:
        return [n for n in self.masks if not torch.equal(self.masks[n], other.masks[n])]

    # ------------------------------------------------------------------ apply to model
    @torch.no_grad()
    def apply(self, dense: DenseStore, only: list[str] | None = None) -> None:
        for name in (only if only is not None else list(self.masks)):
            u = self.by_name[name]
            m = self.masks[name]
            if self.mode == "structured":
                mm = m.to(u.linears["gate"].weight.dtype)
                u.linears["gate"].weight.copy_(dense.get(name, "gate") * mm[:, None])
                u.linears["up"].weight.copy_(dense.get(name, "up") * mm[:, None])
                u.linears["down"].weight.copy_(dense.get(name, "down") * mm[None, :])
            else:
                w = u.linears["w"].weight
                w.copy_(dense.get(name, "w") * m.to(w.dtype))

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        arrays, shapes = {}, {}
        for name, m in self.masks.items():
            a = m.detach().to("cpu").numpy().reshape(-1)
            arrays[name] = np.packbits(a)
            shapes[name] = list(m.shape)
        meta = {"mode": self.mode, "nm": json.dumps(list(self.nm)), "shapes": json.dumps(shapes)}
        np_save(arrays, str(path), metadata=meta)

    def load(self, path: str | Path) -> None:
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as f:
            meta = f.metadata()
        if meta.get("mode") != self.mode:
            raise ValueError(f"mask file mode {meta.get('mode')} != run mode {self.mode}")
        shapes = json.loads(meta["shapes"])
        arrays = np_load(str(path))
        if set(arrays) != set(self.masks):
            raise ValueError("mask file units do not match the model's prunable units")
        for name, packed in arrays.items():
            shape = tuple(shapes[name])
            if shape != tuple(self.masks[name].shape):
                raise ValueError(f"mask shape mismatch for {name}: {shape} vs {tuple(self.masks[name].shape)}")
            n = int(np.prod(shape))
            flat = np.unpackbits(packed)[:n].astype(bool)
            self.masks[name] = torch.from_numpy(flat.reshape(shape)).to(self.masks[name].device)

    def approx_file_bytes(self) -> int:
        return sum(m.numel() for m in self.masks.values()) // 8 + 4096

    def fingerprint(self) -> str:
        """Cheap content fingerprint (pruned counts per unit) used to validate resumes."""
        import hashlib

        h = hashlib.sha256()
        for name in sorted(self.masks):
            m = self.masks[name]
            h.update(name.encode())
            h.update(str(int((~m).sum().item())).encode())
            # a few strided positions so that two different masks with equal counts differ
            flat = m.reshape(-1)
            step = max(1, flat.numel() // 64)
            h.update(bytes(flat[::step][:64].to("cpu").numpy().astype(np.uint8)))
        return h.hexdigest()[:16]
