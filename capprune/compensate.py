"""Weight compensation after structured MLP-channel removal (docs/compensation_feasibility_plan.md, section 2).

For an MLP with kept channels K (out of all channels N), the output weights on K are refit so that the pruned layer
reproduces the dense layer's output on the layer's current inputs:

    W' = W_dense[:, N] C[N, K] (C[K, K] + lambda I)^-1,    C = sum over calibration tokens of h h^T  (float64),
    lambda = ridge_fraction x mean(diag(C[K, K])),

where h are the dense MLP's hidden activations act(gate(x)) * up(x) over all channels. This is ridge-regularized
least-squares reconstruction of the dense layer output, the closed form behind OBS-style structured pruning (as in
SparseGPT). It is not a reproduction of any published method. gate_proj and up_proj are unchanged; no gradient step.

A compensated checkpoint is identified by the hash of (mask hash, the SHA-256 of every refit layer's current
weights). ``CompensatedTrajectory`` persists each step's refit weights with checksums before the next step starts, and
resumes from the last persisted step by loading them (it never recomputes weights to reproduce a persisted step).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from .utils.io import atomic_write_json, read_json


# ============================================================================ the refit (pure)
@torch.no_grad()
def gram_update(C: torch.Tensor, h: torch.Tensor) -> None:
    """C += h^T h, accumulated in float64. ``h``: [tokens, channels]."""
    hd = h.reshape(-1, h.shape[-1]).to(torch.float64)
    C.addmm_(hd.T, hd)


@torch.no_grad()
def hidden(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, act) -> torch.Tensor:
    """The dense MLP's hidden activations over ALL channels, in float32: act(x W_gate^T) * (x W_up^T)."""
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
    return act(xf @ w_gate.to(torch.float32).T) * (xf @ w_up.to(torch.float32).T)


@torch.no_grad()
def refit(w_down: torch.Tensor, C: torch.Tensor, keep: torch.Tensor, ridge_fraction: float = 0.01) -> tuple[torch.Tensor, dict]:
    """Refit the output weights on the kept channels. ``w_down``: the DENSE [d_out, N] weights; ``C``: [N, N] float64;
    ``keep``: bool [N]. Returns (W' [d_out, |K|] float64, info). No channel removed: the dense weights unchanged
    (no refit, no shrinkage)."""
    keep = keep.to(torch.bool)
    K = keep.nonzero().flatten()
    W = w_down.to(torch.float64)
    if bool(keep.all()):
        return W.clone(), {"refit": False, "kept": int(K.numel()), "lambda": 0.0}
    C = C.to(torch.float64)
    ckk = C.index_select(0, K).index_select(1, K)
    lam = float(ridge_fraction) * float(torch.diagonal(ckk).mean())
    A = ckk + lam * torch.eye(K.numel(), dtype=torch.float64, device=C.device)
    L, err = torch.linalg.cholesky_ex(A)
    if int(err) != 0:
        raise RuntimeError(f"Cholesky failed (leading minor {int(err)}): C[K, K] + lambda I is not positive definite")
    B = W @ C.index_select(1, K)                    # [d_out, |K|]
    Wk = torch.cholesky_solve(B.T.contiguous(), L).T  # = B (C_KK + lambda I)^-1, no explicit inverse
    return Wk, {"refit": True, "kept": int(K.numel()), "lambda": lam}


@torch.no_grad()
def conditioning(C: torch.Tensor, keep: torch.Tensor, lam: float) -> dict:
    """Extreme eigenvalues of C[K, K] and its condition number, without and with the ridge."""
    K = keep.to(torch.bool).nonzero().flatten()
    ev = torch.linalg.eigvalsh(C.to(torch.float64).index_select(0, K).index_select(1, K))
    lo, hi = float(ev[0]), float(ev[-1])
    return {"eig_min": lo, "eig_max": hi, "cond": (hi / lo) if lo > 0 else float("inf"),
            "cond_ridge": (hi + lam) / (lo + lam) if lo + lam > 0 else float("inf")}


@torch.no_grad()
def relative_error(w_down: torch.Tensor, C: torch.Tensor, keep: torch.Tensor, w_k: torch.Tensor) -> float:
    """||W_dense h - W_K h_K||^2 / ||W_dense h||^2 summed over the calibration tokens, from C."""
    K = keep.to(torch.bool).nonzero().flatten()
    W = w_down.to(torch.float64)
    Wk = w_k.to(torch.float64)
    C = C.to(torch.float64)
    total = torch.trace(W @ C @ W.T)
    cross = torch.trace(Wk @ C.index_select(0, K) @ W.T)
    own = torch.trace(Wk @ C.index_select(0, K).index_select(1, K) @ Wk.T)
    return float((total - 2 * cross + own) / total)


# ============================================================================ identity
def tensor_sha(t: torch.Tensor) -> str:
    """SHA-256 of a tensor's dtype, shape and bytes (bfloat16 is hashed through its int16 view)."""
    a = t.detach().contiguous().cpu()
    raw = a.view(torch.int16).numpy() if a.dtype == torch.bfloat16 else a.numpy()
    h = hashlib.sha256()
    h.update(f"{a.dtype}|{tuple(a.shape)}|".encode())
    h.update(raw.tobytes())
    return h.hexdigest()


def checkpoint_hash(mask_hash: str, weight_shas: dict[str, str]) -> str:
    """A compensated checkpoint: the hash of (mask, the current refit weights of every refit layer)."""
    payload = json.dumps({"mask": mask_hash, "weights": dict(sorted(weight_shas.items()))}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================ the trajectory (generic, resumable)
class CompensatedTrajectory:
    """Replays cumulative masks ``masks[0..]`` (step t = masks[t - 1]) on a model and refits every layer whose channels
    are removed in a step, in ascending depth, on its current inputs.

    * ``units``: name -> (gate, up, down) ``nn.Linear`` of that MLP, in depth order (dict order).
    * ``dense``: name -> (W_gate, W_up, W_down) dense weights (never modified).
    * ``inputs(name)``: an iterable of the MLP's input activations over the calibration tokens, on the CURRENT model.
    * ``act``: the MLP's activation function.
    * ``mask_hashes``: the hash of each step's mask (``ranker.mask_hash``).

    Step t's refit weights are written to ``out/steps/step_TTT.safetensors`` (+ ``.json`` with SHA-256s and
    diagnostics); the step counts as done only once ``out/manifest.json`` lists it. A crash before that recomputes the
    whole step from the last persisted step."""

    def __init__(self, out: Path, units: dict, dense: dict, inputs: Callable[[str], Iterable[torch.Tensor]], act,
                 masks: list[dict[str, torch.Tensor]], mask_hashes: list[str], ridge_fraction: float = 0.01,
                 crash_hook: Callable[[int, str], None] | None = None, log=None):
        self.out = Path(out)
        self.units, self.dense, self.inputs, self.act = units, dense, inputs, act
        self.masks, self.mask_hashes = masks, mask_hashes
        self.ridge = float(ridge_fraction)
        self.crash_hook = crash_hook
        self.log = log
        (self.out / "steps").mkdir(parents=True, exist_ok=True)
        self.man_p = self.out / "manifest.json"

    # ------------------------------------------------------------------ model state
    @torch.no_grad()
    def _apply_mask(self, name: str, keep: torch.Tensor) -> None:
        g, u, d = self.units[name]
        wg, wu, wd = self.dense[name]
        m = keep.to(g.weight.device)
        g.weight.copy_(wg * m.to(wg.dtype)[:, None])
        u.weight.copy_(wu * m.to(wu.dtype)[:, None])
        d.weight.copy_(wd * m.to(wd.dtype)[None, :])

    @torch.no_grad()
    def _set_down(self, name: str, keep: torch.Tensor, wk: torch.Tensor) -> torch.Tensor:
        """down[:, K] = W' (in the model dtype), zeros elsewhere. Returns the stored (model-dtype) W'."""
        d = self.units[name][2]
        K = keep.to(torch.bool).nonzero().flatten().to(d.weight.device)
        stored = wk.to(device=d.weight.device, dtype=d.weight.dtype)
        d.weight.zero_()
        d.weight.index_copy_(1, K, stored)
        return stored

    def manifest(self) -> dict:
        return read_json(self.man_p) if self.man_p.exists() else {"steps": [], "current": {}}

    def current_at(self, t: int) -> dict[str, dict]:
        """Every refit layer's current weights at persisted step ``t``: name -> {file, step, sha256}."""
        M = self.manifest()
        if t > len(M["steps"]):
            raise RuntimeError(f"step {t} is not persisted (last: {len(M['steps'])})")
        cur = {}
        for s in M["steps"][:t]:
            rec = read_json(self.out / "steps" / Path(s["file"]).with_suffix(".json"))
            for n, sha in rec["weight_sha256"].items():
                cur[n] = {"file": s["file"], "step": s["step"], "sha256": sha}
        return cur

    @torch.no_grad()
    def restore(self, t: int | None = None) -> int:
        """Bring the model to persisted step ``t`` (default: the last) from the PERSISTED weights, checksums verified.
        Returns the step."""
        from safetensors.torch import load_file

        M = self.manifest()
        done = len(M["steps"]) if t is None else int(t)
        for n in self.units:  # dense first
            self._apply_mask(n, torch.ones(self.dense[n][2].shape[1], dtype=torch.bool))
        if not done:
            return 0
        mask = self.masks[done - 1]
        for n in self.units:
            self._apply_mask(n, mask[n])
        for n, cur in self.current_at(done).items():
            f = self.out / "steps" / cur["file"]
            rec = read_json(f.with_suffix(".json"))
            if _sha_file(f) != rec["file_sha256"]:
                raise RuntimeError(f"{f}: SHA-256 does not match its record")
            w = load_file(str(f))[n]
            if tensor_sha(w) != cur["sha256"]:
                raise RuntimeError(f"{f}[{n}]: weight SHA-256 does not match the manifest")
            self._set_down(n, mask[n], w)
        if t is None and self.current_at(done) != M["current"]:
            raise RuntimeError("the manifest's current weights disagree with the step records")
        return done

    def checkpoint_hash_at(self, t: int) -> str:
        return checkpoint_hash(self.mask_hashes[t - 1], {n: v["sha256"] for n, v in self.current_at(t).items()})

    # ------------------------------------------------------------------ one step
    @torch.no_grad()
    def _step(self, t: int, M: dict) -> dict:
        from safetensors.torch import save_file

        t0 = time.time()
        cuda = torch.cuda.is_available() and any(self.units[n][0].weight.is_cuda for n in self.units)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        prev = self.masks[t - 2] if t > 1 else None
        mask = self.masks[t - 1]
        changed = [n for n in self.units if prev is None and not bool(mask[n].all())
                   or prev is not None and not torch.equal(mask[n].cpu(), prev[n].cpu())]
        for n in changed:  # the step's new mask first (the step's layers start from the dense columns of K)
            self._apply_mask(n, mask[n])
        stored, diag = {}, {}
        for n in changed:  # then one refit at a time, ascending depth, each on the current model's inputs
            wg, wu, wd = self.dense[n]
            N = wd.shape[1]
            C = torch.zeros((N, N), dtype=torch.float64, device=wd.device)
            tokens = 0
            tc = time.time()
            for x in self.inputs(n):
                h = hidden(x.to(wd.device), wg, wu, self.act)
                gram_update(C, h)
                tokens += h.shape[0]
            t_gram = time.time() - tc
            keep = mask[n].to(torch.bool).to(wd.device)
            wk, info = refit(wd, C, keep, self.ridge)
            st = self._set_down(n, keep, wk)
            Kd = wd.to(torch.float64).index_select(1, keep.nonzero().flatten())
            diag[n] = {**info, "tokens": tokens, "gram_seconds": t_gram, **conditioning(C, keep, info["lambda"]),
                       "rel_error_before": relative_error(wd, C, keep, Kd),
                       "rel_error_after": relative_error(wd, C, keep, st)}
            stored[n] = st.detach().cpu().contiguous()
            del C
            if self.crash_hook:
                self.crash_hook(t, n)
        # persist the step before the next one starts: the weights, then their record, then the manifest entry
        f = self.out / "steps" / f"step_{t:03d}.safetensors"
        tmp = f.with_suffix(".tmp")
        save_file(stored, str(tmp)) if stored else save_file({"_empty": torch.zeros(1)}, str(tmp))
        tmp.replace(f)
        shas = {n: tensor_sha(w) for n, w in stored.items()}
        current = dict(M["current"])
        for n in stored:
            current[n] = {"file": f.name, "step": t, "sha256": shas[n]}
        ck = checkpoint_hash(self.mask_hashes[t - 1], {n: v["sha256"] for n, v in current.items()})
        if cuda:
            torch.cuda.synchronize()
        rec = {"step": t, "mask_hash": self.mask_hashes[t - 1], "refit_units": changed, "weight_sha256": shas,
               "file_sha256": _sha_file(f), "checkpoint_hash": ck, "diagnostics": diag, "seconds": time.time() - t0,
               "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if cuda else None}
        atomic_write_json(f.with_suffix(".json"), rec)
        M = {"steps": M["steps"] + [{"step": t, "checkpoint_hash": ck, "mask_hash": rec["mask_hash"], "file": f.name}],
             "current": current}
        atomic_write_json(self.man_p, M)  # the commit point
        if self.log:
            self.log.info("step %d: refit %s in %.1f s; checkpoint %s", t, ",".join(changed), rec["seconds"], ck)
        return M

    def run(self, upto: int) -> list[str]:
        """Run (or resume) to step ``upto``. Returns the checkpoint hashes of steps 1..upto."""
        done = self.restore()
        M = self.manifest()
        for t in range(done + 1, upto + 1):
            M = self._step(t, M)
        return [s["checkpoint_hash"] for s in M["steps"][:upto]]
