"""Interleaved deployment benchmark: the dense model vs. each arm's selected checkpoint (``capprune live-bench``).

Every model runs on the same device, runtime, dtype, batch size, prompt (identical random token ids),
prompt length and number of generated tokens.  Each selected checkpoint is materialised (MLP channels
physically removed, pruning/materialize.py).  After a warm-up of every model, the measurement rounds
interleave the models (the starting model rotates every round), so slow drifts of clock speed or
background load hit all models alike.  Per round we take the pruned/dense ratio; a checkpoint is
"faster" ("slower") when the bootstrap 95 % interval of the median ratio lies entirely below (above) 1,
otherwise the speed result is "inconclusive".

Peak GPU memory per model = weight bytes + peak extra allocation during one prefill+decode run of that
model (other models stay resident, so the absolute device peak is not meaningful).
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .config import runs_dir
from .utils.io import atomic_write_json
from .utils.log import get_logger

log = get_logger("live_bench")


def _timer(dev):
    from .eval.bench import _timer as t

    return t(dev)


@torch.no_grad()
def _once(model, ids: torch.Tensor, gen_tokens: int, dev) -> tuple[float, float]:
    start, stop = _timer(dev)
    start()
    out = model(input_ids=ids, use_cache=True)
    t_pre = stop()
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    del out
    start()
    for _ in range(gen_tokens):
        o = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
    t_dec = stop() / max(1, gen_tokens)
    del past
    return t_pre, t_dec


def _ratio_ci(pruned: list[float], dense: list[float], B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    r = np.asarray(pruned) / np.asarray(dense)
    rng = np.random.default_rng(seed)
    boots = [float(np.median(r[rng.integers(0, len(r), len(r))])) for _ in range(B)]
    return float(np.median(r)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _verdict(lo: float, hi: float) -> str:
    return "faster" if hi < 1.0 else "slower" if lo > 1.0 else "inconclusive"


def selected_checkpoints(cfg: dict, methods: list[str] | None = None) -> list[dict]:
    from .supervisor import build_plan
    from .utils.checkpoint import CheckpointStore, NoCheckpoint

    root = runs_dir(cfg)
    out = []
    for item in build_plan(cfg, methods):
        if cfg["methods"][item["method"]].get("kind") == "dense":
            continue
        d = root / item["id"]
        try:
            S, _ = CheckpointStore(d).load_latest()
        except (NoCheckpoint, FileNotFoundError):
            out.append({"arm": item["method"], "run_id": item["id"], "state_id": None, "status": "not run"})
            continue
        done = "search" in S.get("stages_done", [])
        out.append({"arm": item["method"], "run_id": item["id"], "state_id": S.get("state_id"),
                    "status": "search finished" if done else "search running",
                    "mask_path": str(CheckpointStore(d).mask_path(S["state_id"])) if S.get("state_id") else None})
    return out


def live_bench(cfg: dict, methods: list[str] | None = None) -> Path:
    from .model import build_units, load_model
    from .pruning.masks import PruneState
    from .pruning.materialize import materialize, param_bytes
    from .utils.gpu import free_memory

    lb = cfg.get("live_bench") or {}
    L, G = int(lb.get("prompt_len", 512)), int(lb.get("gen_tokens", 128))
    bs, warm, reps = int(lb.get("batch_size", 1)), int(lb.get("warmup", 5)), int(lb.get("repeats", 20))
    mode = cfg["pruning"]["mode"]
    model = load_model(cfg)
    dev = next(model.parameters()).device
    units = build_units(model, mode, cfg["pruning"].get("eligible", []))
    dense_params = sum(p.numel() for p in model.parameters())
    entries = [{"arm": "dense", "run_id": "dense", "state_id": None, "status": "reference"}]
    models = {"dense": model}
    for e in selected_checkpoints(cfg, methods):
        entries.append(e)
        if not e["state_id"]:
            e["note"] = "no accepted pruning step: identical to the dense model" if e["status"] != "not run" else "not run"
            continue
        st = PruneState(units, mode)
        st.load(e["mask_path"])
        t0 = time.perf_counter()
        m, info = materialize(model, st, cfg)  # slices the kept channels out of the dense weights
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        e.update(materialize_seconds=time.perf_counter() - t0, materialize=info, sparsity=st.global_sparsity())
        models[e["arm"]] = m
    for e in entries:
        if e["arm"] in models:
            m = models[e["arm"]]
            e["parameters"] = sum(p.numel() for p in m.parameters())
            e["removed_parameters"] = dense_params - e["parameters"]
            e["weights_bytes"] = param_bytes(m)

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(100, max(101, model.config.vocab_size - 1), (bs, L), generator=g).to(dev)
    names = list(models)
    log.info("benchmarking %s: prompt %d, %d generated tokens, batch %d, %d warm-up + %d interleaved rounds",
             names, L, G, bs, warm, reps)
    for n in names:
        for _ in range(warm):
            _once(models[n], ids, G, dev)
    pre = {n: [] for n in names}
    dec = {n: [] for n in names}
    for r in range(reps):
        for k in range(len(names)):
            n = names[(r + k) % len(names)]
            p, dd = _once(models[n], ids, G, dev)
            pre[n].append(p)
            dec[n].append(dd)
    for n in names:
        e = next(x for x in entries if x["arm"] == n)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            _once(models[n], ids, G, dev)
            e["peak_memory_bytes"] = e["weights_bytes"] + (torch.cuda.max_memory_allocated(dev) - base)
        e.update(prefill_ms_median=statistics.median(pre[n]), decode_ms_per_token_median=statistics.median(dec[n]),
                 prefill_ms=pre[n], decode_ms=dec[n],
                 decode_iqr=float(np.subtract(*np.percentile(dec[n], [75, 25]))),
                 prefill_iqr=float(np.subtract(*np.percentile(pre[n], [75, 25]))))
        if n != "dense":
            for key, series in (("decode", dec), ("prefill", pre)):
                med, lo, hi = _ratio_ci(series[n], series["dense"])
                e[f"{key}_ratio"] = {"median": med, "ci95": [lo, hi], "verdict": _verdict(lo, hi)}
    session = time.strftime("%Y%m%d-%H%M%S")
    res = {"session": session,
           "settings": {"prompt_len": L, "gen_tokens": G, "batch_size": bs, "warmup": warm, "repeats": reps,
                        "dtype": cfg["model"].get("dtype"), "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                        "torch": torch.__version__, "attn_implementation": cfg["model"].get("attn_implementation")},
           "models": entries}
    # one file per session: earlier sessions (e.g. arms benchmarked before others finished) are never overwritten;
    # every session carries its own interleaved dense reference
    out = runs_dir(cfg) / "_live" / f"bench-{session}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, res)
    for n in names:
        if n != "dense":
            del models[n]
    free_memory()
    return out
