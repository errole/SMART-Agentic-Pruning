"""Hardware benchmark: prompt-processing (prefill) latency, token-generation (decode)
latency and peak GPU memory, measured on a *materialised* model with CUDA events."""
from __future__ import annotations

import statistics
import time

import torch

from ..pruning.materialize import param_bytes


def _timer(dev):
    if dev.type == "cuda":
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        return (lambda: s.record()), (lambda: (e.record(), torch.cuda.synchronize(), s.elapsed_time(e))[2])
    t = {}
    return (lambda: t.__setitem__("t", time.perf_counter())), (lambda: (time.perf_counter() - t["t"]) * 1000)


@torch.no_grad()
def benchmark(model, vocab_size: int, prompt_lens: list[int], gen_tokens: int, batch_size: int = 1,
              warmup: int = 3, repeats: int = 10, seed: int = 0, heartbeat=None) -> dict:
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    results = {"weights_bytes": param_bytes(model), "batch_size": batch_size, "by_prompt_len": {}}
    if dev.type == "cuda":
        torch.cuda.synchronize()
        results["allocated_after_load_bytes"] = torch.cuda.memory_allocated(dev)
    for L in prompt_lens:
        ids = torch.randint(100, max(101, vocab_size - 1), (batch_size, L), generator=g).to(dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        prefill, decode = [], []
        for r in range(warmup + repeats):
            start, stop = _timer(dev)
            start()
            out = model(input_ids=ids, use_cache=True)
            t_pre = stop()
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            del out
            start()
            for _ in range(gen_tokens):
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1:].argmax(-1)
            t_dec = stop() / max(1, gen_tokens)
            del past
            if r >= warmup:
                prefill.append(t_pre)
                decode.append(t_dec)
            if heartbeat:
                heartbeat.beat(None, f"bench L={L} rep={r}")
        rec = {
            "prefill_ms_median": statistics.median(prefill),
            "prefill_ms_iqr": _iqr(prefill),
            "decode_ms_per_token_median": statistics.median(decode),
            "decode_ms_per_token_iqr": _iqr(decode),
            "prefill_tokens_per_s": batch_size * L / (statistics.median(prefill) / 1000),
            "decode_tokens_per_s": batch_size / (statistics.median(decode) / 1000),
            "repeats": repeats,
        }
        if dev.type == "cuda":
            rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(dev)
            rec["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(dev)
        results["by_prompt_len"][str(L)] = rec
    if dev.type == "cuda":
        results["device"] = torch.cuda.get_device_name(dev)
    else:
        results["device"] = "cpu"
    return results


def _iqr(xs: list[float]) -> float:
    if len(xs) < 4:
        return 0.0
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]
