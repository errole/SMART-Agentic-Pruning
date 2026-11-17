"""Token-weighted perplexity over fixed-length windows (Wanda/SparseGPT protocol)."""
from __future__ import annotations

import math

import torch

from ..utils.gpu import AdaptiveBatcher


@torch.no_grad()
def perplexity(model, windows: torch.Tensor, batch_store: dict, key: str, initial_bs: int = 4,
               heartbeat=None) -> float:
    """``windows``: LongTensor [n, T]. Returns exp(mean NLL over all predicted tokens)."""
    dev = next(model.parameters()).device
    totals = {"nll": 0.0, "n": 0}

    def run(batch):
        ids = torch.stack(list(batch)).to(dev)
        logits = model(input_ids=ids, use_cache=False).logits
        T = ids.shape[1]
        nll = 0.0
        for s in range(0, T - 1, 512):  # chunked fp32 upcast keeps peak memory low with a 150k vocab
            e = min(T - 1, s + 512)
            lg = logits[:, s:e, :].float()
            nll += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), ids[:, s + 1 : e + 1].reshape(-1), reduction="sum"))
        del logits
        totals["nll"] += nll
        totals["n"] += ids.shape[0] * (T - 1)
        if heartbeat:
            heartbeat.beat(None, f"ppl {totals['n']} tok")
        return [None] * len(batch)

    AdaptiveBatcher(batch_store, key, initial_bs).run(list(windows), run)
    mean = totals["nll"] / max(1, totals["n"])
    return float(math.exp(mean)) if mean < 700 else float("inf")


def stream_to_windows(stream: torch.Tensor, seq_len: int) -> torch.Tensor:
    n = stream.numel() // seq_len
    return stream[: n * seq_len].view(n, seq_len)
