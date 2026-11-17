"""GPU helpers: OOM-adaptive batching and time accounting."""
from __future__ import annotations

import gc
import time
from typing import Callable, Sequence, TypeVar

import torch

from .log import get_logger

log = get_logger("gpu")
T = TypeVar("T")
R = TypeVar("R")


def is_oom(e: BaseException) -> bool:
    if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    msg = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in msg or "cublas_status_alloc_failed" in msg)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class AdaptiveBatcher:
    """Runs ``fn`` over items in batches, halving the batch size on CUDA OOM.

    The learned batch size is written back into ``store[key]`` (which the
    caller checkpoints), so a restarted run does not hit the same OOM again.
    """

    def __init__(self, store: dict, key: str, initial: int, min_bs: int = 1):
        self.store = store
        self.key = key
        self.bs = int(store.get(key, initial))
        self.min_bs = min_bs

    def run(self, items: Sequence[T], fn: Callable[[Sequence[T]], list[R]],
            on_batch: Callable[[int, int], None] | None = None) -> list[R]:
        out: list[R] = []
        i = 0
        while i < len(items):
            batch = items[i : i + self.bs]
            try:
                res = fn(batch)
            except BaseException as e:
                if not is_oom(e) or self.bs <= self.min_bs:
                    raise
                free_memory()
                self.bs = max(self.min_bs, self.bs // 2)
                self.store[self.key] = self.bs
                log.warning("CUDA OOM in %s; retrying with batch size %d", self.key, self.bs)
                continue
            out.extend(res)
            i += len(batch)
            if on_batch is not None:
                on_batch(i, len(items))
        return out


class GpuClock:
    """Accumulates wall-clock seconds of GPU work; the total is checkpointed."""

    def __init__(self, committed: float = 0.0):
        self.committed = float(committed)
        self._t0 = time.perf_counter()

    def total(self) -> float:
        return self.committed + (time.perf_counter() - self._t0)

    def commit(self) -> float:
        now = time.perf_counter()
        self.committed += now - self._t0
        self._t0 = now
        return self.committed


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
