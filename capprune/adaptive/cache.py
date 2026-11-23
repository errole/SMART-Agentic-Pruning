"""The replay input: cached per-item correctness for the dense model and a fixed pool of candidates.

Directory layout (plain JSON so it is easy to inspect, or to produce from another pipeline)::

    <cache>/items.jsonl          one line per benchmark item
        {"item_id": "abstract_algebra/test/60", "domain": "STEM", "split": "pool", "tokens": 187}
        split: "pool" (searched by the sequential test) or "holdout" (final check only)
        tokens: prompt tokens (optional; null if unknown)
    <cache>/models/dense.json    the original dense checkpoint (role "dense")
    <cache>/models/<name>.json   one file per pruned candidate (role "candidate")
        {"name": "wsp20", "role": "candidate", "order": 1,
         "correct": {"<item_id>": 0 or 1, ...},           # every pool AND holdout item
         "latency": {"decode_ms_per_token": 21.3, ...} or null,
         "timing":  {"layout": {...}, "batch_seconds": {"STEM": [...], ...},
                     "setup_seconds": .., "bench_seconds": .., "holdout_seconds": ..} or null,
         "info": {...}}                                    # free-form provenance
    <cache>/build.json           optional provenance written by ``adaptive-build``

``order`` fixes the candidate order of the search (ascending; ties by name).
``timing.batch_seconds[d][j]`` is the measured time to score look j's batch of
domain d (look 0 = the minimum sample) in the fixed question order described by
``timing.layout``; replay reports GPU time only when that layout matches its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluator import question_order


@dataclass
class ModelRecord:
    name: str
    role: str
    order: int
    correct: dict[str, int]
    latency: dict | None
    timing: dict | None
    info: dict


class ReplayCache:
    """Full cache.  Only the scorer and the replay driver touch this; policies never do."""

    def __init__(self, items: list[dict], dense: ModelRecord, candidates: list[ModelRecord], build: dict | None = None):
        self.items = items
        self.dense = dense
        self.candidates = candidates
        self.build = build or {}
        self.domain_of = {r["item_id"]: r["domain"] for r in items}
        self.tokens = {r["item_id"]: r.get("tokens") for r in items}
        self.pool: dict[str, list[str]] = {}
        self.holdout: dict[str, list[str]] = {}
        for r in items:
            (self.pool if r["split"] == "pool" else self.holdout).setdefault(r["domain"], []).append(r["item_id"])
        self._validate()

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {d: len(v) for d, v in self.pool.items()}

    def candidate(self, name: str) -> ModelRecord:
        for c in self.candidates:
            if c.name == name:
                return c
        raise KeyError(f"candidate {name!r} not in cache")

    def _validate(self) -> None:
        ids = [r["item_id"] for r in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("items.jsonl has duplicate item_id values")
        bad = {r["split"] for r in self.items} - {"pool", "holdout"}
        if bad:
            raise ValueError(f"unknown split values {bad} (use pool | holdout)")
        need = set(ids)
        for m in [self.dense, *self.candidates]:
            missing = need - set(m.correct)
            if missing:
                raise ValueError(f"model {m.name}: no correctness for {len(missing)} items, e.g. {sorted(missing)[:3]}")
            if any(v not in (0, 1) for v in m.correct.values()):
                raise ValueError(f"model {m.name}: correctness values must be 0 or 1")

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "ReplayCache":
        root = Path(path)
        with open(root / "items.jsonl", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        models = [_read_model(p) for p in sorted((root / "models").glob("*.json"))]
        dense = [m for m in models if m.role == "dense"]
        if len(dense) != 1:
            raise ValueError(f"{root}: need exactly one model with role 'dense', found {len(dense)}")
        cands = sorted((m for m in models if m.role == "candidate"), key=lambda m: (m.order, m.name))
        build = json.loads((root / "build.json").read_text(encoding="utf-8")) if (root / "build.json").exists() else None
        return cls(items, dense[0], cands, build)


def _read_model(p: Path) -> ModelRecord:
    d = json.loads(p.read_text(encoding="utf-8"))
    return ModelRecord(name=d["name"], role=d["role"], order=int(d.get("order", 0)),
                       correct={k: int(v) for k, v in d["correct"].items()}, latency=d.get("latency"),
                       timing=d.get("timing"), info=d.get("info") or {})


def write_items(root: Path, items: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / ".items.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in items), encoding="utf-8")
    tmp.replace(root / "items.jsonl")


def write_model(root: Path, rec: dict) -> Path:
    (root / "models").mkdir(parents=True, exist_ok=True)
    p = root / "models" / f"{rec['name']}.json"
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


# ---------------------------------------------------------------------------- the replay interface
class BudgetExceeded(RuntimeError):
    pass


class ReplaySource:
    """Reveals one candidate's cached results item by item, in each domain's fixed order.

    It is built from *pool* items only (the holdout never enters), keeps the
    answers in private attributes, and enforces the per-candidate budget.
    """

    def __init__(self, cache: ReplayCache, candidate: str, seed: int, budget: int):
        cand = cache.candidate(candidate)
        self.pool_sizes = cache.pool_sizes
        self._order = {d: question_order(ids, seed, d) for d, ids in cache.pool.items()}
        self._dense = {i: cache.dense.correct[i] for ids in cache.pool.values() for i in ids}
        self._cand = {i: cand.correct[i] for ids in cache.pool.values() for i in ids}
        self._cursor = {d: 0 for d in self._order}
        self._budget = int(budget)
        self.used = 0

    def reveal(self, domain: str, count: int) -> list[tuple[str, int, int]]:
        if self.used + count > self._budget:
            raise BudgetExceeded(f"{self.used} + {count} > budget {self._budget}")
        c = self._cursor[domain]
        ids = self._order[domain][c : c + count]
        if len(ids) != count:
            raise IndexError(f"{domain}: only {len(self._order[domain]) - c} items left, asked for {count}")
        self._cursor[domain] = c + count
        self.used += count
        return [(i, self._dense[i], self._cand[i]) for i in ids]
