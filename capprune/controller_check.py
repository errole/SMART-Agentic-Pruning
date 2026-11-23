"""One real controller proposal, validated, outside any experiment journal (``capprune controller-check``).

Takes a representative CAP-Prune prompt - the exact system/user prompt a previous run sent at some
iteration, read through a read-only SQLite connection so that run's cache is never modified - sends it
through the experiment's own adapter (``LLMController``, with a throwaway cache in ``out_dir``) and checks
the reply with the repo's parser (``parse_response``) and ``sanitize``:

* the JSON parsed on the first attempt (no repair re-ask) and has reasoning / stop_pruning / layer_decisions;
* every proposed unit exists in the prompt's unit table;
* every raw increment lies in [min_increment, max_increment] and keeps the unit under its sparsity cap;
* what ``sanitize`` (the loop's real post-processing) would change, if anything.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from .utils.io import atomic_write_json


def read_prompt(cache_db: Path, iteration: int, work_dir: Path) -> dict:
    """Copy the run's cache (and any WAL) into ``work_dir`` and read the copy, so the run's directory is never
    touched (even a read-only SQLite connection can create -shm/-wal files next to a WAL database)."""
    import shutil

    work_dir.mkdir(parents=True, exist_ok=True)
    copy = work_dir / "prompt-source.sqlite"
    shutil.copy2(cache_db, copy)
    if Path(f"{cache_db}-wal").exists():
        shutil.copy2(f"{cache_db}-wal", f"{copy}-wal")
    con = sqlite3.connect(f"file:{copy.as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT v FROM kv WHERE k = ?", (f"iter{iteration:06d}/a0",)).fetchone()
    finally:
        con.close()
    if row is None:
        raise SystemExit(f"{cache_db}: no cached prompt for iteration {iteration}")
    v = json.loads(row[0])
    return {"system": v["system"], "user": v["user"], "source_model": v["meta"].get("model")}


class _StubState:
    """Just enough of PruneState for ``sanitize``: units, shares and current sparsities from the prompt table."""

    def __init__(self, table: dict[str, tuple[float, float]], mode: str):
        self.by_name = {n: None for n in table}
        self._t = table
        self.mode = mode
        self.nm = (2, 4)

    def unit_sparsity(self, n):
        return self._t[n][1]

    def share(self, n):
        return self._t[n][0]

    def global_sparsity(self):
        return sum(s * sp for s, sp in self._t.values())


def check(cfg: dict, method: str, prompt_db: Path, iteration: int, out_dir: Path) -> dict:
    from .config import run_config
    from .search.controller import LLMController, sanitize
    from .utils.cache import KVCache
    from .utils.runtime import ControllerUnavailable

    rc = run_config(cfg, method, float(cfg["matrix"]["targets"][0]), int(cfg["matrix"]["seeds"][0]))
    p = read_prompt(prompt_db, iteration, out_dir)
    table = {m.group(1): (float(m.group(2)), float(m.group(3)))
             for m in re.finditer(r"^(L\d+\.\w+) \| ([\d.]+) \| \S+ \| \S+ \| ([\d.]+)$", p["user"], re.M)}
    s = rc["search"]
    lo, hi = float(s["min_increment"]), float(s["max_increment"])
    mode = rc["pruning"]["mode"]
    cap = float(rc["pruning"]["structured"]["max_unit_sparsity"] if mode == "structured" else rc["pruning"]["max_unit_sparsity"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cache = KVCache(out_dir / f"controller-check-{stamp}.sqlite")  # throwaway: not an experiment cache
    ctl = LLMController(rc, cache)
    res = {"endpoints": [e["base_url"] for e in ctl.endpoints], "model": ctl.model, "prompt_source": str(prompt_db),
           "prompt_iteration": iteration, "prompt_originally_sent_to": p["source_model"],
           "prompt_chars": len(p["system"]) + len(p["user"]), "units_in_table": len(table)}
    t0 = time.time()
    try:
        parsed, meta = ctl.propose(iteration, p["system"], p["user"])
    except ControllerUnavailable as e:
        res.update(ok=False, error=str(e)[:2000], seconds=time.time() - t0)
        cache.close()
        atomic_write_json(out_dir / f"controller-check-{stamp}.json", res)
        return res
    raw = cache.get(f"iter{iteration:06d}/a0")["raw"]
    cache.close()
    usage = meta.get("usage") or {}
    decs = parsed["layer_decisions"]
    unknown = [d["layer"] for d in decs if d["layer"] not in table]
    out_of_range = [d for d in decs if not lo - 1e-9 <= d["additional_sparsity"] <= hi + 1e-9]
    over_cap = [d for d in decs if d["layer"] in table and table[d["layer"]][1] + d["additional_sparsity"] > cap + 1e-9]
    dup = sorted({d["layer"] for d in decs if sum(x["layer"] == d["layer"] for x in decs) > 1})
    sanitized, notes = sanitize(parsed, _StubState(table, mode), rc, float(rc["run"]["target"]))
    res.update(
        seconds=round(time.time() - t0, 2), usage=usage, repaired="first" in usage, raw_response=raw[:4000],
        parsed={"reasoning": parsed["reasoning"][:600], "stop_pruning": parsed["stop_pruning"], "layer_decisions": decs},
        checks={"json_complete_first_try": "first" not in usage and isinstance(parsed.get("reasoning"), str)
                and isinstance(parsed.get("stop_pruning"), bool) and isinstance(decs, list),
                "has_decisions": bool(decs) or parsed["stop_pruning"],
                "unknown_units": unknown, "increments_outside_range": out_of_range, "over_unit_cap": over_cap,
                "duplicate_units": dup, "allowed_range": [lo, hi], "unit_cap": cap},
        sanitize={"kept": sanitized, "notes": notes,
                  "planned_global_gain": sum(table[n][0] * d for n, d in sanitized)})
    c = res["checks"]
    res["ok"] = bool(c["json_complete_first_try"] and c["has_decisions"] and not unknown and not out_of_range
                     and not over_cap and not dup)
    atomic_write_json(out_dir / f"controller-check-{stamp}.json", res)
    return res
