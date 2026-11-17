"""SQLite key-value caches (crash-safe via WAL + synchronous=FULL).

Two uses:

* **Controller replay cache** - every LLM controller response is stored under
  ``iter{t}/attempt{a}``.  When a run resumes after a crash, the same iteration
  replays the *same* response instead of re-querying a stochastic model, so the
  resumed trajectory is identical to an uninterrupted one.
* **Evaluation cache** - per (model-state id, question id) answers and
  per-state scalar metrics (perplexity, latency).  A crash in the middle of a
  1,000-question test resumes without re-evaluating finished questions.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable


class KVCache:
    def __init__(self, path: str | Path, table: str = "kv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.table = table
        self._lock = threading.Lock()
        self._conn = self._connect()
        with self._conn:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (k TEXT PRIMARY KEY, v TEXT NOT NULL, ts REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        last_err = None
        for _ in range(10):
            try:
                conn = sqlite3.connect(str(self.path), timeout=60, isolation_level=None, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("PRAGMA busy_timeout=60000")
                ok = conn.execute("PRAGMA quick_check").fetchone()[0]
                if ok != "ok":
                    raise sqlite3.DatabaseError(f"quick_check failed: {ok}")
                return conn
            except sqlite3.DatabaseError as e:
                last_err = e
                if "malformed" in str(e) or "quick_check" in str(e) or "not a database" in str(e):
                    # Corrupt cache: caches are re-derivable, so quarantine and start fresh.
                    bad = self.path.with_suffix(self.path.suffix + f".corrupt-{int(time.time())}")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    try:
                        self.path.rename(bad)
                    except OSError:
                        pass
                    continue
                time.sleep(0.5)
        raise last_err  # type: ignore[misc]

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(f"SELECT v FROM {self.table} WHERE k=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        keys = list(keys)
        out: dict[str, Any] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                q = ",".join("?" * len(chunk))
                for k, v in self._conn.execute(f"SELECT k, v FROM {self.table} WHERE k IN ({q})", chunk):
                    out[k] = json.loads(v)
        return out

    def put(self, key: str, value: Any) -> None:
        self.put_many({key: value})

    def put_many(self, items: dict[str, Any]) -> None:
        if not items:
            return
        now = time.time()
        rows = [(k, json.dumps(v, sort_keys=True), now) for k, v in items.items()]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(f"INSERT OR REPLACE INTO {self.table} (k, v, ts) VALUES (?,?,?)", rows)
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def keys_with_prefix(self, prefix: str) -> list[str]:
        with self._lock:
            return [
                r[0]
                for r in self._conn.execute(
                    f"SELECT k FROM {self.table} WHERE k >= ? AND k < ? ORDER BY k", (prefix, prefix + "￿")
                )
            ]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
