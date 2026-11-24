"""Experiment-matrix supervisor.

Each run executes in its own child process (``python -m capprune run ...``), so
a hard crash - segfault, CUDA driver reset, OOM-killer, a hung kernel - only
kills the child.  The supervisor then:

* detects hangs: no heartbeat progress for ``hang_timeout_s`` -> graceful stop,
  then kill, then restart;
* restarts crashed runs with exponential backoff; the restart budget
  (``max_restarts``) only counts crashes with NO checkpoint progress, so a
  run that keeps advancing between crashes is never abandoned;
* treats exit 76 (LLM controller down) as "wait ``controller_wait_s`` and retry";
* never restarts config errors (2), full disks (74);
* waits for a run whose lock is held by another live worker (77);
* records everything in ``runs/_matrix/ledger.json`` - itself resumable: re-running
  ``capprune supervise`` skips finished runs and continues where it stopped;
* makes a final pass over failed runs (``retry_failed_passes``).

Ctrl+C on the supervisor forwards a graceful stop to the worker (which
checkpoints) and exits; nothing is lost.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import run_id, runs_dir
from .utils import runtime as rt
from .utils.io import atomic_write_json, read_json
from .utils.log import get_logger, setup_logging

log = get_logger("supervisor")


def heartbeat_ts(hb: dict, pid: int, started: float) -> float:
    """The heartbeat time of THIS worker, or 0. A heartbeat counts if its pid is the launched process, or if it was
    written by a worker that started after the launch: on Windows a venv's python.exe is a launcher, so the
    interpreter that writes the heartbeat has a different pid from the process the supervisor started (without
    this, every worker was killed ``hang_timeout_s`` after its start whatever its progress). A heartbeat left by an
    earlier, killed worker started before this launch and does not count."""
    if hb.get("pid") == pid or float(hb.get("started", 0.0)) >= started - 5.0:
        return float(hb["ts"])
    return 0.0


def build_plan(cfg: dict, methods=None, targets=None, seeds=None) -> list[dict]:
    mx = cfg.get("matrix", {})
    methods = methods or mx.get("methods") or list(cfg["methods"])
    unknown = [m for m in methods if m not in cfg["methods"]]
    if unknown:
        raise rt.ConfigError(f"unknown method(s) {unknown}; defined: {', '.join(cfg['methods'])}")
    targets = targets or mx.get("targets", [0.2, 0.4])
    seeds = seeds or mx.get("seeds", [0, 1, 2])
    plan, seen = [], set()
    # dense first (shared reference), then seed-major so every method gets seed 0 early
    for m in methods:
        if cfg["methods"][m].get("kind") == "dense" and "dense" not in seen:
            plan.append({"method": m, "target": 0.0, "seed": 0, "id": "dense"})
            seen.add("dense")
    for s in seeds:
        for t in targets:
            for m in methods:
                if cfg["methods"][m].get("kind") == "dense":
                    continue
                rid = run_id(m, float(t), int(s))
                if rid not in seen:
                    plan.append({"method": m, "target": float(t), "seed": int(s), "id": rid})
                    seen.add(rid)
    return plan


class Ledger:
    def __init__(self, root: Path):
        self.path = root / "_matrix" / "ledger.json"
        self.data = read_json(self.path) if self.path.exists() else {}

    def get(self, rid: str) -> dict:
        return self.data.setdefault(rid, {"status": "pending", "attempts": 0, "exits": []})

    def update(self, rid: str, **kw) -> None:
        rec = self.get(rid)
        rec.update(kw)
        rec["updated"] = time.time()
        atomic_write_json(self.path, self.data)


class Supervisor:
    def __init__(self, cfg: dict, config_path: str, overrides: list[str]):
        self.cfg = cfg
        self.config_path = str(Path(config_path).resolve())
        self.overrides = overrides
        self.root = runs_dir(cfg)
        self.root.mkdir(parents=True, exist_ok=True)
        setup_logging(self.root / "_matrix" / "supervisor.log")
        r = cfg.get("reliability", {})
        self.max_restarts = int(r.get("max_restarts", 5))
        self.backoff = list(r.get("restart_backoff_s", [30, 60, 120, 300, 600]))
        self.hang_timeout = float(r.get("hang_timeout_s", 1800))
        self.controller_wait = float(r.get("controller_wait_s", 900))
        self.retry_passes = int(r.get("retry_failed_passes", 1))
        self.grace = float(r.get("stop_grace_s", 120))
        self.ledger = Ledger(self.root)
        self._child: subprocess.Popen | None = None

    # ------------------------------------------------------------------ child management
    def _cmd(self, item: dict) -> list[str]:
        cmd = [sys.executable, "-m", "capprune", "run", "--config", self.config_path, "--method", item["method"],
               "--target", str(item["target"]), "--seed", str(item["seed"])]
        for o in self.overrides:
            cmd += ["--set", o]
        return cmd

    def _spawn(self, item: dict, attempt: int) -> subprocess.Popen:
        logdir = self.root / item["id"] / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        logf = open(logdir / f"attempt-{attempt:02d}.log", "a", encoding="utf-8")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        kw: dict = {"stdout": logf, "stderr": subprocess.STDOUT, "env": env}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        p = subprocess.Popen(self._cmd(item), **kw)
        p._capprune_log = logf  # type: ignore[attr-defined]
        return p

    def _stop_child(self, p: subprocess.Popen, grace: float) -> None:
        if p.poll() is not None:
            return
        try:
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(p.pid, signal.SIGTERM)
        except Exception as e:
            log.warning("could not signal worker %d: %s", p.pid, e)
        try:
            p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            log.error("worker %d did not stop within %.0fs; killing", p.pid, grace)
            try:
                if os.name == "nt":
                    p.kill()
                else:
                    os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
            p.wait()

    def _watch(self, item: dict, p: subprocess.Popen) -> int:
        hb_path = self.root / item["id"] / "heartbeat.json"
        started = time.time()
        while True:
            try:
                return p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if rt.stop_requested():
                log.warning("stop requested: asking worker %d to checkpoint and exit", p.pid)
                self._stop_child(p, self.grace)
                return p.returncode
            last = started
            try:
                last = max(last, heartbeat_ts(read_json(hb_path), p.pid, started))
            except Exception:
                pass
            if time.time() - last > self.hang_timeout:
                log.error("%s: no heartbeat for %.0fs (worker %d hung?) - restarting it", item["id"],
                          time.time() - last, p.pid)
                self._stop_child(p, min(self.grace, 60))
                return 124

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and not rt.stop_requested():
            time.sleep(min(2.0, end - time.time()))

    def _progress(self, rid: str) -> int:
        """Highest checkpoint number on disk - grows whenever the worker makes durable progress."""
        d = self.root / rid / "ckpt"
        best = -1
        if d.exists():
            for f in d.glob("ckpt-*.json"):
                try:
                    best = max(best, int(f.stem.split("-")[1]))
                except ValueError:
                    pass
        return best

    def _run_one(self, item: dict) -> str:
        rid = item["id"]
        restarts = 0
        while not rt.stop_requested():
            rec = self.ledger.get(rid)
            attempt = rec["attempts"] + 1
            self.ledger.update(rid, status="running", attempts=attempt, started=time.time())
            log.info("%s: starting attempt %d", rid, attempt)
            before = self._progress(rid)
            p = self._spawn(item, attempt)
            try:
                code = self._watch(item, p)
            finally:
                p._capprune_log.close()  # type: ignore[attr-defined]
            rec = self.ledger.get(rid)
            self.ledger.update(rid, exits=(rec["exits"] + [code])[-20:])
            if code == rt.EXIT_OK:
                self.ledger.update(rid, status="done", finished=time.time())
                log.info("%s: done", rid)
                return "done"
            if code in (rt.EXIT_CONFIG, rt.EXIT_DISK):
                self.ledger.update(rid, status="failed", last_error=f"exit {code} (not restartable; see logs)")
                log.error("%s: exit %d is not restartable - see %s", rid, code, self.root / rid / "logs")
                return "failed"
            if code == rt.EXIT_INTERRUPTED and rt.stop_requested():
                self.ledger.update(rid, status="interrupted")
                return "interrupted"
            if code == rt.EXIT_LOCKED:
                log.warning("%s: locked by another live worker; waiting for it", rid)
                self._wait_unlock(rid)
                continue
            if code == rt.EXIT_CONTROLLER_UNAVAILABLE:
                log.warning("%s: controller unavailable; retrying in %.0fs", rid, self.controller_wait)
                self.ledger.update(rid, status="waiting_controller")
                self._sleep(self.controller_wait)
                continue  # controller outages do not consume the restart budget
            if self._progress(rid) > before:
                restarts = 0  # crashed, but after durable progress: not a crash loop
            restarts += 1
            if restarts > self.max_restarts:
                self.ledger.update(rid, status="failed", last_error=f"exit {code} after {restarts - 1} restarts")
                log.error("%s: giving up after %d restarts (last exit %s)", rid, restarts - 1, code)
                return "failed"
            wait = self.backoff[min(restarts - 1, len(self.backoff) - 1)]
            log.warning("%s: worker exited with %s; restart %d/%d in %ds (resumes from checkpoint)",
                        rid, code, restarts, self.max_restarts, wait)
            self.ledger.update(rid, status="restarting")
            self._sleep(wait)
        return "interrupted"

    def _wait_unlock(self, rid: str) -> None:
        lock = self.root / rid / "LOCK"
        while lock.exists() and not rt.stop_requested():
            try:
                other = read_json(lock)
                if not rt.pid_alive(int(other.get("pid", -1))):
                    return
            except Exception:
                return
            time.sleep(10)

    # ------------------------------------------------------------------ matrix
    def run(self, plan: list[dict]) -> int:
        lock = rt.RunLock(self.root / "_matrix")
        try:
            lock.acquire()
        except rt.RunLockedError as e:
            log.error("another supervisor is already running this matrix (%s); exiting", e)
            return rt.EXIT_LOCKED
        try:
            return self._run(plan)
        finally:
            lock.release()

    def _run(self, plan: list[dict]) -> int:
        rt.install_signal_handlers()
        log.info("plan: %d runs -> %s", len(plan), self.root)
        for pass_ in range(1 + self.retry_passes):
            for item in plan:
                if rt.stop_requested():
                    break
                rec = self.ledger.get(item["id"])
                if (self.root / item["id"] / "DONE").exists():
                    if rec["status"] != "done":
                        self.ledger.update(item["id"], status="done")
                    continue
                if pass_ > 0 and rec["status"] != "failed":
                    continue
                if pass_ > 0:
                    log.info("retry pass %d for failed run %s", pass_, item["id"])
                self._run_one(item)
        statuses = {i["id"]: self.ledger.get(i["id"])["status"] for i in plan}
        n_done = sum(s == "done" for s in statuses.values())
        log.info("matrix: %d/%d runs done; %s", n_done, len(plan),
                 ", ".join(f"{k}={v}" for k, v in statuses.items() if v != "done") or "all complete")
        if rt.stop_requested():
            return rt.EXIT_INTERRUPTED
        return 0 if n_done == len(plan) else 1
