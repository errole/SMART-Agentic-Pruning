"""Process-level reliability: run lock, heartbeat, graceful stop, exit codes.

Exit codes are a contract with the supervisor (``capprune.supervisor``):

====  ===========================================================
 0    run finished
 2    configuration/usage error            -> do not restart
 74   disk full / unrecoverable I/O        -> do not restart
 75   graceful stop requested (signal)     -> resume later
 76   LLM controller unavailable (paused)  -> restart after a long wait
 77   another live process holds the lock  -> do not restart
 *    anything else (crash, OOM kill)      -> restart with backoff
====  ===========================================================
"""
from __future__ import annotations

import os
import signal
import socket
import threading
import time
from pathlib import Path

from .io import atomic_write_json, read_json

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DISK = 74
EXIT_INTERRUPTED = 75
EXIT_CONTROLLER_UNAVAILABLE = 76
EXIT_LOCKED = 77
NO_RESTART_CODES = {EXIT_OK, EXIT_CONFIG, EXIT_DISK, EXIT_LOCKED}


class GracefulExit(Exception):
    """Raised at a safe point after a stop signal was received."""


class ControllerUnavailable(Exception):
    """All controller endpoints failed; the run pauses and resumes later."""


class ConfigError(Exception):
    pass


# --------------------------------------------------------------------------- stop flag
_stop_event = threading.Event()
_signal_count = 0


def _handler(signum, frame):  # pragma: no cover - exercised via subprocess tests
    global _signal_count
    _signal_count += 1
    _stop_event.set()
    if _signal_count >= 2:
        # Second signal: the user really wants out. Restore default and re-raise.
        raise KeyboardInterrupt


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGBREAK"):  # Windows: CTRL_BREAK_EVENT from the supervisor
        signal.signal(signal.SIGBREAK, _handler)


def stop_requested() -> bool:
    return _stop_event.is_set()


def request_stop() -> None:
    _stop_event.set()


def check_stop() -> None:
    """Call at safe points. Raises ``GracefulExit`` once a stop was requested."""
    if _stop_event.is_set():
        raise GracefulExit()


# --------------------------------------------------------------------------- pid liveness
def pid_alive(pid: int) -> bool:
    """Cross-platform liveness check.

    Never use ``os.kill(pid, 0)`` on Windows: there it calls TerminateProcess.
    """
    if pid <= 0:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False
    except ImportError:  # pragma: no cover
        pass
    if os.name == "nt":  # pragma: no cover
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLockedError(RuntimeError):
    pass


class RunLock:
    """Exclusive lock on a run directory.

    Prevents two workers from writing the same run (e.g. a supervisor restart
    racing a still-alive worker).  A lock left behind by a dead process is
    detected via its PID (and hostname) and taken over.
    """

    def __init__(self, run_dir: str | os.PathLike):
        self.path = Path(run_dir) / "LOCK"
        self.held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        me = {"pid": os.getpid(), "host": socket.gethostname(), "since": time.time()}
        for _ in range(3):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    other = read_json(self.path)
                except Exception:
                    other = {}
                same_host = other.get("host") == me["host"]
                if same_host and other.get("pid") and pid_alive(int(other["pid"])) and other["pid"] != me["pid"]:
                    raise RunLockedError(
                        f"{self.path.parent} is locked by live process {other['pid']} on {other.get('host')}"
                    )
                if not same_host and other and time.time() - float(other.get("since", 0)) < 24 * 3600:
                    raise RunLockedError(
                        f"{self.path.parent} is locked by host {other.get('host')} (remove LOCK if that host is gone)"
                    )
                # Stale lock: owner is dead. Remove and retry.
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                import json

                json.dump(me, f)
                f.flush()
                os.fsync(f.fileno())
            self.held = True
            return
        raise RunLockedError(f"could not acquire {self.path}")

    def release(self) -> None:
        if self.held:
            try:
                other = read_json(self.path)
                if other.get("pid") == os.getpid():
                    os.unlink(self.path)
            except Exception:
                pass
            self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class Heartbeat:
    """Progress heartbeat read by the supervisor to detect hangs.

    ``beat`` is called from the main thread at natural progress points (every
    evaluation batch, every iteration).  Writes are throttled.  Because the
    beat comes from the main thread, a hung CUDA call or a stuck HTTP request
    stops the heartbeat and the supervisor can kill and restart the worker.
    """

    def __init__(self, run_dir: str | os.PathLike, min_interval_s: float = 5.0):
        self.path = Path(run_dir) / "heartbeat.json"
        self.min_interval = min_interval_s
        self._last: float | None = None  # monotonic: a step of the wall clock must not silence the heartbeat
        self.phase = "start"
        self.started = time.time()
        self.launch = os.environ.get("CAPPRUNE_LAUNCH_ID")  # set by the supervisor for the worker it launched
        self.seq = 0

    def beat(self, phase: str | None = None, detail: str = "", force: bool = False) -> None:
        mono = time.monotonic()
        if phase is not None and phase != self.phase:
            self.phase = phase
            force = True
        if not force and self._last is not None and mono - self._last < self.min_interval:
            return
        self._last = mono
        self.seq += 1
        try:
            atomic_write_json(
                self.path,
                {"pid": os.getpid(), "ts": time.time(), "phase": self.phase, "detail": detail, "started": self.started,
                 "launch": self.launch, "seq": self.seq},
                indent=None,
            )
        except OSError:
            pass  # a heartbeat must never crash the run

