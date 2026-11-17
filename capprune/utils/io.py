"""Crash-safe file primitives.

Every state file CAP-Prune writes goes through these helpers so that a crash,
power loss, or a killed process never leaves a half-written file behind:

* ``atomic_write_*`` writes to a temporary file in the same directory, flushes
  and fsyncs it, then ``os.replace``-s it over the destination.  ``os.replace``
  is atomic on both NTFS and POSIX filesystems, so readers see either the old
  or the new file, never a mix.
* On Windows, ``os.replace`` raises ``PermissionError`` when another process
  (OneDrive, antivirus, the search indexer) briefly holds the destination
  open.  We retry with backoff instead of failing the run.
* ``append_jsonl`` appends one record per line and fsyncs; ``read_jsonl``
  tolerates a torn final line (the only possible damage from a crash mid-append).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator

_REPLACE_RETRIES = 12


class DiskSpaceError(RuntimeError):
    """Raised when the output volume is too full to write safely."""


def _fsync_dir(path: Path) -> None:
    # Directory fsync makes the rename itself durable on POSIX. Windows has no
    # equivalent (and NTFS journals metadata), so it is a no-op there.
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _replace_with_retry(src: str, dst: str) -> None:
    delay = 0.05
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp, str(path))
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int | None = 2) -> None:
    text = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False, default=_json_default)
    atomic_write_text(path, text + "\n")


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_sealed_json(path: str | os.PathLike, payload: Any) -> str:
    """Write ``{"payload": ..., "sha256": ...}`` so corruption is detectable on read."""
    digest = sha256_text(canonical_json(payload))
    atomic_write_json(path, {"payload": payload, "sha256": digest})
    return digest


def read_sealed_json(path: str | os.PathLike) -> Any:
    """Read a sealed JSON file; raises ``ValueError`` if the checksum does not match."""
    obj = read_json(path)
    if not isinstance(obj, dict) or "payload" not in obj or "sha256" not in obj:
        raise ValueError(f"{path}: not a sealed JSON file")
    if sha256_text(canonical_json(obj["payload"])) != obj["sha256"]:
        raise ValueError(f"{path}: checksum mismatch (file is corrupt)")
    return obj["payload"]


def append_jsonl(path: str | os.PathLike, record: dict, fsync: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A torn final line from a crash mid-append; everything before it is intact.
                continue


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def short_hash(*parts: Any, n: int = 16) -> str:
    return sha256_text("\x1f".join(canonical_json(p) for p in parts))[:n]


def ensure_free_space(path: str | os.PathLike, need_bytes: int, min_free_gb: float) -> None:
    path = Path(path)
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(str(probe)).free
    if free < need_bytes + int(min_free_gb * (1 << 30)):
        raise DiskSpaceError(
            f"Only {free / (1 << 30):.1f} GiB free on the volume holding {path}; need "
            f"{need_bytes / (1 << 30):.2f} GiB plus a {min_free_gb} GiB safety margin."
        )


def finalize_tmp(tmp: str | os.PathLike, dst: str | os.PathLike) -> None:
    """fsync a fully written temp file and atomically move it into place (with Windows retry)."""
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    _replace_with_retry(str(tmp), str(dst))
    _fsync_dir(Path(dst).parent)


def copy_file_atomic(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        shutil.copyfile(src, tmp)
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        _replace_with_retry(str(tmp), str(dst))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remove_quietly(paths: Iterable[str | os.PathLike]) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass
